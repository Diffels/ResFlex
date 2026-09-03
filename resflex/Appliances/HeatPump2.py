import pandas as pd
import numpy as np
import os
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ..stochastic import uniform_probability_range


@dataclass
class House:
    """Dataclass representing a house with thermal properties"""
    year: str
    floors: int
    ground_surface: float
    wall_surface: float          # opaque external wall area [m2] (glazing removed)
    volume: float
    north_window_surface: float
    east_window_surface: float
    south_window_surface: float
    west_window_surface: float
    tot_window_surface: float
    U_tot: float                 # full steady-state fabric loss coeff [W/K] (walls+roof+floor+glazing)
    U_wall: float                # wall U-value [W/m2K]
    C_env: float                 # envelope thermal capacity [J/K]
    C_air: float                 # air thermal capacity [J/K]

    @staticmethod
    def generate(year, surface, floors):
        """Generate a house and assign thermal properties.

        `surface` is the total heated floor area [m2]; `floors` storeys of 2.5 m.
        Roof and ground-floor areas are the *footprint* (surface / floors), not the
        total floor area, and external wall area is the footprint perimeter times the
        full building height -- the previous version used `surface` for all three,
        overstating the envelope ~`floors`x.
        """
        if year < 1945: year = '< 45'
        elif year < 1970: year = '45-70'
        elif year < 1990: year = '70-90'
        elif year < 2007: year = '90-07'
        else: year = '> 08'
        # Raw data: ProCEBaR (Task 2) -- https://orbi.uliege.be/bitstream/2268/192397/2/160126_BERA_ULg.pdf
        U_wall   = {'< 45': 2.25, '45-70': 1.56, '70-90': 0.98, '90-07': 0.49, '> 08': 0.4}
        U_window = {'< 45': 5.0,  '45-70': 5.0,  '70-90': 3.5,  '90-07': 3.5,  '> 08': 2.0}
        U_roof   = {'< 45': 4.15, '45-70': 3.33, '70-90': 0.77, '90-07': 0.43, '> 08': 0.3}
        U_floor  = {'< 45': 3.38, '45-70': 3.38, '70-90': 1.14, '90-07': 0.73, '> 08': 0.4}
        K_wall   = {'< 45': 76466, '45-70': 74715, '70-90': 75945, '90-07': 75022, '> 08': 74834}
        K_roof   = {'< 45': 7211,  '45-70': 11357, '70-90': 11922, '90-07': 12848, '> 08': 14356}
        K_floor  = {'< 45': 67352, '45-70': 67352, '70-90': 62673, '90-07': 69245, '> 08': 69246}

        b_ground = 0.6                                   # ground-contact loss reduction (EN 12831), account for the fact that the floor slab doesn't lose heat to the outdoor air temperature, but to the ground.

        volume = surface * 2.5
        ground_surface = surface / floors                # footprint = roof area = ground-floor area
        perimeter = 4.0 * (ground_surface ** 0.5)        # square footprint
        gross_wall = round(perimeter * floors * 2.5, 2)  # external wall (opaque + glazed)

        window_north = max(0.0, uniform_probability_range(-0.1, 0.2) * gross_wall / 4)
        window_south = max(0.0, uniform_probability_range(-0.1, 0.3) * gross_wall / 4)
        window_east  = max(0.0, uniform_probability_range(-0.1, 0.3) * gross_wall / 4)
        window_west  = max(0.0, uniform_probability_range(-0.1, 0.3) * gross_wall / 4)
        window_tot = round(window_north + window_east + window_south + window_west, 2)

        wall_surface = max(gross_wall - window_tot, 0.0)  # opaque wall only

        U_tot = (U_wall[year]  * wall_surface + U_roof[year] * ground_surface + U_floor[year] * b_ground * ground_surface + U_window[year] * window_tot)

        C_env = (K_wall[year] * wall_surface + K_roof[year] * ground_surface + K_floor[year] * ground_surface)
        
        C_air = 1.2 * volume * 1005                       # rho * V * cp

        return House(year=year, floors=floors, ground_surface=ground_surface, wall_surface=wall_surface,
                     volume=volume, north_window_surface=window_north, east_window_surface=window_east,
                     south_window_surface=window_south, west_window_surface=window_west,
                     tot_window_surface=window_tot, U_tot=U_tot, U_wall=U_wall[year],
                     C_env=C_env, C_air=C_air)

    def display(self):
        print(f"Year of construction: {self.year}  floors: {self.floors}  footprint: {self.ground_surface:.0f} m2")
        print(f"Wall surface: {self.wall_surface:.0f} m2  Windows: {self.tot_window_surface:.0f} m2")
        print(f"UA_tot: {self.U_tot:.0f} W/K  C_env: {self.C_env/1e6:.1f} MJ/K")


def add_params_HP(config):
    house = House.generate(config['HP_data']['Year'], config['HP_data']['Size'], config['HP_data']['Floors'])
    config['HP_data']['U_wall'] = house.U_wall
    config['HP_data']['C_env'] = house.C_env
    config['HP_data']['C_air'] = house.C_air
    config['HP_data']['U_tot'] = house.U_tot
    config['HP_data']['window_surface'] = house.tot_window_surface
    config['HP_data']['wall_surface'] = house.wall_surface
    config['HP_data']['volume'] = house.volume
    config['HP_data']['north_window_surface'] = house.north_window_surface
    config['HP_data']['east_window_surface'] = house.east_window_surface
    config['HP_data']['south_window_surface'] = house.south_window_surface
    config['HP_data']['west_window_surface'] = house.west_window_surface
    return config


_weather_cache = {}

def _load_weather(weather_path):
    """Cache the weather Excel file so it's only read from disk once per run, not once per household."""
    if weather_path not in _weather_cache:
        _weather_cache[weather_path] = pd.read_excel(weather_path)
    return _weather_cache[weather_path]


def weather_import(house, weather_path, start_day, nb_days):
    """Return (T_out [degC], Q_solar [W]) at 1-min resolution for the simulated window.

    SF: Solar Factor ~ g_value * frame_factor * mean_shading (~0.5 * 0.7 * 0.85).
    """
    SF = 0.3
    weather = _load_weather(weather_path)
    T_out = np.repeat(weather['Temperature C'].values, 60)
    irr_n = np.repeat(weather['I_north W/m²'].values, 60)
    irr_e = np.repeat(weather['I_east W/m²'].values, 60)
    irr_s = np.repeat(weather['I_south W/m²'].values, 60)
    irr_w = np.repeat(weather['I_west W/m²'].values, 60)
    Q_sol = (house['north_window_surface'] * irr_n
             + house['east_window_surface'] * irr_e
             + house['south_window_surface'] * irr_s
             + house['west_window_surface'] * irr_w) * SF
    sl = slice(start_day * 24 * 60, (start_day + nb_days) * 24 * 60)
    return T_out[sl], Q_sol[sl]


# =============================================================================
#  Space-heating RC model
# =============================================================================

def HP_water_heating_curve_supply_temp(T_out, t_sup_cold=45.0, t_sup_mild=32.0, t_out_cold=-7.0, t_out_mild=15.0):
    """Linear interpolation of the heat pump supply temperature as a function of outdoor temperature, 
    with a minimum supply temperature at the coldest outdoor temperature and a maximum supply temperature 
    at the mildest outdoor temperature."""

    frac = np.clip((T_out - t_out_cold) / (t_out_mild - t_out_cold), 0.0, 1.0)
    return t_sup_cold + frac * (t_sup_mild - t_sup_cold)


def cop_curve(cop_rated, t_out_rated=7.0, cop_min=1.6, cop_max=5.5):
    """COP(T_out, T_supply) on a Carnot law, calibrated so it equals `cop_rated` at
    the rating point (input) (outdoor +7 degC, supply = HP_water_heating_curve_supply_temp(7))."""
    
    tsup_r = HP_water_heating_curve_supply_temp(t_out_rated)
    eta = cop_rated * max(tsup_r - t_out_rated, 1.0) / (tsup_r + 273.15)

    def cop(T_out, T_supply):
        return float(np.clip(eta * (T_supply + 273.15) / max(T_supply - T_out, 1.0),cop_min, cop_max))
    
    return cop # Returning the function itself, not the value at a specific T_out


def heating_dynamics(house, sim_days, T_set, T_out, Q_solar, P_nom_elec,
                     cop_fn=None, Q_internal=None, dt=60.0,
                     ACH=0.4, deadband=0.3, f_air_mass=2.0, aux_kW=None):
    """Two-capacity (air + envelope) RC space-heating model with a modulating,
    outdoor-temperature-dependent heat pump and resistive backup.

        C_air  dT_in/dt   = Q_dem + Q_gain - UA_fast*(T_in - T_out) - h_w*(T_in - T_wall)
        C_wall dT_wall/dt =                   h_w*(T_in - T_wall)    - h_w*(T_wall - T_out)

        h_w     = 2 * U_wall * A_wall               -> walls add exactly U_wall*A_wall
        UA_fast = (U_tot - U_wall*A_wall)           -> roof + floor + glazing ...
                  + ACH/3600 * rho*cp * volume      -> ... + ventilation, straight to outside

    Steady state: Q_dem = (U_tot + UA_vent)*(T_in - T_out) - Q_gain, i.e. the *full*
    building UA. (The old model put walls and everything-else in series through the
    wall-only resistance and lost ~6x of the loss.)

    Control: "ideal load" thermostat -- each minute solve for the power that lands
    T_in on the setpoint, clip to HP capacity P_nom_elec*COP(T_out) plus `aux_kW` of
    COP-1 resistive backup. Unconditionally stable for the air node at any `dt`.

    Returns
    -------
    P_elec [kW]  : total electrical draw (heat pump + backup), length sim_days*1440
    T_in  [degC], T_wall [degC]
    diag  (dict) : 'COP', 'P_hp_elec' [kW], 'P_aux_elec' [kW], 'Q_loss' [kW]
    """
    N = sim_days * 24 * 60

    UA_wall = house['U_wall'] * house['wall_surface']   # Model the walls as a single 'slow' node
    UA_others = max(house['U_tot'] - UA_wall, 0.0)      # Model everything else (roof, floor, glazing) as a single 'fast' node
    UA_vent = ACH / 3600.0 * 1.2 * 1005.0 * house['volume']
    UA_fast = UA_others + UA_vent # Air-Outside heat transfer coefficient, W/K, so the fast node adds exactly U_tot - U_wall*A_wall to the total UA
    h_w = 2.0 * UA_wall # Air-wall-Air heat transfer coefficient, W/K, so the walls add exactly U_wall*A_wall to the total UA

    C_air = house['C_air'] * f_air_mass          # + furniture / internal partitions
    C_wall = house['C_env']
    floor_area = house['ground_surface'] * house['floors']

    T_out = np.asarray(T_out, float)[:N]
    Q_sol = np.asarray(Q_solar, float)[:N]

    if not Q_internal:
        Q_int = np.full(N, house.get('q_int_W_per_m2', 3.0) * float(floor_area))
    else:
        Q_int = np.asarray(Q_internal, float)[:N]
        
    Q_gain = Q_int + Q_sol
    T_set = np.asarray(T_set, float)

    if not cop_fn:
        cop_fn = cop_curve(house.get('COP', 3.0))
    aux_cap_W = (aux_kW if aux_kW is not None else P_nom_elec) * 1e3

    P_hp = np.zeros(N)
    P_aux = np.zeros(N)
    COP = np.zeros(N)
    T_in = np.zeros(N)
    T_wall = np.zeros(N)
    Q_loss = np.zeros(N)

    T_in[0] = T_set[0]
    T_wall[0] = 0.5 * (T_set[0] + T_out[0])

    for k in range(1, N):
        Ti, Tw, To, Ts = T_in[k - 1], T_wall[k - 1], T_out[k - 1], T_set[k - 1]
        COP[k] = cop_fn(To, HP_water_heating_curve_supply_temp(To))

        q_hp_cap = P_nom_elec * 1e3 * COP[k]
        q_cap = q_hp_cap + aux_cap_W

        q_ideal = (C_air * (Ts - Ti) / dt
                   + UA_fast * (Ti - To)
                   + h_w * (Ti - Tw)
                   - Q_gain[k - 1])
        q_dem = 0.0 if (Ts - Ti) < -deadband else min(max(q_ideal, 0.0), q_cap)

        q_hp = min(q_dem, q_hp_cap)
        P_hp[k] = q_hp / COP[k] / 1e3
        P_aux[k] = (q_dem - q_hp) / 1e3

        T_in[k] = Ti + dt / C_air * (q_dem + Q_gain[k - 1] - UA_fast * (Ti - To) - h_w * (Ti - Tw))
        T_wall[k] = Tw + dt / C_wall * (h_w * (Ti - Tw) - h_w * (Tw - To))
        Q_loss[k] = (UA_fast * (Ti - To) + h_w * (Ti - Tw)) / 1e3

    diag = {'COP': COP, 'P_hp_elec': P_hp, 'P_aux_elec': P_aux, 'Q_loss': Q_loss}
    return P_hp + P_aux, T_in, T_wall, diag


def HP_simulate(T_set, config):
    hp = config['HP_data']
    weather_path = os.path.join(os.path.dirname(__file__), 'database', 'Meteo2022_Liege.xlsx')
    T_out, Q_solar = weather_import(hp, weather_path, config['start_day'], config['nb_days'])

    cop_fn = cop_curve(hp.get('COP', 3.0))

    P_elec, T_in, T_wall, diag = heating_dynamics(
        hp, config['nb_days'], np.asarray(T_set, float), T_out, Q_solar,
        P_nom_elec=hp['P_nom'], cop_fn=cop_fn, Q_internal=None,
        ACH=hp.get('ACH', 0.4), aux_kW=hp.get('aux_kW', None),
    )

    n = len(T_in)
    Flex_HP = pd.DataFrame({
        'T_set_HP': np.asarray(T_set, float)[:n],
        'T_ref_HP': T_in,
        'T_wall_HP': T_wall,
        'T_out_HP': np.asarray(T_out, float)[:n],
        'P_loss_HP': diag['Q_loss'],
        'COP_HP': diag['COP'],
        'P_aux_HP': diag['P_aux_elec'],
    })
    return P_elec, Flex_HP          # NB: now ELECTRICAL kW (was thermal)
