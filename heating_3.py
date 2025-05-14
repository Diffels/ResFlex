import pandas as pd
import numpy as np
import os
import random
from scipy.integrate import solve_ivp
from dataclasses import dataclass


# Find the path to this file
file_path = os.path.dirname(os.path.realpath(__file__)) 
# Create an absolute path to the Excel file 'Meteo2022_Liege.xlsx'
weather_path = os.path.join(file_path, 'database/Meteo2022_Liege.xlsx')

@dataclass
class Material:
    """Dataclass to store material properties"""
    density: float  # rho kg/m3
    specific_heat: float  # cp J/(kg.K)
    thermal_conductivity: float  # k in W/(m.K)

# Define common materials, see: (2024) "ENRG0001-1 : ENERGY CHALLENGE - Building stock modelling" by Castiaux J. et al
MATERIALS = {
    "air": Material(1.225, 1005, 0.024),
    "polystyrene": Material(55, 1210, 0.042),
    "rock_wool": Material(28, 900, 0.035),
    "liege": Material(120, 1800, 0.039),
    "brick": Material(1920, 835, 0.72),
    "concrete": Material(2600, 640, 0.6),
    "reinforced_concrete": Material(3000, 840, 0.6),
}

@dataclass
class ThermalProperties:
    """Dataclass to store calculated thermal properties"""
    U_walls: float
    U_windows: float
    U_roof: float
    U_ground: float
    ach: float  # Air changes per hour
    C_wall: float
    C_roof: float
    C_ground: float
    theta: float

@dataclass
class House:
    """Dataclass representing a house with thermal properties"""
    year_of_construction: str
    num_floors: int
    ground_surface: float
    ceiling_height: float
    wall_surface: float
    volume: float
    north_window_surface: float
    east_window_surface: float
    south_window_surface: float
    west_window_surface: float
    tot_window_surface: float
    thermal_properties: ThermalProperties

    @staticmethod
    def generate():
        """Generate a random house and assign thermal properties"""
        construction_periods = ['70-80', '80-2000', '2000-2020', 'after-2020']
        areas = {1: [90, 100, 110, 120, 130, 140, 150],  # Single-story houses
                 2: [60, 70, 80, 90, 100, 110],  # Two-story houses
                 3: [50, 60, 70, 80, 90]}  # Three-story houses
        heights = {1: 2.5, 2: 5, 3: 7.5}  # Heights by floors
        
        num_floors = random.choice([1, 2, 3])
        ground_surface = random.choice(areas[num_floors])/2  # Assume half of the area is used
        # print("Ground surface: ", ground_surface*2, "[m2]")
        ceiling_height = heights[num_floors]
        
        # Derived properties
        volume = round(ground_surface * ceiling_height, 2)
        perimeter = round(4 * (ground_surface ** 0.5), 2)  # Assume square-shaped
        wall_surface = round(perimeter * ceiling_height, 2)
        
        # Window surfaces
        window_north = random.uniform(0.1, 0.2) * wall_surface / 4
        window_south = random.uniform(0.1, 0.3) * wall_surface / 4
        window_east = random.uniform(0.1, 0.3) * wall_surface / 4
        window_west = random.uniform(0.1, 0.3) * wall_surface / 4
        window_tot = round(window_north+window_east+window_south+window_west, 2)

        year_of_construction = random.choice(construction_periods)
        thermal_properties = calculate_thermal_properties(year_of_construction)

        '''print(f"-- Generated House Details --")
        print(f"Year of Construction: {year_of_construction}")
        print(f"Number of Floors: {num_floors}")
        print(f"Ground Surface: {ground_surface} m2")
        print(f"Ceiling Height: {ceiling_height} m")
        print(f"Total Volume: {volume} m3")
        print(f"Wall Surface: {wall_surface} m2")
        print(f"Windows Surface (tot): {window_tot} m2")'''
        
        return House(
            year_of_construction=year_of_construction,
            num_floors=num_floors,
            ground_surface=ground_surface,
            ceiling_height=ceiling_height,
            wall_surface=wall_surface,
            volume=volume,
            north_window_surface=window_north,
            east_window_surface=window_east,
            south_window_surface=window_south,
            west_window_surface=window_west,
            tot_window_surface=window_tot,
            thermal_properties=thermal_properties
        )

def calculate_thermal_properties(year_of_construction):
    """Calculates U-values, heat capacity, and infiltration based on construction year."""
    
    # Insulation data per year group
    insulation_data = {
        '70-80': {
            'layers': [("concrete", 0.3)], 
            'U_windows': 6, 'U_roof': 2, 'U_ground': 0.3, 
            'ach': 0.4, 'C_roof': 100000, 'C_ground': 756000 # = cp_concrete * rho_reinforced_concrete * e_concrete = 840x3000x0.3 (J/K.m²)
        },
        '80-2000': {
            'layers': [("concrete", 0.19), ("polystyrene", 0.05), ("air", 0.03), ("brick", 0.08)], 
            'U_windows': 6, 'U_roof': 2, 'U_ground': 0.3, 
            'ach': 0.4, 'C_roof': 100000, 'C_ground': 756000
        },
        '2000-2020': {
            'layers': [("concrete", 0.19), ("rock_wool", 0.1), ("air", 0.03), ("brick", 0.08)], 
            'U_windows': 3, 'U_roof': 2, 'U_ground': 0.3, 
            'ach': None, 'C_roof': 100000, 'C_ground': 756000
        },
        'after-2020': {
            'layers': [("concrete", 0.19), ("liege", 0.2), ("air", 0.03), ("brick", 0.08)], 
            'U_windows': 2, 'U_roof': 2, 'U_ground': 0.3, 
            'ach': None, 'C_roof': 100000, 'C_ground': 756000
        }
    }

    # Retrieve insulation properties based on construction year
    data = insulation_data.get(year_of_construction)

    # Calculate U_walls using thermal resistance sum, layer = [(material_i, e_i)]
    R_total = sum(layer[1] / MATERIALS[layer[0]].thermal_conductivity for layer in data['layers'])
    U_walls = 1 / R_total  # U-value (W/m²K)

    # Compute total wall heat capacity
    C_wall = sum(
        MATERIALS[layer[0]].specific_heat * MATERIALS[layer[0]].density * layer[1]
        for layer in data['layers']
    )

    # Compute theta for heat transfer modeling
    R_d = sum(layer[1] / (2 * MATERIALS[layer[0]].thermal_conductivity) for layer in data['layers'])
    theta = R_d / R_total

    return ThermalProperties(
        U_walls=U_walls,
        U_windows=data['U_windows'],
        U_roof=data['U_roof'],
        U_ground=data['U_ground'],
        ach=data['ach'],
        C_wall=C_wall,
        C_roof=data['C_roof'],
        C_ground=data['C_ground'],
        theta=theta
    )


def thermal_equations(t, T, house: House, T_set, T_out, P_nom, P_irr):
    """ODE system for indoor, wall, roof, and ground temperatures"""
    C = 0.8
    ICF = 6 # ICF is a coefficient that represents how much the room is filled with stuff (tables, chairs...) compared to an empty room
    PHI = 0.5 # Term that indicates how accessible the capacity is over a day
    RHO_AIR = MATERIALS['air'].density  # kg/m^3
    CP_AIR= MATERIALS['air'].specific_heat # J/(kg.Kelvin)
    M_AIR = house.volume * RHO_AIR # kg

    T_in,T_wall,T_roof,T_ground = T

    volume = house.volume
    tot_window_surface = house.tot_window_surface
    A_roof = house.ground_surface
    A_walls = house.wall_surface
    A_ground = house.ground_surface

    # Handle the case where ach = f(T_set), for year_of_construction >=2000
    if not house.thermal_properties.ach:
        T_out_mean=10   # [°C] for 2022
        v_wind_mean=3.5 # [m/s]
        if house.year_of_construction == '2000-2020':
            k1 = 0.1
            k2 = 0.017  
            k3 = 0.049
            ach = k1 + (k2 * (T_set - T_out_mean)) + (k3 * v_wind_mean)
            house.thermal_properties.ach = ach
        elif house.year_of_construction == 'after-2020':
            k1 = 0.1
            k2 = 0.011
            k3 = 0.034
            ach = k1 + (k2 * (T_set - T_out_mean)) + (k3 * v_wind_mean)
            house.thermal_properties.ach = ach
    else:
        ach = house.thermal_properties.ach

    U_roof = house.thermal_properties.U_roof
    U_walls = house.thermal_properties.U_walls
    U_windows = house.thermal_properties.U_windows
    U_ground = house.thermal_properties.U_ground
    C_wall = house.thermal_properties.C_wall
    C_roof = house.thermal_properties.C_roof
    C_ground = house.thermal_properties.C_ground
    theta = house.thermal_properties.theta

    P_in_addition = P_irr # Irradiation
                    
    Q_infiltration = (ach/6)*volume*RHO_AIR*CP_AIR*(T_in - T_out)/600 # divide by 600 (10minutes) and divide ach by 6 because it s air change per hour
    
    Q_cond_ground = (U_roof / theta) * A_roof * (T_ground - T_in)
    Q_cond_walls = (U_walls / theta) * A_walls * (T_wall - T_in)
    Q_cond_windows = U_windows*tot_window_surface*(T_out-T_in)
    

    if(abs(T_set-T_in) <= 0.5):  # if we are close enough to set point temperature
        Q_TU=0
    else : 
        Q_TU = (max(min(C*(T_set-T_in)*P_nom,P_nom),0)) # C=0.5 to decrease the reaction of the thermal unit’s power
    
    Q_cond = Q_cond_ground+Q_cond_walls+Q_cond_windows
    
    dTin_dt = (1/(M_AIR*CP_AIR*ICF))*(Q_TU # thermal unit
                                    + P_in_addition # Irradiation & (Lights, people, appliances) - these last three are not considered here
                                    + Q_cond # conduction
                                    - Q_infiltration) # exfiltration
        
    dTwall_dt = (((U_walls/(1-theta))*A_walls*(T_out-T_wall))-((U_walls/theta)*A_walls*(T_wall-T_in)))/(PHI*C_wall*A_walls)

    dTroof_dt = (((U_roof / (1 - theta)) * A_roof * (T_out - T_roof)) - ((U_roof / theta) * A_roof * (T_roof - T_in))) / (PHI * C_roof * A_roof)

    dTground_dt = (((U_ground / (1 - theta)) * A_ground * (T_out - T_ground)) - ((U_ground / theta) * A_ground * (T_ground - T_in))) / (PHI * C_ground * A_ground)

    return [dTin_dt,dTwall_dt,dTroof_dt,dTground_dt]

def simulate_heating_dynamics(house: House, sim_days, T_set_series, T_out_series, P_irr_series, P_nom=8000, csv=False):
    """Simulate heating dynamics over N days, to compute power consumption of HP."""
    C=0.8
    HP = np.zeros(sim_days*144) # ts : 10min
    P_net = np.zeros(sim_days*144) # ts : 10min
    T_in_series = np.zeros(sim_days*144) # ts : 10min

    prob_heat_pump_flex = random.random()

    initial_guess = [19, 10, 10, 10] # [T_in, T_wall, T_roof, T_ground]

    for day in range(sim_days):
        
        for i in range(144): # 144 x 10 minutes in one day

            ts = day*144+i

            T_set = T_set_series[ts]  # Setpoint temperature
            T_out = T_out_series[ts]  # Outside temperature 
            P_irr = P_irr_series[ts]  # Irradiation

            #if T_out is larger than 19, heat pump do not work
            if T_out > 20:
                HP[ts]=0
                continue
            # Heat pump has a chance to not work between 11 p.m. and 6a.m.
            if prob_heat_pump_flex < 0.5 and (i < 6*6 or i >=23*6):
                HP[ts]=0
                continue

            sol = solve_ivp(
                lambda t, T: thermal_equations(t, T, house, T_set, T_out=T_out, P_nom=P_nom, P_irr=P_irr),
                [0, 10],  # each min
                initial_guess
            )
            
            initial_guess = sol.y[:, -1]  # Update for next iteration

            T_in_start = sol.y[0][0] 
            T_in_end = sol.y[0][-1]
            
            T_wall_start = sol.y[1][0]
            T_wall_end = sol.y[1][-1]

            T_ground_start = sol.y[-1][0]
            T_ground_end = sol.y[-1][-1]
            
            T_in = (T_in_start+T_in_end)/2
            T_wall = (T_wall_start+T_wall_end)/2
            T_ground = (T_ground_start+T_ground_end)/2
            
            if(abs(T_set-T_in) <= 0.5):  # If we are close enough to set point temperature
                HP[ts]= 0
            else: 
                HP[ts] = (max(min(C*(T_set-T_in)*P_nom,P_nom),0))/1000 # In kW
            # if(HP[ts] <= 0):
            #     HP[ts] = 0

            if csv:

                U_roof = house.thermal_properties.U_roof
                U_walls = house.thermal_properties.U_walls
                U_windows = house.thermal_properties.U_windows
                tot_window_surface = house.tot_window_surface
                A_roof = house.ground_surface
                A_walls = house.wall_surface
                theta = house.thermal_properties.theta


                Q_cond_ground = (U_roof / theta) * A_roof * (T_ground - T_in)
                Q_cond_walls = (U_walls / theta) * A_walls * (T_wall - T_in)
                Q_cond_windows = U_windows*tot_window_surface*(T_out-T_in)
                Q_cond = Q_cond_ground+Q_cond_walls+Q_cond_windows
                Q_infiltration = (house.thermal_properties.ach/6)*house.volume*MATERIALS['air'].density*MATERIALS['air'].specific_heat*(T_in - T_out)/600

                P_net[ts] = P_irr + Q_cond - Q_infiltration
                T_in_series[ts] = T_in
            
    return HP, P_net, T_in_series

def irradiation(house: House, weather_path):

    SF = 0.3    # Solar Factor
    # SF = 0.5
    weather = pd.read_excel(weather_path)
    
    # NumPy arrays where elements are repeated 6 times
    irr_n = np.repeat(weather['I_north W/m²'].values, 6)
    irr_e = np.repeat(weather['I_east W/m²'].values, 6)
    irr_s = np.repeat(weather['I_south W/m²'].values, 6)
    irr_w = np.repeat(weather['I_west W/m²'].values, 6)

    Q_dot_North = house.north_window_surface*irr_n*SF
    Q_dot_East = house.east_window_surface*irr_e*SF
    Q_dot_West = house.west_window_surface*irr_w*SF
    Q_dot_South = house.south_window_surface*irr_s*SF
    
    return Q_dot_North+Q_dot_East+Q_dot_West+Q_dot_South

def outside_temperature(weather_path):
    weather = pd.read_excel(weather_path)
    return np.repeat(weather['Temperature C'].values, 6) # Each 10 minutes


def run_space_heating(t_set_series, sim_days, csv=False):

    results = np.zeros(sim_days*144)

    house = House.generate()
    T_set_series = t_set_series # 'shsetting_data = family.sh_day', ts=10 min for sim_days 
    T_out_series = outside_temperature(weather_path) # ts=10min for 1 year
    P_irr_series = irradiation(house, weather_path) # ts=10min for 1 year
    HP, P_net, T_in = simulate_heating_dynamics(house, sim_days, T_set_series, T_out_series, P_irr_series, P_nom=8000, csv=csv) 
    results += HP
        
    power = results # ["HP dynamic modelling [kW]"]
    total_power = power.sum()/4 # kWe (COP = 4)
    total_consumption = total_power / 6 # 6 is the number of time steps per hour

    # print("Price:", total_consumption*0.3, "€"," for ", sim_days, " days.") # 30 cts per kWh
    print("House:", house.year_of_construction, house.volume, "m3")

    if csv:
        # Save results to CSV file
        year = house.year_of_construction
        vol = str(round(house.volume))
        n_floors = str(house.num_floors)

        ICF = 6
        RHO_AIR = MATERIALS['air'].density  # kg/m^3
        CP_AIR= MATERIALS['air'].specific_heat # J/(kg.Kelvin)
        M_AIR = house.volume * RHO_AIR # kg
        eff_K_J = np.repeat(1/(M_AIR*CP_AIR*ICF), len(HP))

        df = pd.DataFrame({'HP dynamic modelling [kW]': results, 'T_in [C°]': T_in, 'T_setpoint [C°]': T_set_series[1:], 'T_out [C°]': T_out_series[:sim_days*144], 'P net=(P_irr+Q_cond-Q_inf) [W]': P_net, 'Efficiency [K/J]': eff_K_J})

        df.to_csv(f"{file_path}\{year}_house_{n_floors}_floors_{vol}m3_for_{sim_days}_days.csv", index=False)



    return results