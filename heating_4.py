import pandas as pd
import numpy as np
import os
import random
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import matplotlib.pyplot as plt 


# Find the path to this file
file_path = os.path.dirname(os.path.realpath(__file__)) 
# Create an absolute path to the Excel file 'Meteo2022_Liege.xlsx'
weather_path = os.path.join(file_path, 'database/Meteo2022_Liege.xlsx')

class Air:
    """Class to store air properties (1 atm, 20°C)"""
    density = 1.225  # rho kg/m3
    specific_heat = 1005  # cp J/(kg.K)
    thermal_conductivity = 0.24  # k in W/(m.K)

@dataclass
class ConductionCoefficients:
    """Dataclass to store the conduction coefficients of the house"""
    U_wall: float
    U_window: float
    U_roof: float
    U_floor: float

@dataclass
class CapacityCoefficients:
    """Dataclass to store the capacity coefficients of the house"""
    K_wall: float
    K_roof: float
    K_floor: float

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
    conduction_coefficients: ConductionCoefficients
    capacity_coefficients: CapacityCoefficients

    @staticmethod
    def generate():
        """Generate a random house and assign thermal properties"""
        # Datasets
        construction_periods = ['< 45', '45-70', '70-90', '90-07', '> 08']
        floors = [1, 2, 3]
        areas = {1: [90, 100, 110, 120, 130, 140, 150],  # Single-story houses
                 2: [60, 70, 80, 90, 100, 110],  # Two-story houses
                 3: [50, 60, 70, 80, 90]}  # Three-story houses
        heights = {1: 2.5, 2: 5, 3: 7.5}  # Heights by floors

        # Raw Data taken from ProCEBaR (Task 2), presentation of the project: https://orbi.uliege.be/bitstream/2268/192397/2/160126_BERA_ULg.pdf

        # U-values expressed in W/(m².K)
        U_wall_values = {'< 45': 2.25, '45-70': 1.56, '70-90': 0.98, '90-07': 0.49, '> 08': 0.4}
        U_window_values = {'< 45': 5, '45-70': 5, '70-90': 3.5, '90-07': 3.5, '> 08': 2}
        U_roof_values = {'< 45': 4.15, '45-70': 3.33, '70-90': 0.77, '90-07': 0.43, '> 08': 0.3}
        U_floor_values = {'< 45': 3.38, '45-70': 3.38, '70-90': 1.14, '90-07': 0.73, '> 08': 0.4}

        # K-values expressed in W/(m².K), from J/(m².K) with ts = 15 minutes
        K_wall_values = {'< 45': 76466/(15*60), '45-70': 74715/(15*60), '70-90': 75945/(15*60), '90-07': 75022/(15*60), '> 08': 74834/(15*60)}
        K_roof_values = {'< 45': 7211/(15*60), '45-70': 11357/(15*60), '70-90': 11922/(15*60), '90-07': 12848/(15*60), '> 08': 14356/(15*60)}
        K_floor_values = {'< 45': 67352/(15*60), '45-70': 67352/(15*60), '70-90': 62673/(15*60), '90-07': 69245/(15*60), '> 08': 69246/(15*60)}


        # Randomly select properties
        year_of_construction = random.choice(construction_periods)
        n_floors = random.choice(floors)
        ground_surface = random.choice(areas[n_floors])
        ceiling_height = heights[n_floors]/2
        
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
        
        return House(
            year_of_construction=year_of_construction,
            num_floors=n_floors,
            ground_surface=ground_surface,
            ceiling_height=ceiling_height,
            wall_surface=wall_surface,
            volume=volume,
            north_window_surface=window_north,
            east_window_surface=window_east,
            south_window_surface=window_south,
            west_window_surface=window_west,
            tot_window_surface=window_tot,
            conduction_coefficients=ConductionCoefficients(
                U_wall=U_wall_values[year_of_construction],
                U_window=U_window_values[year_of_construction],
                U_roof=U_roof_values[year_of_construction],
                U_floor=U_floor_values[year_of_construction]                
            ),
            capacity_coefficients=CapacityCoefficients(
                K_wall=K_wall_values[year_of_construction],
                K_roof=K_roof_values[year_of_construction],
                K_floor=K_floor_values[year_of_construction]
            )
        )
    
    def display(self):
        """Print the house properties"""
        print(f"Year of construction: {self.year_of_construction}")
        print(f"Number of floors: {self.num_floors}")
        print(f"Ground surface: {self.ground_surface}")
        print(f"Volume: {self.volume}")
        print(f"Total window surface: {self.tot_window_surface}")

def irradiation(house: House, weather_path):
    SF = 0.3    # Solar Factor
    weather = pd.read_excel(weather_path)
    
    irr_n = np.repeat(weather['I_north W/m²'].values, 6)
    irr_e = np.repeat(weather['I_east W/m²'].values, 6)
    irr_s = np.repeat(weather['I_south W/m²'].values, 6)
    irr_w = np.repeat(weather['I_west W/m²'].values, 6)

    Q_dot_North = house.north_window_surface * irr_n * SF
    Q_dot_East = house.east_window_surface * irr_e * SF
    Q_dot_West = house.west_window_surface * irr_w * SF
    Q_dot_South = house.south_window_surface * irr_s * SF
    
    return Q_dot_North + Q_dot_East + Q_dot_West + Q_dot_South

def outside_temperature(weather_path):
    weather = pd.read_excel(weather_path)
    return np.repeat(weather['Temperature C'].values, 6)  # Each 10 minutes

def heat_loss(house, T_in, T_out, P_irradiation):
    """Calculate total heat losses including conduction and ventilation."""
    U = house.conduction_coefficients
    A_wall = house.wall_surface
    A_roof = house.ground_surface  # Assuming flat roof
    A_floor = house.ground_surface
    A_window = house.tot_window_surface

    ACH = 0.1  # Air changes per hour [1/h]
    
    U_house = (U.U_wall * A_wall + U.U_roof * A_roof + U.U_floor * A_floor + U.U_wall * A_window)

    # print(U_house)
    P_conduction = U_house * (T_in - T_out)/2
    Q_exfiltration = (ACH/60)*house.volume*Air.density*Air.specific_heat*(T_in - T_out)/60 # in W: [1/min] * m3 * kg/m3 * J/(kg.K) * K / 60s
    P_loss = P_conduction - P_irradiation + Q_exfiltration # Net losses including solar gain
    return P_loss  # in W

def heating_dynamics(t, T_in, house: House, T_set, T_out, HP, P_nom, P_irr, comfort):
    """Differential equation for indoor temperature evolution with HP power tracking."""

    P_loss = heat_loss(house, T_in, T_out, P_irr)

    # Thermal capacity of the envelope W/K
    C_envelope = (house.capacity_coefficients.K_wall * house.wall_surface +
                  house.capacity_coefficients.K_roof * house.ground_surface +
                  house.capacity_coefficients.K_floor * house.ground_surface)
    
    # Thermal capacity of the air expressed in (kg/m3).m3.(J/(kg.K)) = J/K
    C_air = Air.density * house.volume * Air.specific_heat
    # Converted to W/K
    C_air /= 60  # 1 minute to seconds
    
    # Total thermal capacity expressed in W/K
    C_total = (C_envelope + C_air)
    # print(C_total)
    # Temperature evolution equation: 
    dTdt = (HP - P_loss) / C_total

    return [dTdt]

def simulate_heating_dynamics(house, sim_days, start_day, T_set_series, T_out_series, P_irr_series, comfort=0.5, P_nom=8000):
    """Simulate space heating dynamics with controlled HP power."""
    T0 = T_set_series[0]  # Initial indoor temperature
    HP0 = 0  # Initial HP power
    initial_guess = [T0, HP0]  # Initial temperature and HP power

    HP = np.zeros(sim_days * 144)  # Time steps: 10 min intervals
    T = np.zeros(sim_days * 144)  # Indoor temperature at each time step

    prob_heat_pump_flex = 0 #random.random()  # Probability of HP not working 11 PM - 6 AM

    # Counter for consecutive time steps with HP power on
    ts_power_on = 0
    flag_dead_band = False  # Flag for dead band condition

    for day in range(sim_days):
        for i in range(144):  # 144 x 10 min time steps per day
            ts = day * 144 + i
            T_set = T_set_series[ts]  # Not adjusted since only considered sim_days days
            T_out = T_out_series[ts+start_day*144]  # Adjusted for start day
            P_irr = P_irr_series[ts+start_day*144]  # Adjusted for start day

            # Solve heating dynamics
            sol = solve_ivp(
                lambda t, T_in: heating_dynamics(
                    t, T_in, house, T_set, T_out=T_out, HP=HP[ts-1], P_nom=P_nom, P_irr=P_irr, comfort=comfort # First time step uses last value in HP array, ie 0 W.
                ),
                [0, 10],  # 10-minute interval
                [initial_guess[0]],  # Only temperature is being solved
            )

            # Compute average temperature over 10-minute period
            T[ts] = np.mean(sol.y[0])
            initial_guess[0] = T[ts]  # Update initial guess for next step

            # Apply heat pump constraint (e.g., turn off during 11 PM - 6 AM)
            hour = (ts % 144) / 6  # Convert time step to hour
            if (prob_heat_pump_flex > 0.5) and (23 <= hour or hour < 6):
                HP[ts] = 0  # HP is off
            else:
                # Check if temperature is within dead band
                flag_dead_band = (abs(T_set - T[ts]) <= 0.5)  # Dead band condition

                if T[ts] <= T_set:  # If indoor temperature is below setpoint
                    if abs(T_set - T[ts]) <= 0.5 and flag_dead_band:
                        HP[ts] = 0
                        ts_power_on = 0
                    else:
                        HP[ts] = P_nom  # Turn on HP
                        ts_power_on += 1 
                        flag_dead_band = False
                else:
                    flag_dead_band = True
                    HP[ts] = 0
                    ts_power_on = 0
                
                
                # # Check if close to the setpoint temperature and HP has been on for a while (1 hour)
                # if abs(T_set - T[ts]) <= 0.5 and ts_power_on > 6:  
                #     HP[ts] = 0
                #     ts_power_on = 0
                # # Check if temperature is too high (case where T_in > T_set + 0.5) 
                # elif T[ts] - T_set > 3:
                #     HP[ts] = 0
                #     ts_power_on = 0
                # # Otherwise, 
                # else:
                #     HP[ts] = P_nom
                #     ts_power_on += 1
            

    return HP, T



def run_space_heating(T_set_series, sim_days, start_day=0):
    """Run multiple simulations and compute average results for indoor temperature and HP power."""
    
    # results_T = np.zeros(sim_days * 144)
    results = np.zeros(sim_days * 144)

    # comfort_study(sim_days, T_set_series)

    house = House.generate()
    T_out_series = outside_temperature(weather_path)  # External temperature series
    P_irr_series = irradiation(house, weather_path)  # Solar irradiation series

    # Simulate heating system 
    P_HP_series, T = simulate_heating_dynamics(house, sim_days, start_day, T_set_series, T_out_series, P_irr_series, P_nom=8000, comfort=0.5)

    print(len(T))
    print(len(T_set_series))
    plot_temperatures(sim_days, T_set_series, T)

    results += P_HP_series/1e3  # Convert to kW

    total_power = results.sum()/4 # kWe (COP = 4)
    total_consumption = total_power / 6 # 6 is the number of time steps per hour

    # print("Price:", total_consumption*0.3, "€"," for ", sim_days, " days.") # 30 cts per kWh
    
    return results


def plot_temperatures(sim_days, T_set, T_in):
    """Plot indoor temperature and setpoint temperature."""
    time_intervals = range(sim_days * 144)

    plt.figure(figsize=(12, 8))
    plt.plot(time_intervals, T_set[:len(time_intervals)], label="Setpoint Temperature", color="tab:green", linestyle="--")
    plt.plot(time_intervals, T_in[:len(time_intervals)], label="Indoor Temperature", color="tab:blue", alpha=0.5)
    print(sim_days)
    plt.xticks(range(0, sim_days * 144, 6), labels=list(range(0, sim_days * 24)))
    plt.xlabel("Time [h] ")
    plt.ylabel("Temperature (°C)")
    plt.title("Indoor Temperature Over Time with DEAD BAND")
    plt.legend()
    plt.grid()
    plt.show()


def comfort_study(sim_days, T_set_series):
    """Study the impact of comfort factor on heating dynamics."""
    
    house = House.generate()
    P_irr_series = irradiation(house, weather_path)  # Solar irradiation series
    T_out_series = outside_temperature(weather_path)  # External temperature series

    comforts = [0.1, 0.5]  # comfort factors to test
    time_intervals = range(sim_days * 144)

    results = {}  # Store simulation results

    for c in comforts:
        print(f"comfort factor: {c} [-]")
        P_HP_series, T = simulate_heating_dynamics(
            house, sim_days, T_set_series, T_out_series, P_irr_series, P_nom=8000, comfort=c
        )
        results[c] = (P_HP_series, T)  # Store results

    # # Indoor Temperature Plot
    # plt.figure(figsize=(12, 8))
    # for c, (_, T) in results.items():
    #     plt.plot(time_intervals, T, label=f"T in (comfort={c})")

    # plt.plot(time_intervals, T_set_series[:len(time_intervals)], label="Setpoint Temperature", color="tab:green", linestyle="--")
    # plt.plot(time_intervals, T_out_series[:len(time_intervals)], label="Outdoor Temperature", color="tab:orange", linestyle="--")
    # plt.xlabel("Time (hours)")
    # plt.ylabel("Temperature (°C)")
    # plt.title("Indoor Temperature Over Time for Different comfort Factors")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # # Combined Plot with Two Axes
    # plt.close()

    # Plot Indoor Temperature
    plt.figure(figsize=(12, 8))
    for c, (_, T) in results.items():
        plt.plot(time_intervals, T, label=f"Indoor Temp (comfort={c})")
    plt.plot(time_intervals, T_set_series[:len(time_intervals)], label="Setpoint Temperature", color="gold", linestyle="--", alpha=0.8)
    plt.plot(time_intervals, T_out_series[:len(time_intervals)], label="Outdoor Temperature", color="tab:purple", linestyle="--", alpha=0.8)
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (°C)")
    plt.title("Indoor Temperature Over Time for Different comfort Factors")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot HP Power
    plt.figure(figsize=(12, 8))
    for c, (P_HP_series, _) in results.items():
        plt.plot(time_intervals, P_HP_series, label=f"HP Power (comfort={c})", alpha=0.6)
        # plt.fill_between(time_intervals, P_HP_series, alpha=0.2)
    plt.xlabel("Time (hours)")
    plt.ylabel("HP Power (kW)")
    plt.title("HP Power Over Time for Different comfort Factors")
    plt.legend()
    plt.grid()
    plt.show()

    exit()