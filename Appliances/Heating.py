import pandas as pd
import numpy as np
import os
import random
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import matplotlib.pyplot as plt 
import plotly.graph_objects as go

# Find the path to this file
#file_path = os.path.dirname(os.path.realpath(__file__)) 
# Create an absolute path to the Excel file 'Meteo2022_Liege.xlsx'
#weather_path = os.path.join(file_path, 'database/Meteo2022_Liege.xlsx')

@dataclass
class House:
    """Dataclass representing a house with thermal properties"""
    year: str
    floors: int
    ground_surface: float
    wall_surface: float
    volume: float
    north_window_surface: float
    east_window_surface: float
    south_window_surface: float
    west_window_surface: float
    tot_window_surface: float
    U_tot: float
    U_wall: float
    C_env: float
    C_air: float

    @staticmethod
    def generate(year, surface, floors):
        """Generate a random house and assign thermal properties"""
        # Datasets are based on the year of construction
        # The year of construction is a int, but we convert it to an string to match the keys of the dictionaries
        if year < 1945: year = '< 45'
        elif year < 1970: year = '45-70'
        elif year < 1990: year = '70-90'
        elif year < 2007: year = '90-07'
        else: year = '> 08'
        # Raw Data taken from ProCEBaR (Task 2), presentation of the project: https://orbi.uliege.be/bitstream/2268/192397/2/160126_BERA_ULg.pdf

        # U-values expressed in W/(m².K)
        U_wall = {'< 45': 2.25, '45-70': 1.56, '70-90': 0.98, '90-07': 0.49, '> 08': 0.4}
        U_window = {'< 45': 5, '45-70': 5, '70-90': 3.5, '90-07': 3.5, '> 08': 2}
        U_roof = {'< 45': 4.15, '45-70': 3.33, '70-90': 0.77, '90-07': 0.43, '> 08': 0.3}
        U_floor = {'< 45': 3.38, '45-70': 3.38, '70-90': 1.14, '90-07': 0.73, '> 08': 0.4}

        # K-values expressed in J/(m².K)
        K_wall = {'< 45': 76466, '45-70': 74715, '70-90': 75945, '90-07': 75022, '> 08': 74834}
        K_roof = {'< 45': 7211, '45-70': 11357, '70-90': 11922, '90-07': 12848, '> 08': 14356}
        K_floor = {'< 45': 67352, '45-70': 67352, '70-90': 62673, '90-07': 69245, '> 08': 69246}

        
        # Derived properties
        volume = surface * 2.5 # Volume in m3 (assuming 2.5m height per floor)
        ground_surface = surface / floors  # Ground surface in m2
        wall_surface = round(4 * (surface ** 0.5), 2) * floors * 2.5  # Assume square-shaped house for simplicity
        
        # Window surfaces
        window_north = max(0, random.uniform(-0.1, 0.2) * wall_surface / 4)
        window_south = max(0, random.uniform(-0.1, 0.3) * wall_surface / 4)
        window_east = max(0, random.uniform(-0.1, 0.3) * wall_surface / 4)
        window_west = max(0, random.uniform(-0.1, 0.3) * wall_surface / 4)
        window_tot = round(window_north+window_east+window_south+window_west, 2)        
        
        U_tot = (U_wall[year] * wall_surface + U_roof[year] * surface + U_floor[year] * surface + U_window[year] * window_tot)

        C_env = (K_wall[year] * wall_surface +
                    K_roof[year] * ground_surface +
                    K_floor[year] * ground_surface)
        C_air = 1.225 * volume * 1005 # Rho * V * cp

        return House(year=year,floors=floors,ground_surface=ground_surface,wall_surface=wall_surface,volume=volume,
            north_window_surface=window_north,east_window_surface=window_east,south_window_surface=window_south,west_window_surface=window_west,tot_window_surface=window_tot,
            U_tot = U_tot,U_wall = U_wall[year],C_env = C_env,C_air = C_air)
    
    def display(self):
        """Print the house properties"""
        print(f"Year of construction: {self.year_of_construction}")
        print(f"Number of floors: {self.num_floors}")
        print(f"Ground surface: {self.ground_surface}")
        print(f"Volume: {self.volume}")
        print(f"Total window surface: {self.tot_window_surface}")

def weather_import(house: House, weather_path):
    SF = 0.3    # Solar Factor
    weather = pd.read_excel(weather_path)
    
    T_out = np.repeat(weather['Temperature C'].values, 60)  # Each minute

    irr_n = np.repeat(weather['I_north W/m²'].values, 60)
    irr_e = np.repeat(weather['I_east W/m²'].values, 60)
    irr_s = np.repeat(weather['I_south W/m²'].values, 60)
    irr_w = np.repeat(weather['I_west W/m²'].values, 60)

    Q_dot_North = house.north_window_surface * irr_n * SF
    Q_dot_East = house.east_window_surface * irr_e * SF
    Q_dot_West = house.west_window_surface * irr_w * SF
    Q_dot_South = house.south_window_surface * irr_s * SF
    
    return T_out, Q_dot_North + Q_dot_East + Q_dot_West + Q_dot_South

def heating_dynamics(house, sim_days, T_set, T_out, P_irr, P_nom=8000):
    n_ts = 24*60        # Number of time steps in a day (1 min intervals)
    abs = 0.5           # Temperature difference threshold for HP control
    ACH = 0.1           # Air changes per hour [1/h]
    A_wall = house.wall_surface

    ACH = 0.1  # Air changes per hour [1/h]
    
    """Simulate space heating dynamics with controlled HP power."""
    k_wall = 0.5  # Wall thickness coefficient, models how much of the wall is considered at room temperature

    # Timeseries initialization
    HP = np.zeros(sim_days * n_ts)      # HP power for each time step
    T_in = np.zeros(sim_days * n_ts)    # Indoor temperature at each time step
    T_wall = np.zeros(sim_days * n_ts)  # Indoor temperature at each time step
    P_loss = np.zeros(sim_days * n_ts)  # Power loss at each time step

    T_in[0] = T_set[0]  # Initial indoor temperature
    T_wall[0] = T_set[0]  # Initial wall temperature
    HP_isoff = 0  # Initial HP power
    for ts in range(1,sim_days*n_ts):  # Loop over days
        # Solve heating dynamics
        #P_airloss, P_wallloss = heat_loss(house, T_in[ts-1], T_wall[ts-1], T_out[ts-1+n_ts*start_day], P_irr[ts-1+n_ts*start_day])
        P_aircond = k_wall*house.U_tot * (T_in[ts-1] - T_wall[ts-1]) # Divided by 2 to account for half the thickness of the wall
        P_wallloss = (1-k_wall)*house.U_wall * A_wall * (T_wall[ts-1] - T_out[ts-1]) - P_aircond  # Conduction losses through walls divided by 2 to account for half the thickness of the wall
        Q_exfiltration = (ACH/60)*house.C_air*(T_in[ts-1] - T_out[ts-1])/60 # in W: [1/min] * m3 * kg/m3 * J/(kg.K) * K / 60s
        P_airloss = P_aircond - P_irr[ts-1]/10 + Q_exfiltration # Net losses including solar gain
        P_loss[ts] = P_wallloss # Total losses
        
        if T_in[ts-1] < T_set[ts-1]:# - abs*HP_isoff:  
            HP[ts] = P_nom  # HP is on
            HP_isoff = 0
        else: 
            HP[ts] = 0  # HP is off
            HP_isoff = 1

        dTair = (HP[ts] - P_airloss) / (house.C_air+k_wall*house.C_env)
        dTwall = (-P_wallloss) / ((1-k_wall)*house.C_env)
        T_in[ts] = T_in[ts-1] + dTair*60               # Update indoor temperature
        T_wall[ts] = T_wall[ts-1] + dTwall*60          # Update wall temperature
        

    return HP/1e3, T_in, T_wall, P_loss  # Return HP power, indoor and wall temperature, Power losses

def space_heating(T_set, sim_days, start_day, year, size, floors, P_nom):
    house = House.generate(year, size, floors)
    T_out, P_irr = weather_import(house, os.path.dirname(__file__)+'\database\Meteo2022_Liege.xlsx')  # External temperature and Solar irradiation series
    T_out = T_out[start_day*24*60:(start_day+sim_days)*24*60]  # Crop the time series to the simulation period
    P_irr = P_irr[start_day*24*60:(start_day+sim_days)*24*60]  
    # Simulate heating system 
    P_ref, T, T_wall, P_loss = heating_dynamics(house, sim_days, T_set, T_out, P_irr,P_nom)
    # Create DataFrame for the flexibility data
    Flex_HP = pd.DataFrame({'Tset': T_set[:-1],'Tref': T,'Twall': T_wall, 'Tout': T_out,'Ploss': P_loss}, index=None)
    Param_HP = {'Pmax':P_nom, "Utot": house.U_tot, "Cenv": house.C_env, "Cair": house.C_air}   
    return P_ref, Flex_HP, Param_HP



    """Previous data for househol sizes
        floors = [1, 2, 3]
        areas = {1: [90, 100, 110, 120, 130, 140, 150],  # Single-story houses
                 2: [60, 70, 80, 90, 100, 110],  # Two-story houses
                 3: [50, 60, 70, 80, 90]}  # Three-story houses
        heights = {1: 2.5, 2: 5, 3: 7.5}  # Heights by floors
    """