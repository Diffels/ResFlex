import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import os

# Define the boiler model
class Boiler:
    def __init__(self, Volume, Tset, Tbound, Pmax):
        self.Volume = Volume
        self.m = self.Volume  # Mass of water in kg (assuming 1L = 1kg)
        self.c = 4189  # J/(kgK)
        self.C = self.m * self.c  # Thermal capacity
        self.h = 0.8  # W/(m m K)
        self.T_a = 20  # Ambient temperature in K
        self.T_inlet = 15  # Inlet temperature in K
        self.Tset = Tset
        self.Tbound = Tbound
        self.T_min = Tset - Tbound
        self.T_max = Tset + Tbound
        self.T = self.Tset  # Initial temperature in K 
        self.dt = 1  # Time step of 1 minute
        self.Pmax = Pmax  # W
        self.resistor_on = False  # Resistor stateer
        # Compute Diameter and Height from Volume (assuming cylindrical shape and Diameter = Height / 2)
        self.Height = (4 * self.Volume * 1e-3 / math.pi) ** (1/3)  # Convert L to m^3
        self.Diameter = self.Height / 2
        self.area = self.surface_area()  # Surface area in m^2

    def surface_area(self):
        radius_m = self.Diameter / 2
        lateral_area = math.pi * self.Diameter * self.Height
        cap_area = 2 * math.pi * radius_m**2
        return lateral_area + cap_area

    def simulate(self, mass_flow, t_end):
        num_steps = int(t_end/self.dt)
        temperatures = np.zeros(num_steps)
        powers = np.zeros(num_steps)

        temperatures[0] = self.T
        for i in range(1, num_steps):
            dTdt = -(self.h*self.area/self.C)*(temperatures[i-1]-self.T_a)

            flow_lps = mass_flow[i]
            total_extraction = flow_lps * self.dt

            if total_extraction >= self.m:
                temperatures[i] = self.T_inlet
            else:
                dTdt -= (flow_lps*self.c*(temperatures[i-1]-self.T_inlet))/self.C

            if self.resistor_on:
                dTdt += self.Pmax / self.C
                powers[i] = self.Pmax
            else:
                powers[i] = 0

            temperatures[i] = temperatures[i-1] + dTdt * self.dt

            if temperatures[i] <= self.Tset - self.Tbound and not self.resistor_on:
                self.resistor_on = True

            if temperatures[i] >= self.Tset + self.Tbound and self.resistor_on:
                self.resistor_on = False

        return temperatures, powers


# Read the CSV files
Data = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),"database", "TEST_digital_twin_WH_timeseries_243.csv"), sep=';')
temp_cols = [col for col in Data.columns if col.startswith('T_mean_')]
vdot_cols = [col for col in Data.columns if col.startswith('vdot_')]

# DataFrame with extraction flows and initial temperatures
flows_df = Data[vdot_cols]
initial_temperatures = Data[temp_cols].iloc[0] - 273 + 6  # Take the initial row as the initial temperatures

# Read boiler characteristics
characteristics_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),"database","TEST_digital_twin_WH_charact_243.csv"), sep=';')
max_vpp_power = characteristics_df['Electric Power (W)'].sum()

# General parameters
simulation_time = 1440  # 24 hours in minutes
t_end = simulation_time

# Create DataFrames to store results
temperature_results = pd.DataFrame()
power_results = pd.DataFrame()
grouped_power_results = pd.DataFrame(index=np.arange(0, t_end))

# Initialize arrays to store VPP and flexible demand metrics
total_vpp_power = np.zeros(t_end)
total_vpp_energy = np.zeros(t_end)
max_vpp_energy = np.zeros(t_end)
min_vpp_energy = np.zeros(t_end)
flexible_demand_up = np.zeros(t_end)
flexible_demand_down = np.zeros(t_end)

# Loop to simulate each boiler
print(len(characteristics_df), "boilers to simulate...")
for idx in range(len(characteristics_df)):
    # Get the boiler characteristics
    Volume = characteristics_df['Volume (L)'].iloc[idx]
    Pmax = characteristics_df['Electric Power (W)'].iloc[idx]

    # Get the initial temperature from the time file
    initial_temperature = initial_temperatures.iloc[idx]

    # Get the extraction flow for this boiler
    mass_flow = flows_df[vdot_cols[idx]].values  # L/s, corresponding to the boiler's flow

    # Create a boiler instance
    boiler = Boiler(
        Volume=Volume,
        #initial_temperature=initial_temperature,
        Tset=60,  # Example
        Tbound=5,  # Example
        Diameter=Diameter,
        Height=Height,
        Pmax=Pmax
    )

    # Simulate the boiler's behavior for this mass flow and store the results
    temperatures, powers = boiler.simulate(mass_flow, t_end)

    # Store the simulated temperatures and powers in the results DataFrames
    #temperature_results[f'boiler_{idx}'] = temperatures
    #power_results[f'boiler_{idx}'] = powers