import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go

# Define the boiler model
class Boiler:
    def __init__(self, tank_volume, initial_temperature, min_temperature, max_temperature, diameter, height, resistor_power, inlet_temperature):
        self.tank_volume = tank_volume
        self.m = self.tank_volume  # Mass of water in kg (assuming 1L = 1kg)
        self.c = 4189  # J/(kgK)
        self.C = self.m * self.c  # Thermal capacity
        self.h = 0.8  # W/(m m K)
        self.T_a = 23  # Ambient temperature in K
        self.T_inlet = inlet_temperature  # Inlet temperature in K
        self.T = initial_temperature  # Initial temperature in K 
        self.T_min = min_temperature
        self.T_max = max_temperature
        self.dt = 60  # Time step of 1 minute
        self.P_resistor = resistor_power  # W
        self.resistor_on = False  # Resistor state
        self.height = height
        self.diameter = diameter

    def surface_area(self):
        radius_m = self.diameter / 2
        lateral_area = math.pi * self.diameter * self.height
        cap_area = 2 * math.pi * radius_m**2
        return lateral_area + cap_area

    def simulate(self, mass_flow, t_end):
        num_steps = int(t_end/self.dt)
        temperatures = np.zeros(num_steps)
        powers = np.zeros(num_steps)

        temperatures[0] = self.T

        for i in range(1, num_steps):
            area = self.surface_area()
            dTdt = -(self.h*area/self.C)*(temperatures[i-1]-self.T_a)

            flow_lps = mass_flow[i]
            total_extraction = flow_lps * self.dt

            if total_extraction >= self.m:
                temperatures[i] = self.T_inlet
            else:
                dTdt -= (flow_lps*self.c*(temperatures[i-1]-self.T_inlet))/self.C

            if self.resistor_on:
                dTdt += self.P_resistor / self.C
                powers[i] = self.P_resistor
            else:
                powers[i] = 0

            temperatures[i] = temperatures[i-1] + dTdt * self.dt

            if temperatures[i] <= self.T_min and not self.resistor_on:
                self.resistor_on = True

            if temperatures[i] >= self.T_max and self.resistor_on:
                self.resistor_on = False

        return temperatures, powers


# Read the CSV files
Data = pd.read_csv('TEST_digital_twin_WH_timeseries_243.csv', sep=';')
temp_cols = [col for col in Data.columns if col.startswith('T_mean_')]
vdot_cols = [col for col in Data.columns if col.startswith('vdot_')]

# DataFrame with extraction flows and initial temperatures
flows_df = Data[vdot_cols]
initial_temperatures = Data[temp_cols].iloc[0] - 273 + 6  # Take the initial row as the initial temperatures

# Read boiler characteristics
characteristics_df = pd.read_csv('TEST_digital_twin_WH_charact_243.csv', sep=';')
max_vpp_power = characteristics_df['Electric Power (W)'].sum()

# General parameters
simulation_time = 86400  # 24 hours in seconds
t_end = simulation_time
min_temperature = 75
max_temperature = 80
inlet_temperature = 15  # Cold water at 288 K

# Create DataFrames to store results
temperature_results = pd.DataFrame()
power_results = pd.DataFrame()
grouped_power_results = pd.DataFrame(index=np.arange(0, t_end, 60))

# Initialize variables to store VPP and flexible demand
total_vpp_power = np.zeros(int(t_end / 60))
total_vpp_energy = np.zeros(int(t_end / 60))
max_vpp_energy = np.zeros(int(t_end / 60))
min_vpp_energy = np.zeros(int(t_end / 60))
flexible_demand_up = np.zeros(int(t_end / 60))
flexible_demand_down = np.zeros(int(t_end / 60))

# Loop to simulate each boiler
for idx in range(len(characteristics_df)):
    # Get the boiler characteristics
    volume = characteristics_df['Volume (L)'].iloc[idx]
    resistor_power = characteristics_df['Electric Power (W)'].iloc[idx]
    height = characteristics_df['Height (m)'].iloc[idx]
    diameter = characteristics_df['Diameter (m)'].iloc[idx]

    # Get the initial temperature from the time file
    initial_temperature = initial_temperatures.iloc[idx]

    # Get the extraction flow for this boiler
    mass_flow = flows_df[vdot_cols[idx]].values  # L/s, corresponding to the boiler's flow

    # Create a boiler instance
    boiler = Boiler(
        tank_volume=volume,
        initial_temperature=initial_temperature,
        min_temperature=55,  # Example
        max_temperature=60,  # Example
        diameter=diameter,
        height=height,
        resistor_power=resistor_power,
        inlet_temperature=15  # Inlet water temperature, e.g., 288 K
    )

    # Simulate the boiler's behavior for this mass flow and store the results
    temperatures, powers = boiler.simulate(mass_flow, t_end)

    # Store the simulated temperatures and powers in the results DataFrames
    temperature_results[f'boiler_{idx}'] = pd.Series(temperatures)
    power_results[f'boiler_{idx}'] = pd.Series(powers)

    # Calculate accumulated energy and flexible demand for the VPP
    for t in range(len(temperatures)):
        # Energy stored in each boiler at time t
        boiler_energy_t = boiler.C * (temperatures[t] - boiler.T_min) / (boiler.T_max - boiler.T_min)
        total_vpp_energy[t] += boiler_energy_t
        
        # Maximum and minimum energy that can be stored
        max_boiler_energy = boiler.C * (boiler.T_max - boiler.T_min)
        min_boiler_energy = boiler.C * (boiler.T_min - boiler.T_min)
        max_vpp_energy[t] += max_boiler_energy
        min_vpp_energy[t] += min_boiler_energy
        
        # Active power of the boiler and total VPP power
        total_vpp_power[t] += powers[t]
        
        # Flexible demand (up/down)
        if temperatures[t] < boiler.T_max:  # If the temperature hasn't reached the maximum, it can heat more
            flexible_demand_up[t] += resistor_power
        
        if temperatures[t] > boiler.T_min:  # If the temperature hasn't reached the minimum, it can reduce load
            flexible_demand_down[t] += resistor_power
