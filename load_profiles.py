# -*- coding: utf-8 -*-
"""
@authors: duchmax, noedi
    
August 2024
"""

# Import required modules
import os
import pandas as pd
import numpy as np
from Household_mod import Household_mod
from plots import make_demand_plot
from Flexibility import flexibility_window
from ramp_mobility.EV_run import EV_run
from heating_3 import run_space_heating
from utils import index_to_datetime, occ_reshape
from Hot_water import hot_water_elec_consumption
import time
import random
import json
import xarray as xr
from constant import StaticLoad


def get_profiles(config, dwelling_compo):
    '''
    Function that computes the different load profiles.

    Inputs:
        - config (dict): Dictionnay that contains all the inputs defined in Config.xlsx
        - dwelling_compo (list): Containing the dwelling composition.
    
    Outputs: 
        - df (pd.DataFrame): Dataframe containing the results, ie for each time step, the consumption of each
        appliance.
        - times (np.ndarray): Execution time for each simulation.
        - loads (np.ndarray): Total load during the simulation.
    '''
    times = np.zeros(config['nb_households'])

    nminutes = config['nb_days'] * 1440 + 1
    P = np.zeros((config['nb_households'], nminutes))

    for i in range(config['nb_households']):
        start_time = time.time()

        #---Household creation (Base Load) -------------
        family = Household_mod(f"Scenario {i}", members=dwelling_compo, selected_appliances = config['appliances']) # print put in com 
        family.simulate(year = config['year'], ndays = config['nb_days']) # print in com
        df = pd.DataFrame(family.app_consumption)
        #------------------------------

        #---Space Heating -------------
        shsetting_data = family.sh_day
        heating_consumption = run_space_heating(shsetting_data, config['nb_days'], config['start_day'])*1e3 #return an array with powers in kW every 10min, times 1000 to have the results in Watts
        heating_cons_duplicate = [elem for elem in heating_consumption for _ in range(10)]   # To go from 10 to 1 min time step
        heating_cons_duplicate = pd.Series(heating_cons_duplicate)/4                         #divided by the COP of conventional heat pump 
        df['Heating'] = df.get('Heating', 0) + heating_cons_duplicate
        #------------------------------

        #---Hot Water -------------
        if config['HotWater']:
            hot_water = hot_water_elec_consumption(pd.DataFrame({'mDHW':family.mDHW}), config['year'], config['HotWater_max_power'])
            df['HotWater'] = hot_water.tolist()
        #------------------------------

        #---EV -------------
        if config['EV_presence'] >= random.random():
            # Reshaping of occupancy profile 
            occupancy = occ_reshape(family.occ_m, config['plot_ts'])
            # Determining EV parameter:
            sizes=['small', 'medium', 'large']
            config['EV_size'] = np.random.choice(sizes, p=config['prob_EV_size'])
            usages=['short', 'normal', 'long']
            config['EV_usage'] =  np.random.choice(usages, p=config['prob_EV_size'])
            powers=[3.7, 7.4, 11, 22] #kW
            config['EV_charger_power'] =  np.random.choice(powers, p=config['prob_EV_charger_power'])
            # Running EV module
            load_profile, n_charge_not_home =EV_run(occupancy,config)
            EV_profile = pd.DataFrame({'EVCharging':load_profile})
            # EV_flex = pd.DataFrame({'EVCharging':load_profile, 'Occupancy':occupancy})

            if 'EVCharging' not in df.columns:
                df['EVCharging'] =  EV_profile*1000
            else :
                df['EVCharging'] = df['EVCharging'] + EV_profile['EVCharging']*1000
        #------------------------------

        #---Flexibility -------------
        if config['flex_mode']: 
            pass
            # flex_window = flexibility_window(df[config['appliances'].keys()], family.occ_m, config['flex_mode'], flexibility_rate=config['flex_rate'])
        #------------------------------

        
        P[i,:] = family.P
        end_time = time.time()
        execution_time = end_time - start_time
        times[i] = execution_time
        print(f"Simulation {i+1}/{config['nb_households']} is done. Execution time: {execution_time} s.") 

        df = index_to_datetime(df, config['year'],config['plot_ts'])
        StaticLoad_pres = [col for col in StaticLoad if col in df.columns]
        data=df.copy()
        data.loc[:, 'Base Load'] = data[StaticLoad_pres].sum(axis=1)
        data= data.drop(columns=StaticLoad_pres)

        data_array = xr.DataArray(data, dims=['index', 'columns'], coords={'columns': data.columns}) 
        if i == 0 :
            dataset =  xr.Dataset({f'House {i}': data_array})
        else :
            dataset[f'House {i}'] = data_array
    dataset.coords['index'] = data.index

    P = np.array(P)
    
    total_elec = np.sum(P)
    average_total_elec = total_elec/config['nb_households']
    loads=average_total_elec.sum()/60/1000
    
    #df = df/config['nb_households']

    df = index_to_datetime(df, config['year'],config['plot_ts'])
    
    return loads, times, dataset


def simulate(file_path, disp=True):
    '''
    Simulation with a .json file.
    Input:
        - file (str): .json file path describing the configuration of the simulation.
        - disp (bool): Displaying informations about the simulation. 
    Outputs: 
        - df (pd.DataFrame): Dataframe containing the results, ie for each time step, the consumption of each
        appliance.
    '''
    with open(file_path, 'r', encoding="utf-8") as file: # 'utf-8' to avoid "é" issues
        config = json.load(file)  # Load the JSON data into a Python dictionary

    dwelling_compo = []
    for i in range(config['dwelling_nb_compo']):
        dwelling_compo.append(config[f'dwelling_member{i+1}'])

    if sum(config['prob_EV_size']) != 1: 
        raise ValueError(f"Probabilities associated to the EV size are incorrect. {config['prob_EV_size']}")
    if sum(config['prob_EV_usage']) != 1 and config['EV_km_per_year'] == 0: 
        raise ValueError(f"Probabilities associated to the EV usage are incorrect. {config['prob_EV_usage']}")
    if sum(config['prob_EV_charger_power']) != 1: 
        raise ValueError(f"Probabilities associated to the charger powers are incorrect. {config['prob_EV_charger_power']}")
    if (config['nb_days'] + config['start_day'] > 365) or (config['nb_days'] <= 0): 
        raise ValueError(f"Declaration of time horizon incorrect: {config['nb_days']} days to simulate, starting from {config['start_day']}th day of the year.")
    
    loads, times, dataset = get_profiles(config, dwelling_compo)

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Results.xlsx")

    with pd.ExcelWriter(file_path) as writer:
        for idx in dataset.data_vars:
            subset = dataset[idx].to_pandas()
            subset.to_excel(writer, sheet_name=idx)
    df = pd.DataFrame(0, index= range(0,len(dataset['index'].values)), columns=dataset['columns'].values)
    for var in dataset.data_vars :
        data_var = dataset[var].to_pandas()  
        df += data_var.fillna(0)
    df = df.set_index(dataset['index'].values)

    if disp: 
        print("---- Results ----")
        print("Time Horizon: ", config["nb_days"], "day(s).")
        print("Execution time [s]")
        print(f"\tMean: {np.mean(times)}")
        print("Total load [kWh]")
        print(f"\tMean: {round(np.mean(loads), 2)}; STD: {np.std(loads)}")
    
    if config['plot']:
        make_demand_plot(df.index, df, title=f"Load profile for {config['nb_households']} households, for {config['nb_days']} days.")