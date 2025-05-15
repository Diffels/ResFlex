# -*- coding: utf-8 -*-
"""
@authors: duchmax, noedi
    
August 2024
"""

# Import required libraries
import os
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

# Import custom modules
from Household_mod import Household_mod
from EV import EV_run
from Heating import space_heating
from Hot_water import water_boiler
from plots import plot_P, plot_EV

def check_probas(fields, config):
    for field in fields:
        if sum(config[f'P_{field}']) != 1:
            raise Exception(f"Error: {field} probability sum is not equal to 1.")
        if len(config[f'{field}']) != len(config[f'P_{field}']):
            raise Exception(f"Error: {field} probability length is not equal to the number of params.")

def append_recurring(fields, list_param, config):
    for i in list_param:
        for field in fields:
            i[field]= config[field]
    return list_param

def append_appliances(list_param, config):
    app_list = ['WashingMachine', 'DishWasher', 'TumbleDryer', 'WasherDryer']
    values = [np.random.choice([0,1], size=config['nb_households'], p=[1-config['appliances'][f'P_{a}'], config['appliances'][f'P_{a}']]) for a in app_list]
    for i, house in enumerate(list_param):
        house['appliances'] = {}
        for a, appliance in enumerate(app_list):
            house['appliances'][appliance] = int(values[a][i])
    return list_param

def probas_to_list(appliance, field, config):
    probas = config[f'{appliance}_data'][f'P_{field}']
    values = config[f'{appliance}_data'][field]
    return np.random.choice(values, size=config['nb_households'], p=probas)

def append_family(list_param, config):
    probas = config['P_inhabitants']
    values = config['inhabitants']
    family = np.random.choice(values, size=config['nb_households'], p=probas)
    for i, house in enumerate(list_param):
        house['family'] = int(family[i])
        house['occupations'] = ['Random'] * family[i]
    return list_param

def append_flexible(appliance, fields, list_param, config):
    lists = {field: probas_to_list(appliance, field, config) for field in fields}
    app = np.random.choice([1, 0], size=config['nb_households'], p=[config[f'P_{appliance}'], 1 - config[f'P_{appliance}']])
    for i, house in enumerate(list_param):
        house[appliance] = bool(app[i])
        house[f'{appliance}_data'] = {}
        for f in fields:
            house[f'{appliance}_data'][f] = float(lists[f][i])
    return list_param

def get_list_param(config):
    list_param = [{}] * config['nb_households']
    list_param = append_recurring(['nb_days', 'timestep', 'year', 'start_day', 'flexibility'], list_param, config)
    list_param = append_appliances(list_param, config)
    list_param = append_flexible('SpaceHeating',['Year', 'Size', 'Floors','P_th_nom', 'COP'], list_param, config)
    list_param = append_flexible('HotWater', ['Pmax', 'Volume', 'Tset'], list_param, config)
    list_param = append_flexible('EV', ['Consumption', 'Capacity', 'Pmax', 'eta', 'SoC_target', 'Usage'], list_param, config)
    list_param = append_family(list_param, config)
    return list_param

def create_params(config):
    #Check if the config file is valid
    check_probas(['Year', 'Size', 'Floors','P_th_nom', 'COP'], config['SpaceHeating_data'])
    check_probas(['Pmax', 'Volume', 'Tset'], config['HotWater_data'])
    check_probas(['Consumption', 'Capacity', 'Pmax', 'eta', 'SoC_target', 'Usage'], config['EV_data'])
    check_probas(['inhabitants'], config)
    params = get_list_param(config)
    return params

def simulate_all(file_path, plot=False, disp=True):
    '''
    Simulation with a .json file.
    Input:
        - file (str): .json file path describing the configuration of the simulation.
        - disp (bool): Displaying informations about the simulation. 
    Outputs: 
        - df (pd.DataFrame): Dataframe containing the results, ie for each time step, the consumption of each
        appliance.
    '''
    start_time = datetime.now()
    with open(file_path, 'r', encoding="utf-8") as file: # 'utf-8' to avoid "é" issues
        config = json.load(file)  # Load the JSON data into a Python dictionary

    houses_params = create_params(config)
    dic_Param = {}
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Results/Multiple/{current_time}.xlsx")
    flex_filename = filename.replace('.xlsx', '_Flex.xlsx')

    with pd.ExcelWriter(filename) as writer, pd.ExcelWriter(flex_filename) as flex_writer:
        for u in range(config['nb_households']):
            print(f"Simulating house {u+1}/{config['nb_households']}")
            df_P, df_Flex, dic_Param[f"House{u}"] = one_profile(houses_params[u])
            df_P.to_excel(writer, sheet_name=f"House{u}")
            df_Flex.to_excel(flex_writer, sheet_name=f"House{u}")
    
    # Write dic_Param to a JSON file
    with open(filename.replace('.xlsx', '_Param.json'), 'w', encoding="utf-8") as json_file:
        json.dump(dic_Param, json_file, ensure_ascii=False, indent=4)

    time = datetime.now() - start_time
    if disp: 
        print("---- Results ----")
        print("Time Horizon: ", config["nb_days"], "day(s).")
        print(f"Execution time {time} s")
    
    if plot:
        pass #TODO: plot all the results in one figure

def simulate_one(file_path, plot=False, disp=True):
    '''
    Simulation with a .json file.
    Input:
        - file (str): .json file path describing the configuration of the simulation.
        - plot (bool): Plotting the results.
        - disp (bool): Displaying informations about the simulation. 
    Outputs: 
        - Saved files in Results folder
    '''

    start_time = datetime.now()
    with open(file_path, 'r', encoding="utf-8") as file: # 'utf-8' to avoid "é" issues
        config = json.load(file)  # Load the JSON data into a Python dictionary

    df_P, df_Flex, dic_Param = one_profile(config)
    # Write df_P to a CSV file with the current date and time in the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Results/Single/{current_time}.csv")
    df_P.to_csv(csv_filename, index=True)
    df_Flex.to_csv(csv_filename.replace('.csv', '_Flex.csv'), index=True)
    # Write dic_Param to a JSON file
    json_filename = csv_filename.replace('.csv', '_Param.json')
    with open(json_filename, 'w', encoding="utf-8") as json_file:
        json.dump(dic_Param, json_file, ensure_ascii=False, indent=4)

    time = datetime.now() - start_time
    if disp: 
        print("---- Results ----")
        print("Time Horizon: ", config["nb_days"], "day(s).")
        print(f"Execution time {time} s")
        #print(f"Total load {config['timestep']*sum(df_P['Total'])/60} kWh")
    
    if plot:
        plot_P(df_P)#, title=f"Load profile for {config['nb_households']} households, for {config['nb_days']} days.")
        plot_EV(df_Flex['EVCharging'], df_Flex['Occupancy'], df_Flex['Load'], df_Flex['EV_refilled'])
        
def one_profile(config):
    '''
    Function that computes the different load profiles.

    Inputs:
        - config (dict): Dictionnay that contains all the inputs defined in Config.yaml
    
    Outputs: 
        - df_P (pd.DataFrame): Dataframe containing power consumption of each appliance with 1-minute resolution.
        - df_Flex (pd.DataFrame): Dataframe containing flexibility of each appliance with 1-minute resolution.
        - dic_Param (dict): Dictionary containing fixed parameters for the household.
    '''
    
    start_time = time.time()
    start_date = datetime(2024, 1, 1) + pd.Timedelta(days=config["start_day"])
    end_date = start_date + pd.Timedelta(days=config["nb_days"])
    time_index = pd.date_range(start=start_date, end=end_date, freq="min")  # Minute by minute frequency

    #---Household creation (Base Load and indices) -------------
    family = Household_mod(f"Scenario: ", members=config['occupations'], selected_appliances = config['appliances']) # print put in com 
    family.simulate(year = config['year'], ndays = config['nb_days']) # print in com
    df_P = pd.DataFrame(family.app_consumption.copy())
    df_P.index = time_index[:len(df_P)] if len(time_index) >= len(df_P) else time_index


    df_Flex = pd.DataFrame(index=df_P.index)
    dic_Param = {}
    #------------------------------

    #---Space Heating -------------
    if config['SpaceHeating']:
        shsetting_data = family.sh_day
        P_HP, Flex_HP, Param_HP = space_heating(shsetting_data, config['nb_days'], config['start_day'], config['SpaceHeating_data']['Year'], 
                                                config['SpaceHeating_data']['Size'], config['SpaceHeating_data']['Floors'], 
                                                config['SpaceHeating_data']['P_th_nom'])
        df_P['Heating'] = P_HP/config['SpaceHeating_data']['COP'] 
        Param_HP['COP'] = config['SpaceHeating_data']['COP']
        Flex_HP.index = df_Flex.index
        df_Flex = pd.concat([df_Flex, Flex_HP], axis=1)
        dic_Param["HP"] = Param_HP
    #------------------------------

    #---Water Boiler-------------
    if config['HotWater']:
        P_WB, Flex_WB, Param_WB  = water_boiler(pd.DataFrame({'mDHW':family.mDHW}), config['year'],
                                                config['HotWater_data']['Pmax'])
        df_P['HotWater'] = P_WB.tolist()
        Flex_WB.index = df_Flex.index
        df_Flex = pd.concat([df_Flex, Flex_WB], axis=1)
        dic_Param["WB"] = Param_WB
    #------------------------------

    #---EV-----------------
    if config['EV']:        
        # Redefining occupancy profile: (1: Active, 2: Sleeping)-> 1: At Home; (3: Not at home)-> 0: Not at home
        EV_occ = np.where(np.isin(family.occ_m, [1, 2]), 1, 0)
        # Running EV module
        P_EV, Flex_EV, Param_EV = EV_run(EV_occ,config)
        df_P['EVCharging'] =  P_EV.tolist()
        Flex_EV.index = df_Flex.index
        df_Flex = pd.concat([df_Flex, Flex_EV], axis=1)
        dic_Param["EV"] = Param_EV
    #------------------------------

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Simulation is done. Execution time: {execution_time:.2f} s.") 
    return df_P, df_Flex, dic_Param