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
from Appliances.StROBe.Household_mod import Household_mod
from Appliances.EV import EV_run
from Appliances.Heating import space_heating
from Appliances.Hot_water import water_boiler
import utils

def simulate_all(config, filetype='xlsx', plot_res=False, print_res=True):
    houses_params = utils.create_params(config)
    dic_df_P, dic_df_Flex, dic_Params = {}, {}, {}

    for u, house_params in enumerate(houses_params, start=1):
        user = f"House{u}"
        print(f"Simulating {user} ({u}/{len(houses_params)})")
        dic_df_P[user], dic_df_Flex[user], dic_Params[user] = one_profile(house_params)

    if filetype is not None: utils.save_all(config, dic_df_P, dic_df_Flex, dic_Params, houses_params, filetype=filetype)

    if print_res: utils.print_all(config, dic_df_P, dic_df_Flex, dic_Params)
    if plot_res: utils.plot_all(config, dic_df_P, dic_df_Flex, dic_Params)

def simulate_one(config, filetype='csv', plot_res=False, print_res=True):
    df_P, df_Flex, dic_Param = one_profile(config)

    if filetype is not None: utils.save_one(config, df_P, df_Flex, dic_Param)

    if print_res: utils.print_one(config, df_P, df_Flex, dic_Param, time)
    if plot_res: utils.plot_one(config, df_P, df_Flex, dic_Param)

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

    #---Household creation (Base Load and indices) -------------
    family = Household_mod(f"Scenario: ", members=config['occupations'], selected_appliances = config['appliances']) # print put in com 
    family.simulate(year = config['year'], ndays = config['nb_days']) # print in com
    df_P = pd.DataFrame(family.app_consumption.copy(), index=None)
    df_Flex = pd.DataFrame(index=None)
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
        df_Flex = pd.concat([df_Flex, Flex_HP], axis=1)
        dic_Param["HP"] = Param_HP
    #------------------------------

    #---Water Boiler-------------
    if config['HotWater']:
        P_WB, Flex_WB, Param_WB  = water_boiler(pd.DataFrame({'mDHW':family.mDHW}),
                                                config['HotWater_data']['Pmax'])
        df_P['HotWater'] = P_WB.tolist()
        df_Flex = pd.concat([df_Flex, Flex_WB], axis=1)
        dic_Param["WB"] = Param_WB
    #------------------------------

    #---EV-----------------
    if config['EV']:        
        # Redefining occupancy profile: (1: Active, 2: Sleeping)-> 1: At Home; (3: Not at home)-> 0: Not at home
        EV_occ = np.where(np.isin(family.occ_m, [1, 2]), 1, 0)
        # Running EV module
        P_EV, Flex_EV, Param_EV = EV_run(EV_occ,config)
        # EV_flex = pd.DataFrame({'EVCharging':load_profile, 'Occupancy':occupancy})
        df_P['EVCharging'] =  P_EV.tolist()
        df_Flex = pd.concat([df_Flex, Flex_EV], axis=1)
        dic_Param["EV"] = Param_EV
    #------------------------------
    utils.add_indices(df_P, df_Flex, config)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Simulation is done. Execution time: {execution_time:.2f} s.") 
    return df_P, df_Flex, dic_Param

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input_mult.json")
    with open(file_path, 'r', encoding="utf-8") as file: config = json.load(file)  # Load the JSON data into a Python dictionary
    simulate_all(config, filetype=None, plot_res=False, print_res=True)