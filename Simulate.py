# Import required libraries
import os
import time
import json
import numpy as np

# Import custom modules
import Appliances.Appliances as Appliances
import utils
from stochastic import set_seed
set_seed(369) # Need to be modified for different users to set_seed(user_specific_seed)


def simulate_all(config, save=True, plot_res=False, print_res=True):
    houses_params = utils.create_params(config)
    dic_df_P, dic_df_Flex, dic_Params = {}, {}, {}

    for u, house_params in enumerate(houses_params, start=1):
        user = f"household_{u}"
        print(f"Simulating {user} ({u}/{len(houses_params)})")
        dic_df_P[user], dic_df_Flex[user], dic_Params[user] = one_profile(house_params)

    if save: 
        print("Saving results...")
        utils.save_all(config, dic_df_P, dic_df_Flex, dic_Params, houses_params)

    if print_res: utils.print_all(config, dic_df_P, dic_df_Flex, dic_Params)
    if plot_res: utils.plot_all(config, dic_df_P, dic_df_Flex, dic_Params)

def simulate_one(config, save=True, plot_res=False, print_res=True):
    df_P, df_Flex, dic_Param = one_profile(config)

    if save: utils.save_one(config, df_P, df_Flex, dic_Param)

    if print_res: utils.print_one(config, df_P, df_Flex, dic_Param)
    if plot_res:
        # Poster colors:
        colors = ["#646262",  "#95e2ea", '#6c96c2', '#d4524f', '#a87e5b']  # Add hex color codes (same order as nice_cols)
        # New proposed colors:"#c6dee1",'xlabel': 'Time',
        #colors = ["#a7a7a7", "#45bde9", "#3a74e9", "#ea8f45", "#ed5151", "#80d671"]
        dic_plot = {'show': True, 'save': True, 'fontsize': 44, 'figsize_cm':(66, 10), 'title': 'Power Consumption for one Household',
                     'ylabel': 'Power (kW)', 'grid': True, 'legend': True, 'colors': colors} 
        utils.plot_one(df_P, dic_plot, pdf=True)

def add_ComFlex_params(config):
    config["Price_idx"] = 1

    if config["EV"]:
        config['EV_data']['id'] = 1
        config['EV_data']['alpha'] = 1
    if config["HP"]:
        config['HP_data']['id'] = 1
        config['HP_data']['alpha'] = 1
    if config["WB"]:
        config['WB_data']['id'] = 1
        config['WB_data']['alpha'] = 1

    return config


def one_profile(config):
    '''
    Function that computes the different load profiles.

    Inputs:
        - config (dict): Dictionnay that contains all the inputs defined in input_single.json / input_mult.json
    
    Outputs: 
        - df_P (pd.DataFrame): Dataframe containing power consumption of each appliance with 1-minute resolution.
        - df_Flex (pd.DataFrame): Dataframe containing flexibility of each appliance with 1-minute resolution.
        - config (dict): Dictionary containing fixed parameters for the household.
    '''
    
    start_time = time.time()

    config = Appliances.complete_params(config)
    df_P, df_Flex, family = Appliances.get_baseload(config)

    if config['EV']: df_P, df_Flex = Appliances.add_EV(df_P, df_Flex, family, config)
    if config['HP']: df_P, df_Flex = Appliances.add_HP(df_P, df_Flex, family, config)
    if config['WB']: df_P, df_Flex = Appliances.add_WB(df_P, df_Flex, family, config)

    if config['timestep'] > 1:
        df_P, df_Flex = utils.set_timesteps(df_P, df_Flex, config) # changer timestep

    config = add_ComFlex_params(config)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Simulation is done. Execution time: {execution_time:.2f} s.") 
    return df_P, df_Flex, config


if __name__ == '__main__':
    mult = True

    if mult:
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input_mult.json")
        with open(file_path, 'r', encoding="utf-8") as file: config = json.load(file)  # Load the JSON data into a Python dictionary
        simulate_all(config)
    else:
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input_single.json")
        with open(file_path, 'r', encoding="utf-8") as file: config = json.load(file)  # Load the JSON data into a Python dictionary
        simulate_one(config, print_res=False, plot_res=True)
