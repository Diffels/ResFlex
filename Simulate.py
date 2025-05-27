# Import required libraries
import os
import time
import json

# Import custom modules
import Appliances.Appliances as Appliances
import utils

def simulate_all(config, filetype='xlsx', plot_res=False, print_res=True):
    houses_params = utils.create_params(config)
    dic_df_P, dic_df_Flex, dic_Params = {}, {}, {}

    for u, house_params in enumerate(houses_params, start=1):
        user = f"House{u}"
        print(f"Simulating {user} ({u}/{len(houses_params)})")
        dic_df_P[user], dic_df_Flex[user], dic_Params[user] = one_profile(house_params)

    if filetype is not None: utils.save_all(config, filetype, dic_df_P, dic_df_Flex, dic_Params, houses_params)

    if print_res: utils.print_all(config, dic_df_P, dic_df_Flex, dic_Params)
    if plot_res: utils.plot_all(config, dic_df_P, dic_df_Flex, dic_Params)

def simulate_one(config, filetype='csv', plot_res=False, print_res=True):
    df_P, df_Flex, dic_Param = one_profile(config)

    if filetype is not None: utils.save_one(config, filetype, df_P, df_Flex, dic_Param)

    if print_res: utils.print_one(config, df_P, df_Flex, dic_Param)
    if plot_res: utils.plot_one(config, df_P, df_Flex, dic_Param)

def add_ComFlex_params(d):
    d["Name"] = "House"+str(d['id'])
    d["Price_idx"] = 1
    d["Node_idx"] = d['id']
    d['EV_data']['id'] = 1
    d['EV_data']['alpha'] = 1
    d['HP_data']['id'] = 1
    d['HP_data']['alpha'] = 1
    d['WB_data']['id'] = 1
    d['WB_data']['alpha'] = 1

    d['BSS'] = True
    d['BSS_data'] = {"Pmax": 5,
                     "SOC_min": 0.2, "SOC_max": 0.8,
                     "Capacity": 10, "eta": 0.9}

    d['PV'] = True
    d['PV_data'] = {"id": 1, "Pmax": 5}
    return d


def one_profile(config):
    '''
    Function that computes the different load profiles.

    Inputs:
        - config (dict): Dictionnay that contains all the inputs defined in Config.yaml
    
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

    utils.add_indices(df_P, df_Flex, config)

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
        simulate_one(config, plot_res=True)