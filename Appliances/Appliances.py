import pandas as pd
import numpy as np

from .ElectricVehicle2 import EV_simulate#, add_params_EV
from .HeatPump import HP_simulate, add_params_HP
from .WaterBoiler import WB_simulate, add_params_WB
from .StROBe.Household_mod import Household_mod


def complete_params(config):
    config = add_params_HP(config)
    config = add_params_WB(config)
    # config = add_params_EV(config) # Currently no param to be added
    return config

def get_baseload(config):
    #---Household creation (Base Load and occupancy) -------------
    family = Household_mod(f"Scenario: ", members=config['occupations'], selected_appliances = config['appliances']) # print put in com 
    family.simulate(year = config['year'], ndays = config['nb_days']) # print in com
    df_P = pd.DataFrame(family.app_consumption.copy() / 1e3, index=None)
    df_Flex = pd.DataFrame(family.occ_m.copy()[:len(df_P)], index=None)
    df_Flex.columns = ['Occupancy']
    return df_P, df_Flex, family

def add_HP(df_P, df_Flex, family, config):
    P_HP, Flex_HP = HP_simulate(family.sh_day, config)
    df_P['P_HP'] = P_HP/config['HP_data']['COP'] 
    df_Flex = pd.concat([df_Flex, Flex_HP], axis=1)
    return df_P, df_Flex

def add_WB(df_P, df_Flex, family, config):
    P_WB, Flex_WB  = WB_simulate(pd.DataFrame({'mDHW':family.mDHW}),config)
    df_P['P_WB'] = (P_WB/1e3).tolist()
    df_Flex = pd.concat([df_Flex, Flex_WB], axis=1)
    return df_P, df_Flex

def add_EV(df_P, df_Flex, family, config):   
    # Redefining occupancy profile: (1: Active, 2: Sleeping)-> 1: At Home; (3: Not at home)-> 0: Not at home
    EV_occ = np.where(np.isin(family.occ_week[0], [1, 2]), 1, 0)
    # Running EV module
    P_EV, Flex_EV = EV_simulate(EV_occ,config)
    # print(len(P_EV.tolist()))
    # print(len(Flex_EV.tolist()))
    # EV_flex = pd.DataFrame({'EVCharging':load_profile, 'Occupancy':occupancy})
    df_P['P_EV'] =  P_EV.tolist()
    df_Flex['EV'] = Flex_EV.tolist()
    
    return df_P, df_Flex