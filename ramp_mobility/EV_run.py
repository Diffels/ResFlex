# -*- coding: utf-8 -*-
"""
@author: noedi
    
August 2024
"""

# Import required modules
import pandas as pd
import numpy as np
from ramp_mobility.EV_stoch_cons import EV_stoch_cons
from ramp_mobility.EV_occ_daily_profile import EV_occ_daily_profile
from ramp_mobility.config_init_ import config_init_
from typing import Any
from plots import plot_EV
import os



def electric_vehicle(occupancy: np.ndarray[Any, np.dtype[np.bool_]], config: dict, plot=False)-> pd.DataFrame:
    '''
    Code based on ramp-mobility library that computes stochastic Electrical 
    Vehicle load profile for predefined types of user, on yearly or daily basis.
    The profile is sctochastically linked to an occupancy behaviour in 
    EV_occ_daily_profile.py and from a stochastic EV consumption given by EV_stoch_cons.py. 

    The config variable is a dictionnary containing whole configuration used in the simulation. 
    config = {'nb_days'-'start_day'-'country'-'year'-'car'-'usage'-'charger_power'}

    Please refer to the file main.py/base_load.py for further explanations.
    '''
    # If this EV_km_per_year is correctly set, it takes into account nb of km/y instead of usage probabilities
    usage = int(round(config['EV_km_per_year'])) if config['EV_km_per_year'] > 1 else config['EV_usage']

    Driver = config_init_(config['EV_size'], usage, config['country'])

    EV_cons, EV_dist, EV_time = EV_stoch_cons(Driver, config['nb_days'], year=config['year'], country=config['country'], start_day=config['start_day'])

    SOC, bin_charg, EV_refilled, P_ref_EV = EV_occ_daily_profile(EV_cons, occupancy, Driver, config['EV_charger_power'], SOC_init=0.9)

    P_ref_EV = pd.DataFrame(EV_cons, columns=['P_ref_EV'])
    if plot:
        plot_EV(SOC, occupancy, P_ref_EV, EV_refilled)

    Param_EV = {"Charger":config['EV_charger_power']}
    Flex_EV = pd.DataFrame(bin_charg, columns=['Flex_EV'])   
    return P_ref_EV, Flex_EV, Param_EV