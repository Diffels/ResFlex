# -*- coding: utf-8 -*-
"""
@author: noedi
    
August 2024
"""

# Import required modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ramp_mobility.EV_stoch_cons import EV_stoch_cons
from ramp_mobility.EV_occ_daily_profile import EV_occ_daily_profile
from ramp_mobility.config_init_ import config_init_
from typing import Any



def EV_run(occupancy: np.ndarray[Any, np.dtype[np.bool_]], config: dict, plot=True)-> pd.DataFrame:
    '''
    Code based on ramp-mobility library that computes stochastic Electrical 
    Vehicle load profile for predefined types of user, on yearly or daily basis.
    The profile is sctochastically linked to an occupancy behaviour in 
    EV_occ_daily_profile.py and from a stochastic EV consumption given by EV_stoch_cons.py. 

    The config variable is a dictionnary containing whole configuration used in the simulation. 
    config = {'nb_days'-'start_day'-'country'-'year'-'car'-'usage'-'charger_power'}

    Please refer to the file main.py/base_load.py for further explanations.
    '''

    '''CONFIGURATION'''
    nb_days = config['nb_days']
    start_day = config['start_day']
    country = config['country']
    year = config['year']
    car = config['EV_size']
    usage = config['EV_usage']
    charger_pow = config['EV_charger_power']
    km_per_year = config['EV_km_per_year']
    
    # If this input is correctly set, it takes into account nb of km/y instead of usage probabilities
    if km_per_year > 1:
        usage = int(round(km_per_year))

    Driver = config_init_(car, usage, country)

    EV_cons, EV_dist, EV_time = EV_stoch_cons(Driver, nb_days, year=year, country=country, start_day=start_day)

    SOC, bin_charg, EV_refilled, load_profile, EVplug = EV_occ_daily_profile(EV_cons, occupancy, Driver, charger_pow, SOC_init=0.9)
    n_charge_not_home = np.count_nonzero(EV_refilled)

    #df_load_profile.to_excel('EV_load_profile.xlsx', index=False)

    if plot:
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        # Plot SOC
        axs[0].plot(SOC, color='blue')
        axs[0].set_ylabel('SOC (%)')
        axs[0].set_title('State of Charge Over Time')
        axs[0].grid(True)
        # Plot occupancy
        axs[1].plot(occupancy, color='green')
        axs[1].set_ylabel('Occupancy')
        axs[1].set_title('Occupancy Over Time')
        axs[1].grid(True)
        # Plot load
        axs[2].plot(load_profile, color='orange')
        axs[2].set_ylabel('Load (kW)')
        axs[2].set_title('Load Profile Over Time')
        axs[2].grid(True)
        # Plot EV_refilled (assuming they're relevant)
        axs[3].plot(EV_refilled, color='purple', linestyle='--', label='EV Refilled')
        axs[3].set_ylabel('Charging')
        axs[3].set_xlabel('Time [min]')
        axs[3].set_title('EV Charge while not home')
        axs[3].legend()
        axs[3].grid(True)
        plt.tight_layout()
        file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(file_path)
        full_path = os.path.join(current_dir, "EV_plot.svg")
        plt.savefig(full_path)
        plt.close()

    return load_profile, n_charge_not_home, EVplug
    