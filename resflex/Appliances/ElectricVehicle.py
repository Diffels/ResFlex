# -*- coding: utf-8 -*-
"""
@author: noedi
May 2025
"""

# Import required modules
import pandas as pd
import numpy as np
from typing import Any
from datetime import datetime
import random

def EV_simulate(occupancy: np.ndarray[Any, np.dtype[np.bool_]], config: dict)-> pd.DataFrame:
    '''
    Compute the EV load profile for a given occupancy profile and configuration file.
    Inputs:
        - occupancy: occupancy profile of the driver (1 if at home, 0 if not).
        - config: configuration file containing the parameters for the simulation.
    Outputs:
        - EV_profile: EV load profile for the household.
        - Flex_EV: DataFrame containing the EV load profile.
    '''         
    soc_init = config["EV_data"]["SoC_target"]

    EV_profile = np.zeros(1440*config["nb_days"]) # Time Series (minutes) recording the charging power of the EV battery.
    isplug = np.zeros(1440*config["nb_days"]) # Binary Time Series (minutes) describing when EV is plugged, ie when the EV is at home.
 
    for day in range(config["nb_days"]):
        daily_occupancy = occupancy[day*1440:1440*(day+1)] # with 1440 min/day
        daily_profile, daily_isplug, last_soc = EV_daily_profile(day, daily_occupancy, config, soc_init)

        EV_profile[day*1440:1440*(day+1)] = daily_profile
        isplug[day*1440:1440*(day+1)] = daily_isplug
        # Update SOC for the next day
        soc_init = last_soc # SOC at the end of the day

    Flex_EV = pd.DataFrame(isplug, columns=['EV'], index=None)   

    return EV_profile, Flex_EV

def EV_daily_kwh(config, curr_day: int)->float:
    '''
    Function to compute the stochastic daily energy consumption of the EV based on 
    the yearly km, the consumption in kWh/100km and the type of day (weekday/weekend).
    Inputs:
        - yearly_km: average yearly km driven by the EV.
        - consumption: consumption of the EV in kWh/100km.
        - curr_day: current day of the simulation
    Outputs:
        - rand_kwh: random daily energy consumption of the EV in kWh. 
    '''
    # Tunable Parameters
    r_d=0.3 # Random distance variation
    r_cons= 0.3 # Random consumption variation

    # Daily km based on the type of day (-7% for weekdays, +17.5% for weekends), approx. 25% difference.
    weekday =  (datetime(config["year"], 1, 1) + pd.Timedelta(days=config["start_day"] + curr_day - 1)).weekday()  # Monday = 0, Saturday = 5, Sunday = 6
    daily_km = config["EV_data"]["Usage"] / 365
    if weekday <= 4: # Weekday
        daily_km *= 0.93
    else: # Weekend
        daily_km *= 1.175
    rand_dist = round(random.uniform(daily_km*(1-r_d),daily_km*(1+r_d)))

    # Random EV consumption in kWh/100km
    rand_cons = random.uniform(config["EV_data"]["Consumption"]*(1-r_cons), config["EV_data"]["Consumption"]*(1+r_cons))

    rand_kwh = round(rand_cons * rand_dist / 100, 2)
            
    return rand_kwh

    
def prob_charge_not_home(E_journey, E_leaving):
    '''
    Probability the user want to charge EV depending on the energy spent 
    during journey and energy already in the car when leaving.
    Inputs:
        - E_journey: Energy required by the associated journey. [kWh]
        - E_leaving: Energy available in the EV battery. [kWh]
    Outputs:
        - P: Probability of a charging event, outside the home. [-]
    '''
    r = E_journey/E_leaving
    if r > 1.0: # If journey requires more energy than available, charge mandatory.
        P = 1.0
    elif r < 0.05: # Short journeys do not require charge.
        P = 0
    else: 
        P = r
    return P

def EV_daily_profile(day: int, occupancy: np.ndarray[Any, np.dtype[np.bool_]], config, SOC_init: float, SOC_min: float = 0.1):
    '''
    Function to compute the EV daily load profile for a given occupancy profile, a stochastic consumption in kWh, and EV parameters.
    Inputs:
        - stoch_kwh: stochastic daily energy consumption of the EV in kWh.
        - occupancy: occupancy profile of the driver (1 if at home, 0 if not).
        - battery_cap: capacity of the EV battery in kWh.
        - charger_power: power of the EV charger in kW.
        - eta: efficiency of the EV charger.
        - SOC_max: maximum state of charge of the EV battery.
        - SOC_init: initial state of charge of the EV battery.
        - SOC_min: minimum state of charge of the EV battery.
    Outputs:
        - load_profile: EV load profile for the household.
        - isplug: binary Time Series describing when EV is plugged.
        - SOC_profile: state of charge of the EV battery at the end of the day.
        - EV_refilled: Time Series showing the charge outside home, assumed instantaneous at the half of the departure event.
        - SOC_profile: Time Series recording the state of charge of the EV (plot use only).
    '''
    stoch_kwh = EV_daily_kwh(config, day)
    Cap = config["EV_data"]["Capacity"] # [kWh]
    charger_power = config["EV_data"]["Pmax"]
    eta = config["EV_data"]["eta"]
    SOC_max = config["EV_data"]["SoC_target"]
    # Tunable Parameters
    r_ch_notHome = 0.30 # Time ratio of EV charging when not at home from whole daily departure duration.
    var_ch_not_home = 0.05 # Stochastic variation in charging time ratio defined above.
    var_split = 0.25 # Stochastic variation in Energy split between not home windows.
    tol_batt_lim = 0.5 # Tolerance according to battery limits (min/max SOCs) when charge/disch.
    available_stations=[7.4, 11, 22, 50] # [kW], level 2 and 3 of EV chargers
    prob_stations=[0.3, 0.35, 0.3, 0.05] # The charge that occurs outside home is not always the same that home charger
    station_power = np.random.choice(available_stations, p=prob_stations)
    
    SOC_last=0
    E_spent=0
    ''' ----------------- Pre-processing of the occupancy profile ------------------ '''
    # Vectorized way to find transitions from home (1) to away (0)
    depart_idx = np.where((occupancy[:-1] == 1) & (occupancy[1:] == 0))[0] + 1
    arrivals = np.where((occupancy[:-1] == 0) & (occupancy[1:] == 1))[0] + 1
    print(f"Departures: {depart_idx}")
    print(f"Arrivals: {arrivals}")
    print(f"Arrivals: {arrivals}")
    durations = []
    departures = {}
    tot_time_left = 0
    for d in depart_idx:
        a = arrivals[arrivals > d]
        if a.size > 0:
            duration = a[0] - d
            if duration >= 10:
                departures[d] = [duration]
                durations.append(duration)
                tot_time_left += duration
    ''' ------------ Main loop during the day to compute the daily profile ------------- '''
    fully_charged=False
    SOC_profile = np.full(1440, SOC_init) # Time Series recording the SOC of the EV battery (0 if not home).
    EV_refilled=np.zeros(1440) # Time Series recording if a battery re-filled occurs during a departure. (not used here)
    isplug = np.zeros(1440) # Binary Time Series describing when EV is plugged.
    ischarging = np.zeros(1440) # Time Series recording the charging power of the EV battery (0 if not home).

    for i in range(1, len(occupancy)):
        if not occupancy[i]: # Not at home
            if i in departures.keys(): # Event departure
                fully_charged=False
            # Adding departures E_spent [kWh] to dict according to stochastic ratio [%]
                #print("Dep:", departures[i], "-", i)
                t_departure = departures[i][0]
                ratio = t_departure/tot_time_left
                stoch_ratio = round(ratio * random.uniform((1-var_split),(1+var_split)), 2)
                E_spent = stoch_kwh*stoch_ratio
                departures[i].append(E_spent)
                
            # Probability to charge during the departure, function of E_spent and E_leaving
                SOC_last = SOC_profile[i-1]
                E_leaving = SOC_last*Cap # [kWh]
                P_ch_notHome = prob_charge_not_home(E_spent, E_leaving)
                            
                if random.random() <= P_ch_notHome:
                    print(f"Charging outside home at {i} [min] with {round(E_spent,2)} [kWh] and {round(E_leaving,2)} [kWh] in the battery.")
                    t_charge = round(r_ch_notHome*t_departure*random.uniform((1-var_ch_not_home),(1+var_ch_not_home))) # Stochastic charge time [min]
                    E_charge = station_power * t_charge * eta / 60 # [kWh], t_charge in min
                    E_arrive = E_leaving-E_spent+E_charge
                    
                    # Control to avoid not enough charge:
                    # If a long journey occurs and, despite the charge not home, the EV is coming
                    # home with SOC_i < SOC_min, the charge must be longer. In this specific case, EV comes
                    # back home with SOC_min.
                    if E_arrive < SOC_min*Cap:
                        E_charge = SOC_min*Cap + E_spent - E_leaving

                    # Control to avoid to much charge:
                    # Since the charge is supposed to be at half journey, if after the charge
                    # EV is at SOC_max (ie max. charge occured), then SOC_arrive must be SOC_max
                    # diminished by half the journey consumption.
                    if E_arrive > SOC_max*Cap:
                        E_charge = SOC_max*Cap - E_leaving + E_spent/2

                    # Update Energy spent, either half the journey, or charging < E_spent/2
                    E_spent = max(E_spent/2, E_spent-E_charge)
                    
                    if E_spent < 0:
                        raise ValueError(f"Error in EV_occ_daily_profile.py: E_spent less than 0 at {i} min.")
                    half_dep = round(i + t_departure/2)
                    EV_refilled[half_dep] = E_charge
                    temp_duration = departures[i][0]
                    departures.update({i: [temp_duration, E_spent]}) # The Energy spent is diminished by E_charge, dict update. 
                    # print(f"A battery re-filled occured between {i} [min] and {i+t_departure} [min] of {round(E_charge,2)} [kWh] (+{round(100*E_charge/battery_cap, 2)}%).")
                    
            SOC_profile[i]=0
            
        else: # Is at home
            isplug[i]=1
            if i in arrivals: # Event arrival: EV is coming home
                # Update SOC with discharge from previous journey
                SOC_i = SOC_last - E_spent/Cap
                if SOC_i < SOC_min*(1-tol_batt_lim):
                    SOC_i = SOC_min*(1-tol_batt_lim)
                    #raise ValueError(f"Error in daily_EV_profile.py: SOC at {i} [min] is {SOC_i} [-] which is lower than SOC_min ({SOC_min} [-]).") 
                ischarging[i]=1
            else: # Event charge: EV charge at nominal power until SOC_max
                # Update SOC with charge from home charging station
                if fully_charged:
                    # SOC is fully charged, no need to charge.
                    SOC_i = SOC_profile[i-1]
                else:
                    # Charging Event
                    E_charge = charger_power / 60 * eta # kWh
                    new_SOC = SOC_profile[i-1] + E_charge/Cap
                    if new_SOC > SOC_max:
                        SOC_i = SOC_max
                        fully_charged=True
                    else:
                        SOC_i = new_SOC                    
                        ischarging[i]=1
            # Update SOC profile
            SOC_profile[i] = SOC_i
    ''' -------------------------------------------------------------------------------- '''
    isplug[0] = isplug[1] # First minute of the day is outside the loop, assumed to be equal to the second one.
    load_profile = np.multiply(ischarging, charger_power)

    return load_profile, isplug, SOC_profile