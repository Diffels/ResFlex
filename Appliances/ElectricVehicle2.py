# Import required modules
import pandas as pd
import numpy as np
from typing import Any
from datetime import datetime
import random

def get_weekly_journey_times(occupancy: np.ndarray[Any, np.dtype[np.bool_]], config: dict):
    t_dep = np.where((occupancy[:-1] == 1) & (occupancy[1:] == 0))[0] + 1
    t_arr = np.where((occupancy[:-1] == 0) & (occupancy[1:] == 1))[0] + 1

    # Ensure each departure has a corresponding arrival after it
    t_dep = t_dep[t_dep < t_arr[-1]] if len(t_arr) > 0 else t_dep
    t_arr = np.array([a for a in t_arr if a > t_dep[0]]) if len(t_dep) > 0 else t_arr
    # Pair up departures and arrivals
    min_len = min(len(t_dep), len(t_arr))
    t_dep = t_dep[:min_len]
    t_arr = t_arr[:min_len]
    dur_travel = t_arr - t_dep

    # Shift t_dep and t_arr by day_shift*24*60, then take modulo 7*24*60
    day_shift = (datetime(config["year"], 1, 1) + pd.Timedelta(days=config["start_day"])).weekday() # Monday = 0, Saturday = 5, Sunday = 6
    minutes_per_week = 7 * 24 * 60
    shift = day_shift * 24 * 60
    t_dep = (t_dep + shift) % minutes_per_week
    t_arr = (t_arr + shift) % minutes_per_week

    # Sort by t_dep to put smaller values in front, and reorder t_arr and dur_travel accordingly
    sort_idx = np.argsort(t_dep)
    t_dep = t_dep[sort_idx]
    t_arr = t_arr[sort_idx]
    dur_travel = dur_travel[sort_idx]

    # Remove travels that last less than 30 mins
    valid = dur_travel >= 30
    t_dep = t_dep[valid]
    t_arr = t_arr[valid]
    dur_travel = dur_travel[valid]
    # Remove stays that are less than 30 mins
    dur_stay = t_dep[1:] - t_arr[:-1]
    valid = dur_stay >= 30
    t_dep = np.concatenate(([t_dep[0]], t_dep[1:][valid]))
    t_arr = np.concatenate((t_arr[:-1][valid], [t_arr[-1]]))
    dur_travel = t_arr - t_dep

    return t_arr, t_dep, dur_travel

def get_daily_cons(config: dict) -> np.ndarray:
    # Tunable Parameters
    r_dist_w = 0.15  # Random weekly distance variation
    r_dist_d = 0.3  # Random daily distance variation
    r_cons = 0.2  # Random consumption variation

    day_shift = (datetime(config["year"], 1, 1) + pd.Timedelta(days=config["start_day"])).weekday() # Monday = 0, Saturday = 5, Sunday = 6

    weekly_km = round(random.uniform(config["EV_data"]["Usage"]*7*(1-r_dist_w) / 365,config["EV_data"]["Usage"]*7*(1+r_dist_w) / 365))
    # Compute weekday vector for the week, shifted by day_shift assign a daily usage to each day
    # Average multiplied 0.93 for weekdays, 1.175 for weekends
    daily_km = np.array([0.93*weekly_km/7 if (day_shift + d) % 7 <= 4 else 1.175*weekly_km/7 for d in range(7)])
    daily_km_rand = np.round(np.random.uniform(daily_km*(1 - r_dist_d),daily_km*(1 + r_dist_d)))

    # Random EV consumption in kWh/100km for this week
    weekly_cons = random.uniform(config["EV_data"]["Consumption"]*(1-r_cons), config["EV_data"]["Consumption"]*(1+r_cons))
    return np.round(weekly_cons * daily_km_rand / 100, 2)

def charging_outside(E_journey, E_leaving):
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
        print(f"Charging outside: {E_journey/2} kWh")
        return E_journey/2
    elif r < 0.1: # Short journeys do not require charge.
        return E_journey
    else: 
        P = E_journey / 2 if random.random() < r else E_journey
        if P < E_journey: print(f"Charging outside: {P} kWh")
    return P

def weekly_charging(config: dict, t_arr: np.ndarray, t_dep: np.ndarray, dur_travel: np.ndarray) -> np.ndarray:

    # Remove randomly 20% of the driving events
    num_trips = len(t_arr)
    num_remove = int(num_trips * 0.2)
    if num_remove > 0:
        keep_indices = np.sort(np.random.choice(num_trips, num_trips - num_remove, replace=False))
        t_arr = t_arr[keep_indices]
        t_dep = t_dep[keep_indices]
        dur_travel = dur_travel[keep_indices]
    
    # Get t_arr and t_dep per day
    t_arr_day = [t_arr[(t_arr >= d*24*60) & (t_arr < (d+1)*24*60)] for d in range(7)]
    t_dep_day = [t_dep[(t_dep >= d*24*60) & (t_dep < (d+1)*24*60)] for d in range(7)]
    dur_travel_day = [dur_travel[(t_arr >= d*24*60) & (t_arr < (d+1)*24*60)] for d in range(7)]

    # For each day, ensure first t_arr < first t_dep and last t_dep > last t_arr, remove those that aren't
    for arr, dep, dur in zip(t_arr_day, t_dep_day, dur_travel_day):
        if len(arr) == 0 or len(dep) == 0: 
            arr = []
            dep = []
            dur = []
            continue
        # Remove first arr if it is not before first dep
        if arr[0] <= dep[0]: arr = arr[1:] 
        # Remove last dep if it is not after last arr
        if dep[-1] >= arr[-1]: dep = dep[:-1]
        if len(arr) != len(dep): raise ValueError(f"Error in stochastic_kWh: t_arr and t_dep have different lengths for a day.")
    

    daily_kWh = get_daily_cons(config)

    charge_length_min = []
    missing_charge = 0
    for d in range(7):
        dur_tot = sum(dur_travel_day[d]) 
        arr = t_arr_day[d]
        dep = t_dep_day[d]
        if len(arr) == 0 or len(dep) == 0: continue
        # Compute the consumption for each trip
        for i in range(len(arr)):
            cons = daily_kWh[d]*dur_travel_day[d][i] / dur_tot + missing_charge
            cons = charging_outside(cons, config["EV_data"]["Capacity"])
            # Compute the charge length in minutes
            charge_length = int((cons / config["EV_data"]["Pmax"]) * 60 )
            charge_length_min.append(charge_length)
            # Ensure charge length is not longer than the time between arrival and departure
            max_charging_time = (d+1)*24*60-arr[i] if i==len(dep)-1 else dep[i+1] - arr[i]
            if charge_length > max_charging_time: 
                missing_charge += (charge_length - max_charging_time)*config["EV_data"]["Pmax"] / 60
                charge_length = max_charging_time 
                print(f"Missing charge: {missing_charge} kWh")
            else: 
                missing_charge = 0

    # Flatten t_arr_day and t_dep_day back to 1D arrays
    t_ar = np.concatenate([arr for arr in t_arr_day if len(arr) > 0])
    t_dep = np.concatenate([dep for dep in t_dep_day if len(dep) > 0])
    charge_length_min = np.array(charge_length_min)

    # Throw error if t_ar, t_dep, and charge_length_min do not have the same length
    if not (len(t_ar) == len(t_dep) == len(charge_length_min)):
        raise ValueError(f"Error: t_arr, t_dep, and charge_length_min must have the same length. Got lengths: t_arr={len(t_ar)}, t_dep={len(t_dep)}, charge_length_min={len(charge_length_min)}")
    print(f"Weekly charging : {sum(charge_length_min)*config['EV_data']['Pmax']/60} kWh, representing {sum(charge_length_min)*config['EV_data']['Pmax']/(.60*config['EV_data']['Consumption'])} km")
    return t_ar, t_dep, charge_length_min 

def weekly_charging2(config: dict, t_arr: np.ndarray, t_dep: np.ndarray, dur_travel: np.ndarray) -> np.ndarray:

    # Remove randomly 20% of the driving events
    num_trips = len(t_arr)
    num_remove = int(num_trips * 0.2)
    if num_remove > 0:
        keep_indices = np.sort(np.random.choice(num_trips, num_trips - num_remove, replace=False))
        t_arr = t_arr[keep_indices]
        t_dep = t_dep[keep_indices]
        dur_travel = dur_travel[keep_indices]
    

    trip_kwh = get_trip_cons(config, t_arr, t_dep, dur_travel)

    print(f"Daily kWh: {trip_kwh}")
    charge_length_min = np.zeros(len(t_arr))
    missing_charge = 0
    # Compute the consumption for each trip
    for i in range(len(t_arr)):
        c = trip_kwh[i] + missing_charge
        c = charging_outside(c, config["EV_data"]["Capacity"])
        # Compute the charge length in minutes
        charge_length = int((c / config["EV_data"]["Pmax"]) * 60 )
        charge_length_min.append(charge_length)
        # Ensure charge length is not longer than the time between arrival and departure
        max_charging_time = 7*24*60-t_arr[i] if i==len(t_dep)-1 else t_dep[i+1] - t_arr[i]
        print(f"Charge length: {charge_length} min, max charge length: {max_charging_time} min, {t_arr[i]}")
        if charge_length > max_charging_time: 
            missing_charge += (charge_length - max_charging_time)*config["EV_data"]["Pmax"] / 60
            charge_length = max_charging_time 
            print(f"Missing charge: {missing_charge} kWh")
        else: 
            missing_charge = 0

    # Throw error if t_ar, t_dep, and charge_length_min do not have the same length
    if not (len(t_arr) == len(t_dep) == len(charge_length_min)):
        raise ValueError(f"Error: t_arr, t_dep, and charge_length_min must have the same length. Got lengths: t_arr={len(t_arr)}, t_dep={len(t_dep)}, charge_length_min={len(charge_length_min)}")

    return t_arr, t_dep, charge_length_min 

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

    t_arr, t_dep, dur_travel = get_weekly_journey_times(occupancy, config) # Get the departure and arrival times of the EV
    P_EV = np.zeros(7*(int((config["nb_days"]-1)/7)+1)*24*60) # Initialize the EV load profile
    Flex_EV = np.zeros(len(P_EV)) # Initialize the EV flexibility profile

    for w in range(int((config["nb_days"]-1)/7) + 1): # loop for each week
        t_arr_w, t_dep_w, charge_length_min = weekly_charging(config, t_arr, t_dep, dur_travel) 
        for i in range(len(t_arr_w)):
            P_EV[w*7*60*24+t_arr_w[i]:w*7*60*24+t_arr_w[i]+charge_length_min[i]] = config["EV_data"]["Pmax"] # Add the consumption to the EV load profile
            Flex_EV[w*7*60*24+t_arr_w[i]:w*7*60*24+t_dep_w[i]] = 1 # Add the consumption to the EV flexibility profile

    # Trim P_EV and Flex_EV to the correct length
    P_EV = P_EV[:config["nb_days"] * 60 * 24 ]
    Flex_EV = Flex_EV[:config["nb_days"] * 60 * 24 + 1]
    return P_EV, Flex_EV