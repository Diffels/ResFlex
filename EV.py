22# -*- coding: utf-8 -*-
"""
@author: noedi
    
August 2024
"""

# Import required modules
import pandas as pd
import numpy as np
from typing import Any
from datetime import datetime
import random
from plots import plot_EV



def EV_run(occupancy: np.ndarray[Any, np.dtype[np.bool_]], config: dict, plot: bool = False)-> pd.DataFrame:
    '''
    
    '''         
    year = config["year"]
    start_day = config["start_day"]
    capacity = config["EV_data"]["Capacity"]
    consumption = config["EV_data"]["Consumption"]
    charger_power = config["EV_data"]["Pmax"]
    eta = config["EV_data"]["eta"]
    yearly_km = config["EV_data"]["Usage"]
    soc_max = config["EV_data"]["SoC_target"]
    soc_init = soc_max

    EV_profile = np.zeros(1440*config["nb_days"]) # Time Series (minutes) recording the charging power of the EV battery.
    isplug = np.zeros(1440*config["nb_days"]) # Binary Time Series (minutes) describing when EV is plugged, ie when the EV is at home.
    EV_refilled = np.zeros(1440*config["nb_days"]) # Time Series (minutes) showing the charge outside home, assumed instantaneous at the half of the departure event (plot use only).
    SOC_profile = np.zeros(1440*config["nb_days"]) # Time Series (minutes) recording the state of charge of the EV (plot use only).

    for day in range(config["nb_days"]):
        daily_occupancy = occupancy[day*1440:1440*(day+1)] # with 1440 min/day
        type_day = type_of_day(year, start_day, day)

        stoch_kwh = EV_daily_kwh(yearly_km, consumption, type_day)
        daily_profile, daily_isplug, last_soc, daily_EV_refilled, daily_soc = EV_daily_profile(stoch_kwh, daily_occupancy, capacity, charger_power, eta, soc_max, soc_init)

        EV_profile[day*1440:1440*(day+1)] = daily_profile
        isplug[day*1440:1440*(day+1)] = daily_isplug
        EV_refilled[day*1440:1440*(day+1)] = daily_EV_refilled
        SOC_profile[day*1440:1440*(day+1)] = daily_soc
        # Update SOC for the next day
        soc_init = last_soc # SOC at the end of the day

    Param_EV = config['EV_data']

    Flex_EV = pd.DataFrame(isplug, columns=['Flex_EV'], index=None)   
    init_date = datetime(year,1,1,0,0)
    dates = []
    for i in range(len(isplug)):
        dates.append(init_date+pd.Timedelta(minutes=i))
    Flex_EV['DateTime'] = dates
    Flex_EV = Flex_EV.set_index('DateTime')

    #if plot:
    #    plot_EV(SOC_profile, occupancy, EV_profile, EV_refilled)

    return EV_profile, Flex_EV, Param_EV


def EV_daily_kwh(yearly_km: int, consumption: int, type_day: str)->float:
    '''
    Handle 
    '''
    r_d=0.3 #random in distance
    r_cons= 0.3 #random in consumption

    # Daily km based on the type of day (-7% for weekdays, +17.5% for weekends), approx. 25% difference.
    daily_km = yearly_km / 365
    if type_day == 'weekday':
        daily_km *= 0.93
    else:
        daily_km *= 1.175
    rand_dist = round(random.uniform(daily_km*(1-r_d),daily_km*(1+r_d)))

    # Random EV consumption in kWh/100km
    rand_cons = random.uniform(consumption*(1-r_cons), consumption*(1+r_cons))

    rand_kwh = round(rand_cons * rand_dist / 100, 2)
            
    return rand_kwh

def type_of_day(year: int, start_day: int, curr_day: int) -> str:
    '''
    Function to determine the type of day (weekday, weekend) based on the configuration file.
    '''
    current_date = datetime(year, 1, 1) + pd.Timedelta(days=start_day + curr_day - 1) 
    weekday = current_date.weekday()  # Monday = 0, Saturday = 5, Sunday = 6
    if weekday < 5:
        return "weekday"
    else:
        return "weekend"
    
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
    r = 3 * E_journey/E_leaving
    if r > 1.0: # If journey requires more energy than available, charge mandatory.
        P = 1.0
    elif r < 0.05: # Short journeys do not require charge.
        P = 0
    else: 
        P = r
    return P
    
def EV_daily_profile(stoch_kwh: float, occupancy: np.ndarray[Any, np.dtype[np.bool_]], battery_cap: int, charger_power: float, eta: float, SOC_max: float, SOC_init: float, SOC_min: float = 0.1):
    '''

    '''
    r_ch_notHome = 0.30 # Time ratio of EV charging when not at home from whole daily departure duration.
    var_ch_not_home = 0.05 # Stochastic variation in charging time ratio defined above.
    var_split = 0.25 # Stochastic variation in Energy split between not home windows.
    tol_batt_lim = 0.5 # Tolerance according to battery limits (min/max SOCs) when charge/disch.

    # The charge that occurs outside home is not always the same that home charger
    available_stations=[7.4, 11, 22, 50] # [kW], level 2 and 3 of EV chargers
    prob_stations=[0.3, 0.35, 0.3, 0.05]
    station_power = np.random.choice(available_stations, p=prob_stations)
    
    SOC_last=0
    E_spent=0
    
    ''' --------- First loop during the day to compute departures and arrivals --------- '''
    # Dictionnary containg TS where a departure occurs and associated durations (in min)
    departures={}
    # Array containing TS where an arrival occurs (can be derived from departures)
    arrivals=[]
    leave=None
    tot_time_left=0
    # Handling the critic case where user not home at the beginning:
    if occupancy[0] == 0:
        leave = 0
    # Can't nest this loop in the next one because of durations computations!
    for i in range(1, len(occupancy)):
        if occupancy[i] == 0 and occupancy[i-1] == 1:
            leave = i
        elif occupancy[i] == 1 and leave is not None:
            # Record the departure time and duration in a dict
            duration = i - leave
            # Departures that last less than 10 min are considered to not use EV 
            if duration >= 10:
                departures[leave] = [duration]
                # Record the arrivals TS in an array
                arrivals.append(i)
                tot_time_left += duration
                leave = None
    ''' -------------------------------------------------------------------------------- '''

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
                E_leaving = SOC_last*battery_cap
                P_ch_notHome = prob_charge_not_home(E_spent, E_leaving)
                            
                if random.random() <= P_ch_notHome:
                    t_charge = round(r_ch_notHome*t_departure*random.uniform((1-var_ch_not_home),(1+var_ch_not_home))) # Stochastic charge time [min]
                    E_charge = station_power * t_charge * eta / 60 # [kWh], t_charge in min
                    E_arrive = E_leaving-E_spent+E_charge
                    
                    # Control to avoid not enough charge:
                    # If a long journey occurs and, despite the charge not home, the EV is coming
                    # home with SOC_i < SOC_min, the charge must be longer. In this specific case, EV comes
                    # back home with SOC_min.
                    if E_arrive < SOC_min*battery_cap:
                        E_charge = SOC_min*battery_cap + E_spent - E_leaving

                    # Control to avoid to much charge:
                    # Since the charge is supposed to be at half journey, if after the charge
                    # EV is at SOC_max (ie max. charge occured), then SOC_arrive must be SOC_max
                    # diminished by half the journey consumption.
                    if E_arrive > SOC_max*battery_cap:
                        E_charge = SOC_max*battery_cap - E_leaving + E_spent/2

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
                SOC_i = SOC_last - E_spent/battery_cap
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
                    new_SOC = SOC_profile[i-1] + E_charge/battery_cap
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

    return load_profile, isplug, SOC_profile[-1], EV_refilled, SOC_profile