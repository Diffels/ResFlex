# -*- coding: utf-8 -*-
"""
@author: noedi
    
August 2024
"""

# Importing required modules
import numpy as np
import random 
from typing import Any


def prob_charge_notHome_fun(E_journey, E_leaving):
    '''
    Probability the user want to charge EV depending on the energy spent 
    during journey and energy already in the car when leaving.
    Inputs:
        - E_journey: Energy required by the associated journey. [kWh]
        - E_leaving: Energy available in the EV battery. [kWh]
    Outputs:
        - P: Probability of a charging event, outside the home. [-]
    '''
    r = E_journey/E_leaving # [kWh]/[kWh]
    if r > 1.0: # If journey requires more energy than available, charge mandatory.
        P = 1.0
    elif r < 0.05: # Short journeys do not require charge.
        P = 0
    else: 
        P = r
    #print(P)
    return P
    

def EV_occ_daily_profile(EV_cons: np.ndarray[Any, np.dtype[np.float64]], full_occupancy: np.ndarray[Any, np.dtype[np.bool_]], Driver: object, charger_power: float, SOC_init=0.9, disp=False):
    '''
    Function returning an array containing daily profile which splits the stochastic
    EV capacities given by EV_stochastic.py according to a occupancy profile input.
    Inputs:
        - EV_cons (float np.ndarray): Stochastic EV daily consumption(s) to split according to occupancy.
        - full_occupancy (boolean np.ndarray): Occupancy profile. (1: Home; 0: Not Home)
        - User (class User()): User profile to simulate.
        - SOC_init (float [0.0; 1.0]): Initial State Of Charge of the EV battery.
        - disp (boolean): To get information displayed on the console.
    Outputs:
        - SOC_profile (float np.array [0.0; 1.0]): State of charge profile during the day.
        - charging_profile (boolean np.array): EV plugging profile. (1: Plugged; 0: Not plugged)
        - EV_refilled (float np.array [0.0; 1.0]): Indicate if a charging event outside home occurs.
                                                  (Only for graphical representation, no physical interpretation.)
    '''
    
    r_ch_notHome = 0.30 # Time ratio of EV charging when not at home from whole departure duration.
    var_ch_notHome = 0.05 # Stochastic variation in charging time ratio defined above.
    var_split = 0.25 # Stochastic variation in Energy split between not home windows.
    tol_batt_lim = 0.5 # Tolerance according to battery limits when charge/disch.

    # EV Battery parameters
    SOC_max = 0.9 
    SOC_min = 0.1
    eff = 0.90

    # The charge that occurs outside home is not always the same that home charger
    available_stations=[7.4, 11, 22, 50] # [kW], level 2 and 3 of EV chargers,see source in readme.txt
    prob_stations=[0.3, 0.35, 0.3, 0.05]
    station_power = np.random.choice(available_stations, p=prob_stations)

    EV = Driver.App_list[0]
    battery_cap = EV.Battery_cap # [kWh]
    
    minPerDay=1440
    
    list_SOC_profile=np.zeros(len(EV_cons)*minPerDay)
    list_charging_profile=np.zeros(len(EV_cons)*minPerDay)
    list_EV_refilled=np.zeros(len(EV_cons)*minPerDay)
    
    SOC_beginning = SOC_init
    SOC_last=SOC_init
    E_spent=0
    
    for iteration in range(len(EV_cons)):
        # For 1 day:

        occupancy = full_occupancy[iteration*minPerDay:minPerDay*(iteration+1)]
        SOC_profile = np.full(minPerDay, SOC_beginning) # Time Series of SOC filled with init value
        charging_profile = np.zeros(minPerDay) # Binary Time Series describing when EV is plugged.
        EV_refilled=np.zeros(minPerDay) # Time Series recording if a battery re-filled occurs during a departure.
    
        # Dictionnary containg Time Steps where a departure occurs and associated durations
        departures={}
        # Array containing TS where an arrival occurs
        # TODO Maybe a clever way to do that with vectorized operations
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
            
    # --- Main Loop that iterates over the day ---)
    
        fully_charged=False
        for i in range(1, len(occupancy)):
            if not occupancy[i]: # Not at home
                if i in departures.keys(): # Event departure
                    fully_charged=False
    
                # Adding departures E_spent [kWh] to dict according to stochastic ratio [%]
                    #print("Dep:", departures[i], "-", i)
                    t_departure = departures[i][0]
                    ratio = t_departure/tot_time_left
                    stoch_ratio = round(ratio * random.uniform((1-var_split),(1+var_split)), 2)
                    E_spent = EV_cons[iteration]*stoch_ratio
                    departures[i].append(E_spent)
                    
                # Probability to charge during the departure, function of E_spent and E_leaving
                    SOC_last = SOC_profile[i-1]
                    E_leaving = SOC_last*battery_cap
                    P_ch_notHome = prob_charge_notHome_fun(E_spent, E_leaving)
                                
                    if random.random() <= P_ch_notHome:
                        t_charge = round(r_ch_notHome*t_departure*random.uniform((1-var_ch_notHome),(1+var_ch_notHome))) # Stochastic charge time [min]
                        E_charge = station_power / 60 * t_charge * eff # [kWh]
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
                        EV_refilled[half_dep] = E_charge/battery_cap
                        temp_duration = departures[i][0]
                        departures.update({i: [temp_duration, E_spent]}) # The Energy spent is diminished by E_charge, dict update. 
                        if disp:
                            print(f"A battery re-filled occured between {i} [min] and {i+t_departure} [min] of {round(E_charge,2)} [kWh] (+{round(100*E_charge/battery_cap, 2)}%).")
                        
                SOC_profile[i]=0
                
            else: # Is at home
                if i in arrivals: # Event arrival: EV is coming home
                    # Update SOC with discharge from previous journey
                    SOC_i = SOC_last - E_spent/battery_cap
                    if SOC_i < SOC_min*(1-tol_batt_lim):
                        SOC_i = SOC_min*(1-tol_batt_lim)
                        #raise ValueError(f"Error in daily_EV_profile.py: SOC at {i} [min] is {SOC_i} [-] which is lower than SOC_min ({SOC_min} [-]).") 
                    charging_profile[i]=1
                else: # Event charge: EV charge at nominal power until SOC_max
                    # Update SOC with charge from home charging station
                    if fully_charged:
                        # SOC is fully charged, no need to charge.
                        SOC_i = SOC_profile[i-1]
                    else:
                        # Charging Event
                        E_charge = charger_power / 60 * eff # kWh
                        new_SOC = SOC_profile[i-1] + E_charge/battery_cap
                        if new_SOC > SOC_max:
                            SOC_i = SOC_max
                            fully_charged=True
                        else:
                            SOC_i = new_SOC                    
                            charging_profile[i]=1
                        
                SOC_profile[i] = SOC_i

        list_SOC_profile[iteration*minPerDay:minPerDay*(iteration+1)]=SOC_profile
        list_charging_profile[iteration*minPerDay:minPerDay*(iteration+1)]=charging_profile
        list_EV_refilled[iteration*minPerDay:minPerDay*(iteration+1)]=EV_refilled
        
        # Setting up de first SOC of next day.
        SOC_beginning = SOC_profile[-1]
    
    load_profile = np.multiply(list_charging_profile, charger_power)
    
    return (list_SOC_profile, list_charging_profile, list_EV_refilled, load_profile)     
    