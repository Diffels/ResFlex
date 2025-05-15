# -*- coding: utf-8 -*-
"""
Hot_water.py
@authors: duchmax
Purpose: This script contains the function that computes the hot water consumption of a household.
"""

import numpy as np
import pandas as pd
import datetime as dt


def limit_power(power_per_time, max_power):
    """
    This function takes as arguments the power needed to heat the residential heated water for each time step 
    of the simulation and the maximal power that could be delivered by the electrical boiler. 

    It returns a vector of the electrical boiler load.
    """

    power_per_time = np.array(power_per_time)
    over_power = 0
    j=0
    
    for i in range(len(power_per_time)):
        actual_power = power_per_time[i]
        if i > j or j ==0:
            j=i
        if j <=len(power_per_time)-1: 
            while actual_power > max_power : 
                j=j+1
                if j >len(power_per_time)-1: 
                    power_per_time[i] = max_power
                    break

                if power_per_time[j] < max_power:
                    actual_power = actual_power+power_per_time[j]-max_power 
                    if actual_power > max_power:
                        power_per_time[j] = max_power
                    else : 
                        power_per_time[j]= actual_power
                        power_per_time[i] = max_power
                        j=j-1
        else :
            if actual_power > max_power :
                over_power = over_power + actual_power - max_power
                power_per_time[i] = max_power
    if over_power > 0:
        print(f'{over_power/60e3} kWh of hot water energy should be added next day')
    return power_per_time

def water_boiler(mDHW, year, max_power=3):
    """
    This function takes as arguments the dataframe containing the hot water consumption for each time 
    step of the simulation and the maximal power that could be delivered by the electrical boiler.
    It returns a vector of the electrical boiler load.
    """

    T_initial = 14 # Température de l'eau froide (en °C)
    T_final = 60   # Température de consigne (en °C)
    efficiency = 0.9  # Efficacité du chauffe-eau (en fraction)
    specific_heat = 4186  # Capacité thermique spécifique de l'eau (J/kg°C)
    rho = 1 #kg/L
    mDHW['Power'] = mDHW['mDHW'] * rho  *(T_final-T_initial)*specific_heat#[J]
    mDHW['Power'] = mDHW['Power']/(efficiency*3.6e3)#Wh pour chaque minutes
    mDHW['Power'] = mDHW['Power']*60 #kW
    #Power of the electrical boiler is limited to 5 kW 
    limited_Boiler_power = max_power * 1e3#kW
    mDHW['Power_limited'] = limit_power(mDHW['Power'], limited_Boiler_power)

   
    # init_date = dt.datetime(year, 1,1,0,0)
    # dates = []

    # for i in range(len(mDHW)):
    #     dates.append(init_date+dt.timedelta(minutes=i))
    # mDHW['DateTime'] = dates
    # mDHW = mDHW.set_index('DateTime')
    """daily_consumption = mDHW.resample('D').sum().reset_index()"""
    Flex_WB = pd.DataFrame({'Power': mDHW['Power'][:-1], 'Power_limited': mDHW['Power_limited'][:-1]})
    Param_WB = {'T_initial': T_initial, 'T_final': T_final, 'efficiency': efficiency, 'specific_heat': specific_heat, 'rho': rho}
    return mDHW['Power_limited'][:-1], Flex_WB, Param_WB