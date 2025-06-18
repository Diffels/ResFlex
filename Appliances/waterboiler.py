import numpy as np
import pandas as pd

def add_params_WB(config):
    config['WB_data']['T_initial'] = 14 # Température de l'eau froide (en °C)
    config['WB_data']['T_final'] = 60   # Température de consigne (en °C)
    config['WB_data']['efficiency'] = 0.9  # Efficacité du chauffe-eau (en fraction)
    config['WB_data']['specific_heat'] = 4186  # Capacité thermique spécifique de l'eau (J/kg°C)
    config['WB_data']['rho'] = 1 #kg/L
    return config

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

def WB_simulate(mDHW, config):
    """
    This function takes as arguments the dataframe containing the hot water consumption for each time 
    step of the simulation and the maximal power that could be delivered by the electrical boiler.
    It returns a vector of the electrical boiler load.
    """
    wb_data = config['WB_data']
    mDHW['Power'] = mDHW['mDHW'] * wb_data['rho'] * (wb_data['T_final'] - wb_data['T_initial']) * wb_data['specific_heat']  # [J]
    mDHW['Power'] = mDHW['Power'] / (wb_data['efficiency'] * 3.6e3)  # Wh for each minute
    mDHW['Power'] = mDHW['Power'] * 60  # kW
    mDHW['Power_limited'] = limit_power(mDHW['Power'], wb_data['Pmax']*1e3)  

    Flex_WB = pd.DataFrame({'Power': mDHW['Power'][:-1], 'Power_limited': mDHW['Power_limited'][:-1]})
    return mDHW['Power_limited'][:-1], Flex_WB