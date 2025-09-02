import numpy as np
import pandas as pd

def add_params_WB(config):
    config['WB_data']['T_in'] = 14 # Cold water temperature
    config['WB_data']['T_amb'] = 20 # Ambient temperature
    config['WB_data']['specific_heat'] = 4.186  # Capacité thermique spécifique de l'eau (J/kg°C)
    config['WB_data']['rho'] = 1 #kg/L
    config['WB_data']['C_WB'] = 1/(config['WB_data']['rho'] * config['WB_data']['specific_heat'] * config['WB_data']['Volume'])
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
    wb_data=config['WB_data']

    #P_use = mDHW['mDHW']
    P_use = mDHW['mDHW'] * wb_data['rho'] * (wb_data['T_set'] - wb_data['T_amb']) * wb_data['specific_heat']*60  # [J]*60 seconds

    sim_len=len(P_use)
    P_WB = np.zeros(sim_len)
    P_loss = np.zeros(sim_len)
    T_set = np.repeat(wb_data['T_set'], sim_len)
    T_ref = np.repeat(wb_data['T_in'], sim_len)
    T_ref[0] = wb_data['T_set']
    
    for i in range(sim_len-1):
        if T_ref[i] < T_set[i]:P_WB[i] = wb_data['Pmax']
        else: P_WB[i] = 0
        P_loss[i] = 0#C_wall * (T_ref[i] - T_out)
        T_ref[i+1] = T_ref[i] + 1/60 * wb_data['C_WB'] * (P_WB[i] - P_use[i] - P_loss[i])


    #mDHW['Power'] = mDHW['mDHW'] * wb_data['rho'] * (wb_data['T_final'] - wb_data['T_initial']) * wb_data['specific_heat']  # [J]
    #mDHW['Power'] = mDHW['Power'] / (wb_data['efficiency'] * 3.6e3)  # Wh for each minute
    #mDHW['Power'] = mDHW['Power'] * 60  # kW
    #mDHW['Power_limited'] = limit_power(mDHW['Power'], wb_data['Pmax']*1e3)  

    df_Flex = pd.DataFrame({'P_use_WB': P_use[:-1],'P_loss_WB': P_loss[:-1],'T_set_WB': T_set[:-1],'T_ref_WB': T_ref[:-1]})
    return P_WB[:-1], df_Flex
