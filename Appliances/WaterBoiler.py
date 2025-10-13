import numpy as np
import pandas as pd

def add_params_WB(config):
    config['WB_data']['T_in'] = 14 # Cold water temperature
    config['WB_data']['T_amb'] = 20 # Ambient temperature
    config['WB_data']['specific_heat'] = 4.186  # Capacité thermique spécifique de l'eau (kJ/kg°C)
    config['WB_data']['rho'] = 1 #kg/L
    config['WB_data']['C_WB'] = 1/(config['WB_data']['rho'] * config['WB_data']['specific_heat'] * config['WB_data']['Volume'])
    return config

def WB_simulate(mDHW, config):
    """
    This function takes as arguments the dataframe containing the hot water consumption for each time 
    step of the simulation and the maximal power that could be delivered by the electrical boiler.
    It returns a vector of the electrical boiler load.
    """
    wb_data=config['WB_data']
    tol = 1
    T_out = 22


    P_use = mDHW['mDHW'] * wb_data['rho'] * (wb_data['T_set'] - wb_data['T_in']) * wb_data['specific_heat']*10/1e3  # [J]*60 seconds

    sim_len=len(P_use)
    P_WB = np.zeros(sim_len)
    P_loss = np.zeros(sim_len)
    T_set = np.full(sim_len, float(wb_data['T_set']), dtype=float)
    T_ref = np.full(sim_len, float(wb_data['T_in']), dtype=float)
    T_ref[0] = wb_data['T_set']
    
    for i in range(sim_len-1):
        if T_ref[i] < T_set[i] - tol: P_WB[i] = wb_data['Pmax']
        elif T_ref[i] > T_set[i] + tol: P_WB[i] = 0
        elif i>0: P_WB[i] = P_WB[i-1]
        P_loss[i] = 2 * (T_ref[i] - T_out)*1e-3
        T_ref[i+1] = T_ref[i] + 60 * wb_data['C_WB'] * (P_WB[i] - P_use[i] - P_loss[i])


    #mDHW['Power'] = mDHW['mDHW'] * wb_data['rho'] * (wb_data['T_final'] - wb_data['T_initial']) * wb_data['specific_heat']  # [J]
    #mDHW['Power'] = mDHW['Power'] / (wb_data['efficiency'] * 3.6e3)  # Wh for each minute
    #mDHW['Power'] = mDHW['Power'] * 60  # kW
    #mDHW['Power_limited'] = limit_power(mDHW['Power'], wb_data['Pmax']*1e3)  

    df_Flex = pd.DataFrame({'P_use_WB': P_use[:-1],'P_loss_WB': P_loss[:-1],'T_set_WB': T_set[:-1],'T_ref_WB': T_ref[:-1],'Water_use': mDHW['mDHW'][:-1]})
    return P_WB[:-1], df_Flex
