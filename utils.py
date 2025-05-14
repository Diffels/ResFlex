import numpy as np
import datetime as dt
import pandas as pd


def occ_reshape(occ: np.ndarray, ts: float)->np.ndarray:
    '''
    Function that reshape occupancy profile:
        1. Make it boolean. (1: Active, 2: Sleeping)-> 1: At Home; (3: Not at home)-> 0: Not at home
        2. From 10-min time step into 1-min time step. 
    Inputs
        - occ: former occupancy profile
        - ts: simulation time step [min]
    Outputs
        - new_occ: reshaped new occupancy profile
    '''
    nTS = len(occ) 

    new_occ=np.zeros((nTS-1)*ts)
    # Repeat each occupancy value to match the new resolution
    expanded_occ = np.repeat(occ[:-1], ts)
    # Apply the condition to determine whether the driver is home or not.
    new_occ = np.where(np.isin(expanded_occ, [1, 2]), 1, 0)
    
    return new_occ

def occ_reshape2(occ: np.ndarray, ts: float) -> np.ndarray: return np.where(np.isin(np.repeat(occ[:-1], ts), [1, 2]), 1, 0)

def index_to_datetime2(df, year, ts): return df.set_index(pd.date_range(start=dt.datetime(year, 1, 1), periods=len(df), freq='T')).resample(f'{ts}min').mean()


def index_to_datetime(df, year, ts):
    '''
    Function that convert the index of a dataframe into datetime format.'
    Inputs:
        - df: Dataframe to convert
        - year: Year of the simulation
        - ts: Time step of the simulation [min]
    Outputs:
        - df: Dataframe with datetime index
    '''
    init_date = dt.datetime(year, 1,1,0,0)
    dates = []
    for i in range(len(df)):
        dates.append(init_date+dt.timedelta(minutes=i))
    df['DateTime'] = dates
    df = df.set_index('DateTime')
    df10min = df.resample(str(ts)+'min').mean()
    return df10min

