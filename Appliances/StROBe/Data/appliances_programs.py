# NOT OPTIMAL BECAUSE PROFILES ARE COMPUTED AT EACH CALL!

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

'''
How does it works? 

Base case:
family.simulate() in load_profiles.py calls residential.py where it iterates over all appliances and set
power = nominal power (on state), otherwise 1 W (off state). 

Improvement: changes in residential.py lines 699-811: stochastic_load()
In the file residential.py, line 705, if self.name == 'TumbleDryer' ou 'WashingMachine' or 'DishWasher' (prog, boolean variable)
is set up in order to record a program that variates with time (defined in this file) instead of a nominal power.
Then, each time this type of appliance is turned on, associated function defined in this file is called,
generating a profile among 2 choices according to a predefined probability.

Run this file to see plots of the programs. 
'''

def TumbleDryer(P=[0.5, 0.5]):
    
    '''
    Source: Mazidi, M., Malakhatka, E., Steen, D., & Wallbaum, H. (2023). Real-time rolling-horizon energy 
    management of public laundries: A case study in HSB living lab. Energy Conversion and Management: X, 20, 100462.
    '''
    
    rand_choice = np.random.choice([1, 4], p=P)

    if rand_choice == 1:
        # Program 1
        P1_x = np.array(range(5, 65, 5))
        P1_y = np.array([700, 1700, 1750, 1850, 1980, 1950, 2000, 1950, 2000, 1900, 1600, 500])
        cs = CubicSpline(P1_x, P1_y)
        P1_x_interp = np.linspace(min(P1_x), max(P1_x), len(P1_y)*5)
        P1_y_interp = cs(P1_x_interp)
        return P1_y_interp
    else:
        # Program 4
        P4_x = np.array(range(5, 115, 5))
        P4_y = np.array([700, 1700, 1800, 1900, 2020, 1950, 1700, 1800, 1300, 200, 200, 400, 2200, 500, 750, 1250, 400, 1700, 500, 1100, 2750, 1200])  # y-values (power in watts)
        cs = CubicSpline(P4_x, P4_y)
        P4_x_interp = np.linspace(min(P4_x), max(P4_x), len(P4_y)*5)
        P4_y_interp = cs(P4_x_interp)
        return P4_y_interp    

def WashingMachine(P=[0.5, 0.5]):
    '''
    Source: Mazidi, M., Malakhatka, E., Steen, D., & Wallbaum, H. (2023). Real-time rolling-horizon energy 
    management of public laundries: A case study in HSB living lab. Energy Conversion and Management: X, 20, 100462.
    '''
    rand_choice = np.random.choice([1, 4], p=P)

    if rand_choice == 1:
        # Program 1
        P1_x = np.array(range(5, 145, 5))
        P1_y = np.array([300, 250, 250, 550, 500, 1200, 550, 150, 125, 100, 250, 150, 100, 1000, 1480, 1000, 1500, 100, 150, 100, 100, 150, 100, 500, 250, 400, 1450, 1070])
        cs = CubicSpline(P1_x, P1_y)
        P1_x_interp = np.linspace(min(P1_x), max(P1_x), len(P1_y)*5)
        P1_y_interp = cs(P1_x_interp)
        return P1_y_interp
    else:
        # Program 4
        P4_x = np.array(range(5, 220, 5))
        P4_y = np.array([200, 1, 200, 100, 300, 250, 100, 2250, 1050, 100, 120, 120, 120, 200, 120, 550, 1, 400, 1450, 120, 120, 120, 120, 120, 120, 500, 250, 700, 2900, 2200, 1, 120, 120, 120, 250, 120, 1, 550, 120, 1, 1450, 550, 120])  # y-values (power in watts)
        cs = CubicSpline(P4_x, P4_y, bc_type='natural')
        P4_x_interp = np.linspace(min(P4_x), max(P4_x), len(P4_y)*5)
        P4_y_interp = cs(P4_x_interp)
        P4_y_interp = np.clip(P4_y_interp, a_min=1, a_max=None)
        return P4_y_interp     

def DishWasher(P=[0.5, 0.5]):
    '''
    Source: Issi, F., & Kaplan, O. (2018). The determination of load profiles and power consumptions of 
    home appliances. Energies, 11(3), 607.
    '''
    rand_choice = np.random.choice([1, 4], p=P)

    if rand_choice == 1:
        # Program 1 - 55°C economy program
        P1_x = np.array(range(180)) # 3h program, in min
        water_heating_1 = [2250]*8 # 2250 W for 8 min
        water_heating_2 = [2250]*16 # 2250 W for 16 min
        P1_y = np.full(len(P1_x), 1) # Off-state -> 1W
        P1_y[16:24] = water_heating_1
        P1_y[130:146] = water_heating_2
        return P1_y
    else:
        # Program 4 - 65°C program
        P4_x = np.array(range(60)) # 1h program, in min
        water_heating_1 = [2250]*20
        water_heating_2 = [2250]*3
        water_heating_3 = [2250]*8
        P4_y = np.full(len(P4_x), 1)
        P4_y[5:25] = water_heating_1
        P4_y[32:35] = water_heating_2
        P4_y[37:45] = water_heating_3
        return P4_y
