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

<<<<<<< Updated upstream
=======
def save_all(config, dic_df_P, dic_df_Flex, dic_Params, houses_params):
    filetype = config['output']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create a directory for this simulation
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Results", "Multiple", current_time)
    os.makedirs(out_dir, exist_ok=True)
    if filetype == 'xlsx':
        filename = os.path.join(out_dir, "ref.xlsx")
        flex_filename = os.path.join(out_dir, "flex.xlsx")
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            for sheet_name, rows in dic_df_P.items():
                df = pd.DataFrame(rows)  # Convert list of dicts to DataFrame
                df.to_excel(writer, sheet_name=sheet_name, index=True)
        if config['flexibility']:
            with pd.ExcelWriter(flex_filename, engine="xlsxwriter") as writer_F:
                for sheet_name, rows in dic_df_Flex.items():
                    df = pd.DataFrame(rows)  # Convert list of dicts to DataFrame
                    df.to_excel(writer_F, sheet_name=sheet_name, index=True)
    elif filetype == 'csv':
        for u, (df_P, df_Flex) in enumerate(zip(dic_df_P.values(), dic_df_Flex.values())):
            filename = os.path.join(out_dir, f"household_{u+1}_ref.csv")
            flex_filename = os.path.join(out_dir, f"household_{u+1}_flex.csv")
            df_P.to_csv(filename, mode='w', header=True)
            df_Flex.to_csv(flex_filename, mode='w', header=True)


    elif filetype == 'nc':
        filename = os.path.join(out_dir, "ref.nc")
        flex_filename = os.path.join(out_dir, "flex.nc")
        # Convert data to xarray Dataset and save to NetCDF
        ds_P = xr.Dataset({house: xr.DataArray(data=df.values, dims=["time", "variables"], coords={"time": df.index, "variables": df.columns}) for house, df in dic_df_P.items()})
        ds_P.to_netcdf(filename)

        ds_Flex = xr.Dataset({house: xr.DataArray(data=df.values, dims=["time", "variables"], coords={"time": df.index, "variables": df.columns}) for house, df in dic_df_Flex.items()})
        ds_Flex.to_netcdf(flex_filename)

    else:
        raise ValueError(f"Unsupported file type for saving: {filetype}")
    
    with open(os.path.join(out_dir, "users.json"), 'w', encoding="utf-8") as json_file:
        json.dump(dic_Params, json_file, ensure_ascii=False, indent=4)

"""Plotting functions for the simulation results"""

def plot_all(config, dic_df_P, dic_df_Flex, dic_Params):
    return

def plot_one(df_P, dic_plot, pdf=False):
    if not pdf:
        return

    df_P['P_HP'] *= 1e2 # /!\

    # Combine WashingMachine and DishWasher into 'White Goods'
    df_P['White Goods'] = df_P.get('WashingMachine', 0) + df_P.get('DishWasher', 0)

    # Reform columns for better plot former_col -> (new_col)
    nice_cols = {
        'BaseLoad': 'Base Load',
        'White Goods': 'White Goods',
        'P_WB': 'Water Boiler',
        'P_HP': 'Space Heating',
        'P_EV': 'Electric Vehicle'
    }

    # Only keep relevant columns
    df_P = df_P[list(nice_cols.keys())]
    df_P.rename(columns=nice_cols, inplace=True)

    cm = 1/2.54  # centimeters in inches
    size = dic_plot['figsize_cm']
    fig = plt.figure(figsize=(size[0]*cm, size[1]*cm)) #cm

    plt.rcParams.update({'font.size': dic_plot['fontsize']*cm})

    # Prepare data
    x = df_P.index
    y = [df_P[col] for col in df_P.columns]

    # Stacked area plot
    plt.stackplot(x, y, labels=df_P.columns, alpha=1, colors=dic_plot['colors'])
    plt.title(dic_plot['title'])
    #plt.xlabel(dic_plot['xlabel'])
    plt.ylabel(dic_plot['ylabel'])
    if dic_plot['legend']:
        plt.legend(loc='upper center', fontsize=dic_plot['fontsize']*cm, ncol=5)
    plt.grid(dic_plot['grid'])

    # Set x-axis ticks to each day at 12:00 and labels horizontally
    import matplotlib.dates as mdates
    ax = plt.gca()
    # Find all unique days in the index
    days = pd.to_datetime(x).normalize().unique()
    # Set ticks at 12:00 for each day
    ticks = [pd.Timestamp(day) + pd.Timedelta(hours=12) for day in days]
    ax.set_xticks(ticks)
    ax.set_xticklabels([tick.strftime('%d-%b') for tick in ticks], rotation=0, ha='center')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    if dic_plot['save']:
        # Create output directory
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plot_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Results/Single/Plot")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{current_time}.pdf")
        plt.savefig(plot_path, format='pdf')
    if dic_plot['show']:
        plt.show()
    plt.close(fig)

def plot_P(df):    
    fig = go.Figure()
    x = df.index
    cols = df.columns.tolist()

    for idx, key in enumerate(cols):
        fig.add_trace(go.Scatter(
            name = key,
            x = x,
            y = df.loc[x,key],
            stackgroup='one',
            mode='none'          
           ))

    fig.update_layout(title = "Demand for the household",
                      xaxis_title = r'Time',
                      yaxis_title = r'Power [kW]'
                      )
    fig.show()
    return fig

def plot_EV(SOC, occupancy, load_profile, EV_refilled):
    fig = go.Figure()

    # Plot SOC
    fig.add_trace(go.Scatter(
        y=SOC,
        mode='lines',
        name='SOC [%]',
        line=dict(color='blue')
    ))

    # Plot occupancy
    fig.add_trace(go.Scatter(
        y=occupancy,
        mode='lines',
        name='Occupancy [-]',
        line=dict(color='green')
    ))

    # Plot load profile
    fig.add_trace(go.Scatter(
        y=load_profile,
        mode='lines',
        name='Load [kW]',
        line=dict(color='orange')
    ))

    # Plot EV_refilled
    fig.add_trace(go.Scatter(
        y=EV_refilled,
        mode='lines',
        name='EV Refilled [kWh]',
        line=dict(color='purple', dash='dash')
    ))

    fig.update_layout(
        title="EV Metrics Over Time",
        xaxis_title="Time [min]",
        yaxis_title="Values",
        legend_title="Metrics",
        template="plotly_white"
    )

    fig.show()
    return fig

def plot_heating(T, T_wall, T_set, T_out, P_HP):
    """Plot the heating dynamics of the house."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=T, mode='lines', name='Indoor Temperature'))
    fig.add_trace(go.Scatter(y=T_wall, mode='lines', name='Wall Temperature'))
    fig.add_trace(go.Scatter(y=T_set[:len(T)], mode='lines', name='Setpoint Temperature'))
    fig.add_trace(go.Scatter(y=T_out[:len(T)], mode='lines', name='Outdoor Temperature'))
    fig.add_trace(go.Scatter(y=P_HP/400, mode='lines', name='HP Power'))
    fig.update_layout(
        title="Temperature Dynamics",
        xaxis_title="Time Steps",
        yaxis_title="Temperature (C)",
        legend_title="Legend",
        template="plotly"
    )

    fig.show()

"""Printing functions for the simulation results"""
def print_one(config, df_P, df_Flex, dic_Param):
    print("Simulation Results:")
    print("-" * 30)
    print("Power Demand DataFrame:")
    print(df_P.head())
    print("\nFlexibility DataFrame:")
    if config['flexibility']:
        print(df_Flex.head())
    else:
        print("Flexibility is disabled in the configuration.")
    print("\nSimulation Parameters:")
    for key, value in dic_Param.items():
        print(f"{key}: {value}")
    
    print("\nGeneral Statistics:")
    print("-" * 30)
    print(f"Total Consumption: {df_P.sum().sum()/60:.2f} kWh")
    for appliance in df_P.columns:
        print(f"Total Consumption for {appliance}: {df_P[appliance].sum()/60:.2f} kWh")
    return
def print_all(config, dic_df_P, dic_df_Flex, dic_Params):
    return



"""Functions to create the parameters for the simulation
create_params takes a config file and creates a list of parameters for each household
 - check_probas checks if the probabilities sum to 1 and if the length of the probabilities is equal to the length of the values
 - get_list_param creates a list of parameters for each household
    - append_recurring appends the parameters that are the same for all households
    - append_appliances appends the time-shiftable appliances
    - append_family appends the family size and member types 
    - append_flexible appends the flexible appliances
        - probas_to_list takes a list of probabilities and a list of values and returns a list of values based on the probabilities
"""

def set_timesteps(df_P, df_Flex, config):
    """Set the time index of the dataframes based on the configuration and desired timestep."""

    ts = config['timestep']  # timestep in minutes
    # # NB: Trips last at least 30 minutes so no overlapping if ts <= 30min
    if ts > 30:
        print(f"[WARNING] Given timestep ({ts} min) might be too large regarding EV usage patterns. Please check data consistency or reduce below 30 min.")

    start_date = datetime(2024, 1, 1) + pd.Timedelta(days=config["start_day"])
    end_date = start_date + pd.Timedelta(days=config["nb_days"])
    time_index = pd.date_range(start=start_date, end=end_date, freq="1min")
    df_P.index = time_index[:len(df_P)] if len(time_index) >= len(df_P) else time_index
    df_Flex.index = time_index[:len(df_Flex)] if len(time_index) >= len(df_Flex) else time_index
    if config['timestep'] == 1: return df_P, df_Flex
    # ---- Resample signals ----
    df_P = df_P.resample(f"{ts}min").mean()

    df_Flex_newindex = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq=f"{ts}min"))[:-1]
    df_Flex_newindex['Occupancy'] = df_Flex['Occupancy'].resample(f"{ts}min").last() 

    if config["HP"]:
        df_Flex_newindex[['T_set_HP','P_loss_HP']] = df_Flex[['T_set_HP','P_loss_HP']].resample(f'{ts}min').mean()
        df_Flex_newindex[['T_ref_HP', 'T_wall_HP', 'T_out_HP']] = df_Flex[['T_ref_HP', 'T_wall_HP', 'T_out_HP']].resample(f'{ts}min').first()
    

    if config["WB"]:
        df_Flex_newindex[['T_ref_WB']] = df_Flex[['T_ref_WB']].resample(f'{ts}min').first() 
        df_Flex_newindex[['P_use_WB', 'P_loss_WB', 'T_set_WB']] = df_Flex[['P_use_WB', 'P_loss_WB', 'T_set_WB']].resample(f'{ts}min').mean()


    if config["EV"]:
        df_Flex_newindex[['SoC_ref_EV','SoC_arr_EV','EV_plugged','EV_arrival','EV_departure']] = df_Flex[['SoC_ref_EV','SoC_arr_EV','EV_plugged','EV_arrival','EV_departure']].resample(f"{ts}min").max() 


        # df_Flex_newindex["EV_arrival"] = 0
        # df_Flex_newindex["EV_departure"] = 0
        # df_Flex_newindex["EV_plugged"] = 0

        # # ---- Rebuild EV_plugged from SoC ----
        # # Compute discrete derivative of SoC
        # soc = df_Flex_newindex["SoC_ref_EV"]
        # dsoc = soc.diff()

        # # Rule: if SoC decreases, EV must be away
        # df_Flex_newindex.loc[dsoc == 0, "EV_plugged"] = 0  
        # # Rule: if SoC increases, EV must be home
        # df_Flex_newindex.loc[dsoc > 1e-5, "EV_plugged"] = 1 

        # # Constant SoC case:

        # flat = dsoc.abs() <= 1e-5    
        # df_Flex_newindex.loc[flat & (soc >= 0.01), "EV_plugged"] = 1
        # df_Flex_newindex.loc[flat & (soc == 0), "EV_plugged"] = 0

        # # ---- Enforce strict step SoC ----
        # soc_step = soc.copy()
        # soc_step[df_Flex_newindex["EV_plugged"] == 0] = 0  # away → force to 0
        # soc_step[df_Flex_newindex["EV_plugged"] == 1] = soc_step.ffill()  # home → hold last value

        # s = df_Flex_newindex['EV_plugged'].astype(int)
        # diff = s.diff()
        # df_Flex_newindex['EV_departure'] = 0
        # df_Flex_newindex['EV_arrival'] = 0
        # df_Flex_newindex.loc[diff == -1, 'EV_departure'] = 1 # plugged → unplugged
        # df_Flex_newindex.loc[diff == 1, 'EV_arrival'] = 1 # unplugged → plugged 

        # # Assign SOC values where soc_arr is True
        # # Initialize column
        # df_Flex_newindex["SoC_arr_EV"] = 0.0
        # soc_arr_values = df_Flex["SoC_arr_EV"][df_Flex["SoC_arr_EV"] != 0]
        
        # soc_index = 0 
        # for index, ev_arrival in zip(df_Flex_newindex.index, df_Flex_newindex['EV_arrival']):
        #     if ev_arrival:  # if True, EV has arrived
        #         df_Flex_newindex.loc[index, 'SoC_arr_EV'] = soc_arr_values.iloc[soc_index]
        #         soc_index += 1

    # # ---- Assign trip behavior ----
    # # Assume it's per default plugged
    # df_Flex_newindex["EV_plugged"] = 1
    # df_Flex_newindex["EV_arrival"] = 0
    # df_Flex_newindex["EV_departure"] = 0
    # # df_Flex_newindex["SoC_ref_EV"] = 0

    # s = df_Flex['EV_plugged'].astype(int)
    # diff = s.diff()
    # tdep = df_Flex.index[diff == -1]  # plugged → unplugged
    # tret = df_Flex.index[diff == 1]   # unplugged → plugged

    # # If no events:
    # if len(tdep) == 0 or len(tret) == 0:
    #     return df_P, df_Flex_newindex

    # # If the first event is a return -> add artificial departure at first index
    # if tdep[0] > tret[0]:
    #     tdep = pd.Index([df_Flex.index[0]]).append(tdep)
    # # If the last event is a departure -> add artificial return at last index
    # if tdep[-1] > tret[-1]:
    #     tret = tret.append(pd.Index([df_Flex.index[-1]]))

    # trips = pd.DataFrame({"tdep": tdep.values, "tret": tret.values})
    # trips["duration"] = trips["tret"] - trips["tdep"]
    

    # # Clip the duration to ts to avoid overlaping
    # for t in range(len(trips)):
    #     if trips['duration'].iloc[t] < pd.Timedelta(minutes=ts):
    #         new_return = trips.at[t, 'tdep'] + pd.Timedelta(minutes=ts)
    #         if t < len(trips) - 1 and new_return >= trips.at[t+1, 'tdep']:
    #             # avoid overlap with next departure
    #             new_return = trips.at[t+1, 'tdep'] - pd.Timedelta(minutes=ts)
    #         if new_return > df_Flex.index[-1]:
    #             new_return = df_Flex.index[-1]
    #         trips.at[t, 'tret'] = new_return
    #         trips.at[t, 'duration'] = trips.at[t, 'tret'] - trips.at[t, 'tdep']

    #         # Condition if both end up to last index:
    #         if trips.at[t, 'tdep'] == trips.at[t, 'tret']:
    #             trips.drop(t, inplace=True) 

    # # ---- Drop pathological zero-duration trips ----
    # trips = trips[trips['duration'] > pd.Timedelta(0)]

    # dep_idx, ret_idx = [], []
    # for dep, ret in zip(trips['tdep'], trips['tret']):
    #     dep_idx.append(df_Flex_newindex.index.get_indexer([dep], method="nearest")[0])
    #     ret_idx.append(df_Flex_newindex.index.get_indexer([ret], method="nearest")[0])
    
    # col_soc = df_Flex_newindex.columns.get_loc("SoC_ref_EV")
    # # trips["soc_dep"] = df_Flex_newindex.iloc[dep_idx, col_soc].values
    # trips["soc_ret"] = df_Flex_newindex.iloc[ret_idx, col_soc].values

    # # Set the df_Flex boolean vector
    # if dep_idx and ret_idx:
    #     for d, r in zip(dep_idx, ret_idx):
    #         df_Flex_newindex.iloc[d:r, df_Flex_newindex.columns.get_loc("EV_plugged")] = 0
    #         df_Flex_newindex.iloc[d, df_Flex_newindex.columns.get_loc("EV_departure")] = 1
    #         df_Flex_newindex.iloc[r, df_Flex_newindex.columns.get_loc("EV_arrival")] = 1

    #         # df_P.iloc[d:r, df_P.columns.get_loc("P_EV")] = 0
    #         df_Flex_newindex.iloc[d:r, df_Flex_newindex.columns.get_loc("SoC_ref_EV")] = 0


    return df_P, df_Flex_newindex


    

def check_probas(fields, config):
    for field in fields:
        if sum(config[f'P_{field}']) != 1:
            raise Exception(f"Error: {field} probability sum is not equal to 1.")
        if len(config[f'{field}']) != len(config[f'P_{field}']):
            raise Exception(f"Error: {field} probability length is not equal to the number of params.")

def append_recurring(fields, list_param, config):
    for i in list_param:
        for field in fields:
            i[field]= config[field]
    return list_param

def append_appliances(list_param, config):
    app_list = ['WashingMachine', 'DishWasher']
    values = [np.random.choice([0,1], size=config['nb_households'], p=[1-config['appliances'][f'P_{a}'], config['appliances'][f'P_{a}']]) for a in app_list]
    for i, house in enumerate(list_param):
        house['appliances'] = {}
        for a, appliance in enumerate(app_list):
            if appliance == 'WashingMachine' and bool(values[a][i]):
                # Probability of having a TumbleDryer depends on having a WashingMachine
                prob_td = config['appliances']['P_TumbleDryer_given_WM']
                has_td = np.random.choice([0, 1], p=[1 - prob_td, prob_td])
                house['appliances']['TumbleDryer'] = int(has_td)
            house['appliances'][appliance] = int(values[a][i])
    return list_param

def probas_to_list(appliance, field, config):
    probas = config[f'{appliance}_data'][f'P_{field}']
    values = config[f'{appliance}_data'][field]
    return np.random.choice(values, size=config['nb_households'], p=probas)

def append_family(list_param, config):
    probas = config['P_inhabitants']
    values = config['inhabitants']
    family = np.random.choice(values, size=config['nb_households'], p=probas)
    for i, house in enumerate(list_param):
        house['family'] = int(family[i])
        house['occupations'] = ['Random'] * family[i]
    return list_param

def append_flexible(appliance, fields, list_param, config):
    lists = {field: probas_to_list(appliance, field, config) for field in fields}
    app = np.random.choice([1, 0], size=config['nb_households'], p=[config[f'P_{appliance}'], 1 - config[f'P_{appliance}']])
    for i, house in enumerate(list_param):
        house[appliance] = bool(app[i])
        house[f'{appliance}_data'] = {}
        for f in fields:
            if not house[appliance]:
                house[f'{appliance}_data'][f] = None
            else:
                house[f'{appliance}_data'][f] = float(lists[f][i])
    return list_param

def get_list_param(config):
    list_param = [{} for _ in range(config['nb_households'])]  # each house is independent
    list_param = append_recurring(['nb_days', 'timestep', 'year', 'start_day', 'flexibility'], list_param, config)
    list_param = append_appliances(list_param, config)
    list_param = append_flexible('HP',['Year', 'Size', 'Floors','P_nom', 'COP'], list_param, config)
    list_param = append_flexible('WB', ['Pmax', 'Volume', 'T_set'], list_param, config)
    list_param = append_flexible('EV', ['Consumption', 'Capacity', 'Pmax', 'eta', 'SoC_target', 'Usage'], list_param, config)
    list_param = append_flexible('BSS', ['Pmax', 'Capacity', 'SoC_min', 'SoC_max', 'eta'], list_param, config)
    list_param = append_flexible('PV', ['Pmax', 'id'], list_param, config)

    list_param = append_family(list_param, config)
    return list_param


def create_params(config):
    # Check if the config file is valid
    check_probas(['Year', 'Size', 'Floors','P_nom', 'COP'], config['HP_data'])
    check_probas(['Pmax', 'Volume', 'T_set'], config['WB_data'])
    check_probas(['Consumption', 'Capacity', 'Pmax', 'eta', 'SoC_target', 'Usage'], config['EV_data'])
    check_probas(['inhabitants'], config)
    # Create the list of parameters for each household
    params = get_list_param(config)
    return params


def plot_data(df_P, df_flex, title=[]):
    """
    Plots Water Boiler, Heat Pump, and EV data for the full time horizon,
    saves the figure with a timestamp in 'plots/'.
    """

    # Use DataFrame index for time axis (datetime or integer index)
    time_steps = df_P.index

    # Make sure output folder exists
    os.makedirs("plots", exist_ok=True)

    # Timestamp for filename
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    if "Before" in title:
        str_add = "_before_samp"
    elif "After" in title:
        str_add = "_after_samp"
    else: 
        str_add = ""

    filename = f"plots/{timestamp+str_add}.png"

    # Create figure with more vertical spacing
    fig, axes = plt.subplots(3, 1, figsize=(14, 16), constrained_layout=True)

    # ---------------------- Water Boiler ----------------------
    ax1 = axes[0]
    ax1.plot(time_steps, df_flex["T_ref_WB"], label="Température de Référence (°C)", color="blue", lw=2)
    ax1.set_title("Water Boiler", fontsize=14, pad=15)
    ax1.set_ylabel("Temp. (°C)")
    ax1.legend(loc="upper left")

    ax1b = ax1.twinx()
    ax1b.plot(time_steps, df_P["P_WB"], label="Puissance de Référence (kW)", color="green", lw=2, linestyle="--")
    ax1b.plot(time_steps, df_flex["P_use_WB"], label="P_use (kW)", color="orange", lw=2, linestyle="--")
    ax1b.plot(time_steps, df_flex["P_loss_WB"], label="Pertes (kW)", color="red", lw=2, linestyle="--")
    # ax1b.plot(time_steps, df_flex["Water_use"], label="Conso (L)", color="pink", lw=2)
    ax1b.set_ylabel("Puissance (kW)")
    ax1b.legend(loc="upper right")

    # ---------------------- Heat Pump ----------------------
    ax2 = axes[1]
    ax2.plot(time_steps, df_flex["T_set_HP"], label="Température setpoint (°C)", color="purple", lw=2)
    ax2.plot(time_steps, df_flex["T_wall_HP"], label="Température Wall (°C)", color="orange", lw=2)
    ax2.plot(time_steps, df_flex["T_out_HP"], label="Température Extérieure (°C)", color="darkblue", lw=2)
    ax2.plot(time_steps, df_flex["T_ref_HP"], label="Température de Référence (°C)", color="blue", lw=2)
    ax2.set_title("Heat Pump", fontsize=14, pad=15)
    ax2.set_ylabel("Temp. (°C)")
    ax2.set_ylim(0, 40)
    ax2.legend(loc="upper left")

    ax2b = ax2.twinx()
    ax2b.plot(time_steps, df_P["P_HP"], label="Puissance de Référence (kW)", color="green", lw=2, linestyle="--")
    ax2b.plot(time_steps, df_flex["P_loss_HP"], label="Perte de Puissance (kW)", color="red", lw=2, linestyle="--")
    ax2b.set_ylabel("Puissance (kW)")
    ax2b.legend(loc="upper right")

    # ---------------------- EV ----------------------
    ax3 = axes[2]
    ax3.step(time_steps, df_flex["EV_plugged"], label="Statut EV (Branché)", lw=2, color="black")
    ax3.plot(time_steps, df_flex["SoC_ref_EV"], label="SOC", lw=2, color="blue")

    # Mark arrivals and departures
    arr_indices = df_flex.index[df_flex["EV_arrival"] == 1].tolist()
    dep_indices = df_flex.index[df_flex["EV_departure"] == 1].tolist()
    ax3.scatter(arr_indices, df_flex.loc[arr_indices, "EV_plugged"], label="Arrivée", color="green", marker="o", s=60)
    ax3.scatter(dep_indices, df_flex.loc[dep_indices, "EV_plugged"], label="Départ", color="red", marker="x", s=60)

    ax3.set_title("Electric Vehicle", fontsize=14, pad=15)
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("Statut")
    ax3.legend(loc="upper left")

    ax3b = ax3.twinx()
    ax3b.plot(time_steps, df_P["P_EV"] * 1000, label="Puissance de Référence (kW)", color="green", lw=2, linestyle="--")
    ax3b.set_ylabel("Puissance (kW)")
    ax3b.legend(loc="upper right")

    # ---------------------- Final Layout ----------------------
    fig.suptitle("Résumé du Profil:", fontsize=18, y=0.995)
    
    # Save figure
    fig.savefig(filename, dpi=300)
    plt.close(fig)
>>>>>>> Stashed changes
