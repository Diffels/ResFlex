import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import xarray as xr
from datetime import datetime

"""Saving functions to create files with simulation results"""

def save_one(config, filetype, df_P, df_Flex, dic_Param):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    add_indices(df_P, df_Flex, config)  # Add indices to the dataframes
    if filetype == 'csv':
        csv_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Results/Single/{current_time}.csv")
        df_P.to_csv(csv_filename, index=True)
        if config['flexibility']:
            df_Flex.to_csv(csv_filename.replace('.csv', '_Flex.csv'), index=True)
    else:
        raise ValueError(f"Unsupported file type for saving: {filetype}")
    
    json_filename = csv_filename.replace('.csv', '_Param.json')
    with open(json_filename, 'w', encoding="utf-8") as json_file:
        json.dump(dic_Param, json_file, ensure_ascii=False, indent=4)

def save_all(config, filetype, dic_df_P, dic_df_Flex, dic_Params, houses_params):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Results/Multiple/{current_time}."+filetype)
    flex_filename = filename.replace('.'+filetype, '_Flex.'+filetype)
    if filetype == 'xlsx':
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
        with open(filename, 'w', encoding="utf-8") as file, open(flex_filename, 'w', encoding="utf-8") as flex_file:
            for u, (df_P, df_Flex) in enumerate(zip(dic_df_P.values(), dic_df_Flex.values())):
                df_P.to_csv(file, mode='a', header=True if u == 0 else False)
                df_Flex.to_csv(flex_file, mode='a', header=True if u == 0 else False)
    elif filetype == 'nc':
        # Convert data to xarray Dataset and save to NetCDF
        ds_P = xr.Dataset({house: xr.DataArray(data=df.values, dims=["time", "variables"], coords={"time": df.index, "variables": df.columns}) for house, df in dic_df_P.items()})
        ds_P.to_netcdf(filename)

        ds_Flex = xr.Dataset({house: xr.DataArray(data=df.values, dims=["time", "variables"], coords={"time": df.index, "variables": df.columns}) for house, df in dic_df_Flex.items()})
        ds_Flex.to_netcdf(flex_filename)

    else:
        raise ValueError(f"Unsupported file type for saving: {filetype}")

    with open(filename.replace('.xlsx', '_Param.json'), 'w', encoding="utf-8") as json_file:
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

def add_indices(df_P, df_Flex, config):
    start_date = datetime(2024, 1, 1) + pd.Timedelta(days=config["start_day"])
    end_date = start_date + pd.Timedelta(days=config["nb_days"])
    time_index = pd.date_range(start=start_date, end=end_date, freq="min")  # Minute by minute frequency
    df_P.index = time_index[:len(df_P)] if len(time_index) >= len(df_P) else time_index
    df_Flex.index = time_index[:len(df_Flex)] if len(time_index) >= len(df_Flex) else time_index

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
    app_list = ['WashingMachine', 'DishWasher', 'TumbleDryer', 'WasherDryer']
    values = [np.random.choice([0,1], size=config['nb_households'], p=[1-config['appliances'][f'P_{a}'], config['appliances'][f'P_{a}']]) for a in app_list]
    for i, house in enumerate(list_param):
        house['appliances'] = {}
        for a, appliance in enumerate(app_list):
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
            house[f'{appliance}_data'][f] = float(lists[f][i])
    return list_param

def get_list_param(config):
    list_param = [{}] * config['nb_households']
    list_param = append_recurring(['nb_days', 'timestep', 'year', 'start_day', 'flexibility'], list_param, config)
    list_param = append_appliances(list_param, config)
    list_param = append_flexible('HP',['Year', 'Size', 'Floors','P_nom', 'COP'], list_param, config)
    list_param = append_flexible('WB', ['Pmax', 'Volume', 'Tset'], list_param, config)
    list_param = append_flexible('EV', ['Consumption', 'Capacity', 'Pmax', 'eta', 'SoC_target', 'Usage'], list_param, config)
    list_param = append_family(list_param, config)
    return list_param

def create_params(config):
    # Check if the config file is valid
    check_probas(['Year', 'Size', 'Floors','P_nom', 'COP'], config['HP_data'])
    check_probas(['Pmax', 'Volume', 'Tset'], config['WB_data'])
    check_probas(['Consumption', 'Capacity', 'Pmax', 'eta', 'SoC_target', 'Usage'], config['EV_data'])
    check_probas(['inhabitants'], config)
    # Create the list of parameters for each household
    params = get_list_param(config)
    return params