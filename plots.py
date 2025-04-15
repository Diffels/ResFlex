# -*- coding: utf-8 -*-
"""
Imported from load-shifting:
    https://github.com/pielube/loadshifting
    (Sylvain Quoilin)
    
Modified by duchmax
August 2024
"""

# Import required modules
import plotly.graph_objects as go
from constant import defaultcolors, StaticLoad


def make_demand_plot(idx,data,PV = None,title='Consumption', NB_Scenario = 1):
    '''
    Use of plotly to generate a stacked consumption plot, on local server.

    Parameters
    ----------
    idx : DateTime
        Index of the time period to be plotted.
    data : pandas.DataFrame
        Dataframe with the columns to be plotted. Its index should include idx.
    title : str, optional
        Title of the plot. The default is 'Consumption'.

    Returns
    -------
    Plotly figure.

    '''
    
    fig = go.Figure()
    # Base = list(set(data.columns) & set(StaticLoad))
    # data["Base Load"] = data[Base].sum(axis=1)
    # data = data.drop(columns=Base)
    cols = data.columns.tolist()
    if 'BatteryGeneration' in cols:
        cols.remove('BatteryGeneration')

    if PV is not None:
        fig.add_trace(go.Scatter(
                name = 'PV geneartion',
                x = idx,
                y = PV.loc[idx],
                stackgroup='three',
                fillcolor='rgba(255, 255, 126,0.5)',
                mode='none'               # this remove the lines
                          ))        
    if 'Heating' in cols:
        fig.add_trace(go.Scatter(
            name='Heating',
            x=idx,
            y=data.loc[idx, 'Heating'],
            stackgroup='one',
            fillcolor='rgba(255, 0, 0, 0.5)', 
            mode='none'  
            ))
        cols.remove('Heating')  # Remove 'Heating' after adding it to the plot

    for key in cols:
        fig.add_trace(go.Scatter(
            name = key,
            x = idx,
            y = data.loc[idx,key],
            stackgroup='one',
            fillcolor = defaultcolors[key],
            mode='none'               # this remove the lines
           ))

    fig.update_layout(title = title,
                      xaxis_title = r'Dates',
                      yaxis_title = r'Power [W]'
                      )
    fig.show()
    return fig