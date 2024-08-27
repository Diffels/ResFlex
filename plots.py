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
    StaticLoad_pres = [col for col in StaticLoad if col in data.columns]
    # Modification noedi
    # data['Base Load'] = data[StaticLoad_pres].sum(axis=1)
    data=data.copy()
    data.loc[:, 'Base Load'] = data[StaticLoad_pres].sum(axis=1)
    data= data.drop(columns=StaticLoad_pres)

    fig = go.Figure()
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
                      xaxis_title = 'Dates',
                      yaxis_title = 'Puissance (kW)'
                      )
    fig.show()
    return fig