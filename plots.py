import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
        name='SOC (%)',
        line=dict(color='blue')
    ))

    # Plot occupancy
    fig.add_trace(go.Scatter(
        y=occupancy,
        mode='lines',
        name='Occupancy',
        line=dict(color='green')
    ))

    # Plot load profile
    fig.add_trace(go.Scatter(
        y=load_profile,
        mode='lines',
        name='Load (kW)',
        line=dict(color='orange')
    ))

    # Plot EV_refilled
    fig.add_trace(go.Scatter(
        y=EV_refilled,
        mode='lines',
        name='EV Refilled',
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