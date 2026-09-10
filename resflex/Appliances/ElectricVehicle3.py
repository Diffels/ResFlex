# === EV Load Generation Base Module ===
# Author: nodiffels
# Date: October 2025

import pandas as pd
import numpy as np
import os
from typing import Any, Tuple
from datetime import datetime
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from ..stochastic import uniform_probability_centered, probability_event

class EV:
    def __init__(self, config: dict):
        """
        Initialize the EV class with configuration parameters.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing EV parameters such as capacity, SoC target, and max power.
        """
        self.capacity = config["EV_data"]["Capacity"]  # Battery capacity (kWh)
        self.soc_target = config["EV_data"]["SoC_target"]  # Target state of charge (0-1)
        self.pmax = config["EV_data"]["Pmax"]  # Maximum charging power (kW)
        self.usage = config["EV_data"]["Usage"]  # Average daily usage (km/day)
        self.consumption = config["EV_data"]["Consumption"]  # Energy consumption (kWh/100km)

    def calculate_trip_consumption(self, trip_duration: float) -> float:
        """
        Calculate the energy consumption for a trip based on its duration.

        Parameters
        ----------
        trip_duration : float
            Duration of the trip in minutes.

        Returns
        -------
        float
            Energy consumption for the trip in kWh.
        """
        # Convert trip duration to hours and estimate distance traveled
        trip_hours = trip_duration / 60.0
        distance = trip_hours * (self.usage / 24.0)  # Assume uniform daily usage
        return (distance * self.consumption) / 100.0  # Convert to kWh

    def estimate_charging_time(self, energy_needed: float) -> float:
        """
        Estimate the charging time required to replenish the given energy.

        Parameters
        ----------
        energy_needed : float
            Energy to replenish in kWh.

        Returns
        -------
        float
            Charging time in minutes.
        """
        return (energy_needed / self.pmax) * 60.0  # Convert hours to minutes
    
# -------------------------------------------------------------------------------------
# 0. Plotting functions
# -------------------------------------------------------------------------------------

def plot_weekly_trips(
    occupancy: np.ndarray,
    t_dep: np.ndarray,
    t_arr: np.ndarray,
    config: dict,
    show_labels: bool = True
) -> None:
    """
    Plot occupancy and detected trips, showing trip and stay durations in minutes.
    Short durations (<30 min) are highlighted in red.

    Parameters
    ----------
    occupancy : np.ndarray
        Boolean array (1 = at home, 0 = away), minute resolution.
    t_dep : np.ndarray
        Departure times (minutes from start of week).
    t_arr : np.ndarray
        Arrival times (minutes from start of week).
    config : dict
        Configuration containing at least 'year' and 'start_day'.
    show_labels : bool, optional
        Display duration labels on plot (default: True).
    """
    total_minutes = len(occupancy)
    time_axis = np.arange(total_minutes) / 60.0  # hours

    fig, ax = plt.subplots(figsize=(14, 5))

    # Occupancy profile
    ax.plot(time_axis, occupancy, color="black", lw=1, label="Occupancy (1 = Home)")
    ax.fill_between(time_axis, 0, occupancy, color="lightgray", alpha=0.5)

    # Plot each trip
    for i, (dep, arr) in enumerate(zip(t_dep, t_arr)):
        dep_h, arr_h = dep / 60.0, arr / 60.0
        trip_dur = arr - dep  # minutes
        ax.axvspan(dep_h, arr_h, color="tab:red", alpha=0.4, label="Trip" if i == 0 else None)

        if show_labels:
            mid_x = (dep_h + arr_h) / 2
            color = "red" if trip_dur < 30 else "black"
            ax.text(mid_x, 1.05, f"{trip_dur} min", color=color,
                    ha="center", va="bottom", fontsize=8, rotation=0)

        # Plot stay durations between trips
        if i < len(t_dep) - 1:
            stay_dur = t_dep[i + 1] - t_arr[i]
            mid_stay = ((t_arr[i] + t_dep[i + 1]) / 2) / 60.0
            if show_labels:
                color = "red" if stay_dur < 30 else "gray"
                ax.text(mid_stay, 0.1, f"{stay_dur} min", color=color,
                        ha="center", va="bottom", fontsize=8, rotation=0)

    # Formatting
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Occupancy")
    ax.set_title(
        f"Weekly Trips with Durations (Year {config.get('year', 'N/A')}, "
        f"Start Day {config.get('start_day', 0)})"
    )
    ax.set_xlim(0, 7 * 24)
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend(loc="upper right")

    # Vertical day separators
    for d in range(8):
        ax.axvline(d * 24, color="gray", lw=0.5, ls="--")
        if d < 7:
            ax.text(d * 24 + 12, -0.15, f"Day {d+1}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_weekly_trips_and_charging(
    occupancy: np.ndarray,
    trips: pd.DataFrame,
    charges: pd.DataFrame,
    config: dict,
    show_labels: bool = True
) -> None:
    """
    Debug visualization of trips and charging sessions in one plot.

    Parameters
    ----------
    occupancy : np.ndarray
        Boolean array (1 = at home, 0 = away), minute resolution.
    trips : pd.DataFrame
        DataFrame containing ['t_dep', 't_arr', 'duration', 'consumption'].
    charges : pd.DataFrame
        DataFrame containing ['t_start', 't_end', 'charge_length_min', 'charge_kwh'].
    config : dict
        Configuration with at least {'year', 'start_day', 'EV_data': {'Pmax': float}}.
    show_labels : bool, optional
        Display duration and energy labels on the plot (default: True).
    """

    total_minutes = len(occupancy)
    time_axis = np.arange(total_minutes) / 60.0  # convert to hours

    fig, ax = plt.subplots(figsize=(14, 6))

    # --- Occupancy profile (grey background) ---
    ax.plot(time_axis, occupancy, color="black", lw=1, label="Occupancy (1 = Home)")
    ax.fill_between(time_axis, 0, occupancy, color="lightgray", alpha=0.4)

    # --- Plot trips (red spans) ---
    for i, trip in trips.iterrows():
        dep_h = trip["t_dep"] / 60.0
        arr_h = trip["t_arr"] / 60.0
        dur_min = trip["duration"]

        ax.axvspan(dep_h, arr_h, color="tab:red", alpha=0.4, label="Trip" if i == 0 else None)
        if show_labels:
            mid_x = (dep_h + arr_h) / 2
            color = "red" if dur_min < 30 else "black"
            ax.text(mid_x, 1.05, f"{dur_min:.0f} min", color=color,
                    ha="center", va="bottom", fontsize=8)

        # Show stay durations between trips
        if i < len(trips) - 1:
            stay_min = trips.iloc[i + 1]["t_dep"] - trip["t_arr"]
            mid_stay = ((trip["t_arr"] + trips.iloc[i + 1]["t_dep"]) / 2) / 60.0
            if show_labels:
                color = "red" if stay_min < 30 else "gray"
                ax.text(mid_stay, 0.1, f"{stay_min:.0f} min", color=color,
                        ha="center", va="bottom", fontsize=8)

    # --- Plot charging sessions (blue spans) ---
    for j, ch in charges.iterrows():
        start_h = ch["t_start"] / 60.0
        end_h = ch["t_end"] / 60.0
        charge_dur = ch["charge_length_min"]
        charge_kwh = ch["charge_kwh"]

        ax.axvspan(start_h, end_h, color="tab:blue", alpha=0.3, label="Charging" if j == 0 else None)

        if show_labels:
            mid_x = (start_h + end_h) / 2
            ax.text(mid_x, 1.12, f"{charge_dur:.0f} min\n({charge_kwh:.1f} kWh)",
                    color="blue", ha="center", va="bottom", fontsize=8)

    # --- Plot formatting ---
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Occupancy")
    ax.set_title(
        f"Trips and Charging Events (Year {config.get('year', 'N/A')}, "
        f"Start Day {config.get('start_day', 0)})"
    )
    ax.set_xlim(0, 7 * 24)
    ax.set_ylim(-0.1, 1.25)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend(loc="upper right")

    # --- Day separators ---
    for d in range(8):
        ax.axvline(d * 24, color="gray", lw=0.5, ls="--")
        if d < 7:
            ax.text(d * 24 + 12, -0.15, f"Day {d + 1}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_ev_week_debug(
    occupancy: np.ndarray,
    trips: pd.DataFrame,
    charges: pd.DataFrame,
    config: dict,
    SoC_profile: np.ndarray = None,
    show_labels: bool = True
) -> None:
    """
    Debug plot showing EV trips, charging events, and SoC evolution for one simulated week.

    Parameters
    ----------
    occupancy : np.ndarray
        Boolean array (1 = at home, 0 = away), minute resolution.
    trips : pd.DataFrame
        DataFrame with columns ['t_dep', 't_arr', 'consumption'].
    charges : pd.DataFrame
        DataFrame with columns ['t_start', 't_end', 'charge_length_min', 'charge_kwh'].
    config : dict
        Simulation configuration (must include 'EV_data' with capacity, SoC_target, Pmax, etc.).
    SoC_profile : np.ndarray, optional
        State of Charge time series (same length as occupancy), if available.
    show_labels : bool, optional
        Display durations and charge labels (default: True).
    """
    total_minutes = len(occupancy)
    time_axis = np.arange(total_minutes) / 60.0  # hours

    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Occupancy background
    ax1.plot(time_axis, occupancy, color="black", lw=0.8, label="Occupancy (1=Home)")
    ax1.fill_between(time_axis, 0, occupancy, color="lightgray", alpha=0.4)

    # Plot trips (departures to arrivals)
    for i, trip in trips.iterrows():
        dep_h, arr_h = trip["t_dep"] / 60.0, trip["t_arr"] / 60.0
        trip_dur = trip["t_arr"] - trip["t_dep"]
        ax1.axvspan(dep_h, arr_h, color="tab:red", alpha=0.4, label="Trip" if i == 0 else None)
        if show_labels:
            color = "red" if trip_dur < 30 else "black"
            mid = (dep_h + arr_h) / 2
            ax1.text(mid, 1.05, f"{trip_dur:.0f} min", color=color, ha="center", va="bottom", fontsize=8)

        # Show "outside charging" event if it occurred (consumption < expected)
        if "outside_charge_kwh" in trip:
            if trip["outside_charge_kwh"] > 0:
                ax1.plot(dep_h, 1.1, marker="^", color="blue", markersize=8, label="Outside Charging" if i == 0 else None)

    # Plot charging sessions
    for i, ch in charges.iterrows():
        start_h, end_h = ch["t_start"] / 60.0, ch["t_end"] / 60.0
        charge_dur = ch["charge_length_min"]
        ax1.axvspan(start_h, end_h, color="tab:green", alpha=0.4, label="Home Charging" if i == 0 else None)
        if show_labels:
            mid = (start_h + end_h) / 2
            ax1.text(mid, 0.1, f"{charge_dur:.0f} min", color="green", ha="center", va="bottom", fontsize=8)

    # Add day separators
    for d in range(8):
        ax1.axvline(d * 24, color="gray", lw=0.5, ls="--")
        if d < 7:
            ax1.text(d * 24 + 12, -0.15, f"Day {d+1}", ha="center", fontsize=8)

    ax1.set_xlim(0, 7 * 24)
    ax1.set_ylim(-0.2, 1.3)
    ax1.set_xlabel("Time [hours]")
    ax1.set_ylabel("Occupancy / Trips / Charging")
    ax1.set_title(
        f"EV Weekly Behavior Debug (Year {config.get('year','N/A')}, "
        f"Start Day {config.get('start_day','N/A')})"
    )

    # Plot SoC on secondary axis if available
    if SoC_profile is not None:
        ax2 = ax1.twinx()
        ax2.plot(time_axis, SoC_profile, color="tab:blue", lw=1.2, label="State of Charge")
        ax2.set_ylabel("State of Charge (SoC)")
        ax2.set_ylim(0, 1.05)

        # Merge legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    ax1.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

def plot_ev(config, P_EV, Flex_EV):
    """
    Generate an interactive HTML plot for EV simulation results with SoC on a secondary axis.

    Parameters
    ----------
    config : dict
        Simulation configuration, includes 'year', 'start_day', etc.
    P_EV : np.ndarray
        Power consumption profile (kW).
    Flex_EV : pd.DataFrame
        DataFrame with EV plug/SoC indicators.
    """
    import plotly.graph_objects as go

    # Time axis
    total_minutes = len(P_EV)
    time_axis = pd.date_range(
        start=datetime(config["year"], 1, 1) + pd.Timedelta(days=config["start_day"]),
        periods=total_minutes,
        freq="T"
    )

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add power consumption trace
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=P_EV,
        mode="lines",
        name="Power Consumption (kW)",
        line=dict(color="blue", width=1)
    ), secondary_y=False)

    # Add EV plugged trace
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=Flex_EV["EV_plugged"],
        mode="lines",
        name="EV Plugged (1=Yes)",
        line=dict(color="green", width=1, dash="dot"),
        opacity=0.6
    ), secondary_y=False)

    # Add SoC reference trace
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=Flex_EV["SoC_ref_EV"],
        mode="lines",
        name="SoC Reference",
        line=dict(color="orange", width=1)
    ), secondary_y=True)

    # # Add SoC at arrival trace
    # fig.add_trace(go.Scatter(
    #     x=time_axis,
    #     y=Flex_EV["SoC_arr_EV"],
    #     mode="lines",
    #     name="SoC at Arrival",
    #     line=dict(color="red", width=1, dash="dot")
    # ), secondary_y=True)

    # Layout configuration
    fig.update_layout(
        title="EV Simulation Results",
        xaxis_title="Time",
        yaxis_title="Power (kW) / EV Plugged",
        yaxis2_title="State of Charge (SoC)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=600,
        width=1200
    )

    # Save to HTML
    plot_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_file = os.path.join(plot_dir, "EV_simulation_results.html")
    fig.write_html(output_file)
    print(f"Interactive plot saved to {output_file}")


# -------------------------------------------------------------------------------------
# 1. Departures, Arrivals, Travel Durations, and Stochastic consumption for a week
# -------------------------------------------------------------------------------------

def get_weekly_trips_from_occupancy(
    occupancy: np.ndarray[Any, np.dtype[np.bool_]],
    config: dict,
    day_shift_min: float = 0.0,

) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute weekly EV departure/arrival times and travel durations based on occupancy.

    Parameters
    ----------
    occupancy : np.ndarray
        Boolean array (1 = at home, 0 = away). The occupancy profile is at least one week long.
    config : dict
        Simulation configuration, must include 'year' and 'start_day'.

    Returns
    -------
    t_arr : np.ndarray
        Arrival times (in minutes from start of week).
    t_dep : np.ndarray
        Departure times (in minutes from start of week).
    dur_travel : np.ndarray
        Trip durations (in minutes).
    """
    # Find transitions using numpy for speed
    # The np.flatnonzero finds the indices of non-zero (or True) elements in an array.
    occ_prev = occupancy[:-1]
    occ_next = occupancy[1:]
    t_dep = np.flatnonzero((occ_prev == 1) & (occ_next == 0))     # from 1 to 0.
    t_arr = np.flatnonzero((occ_prev == 0) & (occ_next == 1)) + 1 # from 0 to 1.   

    # Ensure each departure has a corresponding arrival after it
    if len(t_arr) > 0:
        t_dep = t_dep[t_dep < t_arr[-1]]
    # Ensure each arrival has a corresponding departure before it
    if len(t_dep) > 0:
        t_arr = t_arr[t_arr > t_dep[0]]

    # Pair up departures and arrivals (both same length)
    min_len = min(len(t_dep), len(t_arr))
    if min_len == 0:
        # No paired departure/arrival this week (e.g. household never left home) -
        # keep the same schema as the normal return path so callers don't have to special-case it.
        return pd.DataFrame({
            't_dep': pd.Series(dtype=int),
            't_arr': pd.Series(dtype=int),
            'duration': pd.Series(dtype=int),
            'daytype': pd.Series(dtype=object),
            'consumption': pd.Series(dtype=float),
        })
    t_dep = t_dep[:min_len]
    t_arr = t_arr[:min_len]
    dur_travel = t_arr - t_dep
        
    if np.any(dur_travel <= 0):
        raise ValueError("Non-positive travel durations found.")

    # Sort by t_dep and reorder arrays
    sort_idx = np.argsort(t_dep)
    t_dep = t_dep[sort_idx]
    t_arr = t_arr[sort_idx]
    dur_travel = dur_travel[sort_idx]
    
    # Remove travels that last less than 30 mins
    valid_travel = dur_travel >= 30
    t_dep, t_arr, dur_travel = t_dep[valid_travel], t_arr[valid_travel], dur_travel[valid_travel]
    # Remove stays that are less than 30 mins
    # Remove short stays (<30min)
    if len(t_dep) > 1:
        dur_stay = t_dep[1:] - t_arr[:-1]
        valid_stay = np.concatenate(([True], dur_stay >= 30))
        t_dep, t_arr, dur_travel = t_dep[valid_stay], t_arr[valid_stay], dur_travel[valid_stay]
    
    # Determine day types for each trip
    day_types = []
    for dep in t_dep:
        day_index = (dep // 1440 + day_shift_min // 1440) % 7  # day of week, shifted by day_shift_min computed in EV_simulate()
        if day_index < 5:
            day_types.append('weekday')
        else:
            day_types.append('weekend')

    trips = pd.DataFrame({
        't_dep': t_dep,
        't_arr': t_arr,
        'duration': dur_travel,
        'daytype': day_types
    })
    
    # print("Trips DataFrame:\n", trips)

    trips["consumption"] = get_trip_consumption(config, trips)


    # # Print the trips:
    # print(f"Trips Departures ({len(trips['t_dep'])}):", trips['t_dep'])
    # print(f"Trips Arrivals ({len(trips['t_arr'])}):", trips['t_arr'])
    # print(f"Trip Durations ({len(trips['duration'])}):", trips['duration'])
    # plot_weekly_trips(occupancy, trips['t_dep'], trips['t_arr'], config)

    return trips

def get_trip_consumption(config: dict, trips: pd.DataFrame) -> np.float64:
    """
    Compute estimated energy consumption for a trip based on duration.

    Parameters
    ----------
    config : dict
        Must include 'EV_data' with keys ['Usage', 'Consumption'].
    trips : pd.DataFrame
        DataFrame containing arrival/departure times and travel durations (minutes).
    Returns
    -------
    trips_consumptions : np.float64
        Estimated energy consumption for the trips (kWh).
    """
    # Tunable Parameters
    r_dist_w = 0.15  # Random weekly distance variation
    r_dispatch = 0.1  # Random daily dispatch variation
    r_cons = 0.2  # Random consumption variation

    km_center = config["EV_data"]["Usage"] * 7 / 365
    cons_center = config["EV_data"]["Consumption"]
    weekly_km = uniform_probability_centered(km_center, r_dist_w * km_center)
    weekly_cons = uniform_probability_centered(cons_center, r_cons * cons_center)
    stoch_weekly_kwh = round((weekly_cons/100) * weekly_km, 2)
    # print("Weekly consumption:", stoch_weekly_kwh)

    # Dispatch the weekly kWh over the trips
    dispatch = trips["duration"] / np.sum(trips["duration"])
    disp_var = uniform_probability_centered(1.0, r_dispatch, size=len(trips))
    stoch_dispatch = disp_var * dispatch
    # print("Dispatch:", stoch_dispatch)
    trips_consumptions =  stoch_dispatch * stoch_weekly_kwh
    # print("Trips Consumption kWh:", trips_consumptions)

    return trips_consumptions

# ---------------------------------------------------------------------
# 3. External Charging Behavior
# ---------------------------------------------------------------------
def charging_outside(E_trip: float, E_leaving: float) -> float:
    """
    Estimate the probability-weighted charging demand outside the home.

    Parameters
    ----------
    E_trip : float
        Energy required for the trip (kWh).
    E_leaving : float
        Energy available in the battery (kWh).

    Returns
    -------
    P : float
        Equivalent external charge energy (kWh).
    """
    
    r = E_trip / E_leaving

    # Tunable Parameters
    r *= 1.0  # Adjust ratio to reflect charging behavior
    short_journey_threshold = 0.2 # Below this ratio, no external charging occurs
    var_ch_outside = 0.1 # Variability in outside charging
        
    if r > 1.0: # If journey requires more energy than available, charge mandatory.
        # print(f"Charging outside: {E_trip/2} kWh")
        stoch_factor_charge = uniform_probability_centered(1.0, var_ch_outside)
        return (E_trip / 2) * stoch_factor_charge
    
    elif r < short_journey_threshold: # Short journeys do not require charge.
        return 0.0
    
    else: # For intermediate journeys, charge with probability r.
        # print(f"Charging outside with probability {r}: ")
        stoch_factor_charge = uniform_probability_centered(1.0, var_ch_outside)
        return (E_trip / 2) * stoch_factor_charge if probability_event(r) else 0.0

# ---------------------------------------------------------------------
# 4. Charging events Generator
# ---------------------------------------------------------------------
def weekly_charging_events(
    config: dict,
    trips: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate charging session start times and durations for one simulated week.

    Parameters
    ----------
    config : dict
        EV and simulation parameters.
        Must include keys:
            EV_data = {
                'Capacity': float (kWh),
                'SoC_target': float [0–1],
                'Pmax': float (kW)
            }
    trips : pd.DataFrame
        Must include columns:
            ['t_dep', 't_arr', 'dur_travel', 'consumption']

    Returns
    -------
    charges : pd.DataFrame
        Columns:
            ['t_start', 't_end', 'charge_length_min', 'charge_kwh']
    """

    EV = config["EV_data"]
    battery = EV["Capacity"]
    Pmax = EV["Pmax"]        # kW
    SoC_target = EV["SoC_target"]

    # State of energy available at home (kWh)
    E_available = battery * SoC_target
    missing_charge = 0.0

    # Result collector
    charges_df = pd.DataFrame({"t_start": [], "t_end": [], "charge_length_min": [], "charge_kwh": []})

    for i, trip in trips.iterrows():
        t_arr = trip["t_arr"]
        t_dep = trip["t_dep"]
        E_trip = trip["consumption"]

        # Potential charging away 
        E_outside = charging_outside(E_trip, E_available)
        E_trip -= E_outside
        
        # Reduce available energy by last trip’s consumption
        E_available -= E_trip
        E_available = max(E_available, 0)

        # Determine energy needed to reach target SoC again
        E_deficit = (battery * SoC_target) - E_available + missing_charge
        E_deficit = max(E_deficit, 0)

        # Compute charging session
        charge_kwh = E_deficit
        charge_length_min = int((charge_kwh / Pmax) * 60)  # in minutes

        # Determine maximum available charging window
        if i < len(trips) - 1:
            next_dep = trips.loc[i + 1, "t_dep"]
            max_charge_time = next_dep - t_arr
        else:
            # Last event — assume until end (7 days)
            max_charge_time = 7 * 24 * 60 - t_arr

        # Adjust for time constraints
        if charge_length_min > max_charge_time:
            # Not enough time to finish charging fully
            charge_length_min = max_charge_time
            missing_charge = (E_deficit - (max_charge_time / 60) * Pmax)
        else:
            missing_charge = 0.0

        # Update available energy after charging
        E_available = min(E_available + (charge_length_min / 60) * Pmax, battery)

        # Record event
        charges = pd.DataFrame([{
            "t_start": t_arr,
            "t_end": t_arr + charge_length_min,
            "charge_length_min": charge_length_min,
            "charge_kwh": charge_kwh
        }])

        charges_df = pd.concat([charges_df, charges], ignore_index=True)

    # Optional sanity checks
    if not all(charges_df["charge_length_min"] >= 0):
        raise ValueError("Negative charging durations detected.")
    if not all(charges_df["t_end"] >= charges_df["t_start"]):
        raise ValueError("Invalid charging event times (t_end < t_start).")

    return charges_df

# ---------------------------------------------------------------------
# 5. EV Simulation main loop
# ---------------------------------------------------------------------
def EV_simulate(
    occupancy: np.ndarray[Any, np.dtype[np.bool_]],
    config: dict
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute EV load profile for a given household occupancy pattern.

    Parameters
    ----------
    occupancy : np.ndarray
        Occupancy profile (1 = home, 0 = away). The profile length is at least one week.
    config : dict
        Simulation configuration, includes EV_data, year, start_day, nb_days.

    Returns
    -------
    P_EV : np.ndarray
        Power consumption profile (W or kW).
    Flex_EV : pd.DataFrame
        DataFrame with EV plug/SoC indicators.
    """
    # Modify occupancy based on day_shift
    day_shift = (datetime(config["year"], 1, 1) + pd.Timedelta(days=config["start_day"])).weekday()
    day_shift_min = day_shift * 24 * 60
    occupancy = np.roll(occupancy, -day_shift_min)

    # Initialize time base
    total_minutes = config.get("nb_days", 7) * 24 * 60
    P_EV = np.zeros(total_minutes)

    # Initialize Flex_EV DataFrame
    Flex_EV = pd.DataFrame(0, index=np.arange(total_minutes),
                           columns=["EV_plugged", "EV_arrival", "EV_departure",
                                    "SoC_ref_EV", "SoC_arr_EV"])
    Flex_EV = Flex_EV.astype({
        "EV_plugged": int,
        "EV_arrival": int,
        "EV_departure": int,
        "SoC_ref_EV": float,
        "SoC_arr_EV": float
    })

    # Weekly iteration placeholder
    weekly_timesteps = 7 * 24 * 60
    n_weeks = int((config["nb_days"] - 1) / 7) + 1 # int(x) with x < 1 gives 0

    for w in range(n_weeks):
        # print(f"--- Week {w} Simulation ---")
        
        current_week_occupancy = occupancy[w * weekly_timesteps:(w + 1) * weekly_timesteps]
            
        # Weekly journey data
        trips = get_weekly_trips_from_occupancy(current_week_occupancy, config, day_shift_min)

        # Trim trips if last week is incomplete
        days_left = config["nb_days"] - w * 7
        if days_left < 7:
            # Trim trips to remaining days using the DataFrame
            valid_trips = trips['t_dep'] < days_left * 24 * 60
            trips = trips[valid_trips].reset_index(drop=True)

        # Based on arrivals/departures, get charging events
        charges = weekly_charging_events(config, trips)

        # Apply events to profile
        for i in range(len(charges)):
            start = int(w * weekly_timesteps + charges["t_start"][i]) #error of rounding?
            end = int(start + charges["charge_length_min"][i]) #error of rounding? 
            P_EV[start:end] = config["EV_data"]["Pmax"]  # kW

        # Update Flex_EV indicators
        for i, trip in trips.iterrows():
            dep_idx = w * weekly_timesteps + trip["t_dep"]
            arr_idx = w * weekly_timesteps + trip["t_arr"]
            Flex_EV.at[arr_idx, "EV_arrival"] = 1
            Flex_EV.at[dep_idx, "EV_departure"] = 1

            # Reference SoC at departure
            SoC_ref = max(0.0, min(1.0, (config["EV_data"]["Capacity"] * config["EV_data"]["SoC_target"] - trip["consumption"]) / config["EV_data"]["Capacity"]))
            Flex_EV.at[dep_idx, "SoC_ref_EV"] = SoC_ref

            # SoC at arrival
            SoC_arr = max(0.0, min(1.0, SoC_ref - trip["consumption"] / config["EV_data"]["Capacity"]))
            Flex_EV.at[arr_idx, "SoC_arr_EV"] = SoC_arr

            # For each arrival, find the next departure
            if i + 1 < len(trips):
                dep = trips.iloc[i + 1]["t_dep"] + w * weekly_timesteps
            else:
                dep = len(Flex_EV) - 1  # Plugged until the end of the week if no more departures

            # Mark as plugged between arrival and departure
            Flex_EV.loc[arr_idx:dep, "EV_plugged"] = 1        

    # plot_weekly_trips_and_charging(occupancy, trips, charges, config)
    #     plot_ev_week_debug(
    #     occupancy=current_week_occupancy,
    #     trips=trips,
    #     charges=charges,
    #     config=config,
    #     SoC_profile=Flex_EV["SoC_ref_EV"].values,  # optional if tracked
    # )


    # # Trim output to simulation length
    # P_EV = P_EV[:total_minutes]
    # Flex_EV = Flex_EV.iloc[:total_minutes]

    if config.get("plot_EV", False):
        plot_ev(config, P_EV, Flex_EV)

    return P_EV, Flex_EV