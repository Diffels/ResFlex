## ResFlex: A residential Load Profile Generator including flexibility

# Introduction

The flexibility of residential loads must be considered to reach an optimal long-term development strategy in distribution networks. However, due to privacy and logistic obstacles, real data for this is not yet available.

This work uses a tool for generating synthetic behaviors of the households' members and extends it to new usage of electricity. Based on that, it stochastically constructs a load profile by generating occupancy patterns inside the home and random appliance usage events depending on the attendance, to construct a baseload for the home. In addition to this fixed load, several types of flexible appliances are also considered with adapted flexibility boundaries.

These load profiles are used to model neighborhoods or small villages, to represent the reaction of a complete low-voltage grid to different tariffs. Users change their consumption patterns within the defined boundaries, reacting optimally to different energy and grid prices. This allows to evaluate the effects of the regulations for grid tariffs on the customers' energy bills, the grid congestion and the DSO revenues.

# Installation and usage

## As a library, from another project

ResFlex is a proper installable package (`resflex`). This packaging currently lives on the `package-version` branch (not yet merged into `main`), so pin the branch explicitly when installing from GitHub:

```
pip install git+https://github.com/Diffels/ResFlex.git@package-version
```

or, for local development against a cloned copy:

```
git clone https://github.com/Diffels/ResFlex.git
cd ResFlex
git checkout package-version
pip install -e .
```

(Once `package-version` is merged into `main`, drop the `@package-version` ref and the `git checkout` step above.)

Then generate profiles with your own parameters, no JSON files or repo layout required — `input_single.json`/`input_mult.json` are just example configs, any dict with the same keys works:

```python
import resflex

resflex.set_seed(42)  # optional, for reproducible runs

config = {...}  # see "Configuration of inputs" below
df_P, df_Flex, params = resflex.simulate_one(config, save=False, plot_res=False, print_res=False)
```

`resflex.simulate_all(config, ...)` works the same way for a stochastic population of households, returning `(dic_df_P, dic_df_Flex, dic_Params)` keyed by household.

## Running from source

Install the predefined environment with conda:

```
conda env create -f environment.yml
conda activate ResFlex
```

in the project directory, then run the simulation with:

```
python -m resflex.Simulate
```

By default this runs `simulate_all`, generating a stochastic population of households from `resflex/input_mult.json`. To instead simulate a single, fully-specified household with `simulate_one`, set `mult = False` in the `if __name__ == '__main__':` block of `resflex/Simulate.py` — this reads `resflex/input_single.json`.

If you want to remove the environment, use:

```
conda remove -n ResFlex --all
```

# Configuration of inputs

Both input files (`resflex/input_single.json`, `resflex/input_mult.json`) share these top-level keys:

    "nb_days": Number of days to simulate [day]
    "timestep": Resampling timestep of the output profiles [min]
    "year": Calendar year to simulate (drives weekday alignment and weather data) [year]
    "start_day": Day-of-year offset at which the simulation starts
    "flexibility": (boolean) Whether to compute/save the flexibility dataframe
    "output": Output file format when saving results: "csv", "xlsx" or "nc"

## `input_single.json` (used by `simulate_one`)

Fully specifies one household — appliances and flexible loads are given directly rather than drawn from probabilities.

    "appliances": {"WashingMachine", "TumbleDryer", "DishWasher", "WasherDryer"}: (0/1) presence of each time-shiftable appliance
    "WB" / "EV" / "HP": (boolean) enable the water boiler / electric vehicle / heat pump for this household
    "WB_data": {"Pmax" (kW), "Volume" (L), "T_set" (°C)}
    "EV_data": {"Consumption" (kWh/100km), "Capacity" (kWh), "Pmax" (kW), "eta", "SoC_target" (0-1), "Usage" (km/year)}
    "HP_data": {"Year" (construction year), "Size" (m²), "Floors", "P_nom" (kW), "COP"}
    "inhabitants": Number of household members
    "occupations": List of one occupation type per member, chosen from: 'Random', 'FTE' (Full Time Employed), 'PTE' (Partial Time Employed), 'U12' (Under 12 y.o.), 'Retired', 'Unemployed', 'School' (Student)

## `input_mult.json` (used by `simulate_all`)

Generates `nb_households` households by drawing each appliance/flexible-load parameter from a probability distribution. Every `P_<field>` array is a probability distribution over the corresponding `<field>` array of possible values (both arrays must be the same length, and probabilities must sum to 1).

    "nb_households": Number of households to generate
    "appliances": {"P_WashingMachine", "P_TumbleDryer_given_WM" (probability of owning a tumble dryer given a washing machine), "P_DishWasher"}: presence probabilities [0-1]
    "P_PV" / "P_BSS" / "P_WB" / "P_EV" / "P_HP": Probability [0-1] that a household has PV / a battery storage system / a water boiler / an EV / a heat pump
    "PV_data", "BSS_data", "WB_data", "EV_data", "HP_data": for each parameter of the appliance (e.g. "Pmax", "Capacity"), a value array and its matching "P_<parameter>" probability array
    "inhabitants" / "P_inhabitants": Possible household sizes and their probability distribution
