import numpy as np
import pandas as pd
import tabulate

users = pd.read_csv("PSCC2026/communityflex_users.csv", sep=";")

# create several cases flex/non flex/high solar


import json
import pandas as pd

def summarize_houses(json_file):
    """
    Summarize houses from a JSON file into a pandas DataFrame.
    Keeps only useful high-level info for plotting.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    summaries = []

    for house_name, house in data.items():
        summary = {
            "House": house_name,
            "Family": house.get("family"),

            # Equipment flags
            "HP": house.get("HP", False),
            "WB": house.get("WB", False),
            "EV": house.get("EV", False),
            "BSS": house.get("BSS", False),
            "PV": house.get("PV", False),

            # Key specs (None if not present)
            "HP_P_nom": house.get("HP_data", {}).get("P_nom"),
            "HP_COP": house.get("HP_data", {}).get("COP"),
            "WB_Pmax": house.get("WB_data", {}).get("Pmax"),
            "WB_Volume": house.get("WB_data", {}).get("Volume"),
            "EV_Pmax": house.get("EV_data", {}).get("Pmax"),
            "EV_Capacity": house.get("EV_data", {}).get("Capacity"),
            "BSS_Pmax": house.get("BSS_data", {}).get("Pmax"),
            "BSS_Capacity": house.get("BSS_data", {}).get("Capacity"),
            "PV_Pmax": house.get("PV_data", {}).get("Pmax"),
        }
        summaries.append(summary)

    return pd.DataFrame(summaries)


# Example usage
if __name__ == "__main__":

    path = "Results/Multiple/2025-09-11_17-22-43/"
    df = summarize_houses(path+"users.json")
    df.to_excel(path+"users_summary.xlsx")
    print(df)

