"""
File: 
parser.py

Description: 
Handles parsing test.csv files and allow for optional filter/adjustment to data.
Returns the raw data and the optional filter
"""
import pandas as pd
from typing import Dict

def parse(enabled_apps: list[str]) -> Dict[str,pd.DataFrame]:
    """
    Read an existing .csv into a DataFrame and returns a Dict of {app_name}->DataFrame
    """
    df = {}
    for app in enabled_apps:
        df[app] = pd.read_csv(f"./tests/{app}dataset.csv",
                              sep=",", header=0, index_col=0,
                              engine="c", quotechar="\"")
     
    return df

def parse_for_ml(enabled_apps: list[str]) -> Dict[str, pd.DataFrame]:
    """
    Reads an existing .csv into a dataframe like parse, but now allows for specfic apps 
    to have their own configurations to the dataframe before being returned.
    """
    
    return {}
