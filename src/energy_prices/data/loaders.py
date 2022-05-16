"""
Functions for loading data
"""
from typing import Tuple

import pandas as pd

from ..config import RAW_PATH


def load_raw() -> pd.DataFrame:
    """
    Loads the raw data and does some basic preprocessing.

    returns:
    train   : training data in a pandas DataFrame
    """
    df = pd.read_csv(RAW_PATH)
    column_mapper = {"RRP":"price", "demand_pos_RRP":"demand_pos_price", "RRP_positive":"price_positive", "demand_neg_RRP":"demand_neg_price", "RRP_negative":"price_negative", "frac_at_neg_RRP":"frac_neg_price"}
    df.rename(columns = column_mapper, inplace = True)

    # Convert datatypes
    df.date = pd.to_datetime(df.date)
    df.school_day = df.school_day.map({"N": False, "Y":True}).astype('bool')
    df.holiday = df.holiday.map({"N": False, "Y":True}).astype('bool')

    # Extract year, month and day of week from data
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['dow'] = df.date.dt.day_of_week
    df['week'] = df.date.dt.isocalendar().week.astype('int')

    # Convert solar exposure from MJ/m^2 to MWh/m^2 (1 MJ = 1/(60*60) MWh)
    df.solar_exposure = df.solar_exposure/3600

    # Set date as index so can do resampling using pandas
    df.set_index('date', inplace=True)
    
    return df
