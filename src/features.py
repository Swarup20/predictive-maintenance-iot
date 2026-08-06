"""
Feature engineering for King County housing data.
Builds tabular features to feed the XGBoost baseline (Week 2).
"""

import pandas as pd
from datetime import datetime

from src.geo_utils import distance_to_city_center

SEATTLE_CENTER = (47.6062, -122.3321)
CURRENT_YEAR = datetime.now().year


def add_house_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'house_age' feature based on yr_built.
    """
    df = df.copy()
    df["house_age"] = CURRENT_YEAR - df["yr_built"]
    return df


def add_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'price_per_sqft' — useful for spotting over/undervalued homes.
    """
    df = df.copy()
    df["price_per_sqft"] = df["price"] / df["sqft_living"].replace(0, 1)
    return df


def add_room_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'bed_bath_ratio' — bedrooms per bathroom, a proxy for house layout type.
    """
    df = df.copy()
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"].replace(0, 1)
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full feature engineering pipeline, chaining all feature functions.
    """
    df = add_house_age(df)
    df = add_price_per_sqft(df)
    df = add_room_ratio(df)
    df = distance_to_city_center(df, *SEATTLE_CENTER)
    return df


if __name__ == "__main__":
    from src.data_loader import load_raw_data, initial_clean

    df = initial_clean(load_raw_data())
    df = build_feature_set(df)

    print(df[["house_age", "price_per_sqft", "bed_bath_ratio", "dist_to_center_km"]].describe())
