"""
Exploratory data analysis for King County housing data.
Summarizes price distribution, correlations, and spatial coverage
before feature engineering (Week 2) begins.
"""

import pandas as pd

from src.data_loader import load_raw_data, initial_clean
from src.geo_utils import distance_to_city_center

SEATTLE_CENTER = (47.6062, -122.3321)


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns basic descriptive statistics for key numeric columns.
    """
    cols = ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "yr_built"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].describe()


def price_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Correlation of each numeric feature with price, sorted descending.
    Helps flag which raw features matter most before spatial features are added.
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()["price"].drop("price").sort_values(ascending=False)
    return corr


def zipcode_price_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median price and property count per zipcode.
    Useful for spotting undervalued/overvalued zip clusters early.
    """
    if "zipcode" not in df.columns:
        raise ValueError("zipcode column not found in dataframe")

    summary = df.groupby("zipcode").agg(
        median_price=("price", "median"),
        count=("price", "size"),
    ).sort_values("median_price", ascending=False)

    return summary


def missing_value_report(df: pd.DataFrame) -> pd.Series:
    """
    Reports missing value counts per column (should be near-zero after
    initial_clean, but useful as a sanity check on raw data).
    """
    return df.isnull().sum().sort_values(ascending=False)


if __name__ == "__main__":
    df = initial_clean(load_raw_data())
    df = distance_to_city_center(df, *SEATTLE_CENTER)

    print("=== Summary Stats ===")
    print(summary_stats(df))

    print("\n=== Price Correlations ===")
    print(price_correlations(df))

    print("\n=== Top 10 Zipcodes by Median Price ===")
    print(zipcode_price_summary(df).head(10))

    print("\n=== Missing Values ===")
    print(missing_value_report(df))
