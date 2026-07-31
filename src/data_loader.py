"""
Data loader for King County House Sales dataset.
Downloads/loads raw housing data and does initial cleaning.
"""

import pandas as pd
import os

RAW_DATA_PATH = "data/kc_house_data.csv"

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw King County housing dataset.
    Dataset source: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
    Download kc_house_data.csv manually and place it in the data/ folder.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download kc_house_data.csv from Kaggle and place it in data/"
        )
    df = pd.read_csv(path)
    return df


def initial_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: drop duplicates, handle missing lat/long,
    keep only relevant columns for spatial modeling.
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=["lat", "long", "price"])

    keep_cols = [
        "id", "price", "bedrooms", "bathrooms", "sqft_living",
        "sqft_lot", "floors", "condition", "grade",
        "yr_built", "lat", "long", "zipcode"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    return df


if __name__ == "__main__":
    df = load_raw_data()
    df_clean = initial_clean(df)
    print(f"Loaded {len(df)} rows, {len(df_clean)} after cleaning")
    print(df_clean.head())
