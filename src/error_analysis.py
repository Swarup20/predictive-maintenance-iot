"""
Error analysis for the XGBoost baseline.
Identifies where the baseline fails hardest — key evidence for why
a spatial/graph-aware model (Week 3-4) is needed.
"""

import pandas as pd
import numpy as np
import xgboost as xgb

from src.data_loader import load_raw_data, initial_clean
from src.features import build_feature_set
from src.train_baseline import prepare_data, FEATURE_COLS


def load_trained_model(path: str = "models/xgboost_baseline.json") -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def compute_errors(df: pd.DataFrame, model: xgb.XGBRegressor) -> pd.DataFrame:
    """
    Adds predicted price and absolute percentage error per row.
    """
    cols = [c for c in FEATURE_COLS if c in df.columns and c != "price_per_sqft"]
    df = df.copy()
    df["predicted_price"] = model.predict(df[cols])
    df["abs_pct_error"] = np.abs(df["price"] - df["predicted_price"]) / df["price"]
    return df


def worst_zipcodes(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Zipcodes where the baseline has the highest average error.
    These are candidate 'gentrifying' or spatially complex areas
    where neighborhood context matters more than raw features.
    """
    summary = df.groupby("zipcode").agg(
        avg_error=("abs_pct_error", "mean"),
        median_price=("price", "median"),
        count=("price", "size"),
    ).sort_values("avg_error", ascending=False)

    return summary[summary["count"] >= 5].head(top_n)


def worst_individual_predictions(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Individual properties with the largest prediction errors.
    """
    cols = ["price", "predicted_price", "abs_pct_error", "zipcode", "lat", "long"]
    return df.sort_values("abs_pct_error", ascending=False)[cols].head(top_n)


if __name__ == "__main__":
    df = initial_clean(load_raw_data())
    df = build_feature_set(df)
    model = load_trained_model()

    df_errors = compute_errors(df, model)

    print("=== Worst Zipcodes (baseline underperforms) ===")
    print(worst_zipcodes(df_errors))

    print("\n=== Worst Individual Predictions ===")
    print(worst_individual_predictions(df_errors))

    df_errors.to_csv("data/baseline_errors.csv", index=False)
    print("\nFull error report saved to data/baseline_errors.csv")
