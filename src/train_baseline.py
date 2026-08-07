"""
XGBoost baseline model for King County housing price prediction.
This is the benchmark the Week 3-4 GNN/spatial model must beat on MAPE.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb

from src.data_loader import load_raw_data, initial_clean
from src.features import build_feature_set

FEATURE_COLS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "condition", "grade", "house_age", "price_per_sqft",
    "bed_bath_ratio", "dist_to_center_km"
]
TARGET_COL = "price"


def prepare_data(df: pd.DataFrame):
    """
    Splits features/target and does train/test split.
    Drops price_per_sqft from features since it leaks target info directly.
    """
    cols = [c for c in FEATURE_COLS if c in df.columns and c != "price_per_sqft"]
    X = df[cols]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost regressor with reasonable default hyperparameters.
    """
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """
    Returns MAPE and RMSE on the test set — MAPE is the key metric
    the spatial/GNN model must beat.
    """
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {"MAPE": mape, "RMSE": rmse}


if __name__ == "__main__":
    df = initial_clean(load_raw_data())
    df = build_feature_set(df)

    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_xgboost(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    print(f"Baseline XGBoost — MAPE: {metrics['MAPE']:.4f}, RMSE: {metrics['RMSE']:,.0f}")

    model.save_model("models/xgboost_baseline.json")
    print("Model saved to models/xgboost_baseline.json")
