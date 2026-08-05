import os
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def load_dataset(file_path):
    """
    Load dataset from CSV file.
    """
    return pd.read_csv(file_path)


def save_dataset(dataframe, file_path):
    """
    Save DataFrame to CSV.
    """
    dataframe.to_csv(file_path, index=False)


def save_model(model, file_path):
    """
    Save trained model.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)


def load_model(file_path):
    """
    Load trained model.
    """
    return joblib.load(file_path)


def create_directory(directory):
    """
    Create directory if it does not exist.
    """
    os.makedirs(directory, exist_ok=True)


def evaluate_model(y_true, y_pred):
    """
    Calculate regression metrics.
    """
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2 Score": r2_score(y_true, y_pred)
    }

    return metrics


def print_metrics(metrics):
    """
    Print evaluation metrics.
    """
    print("=" * 40)
    print("Model Evaluation")
    print("=" * 40)

    for metric, value in metrics.items():
        print(f"{metric:<12}: {value:.4f}")


def dataset_summary(dataframe):
    """
    Display dataset information.
    """
    print("=" * 40)
    print("Dataset Summary")
    print("=" * 40)
    print(f"Rows    : {dataframe.shape[0]}")
    print(f"Columns : {dataframe.shape[1]}")
    print("\nColumn Names:")
    print(list(dataframe.columns))


def check_missing_values(dataframe):
    """
    Display missing values in dataset.
    """
    missing = dataframe.isnull().sum()

    print("=" * 40)
    print("Missing Values")
    print("=" * 40)
    print(missing[missing > 0])


def display_feature_types(dataframe):
    """
    Display data types of features.
    """
    print("=" * 40)
    print("Feature Types")
    print("=" * 40)
    print(dataframe.dtypes)


if __name__ == "__main__":

    print("Utility module loaded successfully.")
