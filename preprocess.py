import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/housing.csv"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load the raw housing dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df):
    """Clean missing and duplicate records."""

    df = df.drop_duplicates()

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

    return df


def prepare_features(df):
    """Prepare numerical features for model training."""

    if "price" in df.columns:
        X = df.drop(columns=["price"])
        y = df["price"]
    else:
        X = df
        y = None

    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    scaler = StandardScaler()

    if len(numeric_columns) > 0:
        X[numeric_columns] = scaler.fit_transform(
            X[numeric_columns]
        )

    return X, y


def save_processed_data(X, y):
    """Save processed features and target."""

    X.to_csv(
        os.path.join(OUTPUT_DIR, "X_processed.csv"),
        index=False
    )

    if y is not None:
        y.to_csv(
            os.path.join(OUTPUT_DIR, "y_processed.csv"),
            index=False
        )


def preprocess():
    """Run the complete preprocessing pipeline."""

    print("=" * 50)
    print("Real Estate Data Preprocessing")
    print("=" * 50)

    df = load_data()

    print("\nCleaning data...")
    df = clean_data(df)

    print(f"Cleaned dataset: {df.shape}")

    print("\nPreparing features...")
    X, y = prepare_features(df)

    save_processed_data(X, y)

    print("\nPreprocessing completed successfully.")
    print(f"Processed data saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    preprocess()
