import os

# ==========================
# Project Directories
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ==========================
# Dataset Paths
# ==========================

DATASET_PATH = os.path.join(DATA_DIR, "housing.csv")
ENGINEERED_DATASET_PATH = os.path.join(DATA_DIR, "housing_engineered.csv")

# ==========================
# Model Paths
# ==========================

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# ==========================
# Target Column
# ==========================

TARGET_COLUMN = "price"

# ==========================
# Training Configuration
# ==========================

TEST_SIZE = 0.20
RANDOM_STATE = 42

# ==========================
# Random Forest Parameters
# ==========================

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE
}

# ==========================
# Output Files
# ==========================

METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.csv")
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "predictions.csv")

# ==========================
# Create Required Directories
# ==========================

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
