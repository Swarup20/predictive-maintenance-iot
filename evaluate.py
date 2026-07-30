import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/housing.csv"
TARGET = "price"

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)
rmse = mean_squared_error(y, predictions) ** 0.5
r2 = r2_score(y, predictions)

print("=" * 40)
print("Model Evaluation")
print("=" * 40)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
