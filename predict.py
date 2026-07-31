import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"

model = joblib.load(MODEL_PATH)

sample_property = {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 2,
    "waterfront": 0,
    "view": 1,
    "condition": 4,
    "grade": 8,
    "sqft_above": 1800,
    "sqft_basement": 0,
    "yr_built": 2015,
    "zipcode": "98178",
    "latitude": 47.5112,
    "longitude": -122.257
}

input_data = pd.DataFrame([sample_property])

predicted_price = model.predict(input_data)

print("=" * 50)
print("Real Estate Price Prediction")
print("=" * 50)
print(f"Predicted Property Price: ${predicted_price[0]:,.2f}")
