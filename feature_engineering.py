import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def create_property_age(self):
        if "yr_built" in self.df.columns:
            self.df["property_age"] = 2026 - self.df["yr_built"]

    def create_house_size_ratio(self):
        if "sqft_living" in self.df.columns and "sqft_lot" in self.df.columns:
            self.df["living_lot_ratio"] = (
                self.df["sqft_living"] /
                self.df["sqft_lot"].replace(0, np.nan)
            )

    def create_bed_bath_ratio(self):
        if "bedrooms" in self.df.columns and "bathrooms" in self.df.columns:
            self.df["bed_bath_ratio"] = (
                self.df["bedrooms"] /
                self.df["bathrooms"].replace(0, np.nan)
            )

    def create_total_rooms(self):
        if "bedrooms" in self.df.columns and "bathrooms" in self.df.columns:
            self.df["total_rooms"] = (
                self.df["bedrooms"] + self.df["bathrooms"]
            )

    def create_price_per_sqft(self):
        if (
            "price" in self.df.columns and
            "sqft_living" in self.df.columns
        ):
            self.df["price_per_sqft"] = (
                self.df["price"] /
                self.df["sqft_living"].replace(0, np.nan)
            )

    def create_location_clusters(self):
        if (
            "latitude" in self.df.columns and
            "longitude" in self.df.columns
        ):
            self.df["location_cluster"] = (
                self.df["latitude"].round(2).astype(str)
                + "_"
                + self.df["longitude"].round(2).astype(str)
            )

    def generate_features(self):
        self.create_property_age()
        self.create_house_size_ratio()
        self.create_bed_bath_ratio()
        self.create_total_rooms()
        self.create_price_per_sqft()
        self.create_location_clusters()

        self.df = self.df.fillna(0)

        return self.df


if __name__ == "__main__":

    df = pd.read_csv("data/housing.csv")

    engineer = FeatureEngineer(df)

    processed_df = engineer.generate_features()

    processed_df.to_csv(
        "data/housing_engineered.csv",
        index=False
    )

    print("Feature engineering completed successfully.")
    print(processed_df.head())
