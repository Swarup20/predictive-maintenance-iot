"""
Geospatial utility functions for distance calculations.
"""

import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance (in km) between two points
    on Earth given their latitude/longitude in decimal degrees.
    Works with scalars or numpy arrays (vectorized).
    """
    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def distance_to_city_center(df: pd.DataFrame, center_lat: float, center_lon: float) -> pd.DataFrame:
    """
    Adds a 'dist_to_center_km' column measuring each property's
    distance to a reference point (e.g. downtown Seattle).
    """
    df = df.copy()
    df["dist_to_center_km"] = haversine_distance(
        df["lat"], df["long"], center_lat, center_lon
    )
    return df


def pairwise_distance_matrix(df: pd.DataFrame, sample_size: int = None) -> np.ndarray:
    """
    Computes a pairwise Haversine distance matrix between all properties.
    Use sample_size for large datasets to avoid memory blowup (NxN matrix).
    """
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    lats = df["lat"].values
    lons = df["long"].values

    n = len(df)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        dist_matrix[i] = haversine_distance(lats[i], lons[i], lats, lons)

    return dist_matrix


if __name__ == "__main__":
    # quick sanity check: distance between two known Seattle points
    d = haversine_distance(47.6062, -122.3321, 47.6205, -122.3493)
    print(f"Test distance: {d:.2f} km")
