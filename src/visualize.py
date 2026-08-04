"""
Spatial visualization utilities for King County housing data.
Generates interactive Folium maps showing price distribution across geography.
"""

import folium
from folium.plugins import HeatMap
import pandas as pd

from src.data_loader import load_raw_data, initial_clean
from src.geo_utils import distance_to_city_center

# Seattle downtown coordinates (reference center point)
SEATTLE_CENTER = (47.6062, -122.3321)


def price_scatter_map(df: pd.DataFrame, sample_size: int = 1000) -> folium.Map:
    """
    Plots individual properties as circle markers colored by price tier.
    """
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    m = folium.Map(location=SEATTLE_CENTER, zoom_start=10, tiles="cartodbpositron")

    price_25 = df["price"].quantile(0.25)
    price_75 = df["price"].quantile(0.75)

    def get_color(price):
        if price < price_25:
            return "green"
        elif price > price_75:
            return "red"
        return "orange"

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["long"]],
            radius=3,
            color=get_color(row["price"]),
            fill=True,
            fill_opacity=0.6,
            popup=f"${row['price']:,.0f}",
        ).add_to(m)

    return m


def price_heatmap(df: pd.DataFrame) -> folium.Map:
    """
    Generates a weighted heatmap of price intensity across the region.
    """
    m = folium.Map(location=SEATTLE_CENTER, zoom_start=10, tiles="cartodbpositron")

    heat_data = [
        [row["lat"], row["long"], row["price"]]
        for _, row in df.iterrows()
    ]
    HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(m)

    return m


if __name__ == "__main__":
    df = initial_clean(load_raw_data())
    df = distance_to_city_center(df, *SEATTLE_CENTER)

    scatter_map = price_scatter_map(df)
    scatter_map.save("notebooks/price_scatter_map.html")

    heat_map = price_heatmap(df)
    heat_map.save("notebooks/price_heatmap.html")

    print("Maps saved to notebooks/ folder")
