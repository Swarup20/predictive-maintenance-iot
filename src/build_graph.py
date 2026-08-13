"""
Builds a K-nearest-neighbor spatial graph from housing data.
Houses = nodes, edges = K nearest neighbors by Haversine distance.
This graph feeds the spatial embedding / GNN model in Week 3-4.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.data_loader import load_raw_data, initial_clean
from src.features import build_feature_set

K_NEIGHBORS = 10


def build_knn_graph(df: pd.DataFrame, k: int = K_NEIGHBORS):
    """
    Builds a KNN graph using lat/long coordinates.
    Uses haversine metric directly via sklearn (requires radians input).
    Returns edge_index (list of [source, target] node pairs) and distances.
    """
    coords = np.radians(df[["lat", "long"]].values)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="haversine")
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # drop the first neighbor (itself, distance 0)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # convert radians back to km
    distances_km = distances * 6371.0

    edge_list = []
    edge_weights = []
    for i in range(len(df)):
        for j_idx in range(k):
            neighbor = indices[i, j_idx]
            dist = distances_km[i, j_idx]
            edge_list.append((i, neighbor))
            edge_weights.append(dist)

    edge_df = pd.DataFrame(edge_list, columns=["source", "target"])
    edge_df["distance_km"] = edge_weights

    return edge_df


def graph_summary(edge_df: pd.DataFrame, df: pd.DataFrame):
    """
    Quick sanity check on the built graph.
    """
    print(f"Nodes: {len(df)}")
    print(f"Edges: {len(edge_df)}")
    print(f"Avg neighbor distance: {edge_df['distance_km'].mean():.3f} km")
    print(f"Max neighbor distance: {edge_df['distance_km'].max():.3f} km")


if __name__ == "__main__":
    df = initial_clean(load_raw_data())
    df = build_feature_set(df)
    df = df.reset_index(drop=True)  # ensure node indices are 0..N-1

    edge_df = build_knn_graph(df, k=K_NEIGHBORS)
    graph_summary(edge_df, df)

    edge_df.to_csv("data/knn_edges.csv", index=False)
    print("\nGraph edges saved to data/knn_edges.csv")
