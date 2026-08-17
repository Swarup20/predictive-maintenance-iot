"""
Generates spatial embeddings from the KNN graph using node2vec-style
random walks. These embeddings capture each house's neighborhood
context and feed directly into the GNN model (Week 4).
"""

import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec

from src.data_loader import load_raw_data, initial_clean
from src.features import build_feature_set
from src.build_graph import build_knn_graph, K_NEIGHBORS

EMBEDDING_DIM = 16
WALK_LENGTH = 10
NUM_WALKS = 80


def build_networkx_graph(edge_df: pd.DataFrame) -> nx.Graph:
    """
    Converts the KNN edge list into a NetworkX graph with distance
    as an edge weight (inverted so closer = stronger connection).
    """
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        dist = max(row["distance_km"], 1e-3)
        weight = 1.0 / dist
        G.add_edge(int(row["source"]), int(row["target"]), weight=weight)
    return G


def generate_embeddings(G: nx.Graph, dim: int = EMBEDDING_DIM) -> pd.DataFrame:
    """
    Runs node2vec random walks over the graph and trains embeddings.
    Returns a dataframe indexed by node id with embedding columns.
    """
    node2vec = Node2Vec(
        G, dimensions=dim, walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS, weight_key="weight", workers=2, quiet=True
    )
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    node_ids = sorted(G.nodes())
    embeddings = np.array([model.wv[str(n)] for n in node_ids])

    cols = [f"emb_{i}" for i in range(dim)]
    emb_df = pd.DataFrame(embeddings, columns=cols)
    emb_df["node_id"] = node_ids

    return emb_df


if __name__ == "__main__":
    df = initial_clean(load_raw_data())
    df = build_feature_set(df)
    df = df.reset_index(drop=True)

    edge_df = build_knn_graph(df, k=K_NEIGHBORS)
    G = build_networkx_graph(edge_df)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    emb_df = generate_embeddings(G, dim=EMBEDDING_DIM)
    emb_df.to_csv("data/spatial_embeddings.csv", index=False)

    print(f"Saved {len(emb_df)} embeddings (dim={EMBEDDING_DIM}) to data/spatial_embeddings.csv")
