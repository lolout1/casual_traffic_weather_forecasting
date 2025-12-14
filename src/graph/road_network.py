"""
Road network construction using cuGraph (with NetworkX fallback).
Builds a segment-level graph from aggregated temporal windows.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config.config import (
    FEATURES_DATA_DIR,
    MODEL_DIR,
    GRAPH_DISTANCE_THRESHOLD,
    MIN_EDGE_WEIGHT,
    MAX_EDGES_PER_NODE,
    GRAPH_FEATURES_JSON,
)
from src.utils.gpu_utils import GPU_AVAILABLE, read_parquet, to_pandas

try:
    import cudf
    import cugraph
    CUGRAPH_AVAILABLE = True
except ImportError:
    CUGRAPH_AVAILABLE = False
    import networkx as nx


class RoadNetworkGraph:
    """Build and analyze road network graph from segment-level statistics."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.node_features = None

    def log(self, message):
        if self.verbose:
            print(f"[RoadNetwork] {message}")

    def load_segment_windows(self):
        path = FEATURES_DATA_DIR / "train_windows.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing windowed features at {path}. Run feature engineering first.")
        self.log(f"Loading aggregated windows from {path}")
        df = read_parquet(path)
        return to_pandas(df) if GPU_AVAILABLE else df

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        lat1_rad = np.deg2rad(lat1)
        lat2_rad = np.deg2rad(lat2)
        dlat = lat2_rad - lat1_rad
        dlon = np.deg2rad(lon2 - lon1)
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return 6371.0 * c

    def build_segment_stats(self, df):
        stats = (
            df.groupby('segment_id')[['segment_lat', 'segment_lon', 'incident_count', 'severity_sum']]
            .agg({
                'segment_lat': 'mean',
                'segment_lon': 'mean',
                'incident_count': 'mean',
                'severity_sum': 'mean'
            })
            .reset_index()
        )
        stats = stats.rename(columns={
            'incident_count': 'incident_density',
            'severity_sum': 'severity_density'
        })
        return stats

    def build_edges(self, stats):
        coords = stats[['segment_lat', 'segment_lon']].values
        node_ids = stats['segment_id'].tolist()
        edges = []
        threshold_km = GRAPH_DISTANCE_THRESHOLD * 111  # convert degrees to ~km

        for idx, (lat, lon) in enumerate(coords):
            distances = self.haversine(lat, lon, coords[:, 0], coords[:, 1])
            order = np.argsort(distances)
            connections = 0
            for neighbor_idx in order[1:]:
                if connections >= MAX_EDGES_PER_NODE:
                    break
                dist = distances[neighbor_idx]
                if dist > threshold_km:
                    continue
                src = node_ids[idx]
                dst = node_ids[neighbor_idx]
                weight = max(MIN_EDGE_WEIGHT, 1 / (dist + 0.05))
                edges.append((src, dst, weight, dist))
                connections += 1

        self.log(f"Constructed {len(edges):,} candidate edges")
        return edges

    def compute_with_cugraph(self, stats, edges):
        edge_df = cudf.DataFrame(edges, columns=['src', 'dst', 'weight', 'distance'])
        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(edge_df, source='src', destination='dst', edge_attr='weight', renumber=False)

        pagerank = cugraph.pagerank(G)
        betweenness = cugraph.betweenness_centrality(G)
        clustering = cugraph.clustering_coefficient(G)
        degree = cugraph.degree(G)

        features = cudf.merge(pagerank, betweenness, on='vertex')
        features = features.merge(clustering, on='vertex', how='left')
        features = features.merge(degree, on='vertex', how='left')
        features = features.rename(columns={
            'vertex': 'segment_id',
            'pagerank': 'pagerank',
            'betweenness_centrality': 'betweenness_centrality',
            'clustering_coefficient': 'clustering_coefficient',
            'degree': 'degree'
        })

        stats_cudf = cudf.from_pandas(stats)
        features = stats_cudf.merge(features, on='segment_id', how='left')
        fill_cols = ['pagerank', 'betweenness_centrality', 'clustering_coefficient', 'degree']
        for col in fill_cols:
            features[col] = features[col].fillna(0)
        features['weighted_degree'] = features['degree'] * features['incident_density']
        return features.to_pandas()

    def compute_with_networkx(self, stats, edges):
        G = nx.Graph()
        for _, row in stats.iterrows():
            G.add_node(row['segment_id'], lat=row['segment_lat'], lon=row['segment_lon'])
        for src, dst, weight, dist in edges:
            G.add_edge(src, dst, weight=weight, distance=dist)

        pagerank = nx.pagerank(G, weight='weight')
        degree_centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, weight='weight', k=min(500, len(G)))
        clustering = nx.clustering(G, weight='weight')
        features = []
        for node in G.nodes():
            features.append({
                'segment_id': node,
                'pagerank': pagerank.get(node, 0.0),
                'degree_centrality': degree_centrality.get(node, 0.0),
                'betweenness_centrality': betweenness.get(node, 0.0),
                'clustering_coefficient': clustering.get(node, 0.0),
                'degree': G.degree(node),
                'weighted_degree': sum(d.get('weight', 1.0) for _, _, d in G.edges(node, data=True))
            })
        feature_df = pd.DataFrame(features)
        feature_df = stats.merge(feature_df, on='segment_id', how='left')
        feature_df[['pagerank', 'degree_centrality', 'betweenness_centrality',
                    'clustering_coefficient', 'degree', 'weighted_degree']] = \
            feature_df[['pagerank', 'degree_centrality', 'betweenness_centrality',
                        'clustering_coefficient', 'degree', 'weighted_degree']].fillna(0.0)
        return feature_df

    def save_features(self, feature_df):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        feature_dict = feature_df.set_index('segment_id').to_dict(orient='index')
        with open(GRAPH_FEATURES_JSON, 'w') as f:
            json.dump(feature_dict, f)
        self.log(f"✓ Saved graph features for {len(feature_dict):,} segments to {GRAPH_FEATURES_JSON}")

    def run(self):
        df = self.load_segment_windows()
        stats = self.build_segment_stats(df)
        edges = self.build_edges(stats)
        if CUGRAPH_AVAILABLE:
            self.log("Computing graph features with cuGraph")
            features = self.compute_with_cugraph(stats, edges)
        else:
            self.log("Computing graph features with NetworkX fallback")
            features = self.compute_with_networkx(stats, edges)
        self.save_features(features)


def main():
    print("\n" + "📊 " * 20)
    print("  AUSTIN SENTINEL - ROAD NETWORK GRAPH")
    print("📊 " * 20 + "\n")
    builder = RoadNetworkGraph(verbose=True)
    builder.run()


if __name__ == "__main__":
    main()
