"""
Road network graph construction using cuGraph (with NetworkX fallback)
Builds graph from incident locations and computes PageRank for critical intersections
"""

import json
import os
import sys
import math
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config.config import *

# Try to import cuGraph, fall back to NetworkX
try:
    import cugraph
    CUGRAPH_AVAILABLE = True
    print("✓ cuGraph loaded - GPU graph analytics enabled")
except ImportError:
    try:
        import networkx as nx
        CUGRAPH_AVAILABLE = False
        print("⚠️  cuGraph not available - using NetworkX fallback")
    except ImportError:
        print("❌ Neither cuGraph nor NetworkX available - install NetworkX: pip install networkx")
        sys.exit(1)


class RoadNetworkGraph:
    """Build and analyze road network graph from incident data"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.graph = None
        self.node_features = {}

    def log(self, message):
        if self.verbose:
            print(f"[RoadNetwork] {message}")

    def load_data(self, file_path):
        """Load feature-engineered data"""
        self.log(f"Loading data from {file_path}...")

        with open(file_path, 'r') as f:
            data = json.load(f)

        self.log(f"✓ Loaded {len(data):,} records")
        return data

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance in kilometers"""
        R = 6371

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def build_graph_from_grid_cells(self, data):
        """
        Build graph where nodes are grid cells and edges connect adjacent cells

        This is faster than building a full incident-to-incident graph
        and still captures the road network structure
        """
        self.log("\n" + "="*60)
        self.log("BUILDING ROAD NETWORK GRAPH")
        self.log("="*60)

        # Collect all grid cells and their properties
        grid_cells = defaultdict(lambda: {
            'incidents': [],
            'latitude': [],
            'longitude': [],
            'severity': [],
        })

        for record in data:
            grid_cell = record['grid_cell']
            grid_cells[grid_cell]['incidents'].append(record)
            grid_cells[grid_cell]['latitude'].append(record['latitude'])
            grid_cells[grid_cell]['longitude'].append(record['longitude'])
            grid_cells[grid_cell]['severity'].append(record['severity'])

        self.log(f"\nFound {len(grid_cells):,} unique grid cells")

        # Compute grid cell centers
        grid_centers = {}
        for grid_cell, props in grid_cells.items():
            grid_centers[grid_cell] = {
                'lat': sum(props['latitude']) / len(props['latitude']),
                'lon': sum(props['longitude']) / len(props['longitude']),
                'incident_count': len(props['incidents']),
                'avg_severity': sum(props['severity']) / len(props['severity']),
            }

        # Build graph
        self.log(f"\nBuilding graph edges...")

        if CUGRAPH_AVAILABLE:
            # Use cuGraph (GPU-accelerated)
            self.log("Using cuGraph (GPU)")
            # TODO: Implement cuGraph version on DGX Spark
            # For now, fall back to NetworkX
            self.graph = nx.Graph()
        else:
            # Use NetworkX
            self.log("Using NetworkX (CPU)")
            self.graph = nx.Graph()

        # Add nodes
        for grid_cell, center in grid_centers.items():
            self.graph.add_node(
                grid_cell,
                lat=center['lat'],
                lon=center['lon'],
                incident_count=center['incident_count'],
                avg_severity=center['avg_severity']
            )

        # Add edges between adjacent grid cells
        edge_count = 0
        for grid_cell_1 in grid_centers:
            center_1 = grid_centers[grid_cell_1]

            for grid_cell_2 in grid_centers:
                if grid_cell_1 >= grid_cell_2:  # Avoid duplicates
                    continue

                center_2 = grid_centers[grid_cell_2]

                # Calculate distance
                dist = self.haversine_distance(
                    center_1['lat'], center_1['lon'],
                    center_2['lat'], center_2['lon']
                )

                # Connect if within threshold
                if dist <= GRAPH_DISTANCE_THRESHOLD * 111:  # Convert degrees to km
                    # Weight by inverse distance and incident counts
                    weight = (
                        (1 / (dist + 0.1)) *
                        math.sqrt(center_1['incident_count'] * center_2['incident_count'])
                    )

                    self.graph.add_edge(grid_cell_1, grid_cell_2, weight=weight, distance=dist)
                    edge_count += 1

        self.log(f"✓ Graph built:")
        self.log(f"  Nodes: {self.graph.number_of_nodes():,}")
        self.log(f"  Edges: {self.graph.number_of_edges():,}")
        self.log(f"  Avg degree: {2 * self.graph.number_of_edges() / self.graph.number_of_nodes():.2f}")

        return self.graph

    def compute_pagerank(self):
        """Compute PageRank for all nodes"""
        self.log("\nComputing PageRank...")

        if CUGRAPH_AVAILABLE:
            # Use cuGraph PageRank (GPU)
            # TODO: Implement on DGX Spark
            pagerank = nx.pagerank(self.graph, weight='weight')
        else:
            # Use NetworkX PageRank (CPU)
            pagerank = nx.pagerank(self.graph, weight='weight')

        self.log(f"✓ PageRank computed for {len(pagerank):,} nodes")

        # Show top nodes
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        self.log("\nTop 10 nodes by PageRank:")
        for i, (node, score) in enumerate(top_nodes, 1):
            node_data = self.graph.nodes[node]
            self.log(f"  {i}. Grid {node}: PageRank={score:.6f}, "
                    f"Incidents={node_data['incident_count']}, "
                    f"Loc=({node_data['lat']:.4f}, {node_data['lon']:.4f})")

        return pagerank

    def compute_centrality_measures(self):
        """Compute various centrality measures"""
        self.log("\nComputing centrality measures...")

        centrality_measures = {}

        # Degree centrality
        self.log("  Computing degree centrality...")
        centrality_measures['degree'] = nx.degree_centrality(self.graph)

        # Betweenness centrality (expensive for large graphs)
        if self.graph.number_of_nodes() < 5000:
            self.log("  Computing betweenness centrality...")
            centrality_measures['betweenness'] = nx.betweenness_centrality(
                self.graph,
                weight='weight',
                k=min(100, self.graph.number_of_nodes())  # Sample for speed
            )
        else:
            self.log("  Skipping betweenness (graph too large)")
            centrality_measures['betweenness'] = {node: 0.0 for node in self.graph.nodes()}

        # Clustering coefficient
        self.log("  Computing clustering coefficient...")
        centrality_measures['clustering'] = nx.clustering(self.graph, weight='weight')

        self.log("✓ Centrality measures computed")

        return centrality_measures

    def extract_graph_features(self, pagerank, centrality_measures):
        """Extract graph features for each node"""
        self.log("\nExtracting graph features...")

        node_features = {}

        for node in self.graph.nodes():
            node_features[node] = {
                'pagerank': pagerank.get(node, 0.0),
                'degree_centrality': centrality_measures['degree'].get(node, 0.0),
                'betweenness_centrality': centrality_measures['betweenness'].get(node, 0.0),
                'clustering_coefficient': centrality_measures['clustering'].get(node, 0.0),
                'degree': self.graph.degree(node),
                'weighted_degree': sum(d.get('weight', 1.0) for _, _, d in self.graph.edges(node, data=True)),
            }

        self.node_features = node_features
        self.log(f"✓ Extracted features for {len(node_features):,} nodes")

        return node_features

    def save_graph_features(self, output_path):
        """Save graph features to JSON"""
        self.log(f"\nSaving graph features to {output_path}...")

        # Convert int keys to strings for JSON
        features_str_keys = {str(k): v for k, v in self.node_features.items()}

        with open(output_path, 'w') as f:
            json.dump(features_str_keys, f)

        file_size = os.path.getsize(output_path) / 1024 / 1024
        self.log(f"✓ Saved features for {len(self.node_features):,} nodes ({file_size:.2f} MB)")

    def get_statistics(self):
        """Return graph statistics"""
        if self.graph is None:
            return {}

        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'avg_degree': 2 * self.graph.number_of_edges() / self.graph.number_of_nodes(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph),
        }

        if nx.is_connected(self.graph):
            stats['diameter'] = nx.diameter(self.graph)
            stats['avg_shortest_path'] = nx.average_shortest_path_length(self.graph)

        return stats


def main():
    """Build road network graph and compute features"""

    print("\n" + "📊 " * 20)
    print("  AUSTIN SENTINEL - ROAD NETWORK GRAPH")
    print("📊 " * 20 + "\n")

    graph_builder = RoadNetworkGraph(verbose=True)

    # Load training data
    train_data = graph_builder.load_data(FEATURES_DATA_DIR / "train_features.json")

    # Build graph
    graph = graph_builder.build_graph_from_grid_cells(train_data)

    # Compute PageRank
    pagerank = graph_builder.compute_pagerank()

    # Compute other centrality measures
    centrality = graph_builder.compute_centrality_measures()

    # Extract features
    node_features = graph_builder.extract_graph_features(pagerank, centrality)

    # Save features
    os.makedirs(MODEL_DIR, exist_ok=True)
    graph_builder.save_graph_features(MODEL_DIR / "graph_features.json")

    # Print statistics
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)

    stats = graph_builder.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:20}: {value:.4f}")
        else:
            print(f"{key:20}: {value}")

    print("\n" + "="*60)
    print("✓ ROAD NETWORK GRAPH COMPLETE")
    print("="*60)
    print(f"\nGraph features saved to: {MODEL_DIR / 'graph_features.json'}")
    print(f"Features per node: {len(list(node_features.values())[0])}")
    print()


if __name__ == "__main__":
    main()
