"""
Feature engineering for traffic incident prediction
Extracts temporal, spatial, and historical features
"""

import json
import os
import sys
from datetime import datetime
from collections import defaultdict
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config.config import *


class FeatureEngineer:
    """Extract features from cleaned traffic incident data"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.grid_stats = None  # Will store historical grid statistics

    def log(self, message):
        if self.verbose:
            print(f"[FeatureEngineer] {message}")

    def load_data(self, file_path):
        """Load cleaned JSON data"""
        self.log(f"Loading data from {file_path}...")

        with open(file_path, 'r') as f:
            data = json.load(f)

        self.log(f"✓ Loaded {len(data):,} records")
        return data

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two points using Haversine formula

        Returns distance in kilometers
        """
        R = 6371  # Earth's radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def extract_temporal_features(self, record):
        """Extract time-based features"""

        try:
            dt = datetime.fromisoformat(record['published_date'].replace('Z', '+00:00'))
        except:
            # Fallback for different date formats
            try:
                dt = datetime.strptime(record['published_date'], '%Y-%m-%dT%H:%M:%S.%fZ')
            except:
                dt = datetime.now()  # Last resort

        features = {
            'hour': dt.hour,
            'day_of_week': dt.weekday(),  # 0 = Monday
            'month': dt.month,
            'day_of_month': dt.day,
            'week_of_year': dt.isocalendar()[1],
            'year': dt.year,

            # Binary flags
            'is_weekend': int(dt.weekday() >= 5),
            'is_rush_hour': int((7 <= dt.hour <= 9) or (16 <= dt.hour <= 19)),
            'is_night': int((22 <= dt.hour) or (dt.hour <= 5)),
            'is_morning': int(6 <= dt.hour <= 11),
            'is_afternoon': int(12 <= dt.hour <= 17),
            'is_evening': int(18 <= dt.hour <= 21),

            # Season (meteorological)
            'season': (dt.month % 12 + 3) // 3,  # 1=winter, 2=spring, 3=summer, 4=fall

            # Cyclic encoding for hour (helps ML models understand circular nature)
            'hour_sin': math.sin(2 * math.pi * dt.hour / 24),
            'hour_cos': math.cos(2 * math.pi * dt.hour / 24),

            # Cyclic encoding for day of week
            'dow_sin': math.sin(2 * math.pi * dt.weekday() / 7),
            'dow_cos': math.cos(2 * math.pi * dt.weekday() / 7),

            # Timestamp for sorting/filtering
            'timestamp': dt.timestamp(),
        }

        return features

    def extract_spatial_features(self, record):
        """Extract location-based features"""

        lat = record['latitude']
        lon = record['longitude']

        # Grid cell assignment
        grid_lat = int((lat - AUSTIN_LAT_MIN) / GRID_SIZE_LAT)
        grid_lon = int((lon - AUSTIN_LON_MIN) / GRID_SIZE_LON)
        grid_cell = grid_lat * 1000 + grid_lon

        # Distance from Austin center (downtown)
        dist_from_center = self.haversine_distance(
            lat, lon,
            AUSTIN_CENTER_LAT, AUSTIN_CENTER_LON
        )

        # Quadrant (relative to center)
        north_of_center = int(lat > AUSTIN_CENTER_LAT)
        east_of_center = int(lon > AUSTIN_CENTER_LON)

        features = {
            'latitude': lat,
            'longitude': lon,
            'grid_lat': grid_lat,
            'grid_lon': grid_lon,
            'grid_cell': grid_cell,
            'distance_from_center': dist_from_center,
            'north_of_center': north_of_center,
            'east_of_center': east_of_center,

            # Quadrant encoding
            'quadrant': north_of_center * 2 + east_of_center,  # 0=SW, 1=SE, 2=NW, 3=NE
        }

        return features

    def compute_grid_statistics(self, data):
        """
        Compute historical statistics for each grid cell

        This creates features like:
        - Average incidents per grid cell
        - Grid cell risk score
        - Most common incident type per cell
        """
        self.log("\nComputing grid statistics from training data...")

        grid_incidents = defaultdict(list)
        grid_severity = defaultdict(list)

        for record in data:
            # Get grid cell
            lat = record['latitude']
            lon = record['longitude']
            grid_lat = int((lat - AUSTIN_LAT_MIN) / GRID_SIZE_LAT)
            grid_lon = int((lon - AUSTIN_LON_MIN) / GRID_SIZE_LON)
            grid_cell = grid_lat * 1000 + grid_lon

            grid_incidents[grid_cell].append(record)
            grid_severity[grid_cell].append(record.get('severity', 0.5))

        # Compute statistics
        grid_stats = {}
        for grid_cell, incidents in grid_incidents.items():
            grid_stats[grid_cell] = {
                'incident_count': len(incidents),
                'avg_severity': sum(grid_severity[grid_cell]) / len(grid_severity[grid_cell]),
                'crash_ratio': sum(1 for i in incidents if i.get('is_crash', 0)) / len(incidents),
                'hazard_ratio': sum(1 for i in incidents if i.get('is_hazard', 0)) / len(incidents),
            }

        # Normalize incident count to risk score (0-1)
        max_incidents = max(stats['incident_count'] for stats in grid_stats.values())
        for grid_cell in grid_stats:
            grid_stats[grid_cell]['risk_score'] = (
                grid_stats[grid_cell]['incident_count'] / max_incidents
            )

        self.grid_stats = grid_stats
        self.log(f"✓ Computed statistics for {len(grid_stats):,} grid cells")

        return grid_stats

    def extract_grid_features(self, record):
        """Extract historical grid-based features"""

        if self.grid_stats is None:
            return {
                'grid_incident_count': 0,
                'grid_avg_severity': 0.5,
                'grid_risk_score': 0.0,
                'grid_crash_ratio': 0.0,
                'grid_hazard_ratio': 0.0,
            }

        # Get grid cell
        lat = record['latitude']
        lon = record['longitude']
        grid_lat = int((lat - AUSTIN_LAT_MIN) / GRID_SIZE_LAT)
        grid_lon = int((lon - AUSTIN_LON_MIN) / GRID_SIZE_LON)
        grid_cell = grid_lat * 1000 + grid_lon

        # Get grid statistics
        stats = self.grid_stats.get(grid_cell, {
            'incident_count': 0,
            'avg_severity': 0.5,
            'risk_score': 0.0,
            'crash_ratio': 0.0,
            'hazard_ratio': 0.0,
        })

        return {
            'grid_incident_count': stats['incident_count'],
            'grid_avg_severity': stats['avg_severity'],
            'grid_risk_score': stats['risk_score'],
            'grid_crash_ratio': stats['crash_ratio'],
            'grid_hazard_ratio': stats['hazard_ratio'],
        }

    def engineer_features(self, data, compute_grid_stats=False):
        """
        Apply feature engineering to all records

        Args:
            data: List of cleaned incident records
            compute_grid_stats: Whether to compute grid statistics (only for training data)
        """
        self.log("\n" + "="*60)
        self.log("FEATURE ENGINEERING")
        self.log("="*60)

        # Compute grid statistics if requested (training data only)
        if compute_grid_stats:
            self.compute_grid_statistics(data)

        self.log(f"\nEngineering features for {len(data):,} records...")

        enhanced_data = []
        for i, record in enumerate(data):
            if i % 50000 == 0 and i > 0:
                self.log(f"  Processed {i:,} records...")

            # Create enhanced record with all original fields
            enhanced_record = record.copy()

            # Add temporal features
            temporal_features = self.extract_temporal_features(record)
            enhanced_record.update(temporal_features)

            # Add spatial features
            spatial_features = self.extract_spatial_features(record)
            enhanced_record.update(spatial_features)

            # Add grid-based features
            grid_features = self.extract_grid_features(record)
            enhanced_record.update(grid_features)

            enhanced_data.append(enhanced_record)

        self.log(f"✓ Feature engineering complete!")
        self.log(f"  Original fields: {len(record.keys())}")
        self.log(f"  Enhanced fields: {len(enhanced_record.keys())}")
        self.log(f"  New features: {len(enhanced_record.keys()) - len(record.keys())}")

        return enhanced_data

    def save_features(self, data, output_path):
        """Save engineered features to JSON"""
        self.log(f"\nSaving features to {output_path}...")

        with open(output_path, 'w') as f:
            json.dump(data, f)

        file_size = os.path.getsize(output_path) / 1024 / 1024
        self.log(f"✓ Saved {len(data):,} records ({file_size:.1f} MB)")

    def save_grid_stats(self, output_path):
        """Save grid statistics for use in validation/test"""
        if self.grid_stats is None:
            self.log("⚠️  No grid statistics to save")
            return

        self.log(f"\nSaving grid statistics to {output_path}...")

        with open(output_path, 'w') as f:
            json.dump(self.grid_stats, f)

        file_size = os.path.getsize(output_path) / 1024 / 1024
        self.log(f"✓ Saved statistics for {len(self.grid_stats):,} grid cells ({file_size:.2f} MB)")

    def load_grid_stats(self, file_path):
        """Load pre-computed grid statistics"""
        self.log(f"Loading grid statistics from {file_path}...")

        with open(file_path, 'r') as f:
            # Convert string keys back to int
            grid_stats_str = json.load(f)
            self.grid_stats = {int(k): v for k, v in grid_stats_str.items()}

        self.log(f"✓ Loaded statistics for {len(self.grid_stats):,} grid cells")


def main():
    """Feature engineering pipeline"""

    print("\n" + "🔧 " * 20)
    print("  AUSTIN SENTINEL - FEATURE ENGINEERING")
    print("🔧 " * 20 + "\n")

    engineer = FeatureEngineer(verbose=True)

    # Ensure output directory exists
    os.makedirs(FEATURES_DATA_DIR, exist_ok=True)

    # Process TRAINING data (compute grid stats)
    print("\n" + "="*60)
    print("Processing TRAINING data")
    print("="*60)

    train_data = engineer.load_data(PROCESSED_DATA_DIR / "train_clean.json")
    train_features = engineer.engineer_features(train_data, compute_grid_stats=True)
    engineer.save_features(
        train_features,
        FEATURES_DATA_DIR / "train_features.json"
    )
    engineer.save_grid_stats(FEATURES_DATA_DIR / "grid_stats.json")

    # Process VALIDATION data (use pre-computed grid stats)
    print("\n" + "="*60)
    print("Processing VALIDATION data")
    print("="*60)

    val_data = engineer.load_data(PROCESSED_DATA_DIR / "val_clean.json")
    val_features = engineer.engineer_features(val_data, compute_grid_stats=False)
    engineer.save_features(
        val_features,
        FEATURES_DATA_DIR / "val_features.json"
    )

    # Process TEST data (use pre-computed grid stats)
    print("\n" + "="*60)
    print("Processing TEST data")
    print("="*60)

    test_data = engineer.load_data(PROCESSED_DATA_DIR / "test_clean.json")
    test_features = engineer.engineer_features(test_data, compute_grid_stats=False)
    engineer.save_features(
        test_features,
        FEATURES_DATA_DIR / "test_features.json"
    )

    print("\n" + "="*60)
    print("✓ FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"\nFeature files saved to: {FEATURES_DATA_DIR}")
    print(f"  - train_features.json ({len(train_features):,} records)")
    print(f"  - val_features.json   ({len(val_features):,} records)")
    print(f"  - test_features.json  ({len(test_features):,} records)")
    print(f"  - grid_stats.json     ({len(engineer.grid_stats):,} grid cells)")

    # Show sample features
    print("\n" + "="*60)
    print("SAMPLE FEATURE SET")
    print("="*60)

    sample = train_features[0]
    print(f"\nFeature count: {len(sample)} features")
    print("\nTemporal features:")
    for key in ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'season']:
        if key in sample:
            print(f"  {key}: {sample[key]}")

    print("\nSpatial features:")
    for key in ['latitude', 'longitude', 'grid_cell', 'distance_from_center', 'quadrant']:
        if key in sample:
            print(f"  {key}: {sample[key]}")

    print("\nGrid-based features:")
    for key in ['grid_incident_count', 'grid_risk_score', 'grid_avg_severity']:
        if key in sample:
            print(f"  {key}: {sample[key]}")

    print()


if __name__ == "__main__":
    main()
