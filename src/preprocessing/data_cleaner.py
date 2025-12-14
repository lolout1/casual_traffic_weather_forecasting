"""
Data cleaning and preprocessing using RAPIDS cuDF (with pandas fallback)
Handles raw traffic incident data and prepares it for feature engineering
"""

from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config.config import *
from src.utils.gpu_utils import (
    DataFrame, Series, read_json, to_datetime,
    GPU_AVAILABLE, get_compute_mode, print_performance_info
)


class DataCleaner:
    """Clean and preprocess traffic incident data on GPU"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.stats = {}

    def log(self, message):
        """Print log message if verbose"""
        if self.verbose:
            print(f"[DataCleaner] {message}")

    def load_data(self, file_path):
        """
        Load JSON data into DataFrame (GPU if available, CPU otherwise)

        Args:
            file_path: Path to JSON file

        Returns:
            DataFrame (cudf.DataFrame or pandas.DataFrame)
        """
        self.log(f"Loading data from {file_path}...")
        self.log(f"Compute mode: {get_compute_mode()}")

        try:
            df = read_json(file_path)
            mode = "GPU" if GPU_AVAILABLE else "CPU"
            self.log(f"✓ Loaded {len(df):,} records on {mode}")
        except Exception as e:
            self.log(f"✗ Error loading data: {e}")
            raise

        self.stats['initial_records'] = len(df)
        return df

    def clean_data(self, df):
        """
        Clean raw traffic incident data

        Steps:
        1. Remove duplicates
        2. Handle missing values
        3. Parse and validate timestamps
        4. Parse and validate coordinates
        5. Normalize incident types
        6. Add severity scores
        """
        self.log("\n" + "="*60)
        self.log("CLEANING DATA")
        self.log("="*60)

        initial_count = len(df)

        # 1. Remove duplicates
        self.log("\n1. Removing duplicates...")
        df = df.drop_duplicates(subset=['traffic_report_id'], keep='first')
        duplicates_removed = initial_count - len(df)
        self.log(f"   Removed {duplicates_removed:,} duplicates")
        self.stats['duplicates_removed'] = duplicates_removed

        # 2. Handle missing values
        self.log("\n2. Handling missing values...")
        missing_before = df.isnull().sum().sum()

        # Remove rows with missing critical fields
        critical_fields = ['published_date', 'latitude', 'longitude']
        for field in critical_fields:
            if field in df.columns:
                missing_count = df[field].isnull().sum()
                if missing_count > 0:
                    self.log(f"   Removing {missing_count:,} rows with missing {field}")
                    df = df[df[field].notnull()]

        # Fill missing non-critical fields
        if 'issue_reported' in df.columns:
            df['issue_reported'] = df['issue_reported'].fillna('UNKNOWN')
        if 'agency' in df.columns:
            df['agency'] = df['agency'].fillna('UNKNOWN')
        if 'address' in df.columns:
            df['address'] = df['address'].fillna('UNKNOWN')

        missing_after = df.isnull().sum().sum()
        self.log(f"   Missing values: {missing_before:,} → {missing_after:,}")
        self.stats['missing_values_handled'] = missing_before - missing_after

        # 3. Parse timestamps
        self.log("\n3. Parsing timestamps...")
        df['published_date'] = to_datetime(df['published_date'], errors='coerce')

        # Remove rows with invalid timestamps
        invalid_timestamps = df['published_date'].isnull().sum()
        if invalid_timestamps > 0:
            self.log(f"   Removing {invalid_timestamps:,} rows with invalid timestamps")
            df = df[df['published_date'].notnull()]

        # Add status timestamp if available
        if 'traffic_report_status_date_time' in df.columns:
            df['status_date'] = to_datetime(
                df['traffic_report_status_date_time'],
                errors='coerce'
            )

        self.stats['invalid_timestamps_removed'] = invalid_timestamps

        # 4. Parse and validate coordinates
        self.log("\n4. Validating coordinates...")

        # Convert to float if they're strings
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)

        # Remove invalid coordinates
        before_coord_filter = len(df)

        # Austin area bounds (with some buffer)
        df = df[
            (df['latitude'] >= AUSTIN_LAT_MIN) &
            (df['latitude'] <= AUSTIN_LAT_MAX) &
            (df['longitude'] >= AUSTIN_LON_MIN) &
            (df['longitude'] <= AUSTIN_LON_MAX)
        ]

        invalid_coords = before_coord_filter - len(df)
        self.log(f"   Removed {invalid_coords:,} rows with invalid coordinates")
        self.stats['invalid_coords_removed'] = invalid_coords

        # 5. Normalize incident types
        self.log("\n5. Normalizing incident types...")
        df['issue_reported_original'] = df['issue_reported']
        df['issue_reported'] = df['issue_reported'].str.strip().str.upper()

        unique_types = df['issue_reported'].nunique()
        self.log(f"   Found {unique_types} unique incident types")
        self.stats['unique_incident_types'] = unique_types

        # 6. Add severity scores
        self.log("\n6. Adding severity scores...")

        # Map incident types to severity scores
        df['severity'] = df['issue_reported'].map(INCIDENT_SEVERITY)
        df['severity'] = df['severity'].fillna(0.5)  # Default severity for unknown types

        self.log(f"   Severity range: {df['severity'].min():.2f} - {df['severity'].max():.2f}")

        # 7. Add binary flags
        self.log("\n7. Adding binary flags...")

        df['is_crash'] = df['issue_reported'].str.contains('CRASH|COLLISION|COLLISN').fillna(False).astype(int)
        df['is_hazard'] = df['issue_reported'].str.contains('HAZARD|DEBRIS').fillna(False).astype(int)
        df['is_stall'] = df['issue_reported'].str.contains('STALL').fillna(False).astype(int)

        # Final statistics
        final_count = len(df)
        records_removed = initial_count - final_count
        retention_rate = (final_count / initial_count) * 100

        self.log("\n" + "="*60)
        self.log("CLEANING SUMMARY")
        self.log("="*60)
        self.log(f"Initial records:    {initial_count:,}")
        self.log(f"Final records:      {final_count:,}")
        self.log(f"Records removed:    {records_removed:,}")
        self.log(f"Retention rate:     {retention_rate:.2f}%")
        self.log("="*60)

        self.stats['final_records'] = final_count
        self.stats['retention_rate'] = retention_rate

        return df

    def save_processed_data(self, df, output_path, format='parquet'):
        """
        Save processed data to disk

        Args:
            df: cuDF DataFrame
            output_path: Path to save file
            format: 'parquet' or 'csv'
        """
        self.log(f"\nSaving processed data to {output_path}...")

        if format == 'parquet':
            df.to_parquet(output_path)
        elif format == 'csv':
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Get file size
        file_size = os.path.getsize(output_path) / 1024 / 1024
        self.log(f"✓ Saved {len(df):,} records ({file_size:.1f} MB)")

    def get_stats(self):
        """Return cleaning statistics"""
        return self.stats


def main():
    """Test data cleaning pipeline"""

    print("\n" + "🚀 " * 20)
    print("  AUSTIN SENTINEL - DATA PREPROCESSING")
    print("🚀 " * 20 + "\n")

    cleaner = DataCleaner(verbose=True)

    # Process training data
    print("\n" + "="*60)
    print("Processing TRAINING data")
    print("="*60)

    train_df = cleaner.load_data(TRAIN_FILE)
    train_df_clean = cleaner.clean_data(train_df)
    cleaner.save_processed_data(
        train_df_clean,
        PROCESSED_DATA_DIR / "train_clean.parquet"
    )

    # Process validation data
    print("\n" + "="*60)
    print("Processing VALIDATION data")
    print("="*60)

    val_df = cleaner.load_data(VAL_FILE)
    val_df_clean = cleaner.clean_data(val_df)
    cleaner.save_processed_data(
        val_df_clean,
        PROCESSED_DATA_DIR / "val_clean.parquet"
    )

    # Process test data
    print("\n" + "="*60)
    print("Processing TEST data")
    print("="*60)

    test_df = cleaner.load_data(TEST_FILE)
    test_df_clean = cleaner.clean_data(test_df)
    cleaner.save_processed_data(
        test_df_clean,
        PROCESSED_DATA_DIR / "test_clean.parquet"
    )

    print("\n" + "="*60)
    print("✓ DATA PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nProcessed files saved to: {PROCESSED_DATA_DIR}")
    print(f"  - train_clean.parquet ({len(train_df_clean):,} records)")
    print(f"  - val_clean.parquet   ({len(val_df_clean):,} records)")
    print(f"  - test_clean.parquet  ({len(test_df_clean):,} records)")
    print()


if __name__ == "__main__":
    main()
