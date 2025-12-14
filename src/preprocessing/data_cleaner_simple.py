"""
Simplified data cleaning using only JSON and standard library
Production version will use RAPIDS cuDF on DGX Spark
"""

import json
import os
import sys
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config.config import *


class SimpleDataCleaner:
    """Clean traffic incident data using standard library"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.stats = {}

    def log(self, message):
        if self.verbose:
            print(f"[DataCleaner] {message}")

    def load_data(self, file_path):
        """Load JSON data"""
        self.log(f"Loading data from {file_path}...")

        with open(file_path, 'r') as f:
            data = json.load(f)

        self.log(f"✓ Loaded {len(data):,} records")
        self.stats['initial_records'] = len(data)
        return data

    def clean_data(self, data):
        """Clean raw traffic incident data"""
        self.log("\n" + "="*60)
        self.log("CLEANING DATA")
        self.log("="*60)

        initial_count = len(data)
        cleaned_data = []
        seen_ids = set()

        for record in data:
            # 1. Remove duplicates
            report_id = record.get('traffic_report_id')
            if report_id in seen_ids:
                continue
            seen_ids.add(report_id)

            # 2. Check critical fields
            if not all(key in record for key in ['published_date', 'latitude', 'longitude']):
                continue

            # 3. Validate coordinates
            try:
                lat = float(record['latitude'])
                lon = float(record['longitude'])

                if not (AUSTIN_LAT_MIN <= lat <= AUSTIN_LAT_MAX and
                        AUSTIN_LON_MIN <= lon <= AUSTIN_LON_MAX):
                    continue

                record['latitude'] = lat
                record['longitude'] = lon

            except (ValueError, TypeError):
                continue

            # 4. Parse timestamp
            try:
                record['published_date'] = record['published_date']
                # datetime.fromisoformat(record['published_date'].replace('Z', '+00:00'))
            except:
                continue

            # 5. Normalize incident type
            issue = record.get('issue_reported', 'UNKNOWN').strip().upper()
            record['issue_reported'] = issue
            record['issue_reported_original'] = record.get('issue_reported', 'UNKNOWN')

            # 6. Add severity
            record['severity'] = INCIDENT_SEVERITY.get(issue, 0.5)

            # 7. Add binary flags
            record['is_crash'] = int('CRASH' in issue or 'COLLISION' in issue or 'COLLISN' in issue)
            record['is_hazard'] = int('HAZARD' in issue or 'DEBRIS' in issue)
            record['is_stall'] = int('STALL' in issue)

            cleaned_data.append(record)

        final_count = len(cleaned_data)
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

        return cleaned_data

    def save_processed_data(self, data, output_path):
        """Save processed data as JSON"""
        self.log(f"\nSaving processed data to {output_path}...")

        with open(output_path, 'w') as f:
            json.dump(data, f)

        file_size = os.path.getsize(output_path) / 1024 / 1024
        self.log(f"✓ Saved {len(data):,} records ({file_size:.1f} MB)")


def main():
    """Process all datasets"""

    print("\n" + "🚀 " * 20)
    print("  AUSTIN SENTINEL - DATA PREPROCESSING")
    print("🚀 " * 20 + "\n")

    cleaner = SimpleDataCleaner(verbose=True)

    # Ensure output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Process training data
    print("\n" + "="*60)
    print("Processing TRAINING data")
    print("="*60)

    train_data = cleaner.load_data(TRAIN_FILE)
    train_clean = cleaner.clean_data(train_data)
    cleaner.save_processed_data(
        train_clean,
        PROCESSED_DATA_DIR / "train_clean.json"
    )

    # Process validation data
    print("\n" + "="*60)
    print("Processing VALIDATION data")
    print("="*60)

    val_data = cleaner.load_data(VAL_FILE)
    val_clean = cleaner.clean_data(val_data)
    cleaner.save_processed_data(
        val_clean,
        PROCESSED_DATA_DIR / "val_clean.json"
    )

    # Process test data
    print("\n" + "="*60)
    print("Processing TEST data")
    print("="*60)

    test_data = cleaner.load_data(TEST_FILE)
    test_clean = cleaner.clean_data(test_data)
    cleaner.save_processed_data(
        test_clean,
        PROCESSED_DATA_DIR / "test_clean.json"
    )

    print("\n" + "="*60)
    print("✓ DATA PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nProcessed files saved to: {PROCESSED_DATA_DIR}")
    print(f"  - train_clean.json ({len(train_clean):,} records)")
    print(f"  - val_clean.json   ({len(val_clean):,} records)")
    print(f"  - test_clean.json  ({len(test_clean):,} records)")

    # Print statistics
    total_records = len(train_clean) + len(val_clean) + len(test_clean)
    print(f"\nTotal processed:  {total_records:,} records")

    # Show incident type distribution
    print("\n" + "="*60)
    print("INCIDENT TYPE DISTRIBUTION")
    print("="*60)

    all_types = [r['issue_reported'] for r in train_clean]
    type_counts = Counter(all_types)

    print(f"\nTop 15 incident types:")
    for incident_type, count in type_counts.most_common(15):
        pct = count / len(train_clean) * 100
        print(f"  {incident_type[:35]:35} {count:6,} ({pct:5.1f}%)")

    print()


if __name__ == "__main__":
    main()
