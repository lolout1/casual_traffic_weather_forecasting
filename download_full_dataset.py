#!/usr/bin/env python3
"""
Download the complete Austin traffic dataset and partition into train/val/test
"""

import requests
import json
import os
from datetime import datetime
import random

def download_full_dataset():
    """Download all available traffic incident data"""
    print('='*70)
    print('  DOWNLOADING FULL AUSTIN TRAFFIC DATASET')
    print('='*70)

    # The API has a limit per request, so we'll need to paginate
    total_records = []
    limit = 50000
    offset = 0
    max_iterations = 10  # ~500k records max

    while True:
        url = f'https://data.austintexas.gov/resource/dx9v-zd7x.json?%24limit={limit}&%24offset={offset}&%24order=published_date%20DESC'

        print(f'\nFetching batch {offset//limit + 1} (offset={offset})...')

        try:
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            data = response.json()

            if not data or len(data) == 0:
                print('  No more records - download complete!')
                break

            total_records.extend(data)
            print(f'  ✓ Got {len(data):,} records')
            print(f'  Total so far: {len(total_records):,}')

            if len(data) < limit:
                print('  Received less than limit - download complete!')
                break

            offset += limit

            if offset >= limit * max_iterations:
                print(f'\n  Reached max iterations ({max_iterations})')
                break

        except Exception as e:
            print(f'  ✗ Error: {e}')
            break

    print('\n' + '='*70)
    print(f'DOWNLOAD COMPLETE: {len(total_records):,} total records')
    print('='*70)

    return total_records


def partition_dataset(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Partition dataset into train/val/test splits

    Uses temporal split (not random) to avoid data leakage:
    - Train: oldest data
    - Val: middle period
    - Test: most recent data
    """

    print('\n' + '='*70)
    print('  PARTITIONING DATASET (TEMPORAL SPLIT)')
    print('='*70)

    # Sort by date
    print('\nSorting by published_date...')
    sorted_data = sorted(data, key=lambda x: x.get('published_date', ''))

    # Calculate split indices
    n = len(sorted_data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split data
    train_data = sorted_data[:train_end]
    val_data = sorted_data[train_end:val_end]
    test_data = sorted_data[val_end:]

    # Show split info
    print(f'\nDataset sizes:')
    print(f'  Train: {len(train_data):,} records ({len(train_data)/n*100:.1f}%)')
    print(f'  Val:   {len(val_data):,} records ({len(val_data)/n*100:.1f}%)')
    print(f'  Test:  {len(test_data):,} records ({len(test_data)/n*100:.1f}%)')
    print(f'  Total: {n:,} records')

    # Show date ranges for each split
    def get_date_range(dataset):
        dates = [r['published_date'] for r in dataset if 'published_date' in r]
        if dates:
            return min(dates), max(dates)
        return None, None

    print(f'\nDate ranges:')
    train_start, train_end_date = get_date_range(train_data)
    val_start, val_end_date = get_date_range(val_data)
    test_start, test_end_date = get_date_range(test_data)

    print(f'  Train: {train_start} to {train_end_date}')
    print(f'  Val:   {val_start} to {val_end_date}')
    print(f'  Test:  {test_start} to {test_end_date}')

    return train_data, val_data, test_data


def save_partitions(train_data, val_data, test_data):
    """Save partitioned datasets to files"""

    print('\n' + '='*70)
    print('  SAVING PARTITIONED DATASETS')
    print('='*70)

    datasets = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for name, data in datasets.items():
        filename = f'austin_traffic_{name}.json'
        print(f'\nSaving {name} set...')

        with open(filename, 'w') as f:
            json.dump(data, f)

        file_size = os.path.getsize(filename) / 1024 / 1024
        print(f'  ✓ {filename} ({file_size:.1f} MB, {len(data):,} records)')

    # Also save full dataset
    all_data = train_data + val_data + test_data
    filename = 'austin_traffic_full.json'
    print(f'\nSaving full dataset...')
    with open(filename, 'w') as f:
        json.dump(all_data, f)

    file_size = os.path.getsize(filename) / 1024 / 1024
    print(f'  ✓ {filename} ({file_size:.1f} MB, {len(all_data):,} records)')


def main():
    """Main function"""

    # Download full dataset
    print('\nStep 1: Downloading dataset...')
    data = download_full_dataset()

    if not data:
        print('✗ No data downloaded!')
        return

    # Partition into train/val/test
    print('\nStep 2: Partitioning dataset...')
    train_data, val_data, test_data = partition_dataset(
        data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Save partitions
    print('\nStep 3: Saving partitions...')
    save_partitions(train_data, val_data, test_data)

    print('\n' + '='*70)
    print('  ✓ COMPLETE - Dataset downloaded and partitioned!')
    print('='*70)
    print('\nFiles created:')
    print('  - austin_traffic_full.json  (complete dataset)')
    print('  - austin_traffic_train.json (70% - training)')
    print('  - austin_traffic_val.json   (15% - validation)')
    print('  - austin_traffic_test.json  (15% - testing)')
    print('\nNext steps:')
    print('  1. Load data with pandas/cuDF')
    print('  2. Feature engineering')
    print('  3. Train models on train set')
    print('  4. Tune on val set')
    print('  5. Final evaluation on test set')
    print()


if __name__ == '__main__':
    main()
