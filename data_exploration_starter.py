#!/usr/bin/env python3
"""
Austin Sentinel - Data Exploration Starter
Run this first to understand the data structure and verify API access.

Usage:
    python data_exploration_starter.py
"""

import requests
import json
from datetime import datetime, timedelta
from collections import Counter
import sys

# Configuration
AUSTIN_API_BASE = "https://data.austintexas.gov/resource/dx9v-zd7x.json"
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "(AustinSentinel, hackathon@example.com)"

def fetch_traffic_sample(limit=100):
    """Fetch sample traffic incident data"""
    print("\n" + "="*60)
    print("  AUSTIN TRAFFIC INCIDENT DATA EXPLORATION")
    print("="*60)

    url = f"{AUSTIN_API_BASE}?$limit={limit}&$order=published_date DESC"

    print(f"\nFetching {limit} most recent incidents...")
    print(f"URL: {url}\n")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        print(f"✓ Successfully fetched {len(data)} records")

        return data
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching data: {e}")
        return None

def analyze_traffic_data(data):
    """Analyze traffic incident data structure"""
    if not data:
        return

    print("\n" + "-"*60)
    print("DATA SCHEMA ANALYSIS")
    print("-"*60)

    # Get all unique fields
    all_fields = set()
    for record in data:
        all_fields.update(record.keys())

    print(f"\nFound {len(all_fields)} fields:")
    for field in sorted(all_fields):
        # Get sample value
        sample = None
        for record in data:
            if field in record and record[field]:
                sample = record[field]
                break

        sample_str = str(sample)[:50] + "..." if len(str(sample)) > 50 else str(sample)
        print(f"  • {field}: {sample_str}")

    # Show sample record
    print("\n" + "-"*60)
    print("SAMPLE RECORD")
    print("-"*60)
    print(json.dumps(data[0], indent=2, default=str))

    # Analyze incident types
    print("\n" + "-"*60)
    print("INCIDENT TYPE DISTRIBUTION")
    print("-"*60)

    issue_counts = Counter(r.get("issue_reported", "UNKNOWN") for r in data)
    for issue, count in issue_counts.most_common(15):
        pct = count / len(data) * 100
        bar = "█" * int(pct / 2)
        print(f"  {issue[:35]:35} {count:4} ({pct:5.1f}%) {bar}")

    # Analyze time distribution
    print("\n" + "-"*60)
    print("HOURLY DISTRIBUTION")
    print("-"*60)

    hours = []
    for r in data:
        if "published_date" in r and r["published_date"]:
            try:
                dt = datetime.fromisoformat(r["published_date"].replace("Z", "+00:00"))
                hours.append(dt.hour)
            except:
                pass

    if hours:
        hour_counts = Counter(hours)
        for hour in range(24):
            count = hour_counts.get(hour, 0)
            pct = count / len(hours) * 100
            bar = "█" * int(pct)
            print(f"  {hour:02d}:00  {count:3} ({pct:5.1f}%) {bar}")

    # Geographic bounds
    print("\n" + "-"*60)
    print("GEOGRAPHIC BOUNDS")
    print("-"*60)

    lats = [float(r["latitude"]) for r in data if r.get("latitude")]
    lons = [float(r["longitude"]) for r in data if r.get("longitude")]

    if lats and lons:
        print(f"  Latitude:  {min(lats):.4f} to {max(lats):.4f}")
        print(f"  Longitude: {min(lons):.4f} to {max(lons):.4f}")
        print(f"  Center:    ({sum(lats)/len(lats):.4f}, {sum(lons)/len(lons):.4f})")

def fetch_weather_sample():
    """Fetch sample weather data from NWS API"""
    print("\n" + "="*60)
    print("  NATIONAL WEATHER SERVICE API EXPLORATION")
    print("="*60)

    headers = {"User-Agent": USER_AGENT}

    # Get current conditions
    print("\nFetching current conditions from Austin-Bergstrom Airport (KAUS)...")

    try:
        url = f"{NWS_API_BASE}/stations/KAUS/observations/latest"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        print("✓ Successfully fetched current weather")

        props = data.get("properties", {})

        print("\n" + "-"*60)
        print("CURRENT CONDITIONS")
        print("-"*60)

        def get_value(measurement):
            if isinstance(measurement, dict):
                return measurement.get("value")
            return measurement

        print(f"  Timestamp:    {props.get('timestamp', 'N/A')}")
        print(f"  Temperature:  {get_value(props.get('temperature'))} °C")
        print(f"  Humidity:     {get_value(props.get('relativeHumidity'))} %")
        print(f"  Wind Speed:   {get_value(props.get('windSpeed'))} km/h")
        print(f"  Wind Dir:     {get_value(props.get('windDirection'))} °")
        print(f"  Visibility:   {get_value(props.get('visibility'))} m")
        print(f"  Pressure:     {get_value(props.get('barometricPressure'))} Pa")
        print(f"  Description:  {props.get('textDescription', 'N/A')}")

        # Show available fields
        print("\n" + "-"*60)
        print("AVAILABLE WEATHER FIELDS")
        print("-"*60)
        for key in sorted(props.keys()):
            value = get_value(props.get(key))
            if value is not None:
                print(f"  • {key}: {value}")

        return data

    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching weather: {e}")
        return None

def get_dataset_stats():
    """Get overall dataset statistics"""
    print("\n" + "="*60)
    print("  DATASET STATISTICS")
    print("="*60)

    # Get total count
    url = f"{AUSTIN_API_BASE}?$select=count(*)"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        total_count = data[0].get("count", "Unknown")
        if isinstance(total_count, (int, float)):
            print(f"\nTotal records in dataset: {int(total_count):,}")
        else:
            print(f"\nTotal records in dataset: {total_count}")

        # Get date range
        url_min = f"{AUSTIN_API_BASE}?$select=min(published_date)"
        url_max = f"{AUSTIN_API_BASE}?$select=max(published_date)"

        resp_min = requests.get(url_min, timeout=30).json()
        resp_max = requests.get(url_max, timeout=30).json()

        min_date = resp_min[0].get("min_published_date", "Unknown")
        max_date = resp_max[0].get("max_published_date", "Unknown")

        print(f"Date range: {min_date} to {max_date}")

    except requests.exceptions.RequestException as e:
        print(f"✗ Error getting stats: {e}")

def generate_code_snippets():
    """Generate code snippets for common operations"""
    print("\n" + "="*60)
    print("  USEFUL CODE SNIPPETS")
    print("="*60)

    snippets = """
# ============================================================
# SNIPPET 1: Fetch traffic data with RAPIDS cuDF
# ============================================================
import cudf
import requests

url = "https://data.austintexas.gov/resource/dx9v-zd7x.json?$limit=50000"
response = requests.get(url)
data = response.json()

# Convert to cuDF DataFrame (GPU-accelerated)
import pandas as pd
df = cudf.DataFrame.from_pandas(pd.DataFrame(data))

# Parse timestamps
df["published_date"] = cudf.to_datetime(df["published_date"])

print(f"Loaded {len(df)} records")
print(df.head())

# ============================================================
# SNIPPET 2: Filter by date range
# ============================================================
from datetime import datetime, timedelta

start_date = datetime.now() - timedelta(days=30)
url = f"https://data.austintexas.gov/resource/dx9v-zd7x.json?" \\
      f"$where=published_date>='{start_date.isoformat()}'" \\
      f"&$limit=50000"

# ============================================================
# SNIPPET 3: Get incidents by type
# ============================================================
url = "https://data.austintexas.gov/resource/dx9v-zd7x.json?" \\
      "$where=issue_reported='CRASH'" \\
      "&$limit=10000"

# ============================================================
# SNIPPET 4: Geographic bounding box (downtown Austin)
# ============================================================
url = "https://data.austintexas.gov/resource/dx9v-zd7x.json?" \\
      "$where=latitude>=30.25 AND latitude<=30.30 " \\
      "AND longitude>=-97.77 AND longitude<=-97.72" \\
      "&$limit=10000"

# ============================================================
# SNIPPET 5: Merge with weather data
# ============================================================
import requests

# Get weather observations
weather_url = "https://api.weather.gov/stations/KAUS/observations"
headers = {"User-Agent": "(AustinSentinel, demo@example.com)"}
weather_resp = requests.get(weather_url, headers=headers)
weather_data = weather_resp.json()

# Extract observations
observations = []
for feature in weather_data.get("features", []):
    props = feature["properties"]
    obs = {
        "timestamp": props.get("timestamp"),
        "temperature_c": props.get("temperature", {}).get("value"),
        "humidity_pct": props.get("relativeHumidity", {}).get("value"),
        "visibility_m": props.get("visibility", {}).get("value"),
    }
    observations.append(obs)

weather_df = pd.DataFrame(observations)
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

# Round to hour for merging
weather_df["hour"] = weather_df["timestamp"].dt.floor("H")
incidents_df["hour"] = incidents_df["published_date"].dt.floor("H")

# Merge
merged = incidents_df.merge(weather_df, on="hour", how="left")

# ============================================================
# SNIPPET 6: Create grid-based risk aggregation
# ============================================================
GRID_SIZE = 0.01  # ~1.1 km

df["grid_lat"] = ((df["latitude"] - 30.10) / GRID_SIZE).astype(int)
df["grid_lon"] = ((df["longitude"] + 97.95) / GRID_SIZE).astype(int)
df["grid_cell"] = df["grid_lat"] * 100 + df["grid_lon"]

# Aggregate by grid cell
risk_by_cell = df.groupby("grid_cell").agg({
    "traffic_report_id": "count",
    "severity": "mean"
}).reset_index()
risk_by_cell.columns = ["grid_cell", "incident_count", "avg_severity"]

# Normalize to risk score
risk_by_cell["risk_score"] = (
    risk_by_cell["incident_count"] / risk_by_cell["incident_count"].max()
)
"""
    print(snippets)

def main():
    """Main exploration function"""
    print("\n" + "🚨 " * 20)
    print("\n    AUSTIN SENTINEL - DATA EXPLORATION")
    print("\n" + "🚨 " * 20)

    # Get dataset stats
    get_dataset_stats()

    # Fetch and analyze traffic data
    traffic_data = fetch_traffic_sample(limit=500)
    if traffic_data:
        analyze_traffic_data(traffic_data)

    # Fetch weather data
    fetch_weather_sample()

    # Show code snippets
    generate_code_snippets()

    print("\n" + "="*60)
    print("  NEXT STEPS")
    print("="*60)
    print("""
    1. Install RAPIDS: conda install -c rapidsai cudf cuml cugraph
    2. Run full data download: python src/data_ingestion/traffic_api.py --days 180
    3. Clean data: python src/preprocessing/clean_incidents.py
    4. Build graph: python src/graph/road_network.py
    5. Train model: python src/models/train.py
    6. Launch dashboard: streamlit run dashboard/app.py
    """)

    print("\n✓ Data exploration complete!\n")

if __name__ == "__main__":
    main()
