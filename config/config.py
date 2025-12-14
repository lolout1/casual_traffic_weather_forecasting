"""
Configuration file for Austin Sentinel project
"""
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESED_RAW_FALLBACK = RAW_DATA_DIR / "archive"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# External data artifacts
WEATHER_PARQUET = EXTERNAL_DATA_DIR / "weather_hourly.parquet"
LOOP_VOLUME_PARQUET = EXTERNAL_DATA_DIR / "loop_detector_hourly.parquet"
EVENTS_PARQUET = EXTERNAL_DATA_DIR / "calendar_events.parquet"
ROAD_SEGMENTS_PARQUET = EXTERNAL_DATA_DIR / "road_segments.parquet"

# Model paths
TRAINED_MODEL_PATH = MODEL_DIR / "xgboost_risk_predictor.pkl"
GRAPH_FEATURES_JSON = MODEL_DIR / "graph_features.json"
GRAPH_FEATURES_PATH = MODEL_DIR / "graph_features.parquet"

# Dataset parameters
TRAIN_FILE = PROJECT_ROOT / "austin_traffic_train.json"
VAL_FILE = PROJECT_ROOT / "austin_traffic_val.json"
TEST_FILE = PROJECT_ROOT / "austin_traffic_test.json"
FULL_FILE = PROJECT_ROOT / "austin_traffic_full.json"

# Grid configuration
GRID_SIZE_LAT = 0.01  # ~1.1 km
GRID_SIZE_LON = 0.01  # ~1.1 km
AUSTIN_LAT_MIN = 30.10
AUSTIN_LAT_MAX = 30.62
AUSTIN_LON_MIN = -98.10
AUSTIN_LON_MAX = -97.44

# Feature engineering parameters
TEMPORAL_FEATURES = [
    'hour', 'day_of_week', 'month', 'day_of_month',
    'is_weekend', 'is_rush_hour', 'is_night',
    'season', 'week_of_year'
]

SPATIAL_FEATURES = [
    'latitude', 'longitude', 'segment_lat', 'segment_lon',
    'segment_distance_from_center'
]

WEATHER_FEATURES = [
    'temperature_c', 'humidity_pct', 'visibility_m',
    'wind_speed_kmh', 'pressure_pa', 'is_raining',
    'is_clear', 'precipitation_mm'
]

LOOP_FEATURES = [
    'avg_speed_kmh', 'vehicle_count', 'congestion_index'
]

EVENT_FEATURES = [
    'is_weekend', 'is_holiday', 'is_school_day', 'is_special_event'
]

GRAPH_FEATURES = [
    'pagerank', 'betweenness_centrality', 'degree',
    'clustering_coefficient', 'core_number'
]

BIN_INTERVAL_MINUTES = int(os.getenv("BIN_INTERVAL_MINUTES", "60"))
TARGET_HORIZON_WINDOWS = int(os.getenv("TARGET_HORIZON_WINDOWS", "1"))
ROLLING_WINDOWS = [1, 4, 12, 24]  # hours
EMA_ALPHA = 0.3
MAX_DATA_LAG_DAYS = int(os.getenv("MAX_DATA_LAG_DAYS", "730"))

# Austin city center (for distance calculations)
AUSTIN_CENTER_LAT = 30.2672
AUSTIN_CENTER_LON = -97.7431

# Weather API configuration
WEATHER_API_BASE = "https://api.weather.gov"
WEATHER_STATION = "KAUS"  # Austin-Bergstrom Airport
USER_AGENT = "(AustinSentinel, hackathon@example.com)"

# Model hyperparameters
XGBOOST_PARAMS = {
    'tree_method': 'gpu_hist',
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}

# Graph construction parameters
GRAPH_DISTANCE_THRESHOLD = 0.005  # ~550m - incidents within this distance are connected
MIN_EDGE_WEIGHT = 0.1
MAX_EDGES_PER_NODE = 10

# Risk prediction parameters
TOP_K_HOTSPOTS = 10
RISK_THRESHOLD = 0.7
PREDICTION_HORIZON_HOURS = 1

# Benchmark parameters
BENCHMARK_ITERATIONS = 5
CPU_THREADS = 96

# Incident type severity weights
INCIDENT_SEVERITY = {
    'CRASH': 1.0,
    'Crash Urgent': 1.0,
    'CRASH URGENT': 1.0,
    'COLLISION': 0.9,
    'COLLISION WITH INJURY': 1.0,
    'COLLISN/ LVNG SCN': 0.8,
    'Crash Service': 0.7,
    'CRASH SERVICE ROAD': 0.7,
    'Traffic Hazard': 0.5,
    'TRFC HAZD/ DEBRIS': 0.4,
    'Stalled Vehicle': 0.3,
    'STALL': 0.3,
    'LOOSE LIVESTOCK': 0.4,
    'VEHICLE FIRE': 0.8,
    'HAZARD IN ROAD': 0.5,
    'BLOCKED DRIV/ HWY': 0.3,
    'AUTO/ PED': 1.0,
}

# Strongly-typed schema for cleaned data
DATA_SCHEMA = {
    'traffic_report_id': 'string',
    'published_date': 'datetime64[ns]',
    'issue_reported': 'string',
    'issue_reported_original': 'string',
    'latitude': 'float64',
    'longitude': 'float64',
    'address': 'string',
    'traffic_report_status': 'string',
    'traffic_report_status_date_time': 'datetime64[ns]',
    'status_date': 'datetime64[ns]',
    'agency': 'string',
    'severity': 'float32',
    'is_crash': 'int8',
    'is_hazard': 'int8',
    'is_stall': 'int8'
}

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR, MODEL_DIR, EXTERNAL_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
