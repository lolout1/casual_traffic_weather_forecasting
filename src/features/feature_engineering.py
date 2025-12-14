"""
GPU-aware feature engineering that builds temporal windows, joins external
exposures, and materializes forward-looking targets for crash risk modeling.
"""

import math
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config.config import (
    PROCESSED_DATA_DIR,
    FEATURES_DATA_DIR,
    MODEL_DIR,
    BIN_INTERVAL_MINUTES,
    TARGET_HORIZON_WINDOWS,
    ROLLING_WINDOWS,
    EMA_ALPHA,
    WEATHER_FEATURES,
    LOOP_FEATURES,
    EVENT_FEATURES,
    AUSTIN_CENTER_LAT,
    AUSTIN_CENTER_LON,
    WEATHER_PARQUET,
    LOOP_VOLUME_PARQUET,
    EVENTS_PARQUET,
    RISK_THRESHOLD,
)
from src.utils.gpu_utils import (
    GPU_AVAILABLE,
    DataFrame,
    read_parquet,
    to_pandas,
    from_pandas,
)

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas should exist but guard anyway
    pd = None


BIN_FREQ = f"{BIN_INTERVAL_MINUTES}min"


def normalize_address(value):
    if value is None:
        return "UNKNOWN"
    return str(value).strip().upper() or "UNKNOWN"


def haversine_vector(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0 * c


class ExternalFeatureLoader:
    """Loads or synthesizes weather, loop-detector, and calendar features."""

    def __init__(self, logger):
        self.log = logger
        self.weather = self._read_optional_parquet(WEATHER_PARQUET, "weather")
        self.loop = self._read_optional_parquet(LOOP_VOLUME_PARQUET, "loop detector")
        self.events = self._read_optional_parquet(EVENTS_PARQUET, "calendar/events")

    def _read_optional_parquet(self, path, label):
        if path.exists():
            self.log(f"Loading {label} features from {path}")
            df = read_parquet(path)
            return to_pandas(df) if GPU_AVAILABLE else df
        self.log(f"⚠️  No {label} file at {path} - generating synthetic baseline")
        return None

    def _generate_weather(self, windows: pd.Series) -> pd.DataFrame:
        unique_windows = pd.DataFrame({'window_start': pd.Series(windows.unique()).sort_values()})
        hours = unique_windows['window_start'].dt.hour
        months = unique_windows['window_start'].dt.month
        unique_windows['temperature_c'] = 18 + 10 * np.sin((months / 12) * 2 * np.pi)
        unique_windows['humidity_pct'] = 60 + 20 * np.cos((hours / 24) * 2 * np.pi)
        unique_windows['wind_speed_kmh'] = 15 + 5 * np.sin((hours / 24) * 2 * np.pi + np.pi / 4)
        unique_windows['pressure_pa'] = 101325 + 500 * np.cos((months / 12) * 2 * np.pi)
        unique_windows['visibility_m'] = 8000 - 2000 * (unique_windows['humidity_pct'] / 100)
        unique_windows['precipitation_mm'] = np.clip(5 * np.random.RandomState(42).rand(len(unique_windows)), 0, 8)
        unique_windows['is_raining'] = (unique_windows['precipitation_mm'] > 1.0).astype(int)
        unique_windows['is_clear'] = (unique_windows['precipitation_mm'] < 0.5).astype(int)
        return unique_windows

    def _generate_loop(self, frame: pd.DataFrame) -> pd.DataFrame:
        key = frame[['segment_id', 'window_start']].drop_duplicates().reset_index(drop=True)
        hashed = pd.util.hash_pandas_object(key['segment_id'], index=False).astype(np.int64)
        hour = key['window_start'].dt.hour
        key['avg_speed_kmh'] = 45 + (hashed % 15) - 5 + 3 * np.sin((hour / 24) * 2 * np.pi)
        key['vehicle_count'] = 50 + (hashed % 40) + 10 * np.cos((hour / 24) * 2 * np.pi)
        key['congestion_index'] = np.clip(1 - key['avg_speed_kmh'] / 70, 0, 1)
        return key

    def _generate_events(self, windows: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({'window_start': pd.Series(windows.unique()).sort_values()})
        df['is_weekend'] = df['window_start'].dt.weekday >= 5
        df['is_holiday'] = df['window_start'].dt.month.isin([1, 7, 12]).astype(int)
        df['is_school_day'] = ((df['window_start'].dt.weekday < 5) & (~df['window_start'].dt.month.isin([6, 7, 8]))).astype(int)
        df['is_special_event'] = (((df['window_start'].dt.day % 5) == 0) & (df['window_start'].dt.hour >= 18)).astype(int)
        return df

    def join_all(self, df: pd.DataFrame) -> pd.DataFrame:
        windows = df['window_start']
        weather = self.weather if self.weather is not None else self._generate_weather(windows)
        loop = self.loop if self.loop is not None else self._generate_loop(df)
        events = self.events if self.events is not None else self._generate_events(windows)

        df = df.merge(weather, on='window_start', how='left')
        df = df.merge(loop, on=['segment_id', 'window_start'], how='left')
        df = df.merge(events, on='window_start', how='left')

        for feature in WEATHER_FEATURES + LOOP_FEATURES + EVENT_FEATURES:
            if feature not in df.columns:
                df[feature] = 0.0

        return df


class TemporalFeatureEngineer:
    """Builds temporal windows with forward-looking targets."""

    def __init__(self, bin_minutes=BIN_INTERVAL_MINUTES):
        self.bin_minutes = bin_minutes
        self.external_loader = ExternalFeatureLoader(self.log)

    def log(self, message):
        print(f"[FeatureEngineer] {message}")

    def load_clean_split(self, split: str):
        path = PROCESSED_DATA_DIR / f"{split}_clean.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing cleaned parquet for split '{split}' at {path}")
        self.log(f"Loading cleaned {split} data from {path}")
        return read_parquet(path)

    def _assign_segments(self, df):
        if GPU_AVAILABLE:
            df['segment_id'] = df['address'].fillna('UNKNOWN').astype('str').str.strip().str.upper()
        else:
            df['segment_id'] = df['address'].fillna('UNKNOWN').astype(str).str.strip().str.upper()

        mask = (df['segment_id'] == "UNKNOWN")
        if mask.any():
            replacement = (
                df.loc[mask, 'latitude'].round(3).astype(str) + "_" +
                df.loc[mask, 'longitude'].round(3).astype(str)
            )
            df.loc[mask, 'segment_id'] = replacement

        return df

    def _aggregate_windows(self, df):
        df['window_start'] = df['published_date'].dt.floor(BIN_FREQ)
        group_cols = ['segment_id', 'window_start']
        agg_df = df.groupby(group_cols).agg({
            'traffic_report_id': 'count',
            'severity': ['sum', 'mean'],
            'is_crash': 'sum',
            'is_hazard': 'sum',
            'is_stall': 'sum',
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()

        pdf = to_pandas(agg_df) if GPU_AVAILABLE else agg_df
        pdf.columns = [
            '_'.join(col).rstrip('_') if isinstance(col, tuple) else col
            for col in pdf.columns
        ]

        pdf = pdf.rename(columns={
            'traffic_report_id_count': 'incident_count',
            'severity_sum': 'severity_sum',
            'severity_mean': 'severity_mean',
            'is_crash_sum': 'crash_count',
            'is_hazard_sum': 'hazard_count',
            'is_stall_sum': 'stall_count',
            'latitude_mean': 'segment_lat',
            'longitude_mean': 'segment_lon'
        })

        return pdf

    def _add_distance_feature(self, pdf: pd.DataFrame) -> pd.DataFrame:
        distances = haversine_vector(
            pdf['segment_lat'].values,
            pdf['segment_lon'].values,
            np.full(len(pdf), AUSTIN_CENTER_LAT),
            np.full(len(pdf), AUSTIN_CENTER_LON)
        )
        pdf['segment_distance_from_center'] = distances
        return pdf

    def _add_rolling_features(self, pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values(['segment_id', 'window_start'])

        for window in ROLLING_WINDOWS:
            roll = (
                pdf.groupby('segment_id')['incident_count']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            pdf[f'rolling_incident_mean_{window}h'] = roll

            sev_roll = (
                pdf.groupby('segment_id')['severity_sum']
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            pdf[f'rolling_severity_sum_{window}h'] = sev_roll

        ema = (
            pdf.groupby('segment_id')['incident_count']
            .apply(lambda s: s.ewm(alpha=EMA_ALPHA).mean())
        )
        pdf['ema_incident_count'] = ema.reset_index(level=0, drop=True)

        return pdf

    def _add_targets(self, pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values(['segment_id', 'window_start'])
        group = pdf.groupby('segment_id')
        pdf['target_incident_count_next'] = group['incident_count'].shift(-TARGET_HORIZON_WINDOWS)
        pdf['target_severity_next'] = group['severity_sum'].shift(-TARGET_HORIZON_WINDOWS)
        pdf['target_has_incident_next'] = (pdf['target_incident_count_next'] > 0).astype(float)
        pdf['target_high_risk_next'] = (pdf['target_severity_next'] >= RISK_THRESHOLD).astype(float)
        pdf = pdf.dropna(subset=['target_incident_count_next', 'target_severity_next'])
        return pdf

    def build_split(self, split: str, compute_targets: bool = True):
        df = self.load_clean_split(split)
        df = self._assign_segments(df)
        pdf = self._aggregate_windows(df)
        pdf = self._add_distance_feature(pdf)
        pdf = self.external_loader.join_all(pdf)
        pdf = self._add_rolling_features(pdf)
        if compute_targets:
            pdf = self._add_targets(pdf)

        output_path = FEATURES_DATA_DIR / f"{split}_windows.parquet"
        FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
        pdf.to_parquet(output_path, index=False)
        self.log(f"✓ Saved {len(pdf):,} windowed rows to {output_path}")

        if split == "train":
            stats_path = FEATURES_DATA_DIR / "segment_stats.json"
            stats = (
                pdf.groupby('segment_id')[['segment_lat', 'segment_lon', 'segment_distance_from_center']]
                .mean()
                .reset_index()
            )
            stats.to_json(stats_path, orient='records')
            self.log(f"✓ Persisted segment stats to {stats_path}")


def main():
    engineer = TemporalFeatureEngineer()
    for split in ("train", "val", "test"):
        engineer.build_split(split, compute_targets=True)


if __name__ == "__main__":
    main()
