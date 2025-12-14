"""
XGBoost training using forward-looking severity targets with class balancing.
"""

import json
import os
import sys
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config.config import (
    FEATURES_DATA_DIR,
    MODEL_DIR,
    GRAPH_FEATURES_JSON,
    TRAINED_MODEL_PATH,
    XGBOOST_PARAMS,
    RISK_THRESHOLD,
)

try:
    from cuml import XGBRegressor as cuMLXGBRegressor
    CUML_AVAILABLE = True
    print("✓ cuML loaded - GPU-accelerated XGBoost enabled")
except ImportError:
    try:
        from xgboost import XGBRegressor
        CUML_AVAILABLE = False
        print("⚠️  cuML not available - using XGBoost CPU fallback")
    except ImportError:
        print("❌ XGBoost not available - install: pip install xgboost")
        sys.exit(1)


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Balance rare crashes/high-risk events as recommended in 2024 studies."""
    positives = df['target_has_incident_next'].sum()
    negatives = len(df) - positives
    if positives == 0 or negatives == 0:
        return np.ones(len(df))

    pos_weight = len(df) / (2 * positives)
    neg_weight = len(df) / (2 * negatives)
    weights = np.where(df['target_has_incident_next'] >= 1, pos_weight, neg_weight)

    # Emphasize high-severity windows (focal-style reweighting)
    weights = np.where(df['target_high_risk_next'] >= 1, weights * 1.5, weights)
    return weights


class TrafficRiskPredictor:
    """Train XGBoost model for segment-level crash risk prediction."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.model = None
        self.feature_names = None
        self.graph_features = None

    def log(self, message: str):
        if self.verbose:
            print(f"[ModelTrainer] {message}")

    def load_data(self, path: os.PathLike) -> pd.DataFrame:
        self.log(f"Loading data from {path}...")
        df = pd.read_parquet(path)
        self.log(f"✓ Loaded {len(df):,} rows")
        return df

    def load_graph_features(self, path: os.PathLike):
        self.log(f"Loading graph features from {path}...")
        if not os.path.exists(path):
            self.log("⚠️  Graph feature file missing; graph inputs will be zero.")
            self.graph_features = pd.DataFrame()
            return
        with open(path, 'r') as f:
            data = json.load(f)
        graph_df = pd.DataFrame.from_dict(data, orient='index').reset_index()
        graph_df = graph_df.rename(columns={'index': 'segment_id'})
        self.graph_features = graph_df
        self.log(f"✓ Loaded {len(graph_df):,} segment embeddings")

    def merge_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.graph_features is None or self.graph_features.empty:
            for col in ['pagerank', 'degree_centrality', 'betweenness_centrality',
                        'clustering_coefficient', 'degree', 'weighted_degree',
                        'segment_lat', 'segment_lon']:
                if col not in df.columns:
                    df[col] = 0.0
            return df
        graph_df = self.graph_features.add_prefix('graph_')
        graph_df = graph_df.rename(columns={'graph_segment_id': 'segment_id'})
        df = df.merge(graph_df, on='segment_id', how='left')
        graph_cols = [c for c in df.columns if c.startswith('graph_')]
        df[graph_cols] = df[graph_cols].fillna(0.0)
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        df = self.merge_graph_features(df)
        df = df.dropna(subset=['target_severity_next', 'target_incident_count_next'])

        exclude_fields = {
            'segment_id',
            'window_start',
            'target_incident_count_next',
            'target_severity_next',
            'target_has_incident_next',
            'target_high_risk_next'
        }

        feature_cols = [c for c in df.columns if c not in exclude_fields]
        features = df[feature_cols].fillna(0.0)
        target = df['target_severity_next'].astype(float)

        self.feature_names = feature_cols
        self.log(f"✓ Prepared {len(features):,} samples with {len(feature_cols)} features")
        return features.values, target.values, df

    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        self.log("\n" + "=" * 60)
        self.log("TRAINING XGBOOST MODEL")
        self.log("=" * 60)

        if CUML_AVAILABLE:
            self.log("Using cuML XGBoost (GPU)")
            self.model = cuMLXGBRegressor(**XGBOOST_PARAMS)
        else:
            self.log("Using XGBoost (CPU/GPU via xgboost)")
            params = XGBOOST_PARAMS.copy()
            params['tree_method'] = 'hist'
            params.pop('device', None)
            self.model = XGBRegressor(**params)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight

        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.log(f"Validation set size: {len(X_val):,}")
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False, **fit_kwargs)
        else:
            self.model.fit(X_train, y_train, **fit_kwargs)

        self.log("✓ Training complete")

    def evaluate(self, X, y, df, dataset_name="Test"):
        self.log(f"\n{'='*60}\nEVALUATING ON {dataset_name.upper()}\n{'='*60}")
        preds = self.model.predict(X)

        mse = mean_squared_error(y, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)

        threshold = RISK_THRESHOLD
        y_binary = (y >= threshold).astype(int)
        pred_binary = (preds >= threshold).astype(int)

        acc = accuracy_score(y_binary, pred_binary)
        prec = precision_score(y_binary, pred_binary, zero_division=0)
        rec = recall_score(y_binary, pred_binary, zero_division=0)
        f1 = f1_score(y_binary, pred_binary, zero_division=0)

        true_high = df['target_high_risk_next'].astype(int)
        pred_high = (preds >= threshold).astype(int)
        high_prec = precision_score(true_high, pred_high, zero_division=0)
        high_rec = recall_score(true_high, pred_high, zero_division=0)

        self.log(f"  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}")
        self.log(f"  Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
        self.log(f"  High-risk Precision: {high_prec:.4f}  High-risk Recall: {high_rec:.4f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'high_risk_precision': high_prec,
            'high_risk_recall': high_rec,
        }

    def feature_importance(self, top_k=20):
        if not hasattr(self.model, 'feature_importances_'):
            self.log("Feature importances unavailable")
            return
        importances = self.model.feature_importances_
        ranking = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)[:top_k]
        self.log("Top feature importances:")
        for idx, (name, score) in enumerate(ranking, 1):
            self.log(f"  {idx:2d}. {name:35} {score:.6f}")
        return ranking

    def save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRAINED_MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(MODEL_DIR / "feature_names.json", 'w') as f:
            json.dump({'feature_names': self.feature_names}, f)
        self.log(f"✓ Model saved to {TRAINED_MODEL_PATH}")


def main():
    print("\n" + "🤖 " * 20)
    print("  AUSTIN SENTINEL - MODEL TRAINING")
    print("🤖 " * 20 + "\n")

    trainer = TrafficRiskPredictor(verbose=True)
    trainer.load_graph_features(GRAPH_FEATURES_JSON)

    train_df = trainer.load_data(FEATURES_DATA_DIR / "train_windows.parquet")
    X_train, y_train, train_full = trainer.prepare_features(train_df)
    weights = compute_sample_weights(train_full)

    val_df = trainer.load_data(FEATURES_DATA_DIR / "val_windows.parquet")
    X_val, y_val, val_full = trainer.prepare_features(val_df)

    trainer.train(X_train, y_train, X_val, y_val, sample_weight=weights)

    val_metrics = trainer.evaluate(X_val, y_val, val_full, dataset_name="Validation")

    test_df = trainer.load_data(FEATURES_DATA_DIR / "test_windows.parquet")
    X_test, y_test, test_full = trainer.prepare_features(test_df)
    test_metrics = trainer.evaluate(X_test, y_test, test_full, dataset_name="Test")

    trainer.feature_importance(top_k=20)
    trainer.save()

    print("\n" + "="*60)
    print("✓ MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"\nValidation R²: {val_metrics['r2']:.4f} | RMSE: {val_metrics['rmse']:.4f}")
    print(f"Validation precision/recall: {val_metrics['precision']:.4f} / {val_metrics['recall']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f} | RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test precision/recall: {test_metrics['precision']:.4f} / {test_metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
