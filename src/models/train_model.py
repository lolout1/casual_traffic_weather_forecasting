"""
XGBoost model training for traffic incident risk prediction
Uses cuML on DGX Spark, falls back to xgboost for development
"""

import json
import os
import sys
import pickle
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config.config import *

# Try to import cuML, fall back to xgboost
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


class TrafficRiskPredictor:
    """Train XGBoost model for grid-based traffic risk prediction"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.model = None
        self.feature_names = None
        self.graph_features = None

    def log(self, message):
        if self.verbose:
            print(f"[ModelTrainer] {message}")

    def load_data(self, file_path):
        """Load feature data"""
        self.log(f"Loading data from {file_path}...")

        with open(file_path, 'r') as f:
            data = json.load(f)

        self.log(f"✓ Loaded {len(data):,} records")
        return data

    def load_graph_features(self, file_path):
        """Load graph features"""
        self.log(f"Loading graph features from {file_path}...")

        with open(file_path, 'r') as f:
            graph_features_str = json.load(f)

        # Convert string keys back to int
        self.graph_features = {int(k): v for k, v in graph_features_str.items()}

        self.log(f"✓ Loaded features for {len(self.graph_features):,} grid cells")

    def merge_graph_features(self, record):
        """Merge graph features into record"""

        grid_cell = record.get('grid_cell', -1)

        if self.graph_features and grid_cell in self.graph_features:
            graph_feats = self.graph_features[grid_cell]
        else:
            # Default values for missing grid cells
            graph_feats = {
                'pagerank': 0.0,
                'degree_centrality': 0.0,
                'betweenness_centrality': 0.0,
                'clustering_coefficient': 0.0,
                'degree': 0,
                'weighted_degree': 0.0,
            }

        # Add graph features to record
        record_copy = record.copy()
        for key, value in graph_feats.items():
            record_copy[f'graph_{key}'] = value

        return record_copy

    def prepare_features(self, data):
        """
        Prepare feature matrix (X) and target vector (y)

        Target: Predict incident count per grid cell per hour
        """
        self.log("\nPreparing features for training...")

        # Merge graph features
        data_with_graph = [self.merge_graph_features(r) for r in data]

        # Define feature columns (exclude target and metadata)
        exclude_fields = {
            'traffic_report_id', 'published_date', 'issue_reported',
            'issue_reported_original', 'location', 'address', 'agency',
            'traffic_report_status', 'traffic_report_status_date_time',
            'status_date', 'timestamp'  # Exclude timestamp from features
        }

        # Get feature names
        sample_record = data_with_graph[0]
        feature_names = [k for k in sample_record.keys() if k not in exclude_fields]

        self.feature_names = feature_names
        self.log(f"✓ Using {len(feature_names)} features")

        # Create feature matrix
        X = []
        for record in data_with_graph:
            features = [record.get(fname, 0.0) for fname in feature_names]
            X.append(features)

        # Create target: severity score (0-1)
        y = [record.get('severity', 0.5) for record in data_with_graph]

        self.log(f"✓ Prepared {len(X):,} samples")
        self.log(f"  Features shape: ({len(X)}, {len(feature_names)})")
        self.log(f"  Target range: [{min(y):.2f}, {max(y):.2f}]")

        return X, y, data_with_graph

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model"""

        self.log("\n" + "="*60)
        self.log("TRAINING XGBOOST MODEL")
        self.log("="*60)

        if CUML_AVAILABLE:
            self.log("\nUsing cuML XGBoost (GPU)")
            self.model = cuMLXGBRegressor(**XGBOOST_PARAMS)
        else:
            self.log("\nUsing XGBoost (CPU - will use GPU if available)")
            # XGBoost can use GPU even without cuML
            params = XGBOOST_PARAMS.copy()
            self.model = XGBRegressor(**params)

        # Train
        self.log(f"\nTraining on {len(X_train):,} samples...")
        self.log(f"Model parameters: {len(self.feature_names)} features")

        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.log(f"Using validation set: {len(X_val):,} samples")

            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        self.log("✓ Training complete!")

        return self.model

    def evaluate_model(self, X, y, dataset_name="Test"):
        """Evaluate model performance"""

        self.log(f"\n{'='*60}")
        self.log(f"EVALUATING ON {dataset_name.upper()} SET")
        self.log(f"{'='*60}")

        # Predictions
        y_pred = self.model.predict(X)

        # Compute metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        self.log(f"\nRegression Metrics:")
        self.log(f"  RMSE:  {rmse:.4f}")
        self.log(f"  MAE:   {mae:.4f}")
        self.log(f"  R²:    {r2:.4f}")

        # Compute classification metrics (high-risk prediction)
        threshold = 0.7  # High severity threshold

        y_binary = [1 if val >= threshold else 0 for val in y]
        y_pred_binary = [1 if val >= threshold else 0 for val in y_pred]

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_binary, y_pred_binary)
        precision = precision_score(y_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_binary, y_pred_binary, zero_division=0)

        self.log(f"\nClassification Metrics (threshold={threshold}):")
        self.log(f"  Accuracy:  {accuracy:.4f}")
        self.log(f"  Precision: {precision:.4f}")
        self.log(f"  Recall:    {recall:.4f}")
        self.log(f"  F1 Score:  {f1:.4f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def get_feature_importance(self, top_k=20):
        """Get feature importance from trained model"""

        self.log(f"\nTop {top_k} Most Important Features:")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            self.log("  Feature importances not available")
            return

        # Sort features by importance
        feature_imp = list(zip(self.feature_names, importances))
        feature_imp.sort(key=lambda x: x[1], reverse=True)

        for i, (name, importance) in enumerate(feature_imp[:top_k], 1):
            self.log(f"  {i:2d}. {name:30} {importance:.6f}")

        return feature_imp[:top_k]

    def save_model(self, model_path, feature_names_path):
        """Save trained model and feature names"""

        self.log(f"\nSaving model to {model_path}...")

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save feature names
        with open(feature_names_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'num_features': len(self.feature_names)
            }, f)

        model_size = os.path.getsize(model_path) / 1024 / 1024
        self.log(f"✓ Model saved ({model_size:.2f} MB)")


def main():
    """Training pipeline"""

    print("\n" + "🤖 " * 20)
    print("  AUSTIN SENTINEL - MODEL TRAINING")
    print("🤖 " * 20 + "\n")

    trainer = TrafficRiskPredictor(verbose=True)

    # Load graph features
    trainer.load_graph_features(MODEL_DIR / "graph_features.json")

    # Load and prepare training data
    print("\n" + "="*60)
    print("LOADING TRAINING DATA")
    print("="*60)

    train_data = trainer.load_data(FEATURES_DATA_DIR / "train_features.json")
    X_train, y_train, train_with_graph = trainer.prepare_features(train_data)

    # Load and prepare validation data
    print("\n" + "="*60)
    print("LOADING VALIDATION DATA")
    print("="*60)

    val_data = trainer.load_data(FEATURES_DATA_DIR / "val_features.json")
    X_val, y_val, val_with_graph = trainer.prepare_features(val_data)

    # Train model
    model = trainer.train_model(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    val_metrics = trainer.evaluate_model(X_val, y_val, dataset_name="Validation")

    # Load and evaluate on test set
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)

    test_data = trainer.load_data(FEATURES_DATA_DIR / "test_features.json")
    X_test, y_test, test_with_graph = trainer.prepare_features(test_data)

    test_metrics = trainer.evaluate_model(X_test, y_test, dataset_name="Test")

    # Feature importance
    trainer.get_feature_importance(top_k=20)

    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)

    trainer.save_model(
        MODEL_DIR / "xgboost_risk_predictor.pkl",
        MODEL_DIR / "feature_names.json"
    )

    # Final summary
    print("\n" + "="*60)
    print("✓ MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"\nModel saved to: {MODEL_DIR}")
    print(f"  - xgboost_risk_predictor.pkl")
    print(f"  - feature_names.json")

    print(f"\nValidation Performance:")
    print(f"  R² Score:  {val_metrics['r2']:.4f}")
    print(f"  RMSE:      {val_metrics['rmse']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")

    print(f"\nTest Performance:")
    print(f"  R² Score:  {test_metrics['r2']:.4f}")
    print(f"  RMSE:      {test_metrics['rmse']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")

    print()


if __name__ == "__main__":
    main()
