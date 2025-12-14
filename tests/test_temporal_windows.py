import pandas as pd
from datetime import datetime, timedelta

from src.features.feature_engineering import TemporalFeatureEngineer
from src.models.train_model import compute_sample_weights


def test_target_generation_alignment():
    engineer = TemporalFeatureEngineer()
    base_time = datetime(2025, 1, 1, 8, 0, 0)
    rows = []
    for i in range(3):
        rows.append({
            'segment_id': 'SEG-1',
            'window_start': base_time + timedelta(hours=i),
            'incident_count': i,
            'severity_sum': float(i)
        })
    df = pd.DataFrame(rows)
    enriched = engineer._add_targets(df.copy())  # pylint: disable=protected-access
    assert len(enriched) == 2  # last row drops due to missing future horizon
    assert enriched.iloc[0]['target_incident_count_next'] == 1
    assert enriched.iloc[1]['target_incident_count_next'] == 2


def test_sample_weighting_prioritizes_crashes():
    df = pd.DataFrame({
        'target_has_incident_next': [0, 1, 0, 1],
        'target_high_risk_next': [0, 1, 0, 0]
    })
    weights = compute_sample_weights(df)
    assert weights[1] > weights[0]
    assert weights[1] > weights[3]  # high-risk boosts weight further
