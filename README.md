# Austin Sentinel
## Segment-Level Traffic Risk Intelligence for DGX Spark Frontier Hackathon

Austin Sentinel ingests Austin’s live traffic-incident feed, cleans it into GPU-friendly tables, fuses contextual signals (history, weather, congestion, calendar), and ranks upcoming high-risk road segments so response teams can stage assets proactively. Weather is optional, but the pipeline is designed to absorb it with zero code changes once data is available.

---

## System Overview

| Layer | What It Does | Key Files |
|-------|--------------|-----------|
| **Ingestion & Cleaning** | Deduplicates records, validates geometry, normalizes incident taxonomy, enforces typed Parquet schema, and drops stale (>730 day) entries. | `src/preprocessing/data_cleaner.py` |
| **Temporal Windowing & Features** | Bins incidents into hourly (configurable) windows per road segment, computes rolling/EMA history, attaches optional weather/loop-detector/event exposures, and creates forward targets (next-hour counts & severity). | `src/features/feature_engineering.py` |
| **Road Network Graph** | Builds a segment-level k-NN graph, computes cuGraph (or NetworkX) metrics like PageRank/betweenness, and stores embeddings for downstream models. | `src/graph/road_network.py` |
| **Modeling** | Trains an XGBoost regressor (cuML when available) to predict upcoming severity; includes crash-aware sample weighting, evaluation, and feature attribution. | `src/models/train_model.py` |
| **Automation** | One-touch job script for SLURM/DGX that executes the full stack with optional stage skips. | `train_pipeline.sh` |

The entire workflow operates on Parquet/Arrow data so cuDF/cuGraph can process everything in-GPU memory on the DGX Spark once RAPIDS is installed (`install_rapids.sh`).

---

## Pipeline Details

1. **Data Cleaning (`src/preprocessing/data_cleaner.py`)**
   - Loads `austin_traffic_{train,val,test}.json` with cuDF if available, pandas otherwise.
   - Removes duplicates/missing coordinates, clamps lat/lon to Austin bounds, normalizes issue strings, attaches severity + crash/hazard/stall flags.
   - Enforces an explicit schema (see `config/config.py`) and removes stale records beyond `MAX_DATA_LAG_DAYS`.
   - Outputs Parquet splits under `data/processed/`.

2. **Temporal Features (`src/features/feature_engineering.py`)**
   - Maps incidents to road segments (address string fallback to lat/lon hash).
   - Floors timestamps to `BIN_INTERVAL_MINUTES` (default 60) and aggregates incidents/severity counts per segment/window.
   - Computes rolling means over 1/4/12/24 h windows plus an exponential moving average.
   - Joins external exposures (hourly weather, loop speed/volume, calendar events). If `/data/external/*.parquet` are absent, deterministic synthetic baselines are generated so the pipeline still runs.
   - Creates targets by shifting each segment’s counts/severity forward (`TARGET_HORIZON_WINDOWS`) and exports `*_windows.parquet`.

3. **Graph Construction (`src/graph/road_network.py`)**
   - Summarizes each segment’s centroid and density, builds a nearest-neighbor graph (cap edges per node), and runs cuGraph metrics when GPUs are available (falls back to NetworkX otherwise).
   - Writes `models/graph_features.json` so modeling can join graph embeddings by `segment_id`.

4. **Model Training (`src/models/train_model.py`)**
   - Merges windowed features with graph embeddings, balances samples via class-aware weights, and trains XGBoost (`tree_method='hist'` on CPU, `'gpu_hist'` when cuML is active).
   - Reports regression + high-risk classification metrics and stores `models/xgboost_risk_predictor.pkl` plus the feature manifest.

---

## Running the Pipeline

### Quick Local Run (CPU fallback)
```bash
python3 src/preprocessing/data_cleaner.py
python3 src/features/feature_engineering.py
python3 src/graph/road_network.py
python3 src/models/train_model.py
```

### DGX / SLURM (preferred)
```bash
sbatch -p gpu1 --exclusive train_pipeline.sh
```

Useful environment flags:
- `USE_SIMPLE_CLEANER=1` to force the legacy JSON cleaner.
- `SKIP_DATA_CLEAN=1`, `SKIP_FEATURE_ENG=1`, etc., to iterate on single stages.
- `CONDA_ENV_NAME=austin-sentinel` (default) for RAPIDS/cuML environments.

### External Data Drops
Place optional contextual datasets here (all Parquet):
```
data/external/weather_hourly.parquet
data/external/loop_detector_hourly.parquet
data/external/calendar_events.parquet
```
Schemas are documented in `config/config.py`; additional feeds can be added by extending `ExternalFeatureLoader`.

---

## Outputs & Artifacts

| Path | Description |
|------|-------------|
| `data/processed/train_clean.parquet` | Typed incident records ready for GPU processing. |
| `data/features/train_windows.parquet` | Segment-window aggregates with history, exposures, and future targets. |
| `models/graph_features.json` | Segment-level graph metrics (PageRank, betweenness, etc.). |
| `models/xgboost_risk_predictor.pkl` | Latest trained model (CPU-trained unless RAPIDS/cuML is active). |
| `models/feature_names.json` | Feature manifest for inference. |

---

## Current Performance Snapshot (CPU run)
- Validation RMSE ≈ 0.314, precision/recall ≈ 0.52 / 0.35 for “high-risk” windows.
- Model currently trains with synthetic exposures (no real weather/loop feeds). Expect significant gains once true contextual data is available and GPU acceleration is enabled.

---

## Roadmap / Future Directions

1. **Real Contextual Data** – Replace synthetic weather/loop/event placeholders with actual hourly feeds to unlock the >10% AUC gains cited in recent traffic-risk papers.
2. **cuDF Everywhere** – Install RAPIDS on DGX (`bash install_rapids.sh`) so cleaning, feature engineering, and graph analytics run entirely on GPU, reducing latency and matching the hackathon “Spark Story”.
3. **Tow-Truck Optimization** – Add a cuOpt layer that takes predicted hotspots plus fleet constraints and outputs optimal staging plans.
4. **Dashboard & Alerting** – Surface the top-N predicted segments, their drivers (weather, history, graph centrality), and recommended actions via Streamlit or a lightweight API.
5. **Model Quality** – Experiment with gradient-boosted classification heads (probability of ≥1 incident) and fine-tune loss weighting, especially once richer exposures are live.

Built for proactive, data-driven traffic response in Austin. 🚦
