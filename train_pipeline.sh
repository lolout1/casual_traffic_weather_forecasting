#!/bin/bash
#SBATCH --job-name=austin-train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

cleanup() {
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    log "Training job failed with exit code $exit_code"
  else
    log "Training job completed successfully"
  fi
}
trap cleanup EXIT

print_section() {
  log "------------------------------------------------------------"
  log "$1"
  log "------------------------------------------------------------"
}

maybe_activate_conda() {
  local env_name="${CONDA_ENV_NAME:-austin-sentinel}"
  if [[ "${SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
    log "Skipping conda activation because SKIP_CONDA_ACTIVATE=1"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
      log "Activating conda environment: $env_name"
      conda activate "$env_name"
    else
      log "Conda environment '$env_name' not found. Continuing without activation."
    fi
  else
    log "Conda is not available on PATH. Continuing with current environment."
  fi
}

run_step() {
  local step_name="$1"
  shift
  print_section "Starting: $step_name"
  "$@"
  log "Finished: $step_name"
}

print_section "Austin Sentinel training pipeline starting on host $(hostname)"

maybe_activate_conda

if command -v nvidia-smi >/dev/null 2>&1; then
  print_section "GPU inventory"
  nvidia-smi
fi

mkdir -p data/processed data/features models

if [[ "${SKIP_DATA_CLEAN:-0}" != "1" ]]; then
  if [[ "${USE_SIMPLE_CLEANER:-0}" == "1" ]]; then
    run_step "Data cleaning (simple fallback)" python3 src/preprocessing/data_cleaner_simple.py
  else
    run_step "Data cleaning (GPU-ready)" python3 src/preprocessing/data_cleaner.py
  fi
else
  log "Skipping data cleaning because SKIP_DATA_CLEAN=1"
fi

if [[ "${SKIP_FEATURE_ENG:-0}" != "1" ]]; then
  run_step "Temporal window + exposure engineering" python3 src/features/feature_engineering.py
else
  log "Skipping feature engineering because SKIP_FEATURE_ENG=1"
fi

if [[ "${SKIP_GRAPH_BUILD:-0}" != "1" ]]; then
  run_step "Graph construction" python3 src/graph/road_network.py
else
  log "Skipping graph construction because SKIP_GRAPH_BUILD=1"
fi

if [[ "${SKIP_MODEL_TRAIN:-0}" != "1" ]]; then
  run_step "Model training" python3 src/models/train_model.py
else
  log "Skipping model training because SKIP_MODEL_TRAIN=1"
fi
