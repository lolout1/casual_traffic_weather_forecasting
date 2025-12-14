#!/bin/bash
# Installation script for RAPIDS on DGX Spark

echo "======================================================================"
echo "  Installing RAPIDS for Austin Sentinel"
echo "======================================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment
echo ""
echo "Creating conda environment 'austin-sentinel'..."
conda create -n austin-sentinel python=3.10 -y

# Activate environment
echo ""
echo "Activating environment..."
source activate austin-sentinel

# Install RAPIDS (for CUDA 12.0)
echo ""
echo "Installing RAPIDS libraries..."
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=24.02 \
    cuml=24.02 \
    cugraph=24.02 \
    python=3.10 \
    cuda-version=12.0 \
    -y

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install streamlit plotly requests

echo ""
echo "======================================================================"
echo "  ✓ Installation Complete!"
echo "======================================================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate austin-sentinel"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import cudf; print(f\"cuDF version: {cudf.__version__}\")'"
echo ""
