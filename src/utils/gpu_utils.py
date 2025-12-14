"""
GPU/CPU abstraction utilities
Automatically uses RAPIDS if available, falls back to pandas/sklearn
"""

import sys
import warnings

# Try to import RAPIDS libraries
GPU_AVAILABLE = False
try:
    import cudf
    import cupy as cp
    import cuml
    import cugraph
    GPU_AVAILABLE = True
    print("✓ RAPIDS libraries loaded - GPU acceleration enabled")
except ImportError:
    import pandas as pd
    import numpy as np
    warnings.warn(
        "⚠️  RAPIDS not found - falling back to CPU (pandas/numpy). "
        "Install RAPIDS for GPU acceleration: run ./install_rapids.sh"
    )


# DataFrame abstraction
if GPU_AVAILABLE:
    DataFrame = cudf.DataFrame
    Series = cudf.Series
    read_json = cudf.read_json
    read_csv = cudf.read_csv
    read_parquet = cudf.read_parquet
    to_datetime = cudf.to_datetime
    concat = cudf.concat
    array_lib = cp  # CuPy
else:
    import pandas as pd
    import numpy as np
    DataFrame = pd.DataFrame
    Series = pd.Series
    read_json = pd.read_json
    read_csv = pd.read_csv
    read_parquet = pd.read_parquet
    to_datetime = pd.to_datetime
    concat = pd.concat
    array_lib = np  # NumPy


def get_compute_mode():
    """Return current compute mode"""
    return "GPU (RAPIDS)" if GPU_AVAILABLE else "CPU (pandas/numpy)"


def from_pandas(df):
    """Convert pandas DataFrame to GPU if available"""
    if GPU_AVAILABLE:
        return cudf.from_pandas(df)
    return df


def to_pandas(df):
    """Convert to pandas DataFrame"""
    if GPU_AVAILABLE and isinstance(df, cudf.DataFrame):
        return df.to_pandas()
    return df


def get_array(data):
    """Get appropriate array type (cupy or numpy)"""
    if GPU_AVAILABLE:
        if isinstance(data, cp.ndarray):
            return data
        return cp.array(data)
    else:
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)


class GPUFallbackTimer:
    """Timer that works with both GPU and CPU operations"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        if GPU_AVAILABLE:
            cp.cuda.Stream.null.synchronize()  # Sync GPU
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if GPU_AVAILABLE:
            cp.cuda.Stream.null.synchronize()  # Sync GPU
        import time
        self.end_time = time.time()

    def elapsed(self):
        """Return elapsed time in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def print_performance_info():
    """Print current performance configuration"""
    print("\n" + "="*60)
    print("PERFORMANCE CONFIGURATION")
    print("="*60)
    print(f"Compute Mode: {get_compute_mode()}")

    if GPU_AVAILABLE:
        print(f"cuDF Version: {cudf.__version__}")
        print(f"CuPy Version: {cp.__version__}")
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
            print(f"GPU Count: {gpu_count}")
            if gpu_count > 0:
                props = cp.cuda.runtime.getDeviceProperties(0)
                print(f"GPU 0: {props['name'].decode()}")
                print(f"GPU Memory: {props['totalGlobalMem'] / 1024**3:.1f} GB")
        except:
            print("GPU info unavailable")
    else:
        import multiprocessing
        print(f"CPU Cores: {multiprocessing.cpu_count()}")

    print("="*60 + "\n")
