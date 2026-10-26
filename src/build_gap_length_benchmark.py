"""
Build Benchmark Datasets for Different Gap Lengths
Creates datasets for gap lengths 10, 20, and 50 timesteps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
import json
import glob
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path("data/raw/uploads/phm_data_challenge_2018")
BENCHMARK_BASE = Path("data/benchmark_v2")
GAP_BENCHMARK_BASE = Path("data/gap_length_benchmark")

SENSOR_COLUMNS = [
    "IONGAUGEPRESSURE",
    "ETCHBEAMVOLTAGE",
    "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE",
    "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE",
    "ROTATIONSPEED",
    "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION",
    "ETCHSOURCEUSAGE",
    "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION"
]

METADATA_COLUMNS = ["time", "Tool", "stage", "Lot", "runnum", "recipe", "recipe_step"]

TRAIN_FILES = [
    "05_M02_DC_train.csv",
    "07_M02_DC_train.csv"
]

TEST_FILES = [
    "01_M02_DC_test.csv",
    "02_M02_DC_test.csv",
    "03_M01_DC_test.csv",
    "04_M01_DC_test.csv",
    "06_M01_DC_test.csv"
]

GAP_LENGTHS = [10, 20, 50]
CONTEXT_LENGTH = 50
STRIDE = 100

print("=" * 70)
print("BUILDING GAP LENGTH BENCHMARKS")
print("=" * 70)
print(f"Gap lengths: {GAP_LENGTHS}")
print(f"Context length: {CONTEXT_LENGTH}")
print(f"Stride: {STRIDE}")
print()

# Load the existing scaler from benchmark_v2
with open(BENCHMARK_BASE / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Loaded existing scaler from benchmark_v2")
print(f"Scaler features: {scaler.n_features_in_}")
print()


def load_and_preprocess_file(file_path: Path) -> pd.DataFrame:
    """Load and preprocess a single CSV file."""
    df = pd.read_csv(file_path)
    df = df.sort_values("time").reset_index(drop=True)
    df = df.ffill().bfill()
    return df


def identify_runs(df: pd.DataFrame) -> List[Tuple[int, int, int, int]]:
    """Identify contiguous (Lot, runnum) runs."""
    runs = []
    current_run = None
    start_idx = 0
    
    for idx, row in df.iterrows():
        run_id = (row["Lot"], row["runnum"])
        
        if current_run is None:
            current_run = run_id
            start_idx = idx
        elif run_id != current_run:
            runs.append((start_idx, idx, current_run[0], current_run[1]))
            current_run = run_id
            start_idx = idx
    
    if current_run is not None:
        runs.append((start_idx, len(df), current_run[0], current_run[1]))
    
    return runs


def split_runs_temporally(runs: List[Tuple[int, int, int, int]], train_ratio: float = 0.8) -> Tuple[List, List]:
    """Split runs temporally: first 80% to train, last 20% to validation."""
    n_runs = len(runs)
    split_idx = int(n_runs * train_ratio)
    return runs[:split_idx], runs[split_idx:]


def extract_run_data(df: pd.DataFrame, runs: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Extract sensor data for specified runs."""
    all_data = []
    for start_idx, end_idx, _, _ in runs:
        run_data = df.iloc[start_idx:end_idx][SENSOR_COLUMNS].values
        all_data.append(run_data)
    
    if all_data:
        return np.vstack(all_data)
    else:
        return np.array([]).reshape(0, len(SENSOR_COLUMNS))


def generate_sequences_within_runs(
    df: pd.DataFrame,
    runs: List[Tuple[int, int, int, int]],
    scaler: StandardScaler,
    context_length: int,
    gap_length: int,
    stride: int
) -> Dict:
    """Generate masked reconstruction sequences within individual runs."""
    context_before_list = []
    context_after_list = []
    masked_segment_list = []
    lot_list = []
    runnum_list = []
    start_idx_list = []
    
    required_length = context_length + gap_length + context_length
    
    for start_idx, end_idx, lot, runnum in runs:
        run_df = df.iloc[start_idx:end_idx]
        run_length = len(run_df)
        
        if run_length < required_length:
            continue
        
        run_sensors = run_df[SENSOR_COLUMNS].values
        run_normalized = scaler.transform(run_sensors)
        
        for seq_start in range(0, run_length - required_length + 1, stride):
            seq_end = seq_start + required_length
            
            ctx_before = run_normalized[seq_start:seq_start + context_length]
            masked = run_normalized[seq_start + context_length:seq_start + context_length + gap_length]
            ctx_after = run_normalized[seq_start + context_length + gap_length:seq_end]
            
            context_before_list.append(ctx_before.astype(np.float32))
            context_after_list.append(ctx_after.astype(np.float32))
            masked_segment_list.append(masked.astype(np.float32))
            lot_list.append(lot)
            runnum_list.append(runnum)
            start_idx_list.append(start_idx + seq_start)
    
    return {
        "context_before": np.array(context_before_list, dtype=np.float32),
        "context_after": np.array(context_after_list, dtype=np.float32),
        "masked_segment": np.array(masked_segment_list, dtype=np.float32),
        "Lot": np.array(lot_list, dtype=np.int32),
        "runnum": np.array(runnum_list, dtype=np.int32),
        "start_idx": np.array(start_idx_list, dtype=np.int32)
    }


def save_npz(data: Dict, file_path: Path):
    """Save data as NPZ file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file_path, **data)
    print(f"  Saved: {file_path}")


# Load training files
print("Loading training files...")
train_dfs = {}
for train_file in TRAIN_FILES:
    file_path = DATASET_PATH / "train" / train_file
    df = load_and_preprocess_file(file_path)
    train_dfs[train_file] = df
    print(f"  Loaded {train_file}: {len(df):,} rows")

# Identify runs in each training file
print("\nIdentifying runs in training files...")
train_runs_by_file = {}
for train_file, df in train_dfs.items():
    runs = identify_runs(df)
    train_runs_by_file[train_file] = runs
    print(f"  {train_file}: {len(runs)} runs")

# Split runs temporally for each file
print("\nSplitting runs temporally (80% train, 20% val)...")
train_runs_split = {}
val_runs_split = {}
for train_file, runs in train_runs_by_file.items():
    train_runs, val_runs = split_runs_temporally(runs, train_ratio=0.8)
    train_runs_split[train_file] = train_runs
    val_runs_split[train_file] = val_runs
    print(f"  {train_file}: {len(train_runs)} train runs, {len(val_runs)} val runs")

# Generate sequences for each gap length
for gap_length in GAP_LENGTHS:
    print("\n" + "=" * 70)
    print(f"GAP LENGTH: {gap_length}")
    print("=" * 70)
    
    gap_dir = GAP_BENCHMARK_BASE / f"gap_{gap_length}"
    gap_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy scaler
    scaler_path = gap_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler to {scaler_path}")
    
    manifest = {
        "config": {
            "context_length": CONTEXT_LENGTH,
            "gap_length": gap_length,
            "stride": STRIDE
        },
        "sensor_columns": SENSOR_COLUMNS,
        "files": {}
    }
    
    # Generate training sequences
    print("\nGenerating sequences for training files...")
    for train_file in TRAIN_FILES:
        df = train_dfs[train_file]
        train_runs = train_runs_split[train_file]
        val_runs = val_runs_split[train_file]
        
        train_data = generate_sequences_within_runs(
            df, train_runs, scaler,
            CONTEXT_LENGTH, gap_length, STRIDE
        )
        
        val_data = generate_sequences_within_runs(
            df, val_runs, scaler,
            CONTEXT_LENGTH, gap_length, STRIDE
        )
        
        train_filename = f"train_{train_file.replace('.csv', '.npz')}"
        train_path = gap_dir / train_filename
        save_npz(train_data, train_path)
        
        val_filename = f"val_{train_file.replace('.csv', '.npz')}"
        val_path = gap_dir / val_filename
        save_npz(val_data, val_path)
        
        manifest["files"][train_file] = {
            "type": "train",
            "train_sequences": len(train_data["context_before"]),
            "val_sequences": len(val_data["context_before"]),
            "train_runs": len(train_runs),
            "val_runs": len(val_runs)
        }
        
        print(f"  {train_file}:")
        print(f"    Train: {len(train_data['context_before']):,} sequences")
        print(f"    Val: {len(val_data['context_before']):,} sequences")
    
    # Generate test sequences
    print("\nGenerating sequences for test files...")
    for test_file in TEST_FILES:
        file_path = DATASET_PATH / "test" / test_file
        df = load_and_preprocess_file(file_path)
        print(f"  Loaded {test_file}: {len(df):,} rows")
        
        runs = identify_runs(df)
        print(f"    {len(runs)} runs")
        
        test_data = generate_sequences_within_runs(
            df, runs, scaler,
            CONTEXT_LENGTH, gap_length, STRIDE
        )
        
        test_filename = f"test_{test_file.replace('.csv', '.npz')}"
        test_path = gap_dir / test_filename
        save_npz(test_data, test_path)
        
        manifest["files"][test_file] = {
            "type": "test",
            "test_sequences": len(test_data["context_before"]),
            "test_runs": len(runs)
        }
        
        print(f"    Test: {len(test_data['context_before']):,} sequences")
    
    # Save manifest
    config_path = gap_dir / "manifest.json"
    with open(config_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest to {config_path}")

print("\n" + "=" * 70)
print("BENCHMARK CREATION COMPLETE")
print("=" * 70)
print(f"Benchmarks saved to: {GAP_BENCHMARK_BASE}")
print()
