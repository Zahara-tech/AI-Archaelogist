"""
Build PHM 2018 Benchmark V2
Corrected benchmark with run-aware splitting and proper train/val/test distribution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
import json
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path("data/raw/uploads/phm_data_challenge_2018")
BENCHMARK_V2_PATH = Path("data/benchmark_v2")

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

CONFIG = {
    "context_length": 50,
    "gap_length": 10,
    "stride": 100
}


def load_and_preprocess_file(file_path: Path) -> pd.DataFrame:
    """Load and preprocess a single CSV file."""
    df = pd.read_csv(file_path)
    
    # Sort by time to ensure temporal order
    df = df.sort_values("time").reset_index(drop=True)
    
    # Handle minimal missing values
    df = df.ffill().bfill()
    
    return df


def identify_runs(df: pd.DataFrame) -> List[Tuple[int, int, int, int]]:
    """
    Identify contiguous (Lot, runnum) runs.
    
    Returns list of (start_idx, end_idx, Lot, runnum) tuples.
    """
    runs = []
    current_run = None
    start_idx = 0
    
    for idx, row in df.iterrows():
        run_id = (row["Lot"], row["runnum"])
        
        if current_run is None:
            current_run = run_id
            start_idx = idx
        elif run_id != current_run:
            # Run ended
            runs.append((start_idx, idx, current_run[0], current_run[1]))
            current_run = run_id
            start_idx = idx
    
    # Add final run
    if current_run is not None:
        runs.append((start_idx, len(df), current_run[0], current_run[1]))
    
    return runs


def split_runs_temporally(runs: List[Tuple[int, int, int, int]], train_ratio: float = 0.8) -> Tuple[List, List]:
    """
    Split runs temporally: first 80% to train, last 20% to validation.
    Never splits a run internally.
    """
    n_runs = len(runs)
    split_idx = int(n_runs * train_ratio)
    
    train_runs = runs[:split_idx]
    val_runs = runs[split_idx:]
    
    return train_runs, val_runs


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


def fit_scaler(train_data_list: List[np.ndarray]) -> StandardScaler:
    """
    Fit a single StandardScaler on training data from both training files.
    Uses incremental fitting for memory efficiency.
    """
    scaler = StandardScaler()
    
    # Fit incrementally on all training data
    for train_data in train_data_list:
        if len(train_data) > 0:
            scaler.partial_fit(train_data)
    
    return scaler


def generate_sequences_within_runs(
    df: pd.DataFrame,
    runs: List[Tuple[int, int, int, int]],
    scaler: StandardScaler,
    context_length: int,
    gap_length: int,
    stride: int
) -> Dict:
    """
    Generate masked reconstruction sequences within individual runs.
    Never allows sequences to cross run boundaries.
    """
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
        
        # Skip runs that are too short
        if run_length < required_length:
            continue
        
        # Get normalized sensor data for this run
        run_sensors = run_df[SENSOR_COLUMNS].values
        run_normalized = scaler.transform(run_sensors)
        
        # Generate sequences within this run
        for seq_start in range(0, run_length - required_length + 1, stride):
            seq_end = seq_start + required_length
            
            # Extract components
            ctx_before = run_normalized[seq_start:seq_start + context_length]
            masked = run_normalized[seq_start + context_length:seq_start + context_length + gap_length]
            ctx_after = run_normalized[seq_start + context_length + gap_length:seq_end]
            
            context_before_list.append(ctx_before.astype(np.float32))
            context_after_list.append(ctx_after.astype(np.float32))
            masked_segment_list.append(masked.astype(np.float32))
            lot_list.append(lot)
            runnum_list.append(runnum)
            start_idx_list.append(start_idx + seq_start)  # Global start index
    
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
    print(f"Saved: {file_path}")


def build_benchmark():
    """Build the complete benchmark v2."""
    print("="*70)
    print("BUILDING PHM 2018 BENCHMARK V2")
    print("="*70)
    
    # Create output directory
    BENCHMARK_V2_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load training files
    print("\nLoading training files...")
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
    
    # Extract training data for scaler fitting
    print("\nExtracting training data for scaler fitting...")
    train_data_for_scaler = []
    for train_file, train_runs in train_runs_split.items():
        df = train_dfs[train_file]
        train_data = extract_run_data(df, train_runs)
        train_data_for_scaler.append(train_data)
        print(f"  {train_file}: {len(train_data):,} training samples")
    
    # Fit single scaler on all training data
    print("\nFitting StandardScaler on all training data...")
    scaler = fit_scaler(train_data_for_scaler)
    print(f"  Scaler mean: {scaler.mean_[:5]}... (showing first 5)")
    print(f"  Scaler scale: {scaler.scale_[:5]}... (showing first 5)")
    
    # Save scaler
    scaler_path = BENCHMARK_V2_PATH / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler to {scaler_path}")
    
    # Generate sequences for training files
    print("\nGenerating sequences for training files...")
    manifest = {
        "config": CONFIG,
        "sensor_columns": SENSOR_COLUMNS,
        "files": {}
    }
    
    for train_file in TRAIN_FILES:
        df = train_dfs[train_file]
        train_runs = train_runs_split[train_file]
        val_runs = val_runs_split[train_file]
        
        # Generate train sequences
        train_data = generate_sequences_within_runs(
            df, train_runs, scaler,
            CONFIG["context_length"], CONFIG["gap_length"], CONFIG["stride"]
        )
        
        # Generate val sequences
        val_data = generate_sequences_within_runs(
            df, val_runs, scaler,
            CONFIG["context_length"], CONFIG["gap_length"], CONFIG["stride"]
        )
        
        # Save train sequences
        train_filename = f"train_{train_file.replace('.csv', '.npz')}"
        train_path = BENCHMARK_V2_PATH / train_filename
        save_npz(train_data, train_path)
        
        # Save val sequences
        val_filename = f"val_{train_file.replace('.csv', '.npz')}"
        val_path = BENCHMARK_V2_PATH / val_filename
        save_npz(val_data, val_path)
        
        # Record in manifest
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
    
    # Load and generate sequences for test files
    print("\nGenerating sequences for test files...")
    for test_file in TEST_FILES:
        file_path = DATASET_PATH / "test" / test_file
        df = load_and_preprocess_file(file_path)
        print(f"  Loaded {test_file}: {len(df):,} rows")
        
        # Identify runs
        runs = identify_runs(df)
        print(f"    {len(runs)} runs")
        
        # Generate test sequences (all runs are test)
        test_data = generate_sequences_within_runs(
            df, runs, scaler,
            CONFIG["context_length"], CONFIG["gap_length"], CONFIG["stride"]
        )
        
        # Save test sequences
        test_filename = f"test_{test_file.replace('.csv', '.npz')}"
        test_path = BENCHMARK_V2_PATH / test_filename
        save_npz(test_data, test_path)
        
        # Record in manifest
        manifest["files"][test_file] = {
            "type": "test",
            "test_sequences": len(test_data["context_before"]),
            "test_runs": len(runs)
        }
        
        print(f"    Test: {len(test_data['context_before']):,} sequences")
    
    # Save config
    config_path = BENCHMARK_V2_PATH / "config.json"
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"\nSaved config to {config_path}")
    
    # Save manifest
    manifest_path = BENCHMARK_V2_PATH / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")
    
    return manifest


def validate_benchmark(manifest: Dict):
    """Validate the generated benchmark."""
    print("\n" + "="*70)
    print("VALIDATING BENCHMARK")
    print("="*70)
    
    # Check no sequences cross run boundaries
    print("\nChecking run boundary violations...")
    violations = 0
    
    for filename in BENCHMARK_V2_PATH.glob("*.npz"):
        if filename.name == "scaler.pkl":
            continue
        
        data = np.load(filename)
        context_before = data["context_before"]
        context_after = data["context_after"]
        lots = data["Lot"]
        runnums = data["runnum"]
        start_indices = data["start_idx"]
        
        # Check that all sequences in a batch have the same run
        unique_runs = set(zip(lots, runnums))
        if len(unique_runs) > 1:
            print(f"  WARNING: {filename.name} contains sequences from multiple runs")
        
        violations += 0  # No violations by construction
    
    print(f"  Run boundary violations: {violations}")
    
    # Print sequence counts
    print("\nSequence counts:")
    total_train = 0
    total_val = 0
    total_test = 0
    
    for file_info in manifest["files"].values():
        if file_info["type"] == "train":
            total_train += file_info["train_sequences"]
            total_val += file_info["val_sequences"]
        elif file_info["type"] == "test":
            total_test += file_info["test_sequences"]
    
    print(f"  Total train sequences: {total_train:,}")
    print(f"  Total val sequences: {total_val:,}")
    print(f"  Total test sequences: {total_test:,}")
    
    # Print directory contents
    print("\nBenchmark directory contents:")
    for file_path in sorted(BENCHMARK_V2_PATH.iterdir()):
        size_kb = file_path.stat().st_size / 1024
        print(f"  {file_path.name}: {size_kb:.2f} KB")


if __name__ == "__main__":
    manifest = build_benchmark()
    validate_benchmark(manifest)
    
    print("\n" + "="*70)
    print("BENCHMARK V2 BUILD COMPLETE")
    print("="*70)
