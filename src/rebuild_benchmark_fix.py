"""
Rebuild benchmark v2 with verified normalization
This script ensures the scaler is correctly fitted and applied to all data
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
    df = pd.read_csv(file_path)
    df = df.sort_values("time").reset_index(drop=True)
    df = df.ffill().bfill()
    return df

def identify_runs(df: pd.DataFrame) -> List[Tuple[int, int, int, int]]:
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
    n_runs = len(runs)
    split_idx = int(n_runs * train_ratio)
    return runs[:split_idx], runs[split_idx:]

def extract_run_data(df: pd.DataFrame, runs: List[Tuple[int, int, int, int]]) -> np.ndarray:
    all_data = []
    for start_idx, end_idx, _, _ in runs:
        run_data = df.iloc[start_idx:end_idx][SENSOR_COLUMNS].values
        all_data.append(run_data)
    return np.vstack(all_data) if all_data else np.array([]).reshape(0, len(SENSOR_COLUMNS))

def fit_scaler(train_data_list: List[np.ndarray]) -> StandardScaler:
    scaler = StandardScaler()
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
        
        # Get raw sensor data for this run
        run_sensors = run_df[SENSOR_COLUMNS].values
        
        # Normalize the entire run
        run_normalized = scaler.transform(run_sensors)
        
        # Generate sequences within this run
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
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file_path, **data)
    print(f"Saved: {file_path}")

def verify_normalization(data_dict: Dict, name: str):
    """Verify that data is properly normalized (mean≈0, std≈1)"""
    context_before = data_dict["context_before"]
    context_after = data_dict["context_after"]
    masked_segment = data_dict["masked_segment"]
    
    print(f"\n{name} normalization check:")
    print(f"  context_before: mean={context_before.mean():.6f}, std={context_before.std():.6f}")
    print(f"  context_after: mean={context_after.mean():.6f}, std={context_after.std():.6f}")
    print(f"  masked_segment: mean={masked_segment.mean():.6f}, std={masked_segment.std():.6f}")
    
    # Check if close to standard normal
    if abs(context_before.mean()) > 0.1 or abs(context_before.std() - 1.0) > 0.1:
        print(f"  WARNING: context_before not properly normalized!")
    if abs(masked_segment.mean()) > 0.1 or abs(masked_segment.std() - 1.0) > 0.1:
        print(f"  WARNING: masked_segment not properly normalized!")

def rebuild_benchmark():
    print("="*70)
    print("REBUILDING PHM 2018 BENCHMARK V2 WITH VERIFIED NORMALIZATION")
    print("="*70)
    
    BENCHMARK_V2_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load training files
    print("\nLoading training files...")
    train_dfs = {}
    for train_file in TRAIN_FILES:
        file_path = DATASET_PATH / "train" / train_file
        df = load_and_preprocess_file(file_path)
        train_dfs[train_file] = df
        print(f"  Loaded {train_file}: {len(df):,} rows")
    
    # Identify runs
    print("\nIdentifying runs in training files...")
    train_runs_by_file = {}
    for train_file, df in train_dfs.items():
        runs = identify_runs(df)
        train_runs_by_file[train_file] = runs
        print(f"  {train_file}: {len(runs)} runs")
    
    # Split runs
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
    
    # Fit scaler
    print("\nFitting StandardScaler on all training data...")
    scaler = fit_scaler(train_data_for_scaler)
    print(f"  Scaler mean (first 5): {scaler.mean_[:5]}")
    print(f"  Scaler scale (first 5): {scaler.scale_[:5]}")
    
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
        
        train_data = generate_sequences_within_runs(
            df, train_runs, scaler,
            CONFIG["context_length"], CONFIG["gap_length"], CONFIG["stride"]
        )
        
        val_data = generate_sequences_within_runs(
            df, val_runs, scaler,
            CONFIG["context_length"], CONFIG["gap_length"], CONFIG["stride"]
        )
        
        # Verify normalization
        verify_normalization(train_data, f"{train_file} train")
        verify_normalization(val_data, f"{train_file} val")
        
        train_filename = f"train_{train_file.replace('.csv', '.npz')}"
        train_path = BENCHMARK_V2_PATH / train_filename
        save_npz(train_data, train_path)
        
        val_filename = f"val_{train_file.replace('.csv', '.npz')}"
        val_path = BENCHMARK_V2_PATH / val_filename
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
            CONFIG["context_length"], CONFIG["gap_length"], CONFIG["stride"]
        )
        
        # Verify normalization
        verify_normalization(test_data, f"{test_file} test")
        
        test_filename = f"test_{test_file.replace('.csv', '.npz')}"
        test_path = BENCHMARK_V2_PATH / test_filename
        save_npz(test_data, test_path)
        
        manifest["files"][test_file] = {
            "type": "test",
            "test_sequences": len(test_data["context_before"]),
            "test_runs": len(runs)
        }
        
        print(f"    Test: {len(test_data['context_before']):,} sequences")
    
    # Save manifest
    manifest_path = BENCHMARK_V2_PATH / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest to {manifest_path}")
    
    print("\n" + "="*70)
    print("BENCHMARK V2 REBUILD COMPLETE")
    print("="*70)
    print("\nIMPORTANT: You must now RETRAIN the model on the corrected data.")
    print("Run: python src/train_attention_seq2seq_v2.py")

if __name__ == "__main__":
    rebuild_benchmark()
