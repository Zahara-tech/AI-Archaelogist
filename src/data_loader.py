"""
PHM 2018 Data Loader Module
Reproducible loading of sensor time-series data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import pickle

DATASET_PATH = Path("data/raw/uploads/phm_data_challenge_2018")
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


class PHM2018Loader:
    """Loads and preprocesses PHM 2018 dataset."""
    
    def __init__(self, dataset_path: Path = DATASET_PATH):
        self.dataset_path = dataset_path
        self.train_files = []
        self.test_files = []
        self._discover_files()
    
    def _discover_files(self):
        """Discover train and test files."""
        self.train_files = list((self.dataset_path / "train").glob("*.csv"))
        self.test_files = list((self.dataset_path / "test").glob("*.csv"))
        print(f"Found {len(self.train_files)} train files, {len(self.test_files)} test files")
    
    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single CSV file."""
        df = pd.read_csv(file_path)
        
        # Sort by time to ensure temporal order
        df = df.sort_values("time").reset_index(drop=True)
        
        # Handle minimal missing values
        df = df.ffill().bfill()
        
        return df
    
    def load_all_data(self) -> List[pd.DataFrame]:
        """Load all train and test data as separate DataFrames to avoid memory issues."""
        train_dfs = []
        for file_path in self.train_files:
            df = self.load_file(file_path)
            df["source_file"] = file_path.name
            train_dfs.append(df)
        
        test_dfs = []
        for file_path in self.test_files:
            df = self.load_file(file_path)
            df["source_file"] = file_path.name
            test_dfs.append(df)
        
        total_train_rows = sum(len(df) for df in train_dfs)
        total_test_rows = sum(len(df) for df in test_dfs)
        
        print(f"Loaded {total_train_rows:,} train rows in {len(train_dfs)} files")
        print(f"Loaded {total_test_rows:,} test rows in {len(test_dfs)} files")
        
        return train_dfs, test_dfs
    
    def get_sensor_data(self, df: pd.DataFrame) -> np.ndarray:
        """Extract sensor columns as numpy array."""
        return df[SENSOR_COLUMNS].values
    
    def get_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract metadata columns."""
        return df[METADATA_COLUMNS].copy()
    
    def normalize_sensors(self, data: np.ndarray, fit: bool = True, scaler=None):
        """Normalize sensor data using StandardScaler."""
        from sklearn.preprocessing import StandardScaler
        
        if fit:
            scaler = StandardScaler()
            normalized = scaler.fit_transform(data)
        else:
            if scaler is None:
                raise ValueError("scaler must be provided when fit=False")
            normalized = scaler.transform(data)
        
        return normalized, scaler
    
    def save_preprocessed(self, data: Dict, save_path: Path):
        """Save preprocessed data."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved preprocessed data to {save_path}")
    
    def load_preprocessed(self, load_path: Path) -> Dict:
        """Load preprocessed data."""
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded preprocessed data from {load_path}")
        return data


if __name__ == "__main__":
    # Test the loader with a single file for prototyping
    loader = PHM2018Loader()
    train_dfs, test_dfs = loader.load_all_data()
    
    # Use first train file for prototyping
    train_data = train_dfs[0]
    test_data = test_dfs[0]
    
    print(f"\nUsing subset for prototyping:")
    print(f"  Train: {len(train_data):,} rows from {train_data['source_file'].iloc[0]}")
    print(f"  Test: {len(test_data):,} rows from {test_data['source_file'].iloc[0]}")
    
    # Extract sensor data
    train_sensors = loader.get_sensor_data(train_data)
    test_sensors = loader.get_sensor_data(test_data)
    
    print(f"\nSensor data shape:")
    print(f"  Train: {train_sensors.shape}")
    print(f"  Test: {test_sensors.shape}")
    
    # Normalize
    train_normalized, scaler = loader.normalize_sensors(train_sensors, fit=True)
    test_normalized, _ = loader.normalize_sensors(test_sensors, fit=False, scaler=scaler)
    
    print(f"\nNormalized data shape:")
    print(f"  Train: {train_normalized.shape}")
    print(f"  Test: {test_normalized.shape}")
    
    # Save only essential data to avoid memory issues
    preprocessed = {
        "train_sensors": train_normalized,
        "test_sensors": test_normalized,
        "scaler": scaler,
        "sensor_columns": SENSOR_COLUMNS,
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "train_file": train_data['source_file'].iloc[0],
        "test_file": test_data['source_file'].iloc[0]
    }
    
    loader.save_preprocessed(preprocessed, Path("data/preprocessed/phm2018_subset.pkl"))
