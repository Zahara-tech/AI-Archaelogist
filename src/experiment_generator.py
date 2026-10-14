"""
Experiment Generator for Missing-Event Reconstruction
Creates masked sequences with ground truth for evaluation.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import pickle
import json


class MaskedSequenceGenerator:
    """Generates masked sequences for reconstruction task."""
    
    def __init__(
        self,
        context_length: int = 100,  # timesteps before and after gap
        gap_length: int = 20,  # timesteps to mask
        stride: int = 150,  # stride between sequences
        min_sequence_length: int = 300  # minimum total sequence length
    ):
        self.context_length = context_length
        self.gap_length = gap_length
        self.stride = stride
        self.min_sequence_length = min_sequence_length
        self.required_length = context_length * 2 + gap_length
    
    def generate_sequences(
        self,
        data: np.ndarray,
        split: str = "train"
    ) -> List[Dict]:
        """
        Generate masked sequences from sensor data.
        
        Args:
            data: Normalized sensor data (n_timesteps, n_sensors)
            split: "train", "val", or "test"
            
        Returns:
            List of sequence dictionaries with:
            - context_before: sensor data before gap
            - context_after: sensor data after gap
            - masked_segment: ground truth (hidden during prediction)
            - indices: original indices in data
        """
        sequences = []
        n_timesteps = data.shape[0]
        
        # Generate sequences with stride
        for start_idx in range(0, n_timesteps - self.required_length, self.stride):
            end_idx = start_idx + self.required_length
            
            if end_idx > n_timesteps:
                break
            
            # Extract segments
            context_before = data[start_idx:start_idx + self.context_length]
            masked_segment = data[start_idx + self.context_length:start_idx + self.context_length + self.gap_length]
            context_after = data[start_idx + self.context_length + self.gap_length:end_idx]
            
            sequence = {
                "context_before": context_before,
                "context_after": context_after,
                "masked_segment": masked_segment,
                "indices": (start_idx, end_idx),
                "split": split
            }
            sequences.append(sequence)
        
        print(f"Generated {len(sequences)} sequences for {split} split")
        return sequences
    
    def create_train_val_test_split(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        val_ratio: float = 0.2
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create train/validation/test splits without temporal leakage.
        
        Test data is from separate files (no leakage).
        Validation is split from training data using temporal split.
        """
        # Generate test sequences from test data
        test_sequences = self.generate_sequences(test_data, split="test")
        
        # Temporal split for train/val from training data
        split_idx = int(len(train_data) * (1 - val_ratio))
        
        train_data_split = train_data[:split_idx]
        val_data_split = train_data[split_idx:]
        
        train_sequences = self.generate_sequences(train_data_split, split="train")
        val_sequences = self.generate_sequences(val_data_split, split="val")
        
        print(f"\nSplit summary:")
        print(f"  Train: {len(train_sequences)} sequences")
        print(f"  Val: {len(val_sequences)} sequences")
        print(f"  Test: {len(test_sequences)} sequences")
        
        return train_sequences, val_sequences, test_sequences
    
    def save_sequences(self, sequences: List[Dict], save_path: Path):
        """Save generated sequences."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(sequences, f)
        print(f"Saved {len(sequences)} sequences to {save_path}")
    
    def load_sequences(self, load_path: Path) -> List[Dict]:
        """Load generated sequences."""
        with open(load_path, "rb") as f:
            sequences = pickle.load(f)
        print(f"Loaded {len(sequences)} sequences from {load_path}")
        return sequences
    
    def save_config(self, save_path: Path):
        """Save generator configuration."""
        config = {
            "context_length": self.context_length,
            "gap_length": self.gap_length,
            "stride": self.stride,
            "min_sequence_length": self.min_sequence_length,
            "required_length": self.required_length
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {save_path}")


if __name__ == "__main__":
    # Load preprocessed data
    import sys
    sys.path.append(".")
    from data_loader import PHM2018Loader
    
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    
    train_sensors = preprocessed["train_sensors"]
    test_sensors = preprocessed["test_sensors"]
    
    print(f"\nLoaded preprocessed data:")
    print(f"  Train: {train_sensors.shape}")
    print(f"  Test: {test_sensors.shape}")
    
    # Create generator
    generator = MaskedSequenceGenerator(
        context_length=50,
        gap_length=10,
        stride=100
    )
    
    # Generate sequences
    train_seqs, val_seqs, test_seqs = generator.create_train_val_test_split(
        train_sensors,
        test_sensors,
        val_ratio=0.2
    )
    
    # Save sequences
    generator.save_sequences(train_seqs, Path("data/sequences/train_sequences.pkl"))
    generator.save_sequences(val_seqs, Path("data/sequences/val_sequences.pkl"))
    generator.save_sequences(test_seqs, Path("data/sequences/test_sequences.pkl"))
    
    # Save config
    generator.save_config(Path("data/sequences/config.json"))
    
    # Print sample sequence info
    if train_seqs:
        sample = train_seqs[0]
        print(f"\nSample sequence:")
        print(f"  Context before shape: {sample['context_before'].shape}")
        print(f"  Masked segment shape: {sample['masked_segment'].shape}")
        print(f"  Context after shape: {sample['context_after'].shape}")
