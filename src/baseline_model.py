"""
Baseline Reconstruction Model
Simple interpolation-based reconstruction for comparison.
"""

import numpy as np
from typing import List, Dict
from pathlib import Path
import pickle
import json


class InterpolationBaseline:
    """Linear interpolation baseline for sensor reconstruction."""
    
    def __init__(self):
        self.name = "Linear Interpolation Baseline"
    
    def predict(self, context_before: np.ndarray, context_after: np.ndarray, gap_length: int) -> np.ndarray:
        """
        Predict masked segment using linear interpolation.
        
        Args:
            context_before: (context_length, n_sensors)
            context_after: (context_length, n_sensors)
            gap_length: number of timesteps to predict
            
        Returns:
            Predicted segment (gap_length, n_sensors)
        """
        # Get last value before gap and first value after gap
        last_before = context_before[-1]  # (n_sensors,)
        first_after = context_after[0]  # (n_sensors,)
        
        # Linear interpolation for each sensor
        n_sensors = context_before.shape[1]
        
        predictions = np.zeros((gap_length, n_sensors))
        
        for i in range(gap_length):
            alpha = (i + 1) / (gap_length + 1)
            predictions[i] = last_before * (1 - alpha) + first_after * alpha
        
        return predictions
    
    def evaluate_batch(self, sequences: List[Dict]) -> Dict[str, float]:
        """
        Evaluate baseline on a batch of sequences.
        
        Args:
            sequences: List of sequence dicts with context_before, context_after, masked_segment
            
        Returns:
            Dictionary of metrics
        """
        predictions = []
        ground_truths = []
        
        for seq in sequences:
            gap_length = seq["masked_segment"].shape[0]
            pred = self.predict(seq["context_before"], seq["context_after"], gap_length)
            predictions.append(pred)
            ground_truths.append(seq["masked_segment"])
        
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        
        # Calculate metrics
        mse = np.mean((predictions - ground_truths) ** 2)
        mae = np.mean(np.abs(predictions - ground_truths))
        
        # Per-sensor metrics
        sensor_mse = np.mean((predictions - ground_truths) ** 2, axis=(0, 1))
        sensor_mae = np.mean(np.abs(predictions - ground_truths), axis=(0, 1))
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "sensor_mse": sensor_mse.tolist(),
            "sensor_mae": sensor_mae.tolist(),
            "n_sequences": len(sequences)
        }
        
        return metrics
    
    def save_predictions(self, sequences: List[Dict], save_path: Path):
        """Save predictions for analysis."""
        predictions = []
        ground_truths = []
        
        for seq in sequences:
            gap_length = seq["masked_segment"].shape[0]
            pred = self.predict(seq["context_before"], seq["context_after"], gap_length)
            predictions.append(pred)
            ground_truths.append(seq["masked_segment"])
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump({
                "predictions": predictions,
                "ground_truths": ground_truths
            }, f)
        
        print(f"Saved predictions to {save_path}")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from experiment_generator import MaskedSequenceGenerator
    
    # Load test sequences
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    # Create baseline
    baseline = InterpolationBaseline()
    
    # Evaluate
    print(f"Evaluating {baseline.name} on {len(test_sequences)} test sequences...")
    metrics = baseline.evaluate_batch(test_sequences)
    
    print(f"\nBaseline Results:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Sequences: {metrics['n_sequences']}")
    
    # Save metrics
    metrics_path = Path("results/baseline_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save predictions
    baseline.save_predictions(test_sequences, Path("results/baseline_predictions.pkl"))
