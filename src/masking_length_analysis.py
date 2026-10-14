"""
Analyze performance by masking/gap length.
Test baseline across different gap sizes.
"""

import numpy as np
import json
import csv
from pathlib import Path
from typing import List, Dict
import sys

sys.path.append("src")
from data_loader import PHM2018Loader
from baseline_model import InterpolationBaseline
from experiment_generator import MaskedSequenceGenerator


def test_baseline_by_gap_length():
    """Test baseline performance across different gap lengths."""
    print("Testing baseline across different gap lengths...")
    
    # Load data
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    train_sensors = preprocessed["train_sensors"]
    test_sensors = preprocessed["test_sensors"]
    
    # Test different gap lengths
    gap_lengths = [5, 10, 15, 20, 30, 50]
    context_length = 50
    stride = 200
    
    results = []
    
    for gap_length in gap_lengths:
        print(f"\nTesting gap_length={gap_length}")
        
        # Generate sequences
        generator = MaskedSequenceGenerator(
            context_length=context_length,
            gap_length=gap_length,
            stride=stride
        )
        
        test_sequences = generator.generate_sequences(test_sensors, split="test")
        
        if len(test_sequences) == 0:
            print(f"  No sequences generated for gap_length={gap_length}")
            continue
        
        # Evaluate baseline
        baseline = InterpolationBaseline()
        predictions = []
        ground_truths = []
        
        for seq in test_sequences:
            pred = baseline.predict(seq["context_before"], seq["context_after"], gap_length)
            predictions.append(pred)
            ground_truths.append(seq["masked_segment"])
        
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        
        mse = np.mean((predictions - ground_truths) ** 2)
        mae = np.mean(np.abs(predictions - ground_truths))
        
        result = {
            "gap_length": gap_length,
            "n_sequences": len(test_sequences),
            "mse": float(mse),
            "mae": float(mae)
        }
        results.append(result)
        
        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, Sequences: {len(test_sequences)}")
    
    # Save results
    with open("results/masking_length_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save CSV
    with open("results/masking_length_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gap_length", "n_sequences", "mse", "mae"])
        for r in results:
            writer.writerow([r["gap_length"], r["n_sequences"], r["mse"], r["mae"]])
    
    print(f"\nSaved masking length analysis")
    return results


if __name__ == "__main__":
    results = test_baseline_by_gap_length()
    
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE BY GAP LENGTH")
    print("="*60)
    print(f"{'Gap Length':<12} {'Sequences':<12} {'MSE':<15} {'MAE':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['gap_length']:<12} {r['n_sequences']:<12} {r['mse']:<15.6f} {r['mae']:<15.6f}")
