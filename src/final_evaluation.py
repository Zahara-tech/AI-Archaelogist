"""
Final evaluation comparing all three methods.
"""

import json
from pathlib import Path


def compare_all_methods():
    """Compare baseline, LSTM, and BiLSTM results."""
    
    # Load all metrics
    with open("results/baseline_metrics.json", "r") as f:
        baseline = json.load(f)
    
    with open("results/lstm_metrics.json", "r") as f:
        lstm = json.load(f)
    
    with open("results/bilstm_metrics.json", "r") as f:
        bilstm = json.load(f)
    
    # Load per-sensor metrics
    with open("results/baseline_per_sensor.json", "r") as f:
        baseline_per_sensor = json.load(f)["per_sensor_metrics"]
    
    with open("results/lstm_per_sensor.json", "r") as f:
        lstm_per_sensor = json.load(f)["per_sensor_metrics"]
    
    # Load masking length analysis
    with open("results/masking_length_metrics.json", "r") as f:
        masking_length = json.load(f)
    
    # Comprehensive comparison
    comparison = {
        "dataset": "PHM 2018 Ion Mill Etch Tool",
        "task": "Sensor Value Reconstruction",
        "gap_length": 10,
        "context_length": 50,
        "n_test_sequences": 17035,
        
        "overall_metrics": {
            "baseline": {
                "mse": baseline["mse"],
                "mae": baseline["mae"]
            },
            "lstm": {
                "mse": lstm["mse"],
                "mae": lstm["mae"]
            },
            "bilstm": {
                "mse": bilstm["mse"],
                "mae": bilstm["mae"]
            }
        },
        
        "relative_performance": {
            "lstm_vs_baseline_mse_ratio": lstm["mse"] / baseline["mse"],
            "lstm_vs_baseline_mae_ratio": lstm["mae"] / baseline["mae"],
            "bilstm_vs_baseline_mse_ratio": bilstm["mse"] / baseline["mse"],
            "bilstm_vs_baseline_mae_ratio": bilstm["mae"] / baseline["mae"],
            "bilstm_vs_lstm_mse_ratio": bilstm["mse"] / lstm["mse"],
            "bilstm_vs_lstm_mae_ratio": bilstm["mae"] / lstm["mae"]
        },
        
        "masking_length_analysis": masking_length,
        
        "per_sensor_comparison": {}
    }
    
    # Per-sensor comparison
    for sensor in baseline_per_sensor.keys():
        comparison["per_sensor_comparison"][sensor] = {
            "baseline_mse": baseline_per_sensor[sensor]["mse"],
            "lstm_mse": lstm_per_sensor[sensor]["mse"],
            "lstm_worse_by": lstm_per_sensor[sensor]["mse"] / baseline_per_sensor[sensor]["mse"]
        }
    
    # Save comprehensive comparison
    with open("results/final_model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print("="*70)
    print("FINAL MODEL COMPARISON")
    print("="*70)
    print(f"\nOverall Metrics:")
    print(f"{'Method':<15} {'MSE':<15} {'MAE':<15} {'vs Baseline MSE'}")
    print("-" * 70)
    print(f"{'Baseline':<15} {baseline['mse']:<15.6f} {baseline['mae']:<15.6f} {'1.00x'}")
    print(f"{'LSTM':<15} {lstm['mse']:<15.6f} {lstm['mae']:<15.6f} {lstm['mse']/baseline['mse']:.2f}x worse")
    print(f"{'BiLSTM':<15} {bilstm['mse']:<15.6f} {bilstm['mae']:<15.6f} {bilstm['mse']/baseline['mse']:.2f}x worse")
    
    print(f"\nBiLSTM vs LSTM:")
    print(f"  MSE improvement: {(1 - bilstm['mse']/lstm['mse'])*100:.1f}%")
    print(f"  MAE improvement: {(1 - bilstm['mae']/lstm['mae'])*100:.1f}%")
    
    print(f"\nMasking Length Analysis (Baseline):")
    print(f"{'Gap Length':<12} {'MSE':<15} {'MAE':<15}")
    print("-" * 45)
    for entry in masking_length:
        print(f"{entry['gap_length']:<12} {entry['mse']:<15.6f} {entry['mae']:<15.6f}")
    
    return comparison


if __name__ == "__main__":
    comparison = compare_all_methods()
    print(f"\nSaved comprehensive comparison to results/final_model_comparison.json")
