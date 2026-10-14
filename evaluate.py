"""
Evaluation script for trained reconstruction models.
Usage: python evaluate.py --model lstm --checkpoint models/best_model.pt
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.append("src")
from data_loader import PHM2018Loader
from experiment_generator import MaskedSequenceGenerator
from baseline_model import InterpolationBaseline
from lstm_model import LSTMReconstructor, LSTMModelTrainer, ReconstructionDataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction model")
    parser.add_argument("--model", type=str, choices=["baseline", "lstm"], default="lstm",
                        help="Model type to evaluate")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    return parser.parse_args()


def evaluate_baseline(args):
    """Evaluate baseline model."""
    print("Evaluating baseline model...")
    
    # Load sequences
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    baseline = InterpolationBaseline()
    metrics = baseline.evaluate_batch(test_sequences)
    
    print(f"\nBaseline Results:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Sequences: {metrics['n_sequences']}")
    
    # Save predictions
    baseline.save_predictions(test_sequences, Path("results/baseline_predictions.pkl"))
    
    return metrics


def evaluate_lstm(args):
    """Evaluate LSTM model."""
    print("Evaluating LSTM model...")
    
    # Load sequences
    generator = MaskedSequenceGenerator()
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    # Load config
    with open("data/sequences/config.json", "r") as f:
        config = json.load(f)
    
    # Create dataset
    test_dataset = ReconstructionDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get dimensions
    sample = test_sequences[0]
    input_dim = sample["context_before"].shape[1]
    output_dim = input_dim
    output_length = config["gap_length"]
    
    # Create model
    model = LSTMReconstructor(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        output_length=output_length,
        dropout=0.3
    )
    
    # Load checkpoint
    trainer = LSTMModelTrainer(model)
    trainer.load_checkpoint(args.checkpoint)
    
    # Evaluate
    metrics = trainer.evaluate(test_loader)
    
    print(f"\nLSTM Test Results:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    
    # Save metrics
    metrics_path = Path("results/lstm_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved metrics to {metrics_path}")
    
    return metrics


def compare_models():
    """Compare baseline and LSTM models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Load baseline metrics
    baseline_path = Path("results/baseline_metrics.json")
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            baseline_metrics = json.load(f)
        print(f"\nBaseline:")
        print(f"  MSE: {baseline_metrics['mse']:.6f}")
        print(f"  MAE: {baseline_metrics['mae']:.6f}")
    else:
        print("Baseline metrics not found. Run with --model baseline first.")
        baseline_metrics = None
    
    # Load LSTM metrics
    lstm_path = Path("results/lstm_metrics.json")
    if lstm_path.exists():
        with open(lstm_path, "r") as f:
            lstm_metrics = json.load(f)
        print(f"\nLSTM:")
        print(f"  MSE: {lstm_metrics['mse']:.6f}")
        print(f"  MAE: {lstm_metrics['mae']:.6f}")
    else:
        print("LSTM metrics not found. Run with --model lstm first.")
        lstm_metrics = None
    
    # Compare
    if baseline_metrics and lstm_metrics:
        print(f"\nComparison:")
        mse_improvement = (baseline_metrics['mse'] - lstm_metrics['mse']) / baseline_metrics['mse'] * 100
        mae_improvement = (baseline_metrics['mae'] - lstm_metrics['mae']) / baseline_metrics['mae'] * 100
        
        if mse_improvement > 0:
            print(f"  MSE improvement: {mse_improvement:.2f}% (LSTM better)")
        else:
            print(f"  MSE improvement: {abs(mse_improvement):.2f}% (Baseline better)")
        
        if mae_improvement > 0:
            print(f"  MAE improvement: {mae_improvement:.2f}% (LSTM better)")
        else:
            print(f"  MAE improvement: {abs(mae_improvement):.2f}% (Baseline better)")
        
        # Save comparison
        comparison = {
            "baseline_mse": baseline_metrics['mse'],
            "baseline_mae": baseline_metrics['mae'],
            "lstm_mse": lstm_metrics['mse'],
            "lstm_mae": lstm_metrics['mae'],
            "mse_improvement_pct": mse_improvement,
            "mae_improvement_pct": mae_improvement
        }
        
        comparison_path = Path("results/model_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nSaved comparison to {comparison_path}")


if __name__ == "__main__":
    args = parse_args()
    
    if args.model == "baseline":
        evaluate_baseline(args)
    elif args.model == "lstm":
        evaluate_lstm(args)
    
    # Always show comparison if both metrics exist
    compare_models()
