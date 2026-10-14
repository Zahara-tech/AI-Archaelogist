"""
Training script for missing-event reconstruction models.
Usage: python train.py --model baseline --config config.json
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.append("src")
from data_loader import PHM2018Loader
from experiment_generator import MaskedSequenceGenerator
from baseline_model import InterpolationBaseline
from lstm_model import LSTMReconstructor, LSTMModelTrainer, ReconstructionDataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train reconstruction model")
    parser.add_argument("--model", type=str, choices=["baseline", "lstm"], default="lstm",
                        help="Model type to train")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="LSTM hidden dimension")
    parser.add_argument("--context_length", type=int, default=50, help="Context length before/after gap")
    parser.add_argument("--gap_length", type=int, default=10, help="Gap length to predict")
    parser.add_argument("--stride", type=int, default=100, help="Stride between sequences")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    return parser.parse_args()


def train_baseline(args):
    """Train baseline (no actual training, just evaluate)."""
    print("Training baseline model (evaluation only)...")
    
    # Load sequences
    generator = MaskedSequenceGenerator(
        context_length=args.context_length,
        gap_length=args.gap_length,
        stride=args.stride
    )
    
    # Check if sequences exist, otherwise generate
    train_path = Path("data/sequences/train_sequences.pkl")
    if not train_path.exists():
        print("Sequences not found. Generating...")
        loader = PHM2018Loader()
        preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
        train_sensors = preprocessed["train_sensors"]
        test_sensors = preprocessed["test_sensors"]
        
        train_seqs, val_seqs, test_seqs = generator.create_train_val_test_split(
            train_sensors, test_sensors, val_ratio=args.val_ratio
        )
        
        generator.save_sequences(train_seqs, train_path)
        generator.save_sequences(val_seqs, Path("data/sequences/val_sequences.pkl"))
        generator.save_sequences(test_seqs, Path("data/sequences/test_sequences.pkl"))
        generator.save_config(Path("data/sequences/config.json"))
    
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    baseline = InterpolationBaseline()
    metrics = baseline.evaluate_batch(test_sequences)
    
    print(f"\nBaseline Results:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    
    # Save metrics
    metrics_path = Path("results/baseline_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved metrics to {metrics_path}")


def train_lstm(args):
    """Train LSTM model."""
    print("Training LSTM model...")
    
    # Load or generate sequences
    generator = MaskedSequenceGenerator(
        context_length=args.context_length,
        gap_length=args.gap_length,
        stride=args.stride
    )
    
    train_path = Path("data/sequences/train_sequences.pkl")
    if not train_path.exists():
        print("Sequences not found. Generating...")
        loader = PHM2018Loader()
        preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
        train_sensors = preprocessed["train_sensors"]
        test_sensors = preprocessed["test_sensors"]
        
        train_seqs, val_seqs, test_seqs = generator.create_train_val_test_split(
            train_sensors, test_sensors, val_ratio=args.val_ratio
        )
        
        generator.save_sequences(train_seqs, train_path)
        generator.save_sequences(val_seqs, Path("data/sequences/val_sequences.pkl"))
        generator.save_sequences(test_seqs, Path("data/sequences/test_sequences.pkl"))
        generator.save_config(Path("data/sequences/config.json"))
    
    train_sequences = generator.load_sequences(train_path)
    val_sequences = generator.load_sequences(Path("data/sequences/val_sequences.pkl"))
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    # Create datasets
    train_dataset = ReconstructionDataset(train_sequences)
    val_dataset = ReconstructionDataset(val_sequences)
    test_dataset = ReconstructionDataset(test_sequences)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get dimensions
    sample = train_sequences[0]
    context_length = args.context_length * 2
    input_dim = sample["context_before"].shape[1]
    output_dim = input_dim
    output_length = args.gap_length
    
    print(f"\nModel configuration:")
    print(f"  Context length: {context_length}")
    print(f"  Input dim: {input_dim}")
    print(f"  Output length: {output_length}")
    print(f"  Output dim: {output_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Create model
    model = LSTMReconstructor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        output_dim=output_dim,
        output_length=output_length,
        dropout=0.3
    )
    
    # Train
    trainer = LSTMModelTrainer(model, learning_rate=args.learning_rate)
    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=5)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"\nTest Results:")
    print(f"  MSE: {test_metrics['mse']:.6f}")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    
    # Save metrics
    metrics_path = Path("results/lstm_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"Saved metrics to {metrics_path}")
    
    # Save training config
    config = {
        "model": "lstm",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "context_length": args.context_length,
        "gap_length": args.gap_length,
        "stride": args.stride,
        "val_ratio": args.val_ratio,
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses
    }
    
    config_path = Path("experiments/lstm_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    args = parse_args()
    
    if args.model == "baseline":
        train_baseline(args)
    elif args.model == "lstm":
        train_lstm(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")
