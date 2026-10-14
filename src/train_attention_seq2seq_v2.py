"""
Train Attention Seq2Seq Model V2
Uses corrected benchmark_v2 data with both training files.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import json
import time
import sys

sys.path.append("src")
from attention_seq2seq_model import AttentionSeq2SeqReconstructor


BENCHMARK_V2_PATH = Path("data/benchmark_v2")

TRAIN_FILES = [
    "train_05_M02_DC_train.npz",
    "train_07_M02_DC_train.npz"
]

VAL_FILES = [
    "val_05_M02_DC_train.npz",
    "val_07_M02_DC_train.npz"
]

HYPERPARAMETERS = {
    "context_length": 50,
    "gap_length": 10,
    "batch_size": 64,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "attention_dim": 128,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    "teacher_forcing_ratio": 0.5,
    "maximum_epochs": 20,
    "early_stopping_patience": 5
}


class BenchmarkV2Dataset(Dataset):
    """Dataset for benchmark_v2 NPZ files."""
    
    def __init__(self, npz_files):
        self.context_before = []
        self.context_after = []
        self.targets = []
        
        for npz_file in npz_files:
            file_path = BENCHMARK_V2_PATH / npz_file
            data = np.load(file_path)
            
            self.context_before.append(data["context_before"])
            self.context_after.append(data["context_after"])
            self.targets.append(data["masked_segment"])
        
        # Concatenate all files
        self.context_before = np.concatenate(self.context_before, axis=0)
        self.context_after = np.concatenate(self.context_after, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        
        print(f"Loaded {len(self.context_before)} sequences from {len(npz_files)} files")
    
    def __len__(self):
        return len(self.context_before)
    
    def __getitem__(self, idx):
        return {
            "context_before": torch.FloatTensor(self.context_before[idx]),
            "context_after": torch.FloatTensor(self.context_after[idx]),
            "target": torch.FloatTensor(self.targets[idx])
        }


class AttentionSeq2SeqTrainer:
    """Trainer for Attention Seq2Seq model."""
    
    def __init__(self, model, device, hyperparameters):
        self.model = model.to(device)
        self.device = device
        self.hyperparameters = hyperparameters
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparameters["learning_rate"],
            weight_decay=hyperparameters["weight_decay"]
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            context_before = batch["context_before"].to(self.device)
            context_after = batch["context_after"].to(self.device)
            target = batch["target"].to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(
                context_before,
                context_after,
                target=target,
                teacher_forcing_ratio=self.hyperparameters["teacher_forcing_ratio"]
            )
            
            loss = self.criterion(predictions, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.hyperparameters["gradient_clip"]
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model (no teacher forcing)."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                context_before = batch["context_before"].to(self.device)
                context_after = batch["context_after"].to(self.device)
                target = batch["target"].to(self.device)
                
                predictions = self.model(
                    context_before,
                    context_after,
                    target=None,
                    teacher_forcing_ratio=0.0
                )
                
                loss = self.criterion(predictions, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader):
        """Train the model with early stopping."""
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.hyperparameters["maximum_epochs"]):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{self.hyperparameters['maximum_epochs']} - "
                  f"Train MSE: {train_loss:.6f}, Val MSE: {val_loss:.6f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                patience_counter = 0
                self.save_checkpoint("attention_seq2seq_v2_best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.hyperparameters["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        
        return {
            "total_epochs": epoch + 1,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_time": total_time
        }
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "hyperparameters": self.hyperparameters
        }
        
        save_path = Path("models") / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint_path = Path(filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_epoch = checkpoint["best_epoch"]
        print(f"Loaded checkpoint from {checkpoint_path}")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_checkpoint(model, device, checkpoint_path):
    """Verify the loaded checkpoint."""
    print("\n" + "="*70)
    print("CHECKPOINT VERIFICATION")
    print("="*70)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"\nParameter count: {count_parameters(model):,}")
    
    # Verify shapes with a test batch
    model.eval()
    batch_size = 4
    context_length = 50
    gap_length = 10
    sensor_count = 17
    
    context_before = torch.randn(batch_size, context_length, sensor_count).to(device)
    context_after = torch.randn(batch_size, context_length, sensor_count).to(device)
    
    with torch.no_grad():
        # Inference without teacher forcing
        predictions = model(context_before, context_after, target=None, teacher_forcing_ratio=0.0)
    
    print(f"Input shapes:")
    print(f"  context_before: {context_before.shape}")
    print(f"  context_after: {context_after.shape}")
    
    print(f"\nOutput shape:")
    print(f"  predictions: {predictions.shape}")
    print(f"  Expected: ({batch_size}, {gap_length}, {sensor_count})")
    
    shape_ok = predictions.shape == (batch_size, gap_length, sensor_count)
    print(f"  Shape correct: {'YES' if shape_ok else 'NO'}")
    
    # Check prediction variance
    variance = np.var(predictions.cpu().numpy(), axis=1)
    mean_variance = np.mean(variance)
    print(f"\nPrediction variance across timesteps: {mean_variance:.6f}")
    print(f"  Varying predictions: {'YES' if mean_variance > 0.001 else 'NO'}")
    
    return shape_ok, mean_variance > 0.001


def main():
    print("="*70)
    print("TRAINING ATTENTION SEQ2SEQ MODEL V2")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load training data
    print("\nLoading training data...")
    train_dataset = BenchmarkV2Dataset(TRAIN_FILES)
    train_loader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMETERS["batch_size"],
        shuffle=True,
        num_workers=0
    )
    
    # Load validation data
    print("\nLoading validation data...")
    val_dataset = BenchmarkV2Dataset(VAL_FILES)
    val_loader = DataLoader(
        val_dataset,
        batch_size=HYPERPARAMETERS["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nCreating model...")
    model = AttentionSeq2SeqReconstructor(
        input_dim=17,
        hidden_dim=HYPERPARAMETERS["hidden_dim"],
        num_layers=HYPERPARAMETERS["num_layers"],
        output_dim=17,
        output_length=HYPERPARAMETERS["gap_length"],
        dropout=HYPERPARAMETERS["dropout"],
        attention_dim=HYPERPARAMETERS["attention_dim"]
    )
    
    print(f"Parameter count: {count_parameters(model):,}")
    
    # Create trainer
    trainer = AttentionSeq2SeqTrainer(model, device, HYPERPARAMETERS)
    
    # Train
    print("\nStarting training...")
    training_results = trainer.train(train_loader, val_loader)
    
    # Save training history
    training_history = {
        "hyperparameters": HYPERPARAMETERS,
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses,
        "training_results": training_results
    }
    
    history_path = Path("results/attention_seq2seq_v2_training.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"\nSaved training history to {history_path}")
    
    # Reload best checkpoint and verify
    print("\nReloading best checkpoint for verification...")
    checkpoint_path = Path("models/attention_seq2seq_v2_best_model.pt")
    
    # Create fresh model for verification
    model_verify = AttentionSeq2SeqReconstructor(
        input_dim=17,
        hidden_dim=HYPERPARAMETERS["hidden_dim"],
        num_layers=HYPERPARAMETERS["num_layers"],
        output_dim=17,
        output_length=HYPERPARAMETERS["gap_length"],
        dropout=HYPERPARAMETERS["dropout"],
        attention_dim=HYPERPARAMETERS["attention_dim"]
    )
    
    trainer_verify = AttentionSeq2SeqTrainer(model_verify, device, HYPERPARAMETERS)
    trainer_verify.load_checkpoint(checkpoint_path)
    
    shape_ok, variance_ok = verify_checkpoint(model_verify, device, checkpoint_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Total epochs: {training_results['total_epochs']}")
    print(f"  Best epoch: {training_results['best_epoch']}")
    print(f"  Best val loss: {training_results['best_val_loss']:.6f}")
    print(f"  Total time: {training_results['total_time']:.2f}s")
    print(f"  Shape verification: {'PASSED' if shape_ok else 'FAILED'}")
    print(f"  Variance verification: {'PASSED' if variance_ok else 'FAILED'}")


if __name__ == "__main__":
    main()
