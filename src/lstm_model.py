"""
LSTM-based Reconstruction Model
Learns to predict masked sensor segments from temporal context.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from pathlib import Path
import pickle
import json


class ReconstructionDataset(Dataset):
    """Dataset for masked sequence reconstruction."""
    
    def __init__(self, sequences: List[Dict]):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Concatenate context before and after
        context = np.concatenate([seq["context_before"], seq["context_after"]], axis=0)
        target = seq["masked_segment"]
        
        return {
            "context": torch.FloatTensor(context),
            "target": torch.FloatTensor(target)
        }


class LSTMReconstructor(nn.Module):
    """LSTM model for sensor reconstruction with improved architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        output_length: int,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.output_length = output_length
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Better decoder with residual connections
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Learnable interpolation bias
        self.interpolation_bias = nn.Parameter(torch.zeros(1, output_length, output_dim))
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            context: (batch_size, context_length, input_dim)
            
        Returns:
            predictions: (batch_size, output_length, output_dim)
        """
        batch_size = context.size(0)
        
        # Encode context
        lstm_out, (h_n, c_n) = self.lstm(context)
        
        # Use final hidden state for reconstruction
        final_hidden = h_n[-1]  # (batch_size, hidden_dim)
        
        # Repeat for each output timestep
        repeated_hidden = final_hidden.unsqueeze(1).repeat(1, self.output_length, 1)
        # (batch_size, output_length, hidden_dim)
        
        # Decode to sensor values
        predictions = self.decoder(repeated_hidden)
        
        # Add learnable bias
        predictions = predictions + self.interpolation_bias
        
        return predictions


class LSTMModelTrainer:
    """Trainer for LSTM reconstruction model."""
    
    def __init__(
        self,
        model: LSTMReconstructor,
        learning_rate: float = 0.0005,
        weight_decay: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            context = batch["context"].to(self.device)
            target = batch["target"].to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(context)
            loss = self.criterion(predictions, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                context = batch["context"].to(self.device)
                target = batch["target"].to(self.device)
                
                predictions = self.model(context)
                loss = self.criterion(predictions, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 5
    ):
        """Train the model with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        save_path = Path("models") / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = Path(filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for batch in test_loader:
                context = batch["context"].to(self.device)
                target = batch["target"].to(self.device)
                
                pred = self.model(context)
                predictions.append(pred.cpu().numpy())
                ground_truths.append(target.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        ground_truths = np.concatenate(ground_truths, axis=0)
        
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
            "sensor_mae": sensor_mae.tolist()
        }
        
        return metrics


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from experiment_generator import MaskedSequenceGenerator
    
    # Load sequences
    generator = MaskedSequenceGenerator()
    train_sequences = generator.load_sequences(Path("data/sequences/train_sequences.pkl"))
    val_sequences = generator.load_sequences(Path("data/sequences/val_sequences.pkl"))
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    # Load config
    with open("data/sequences/config.json", "r") as f:
        config = json.load(f)
    
    # Create datasets
    train_dataset = ReconstructionDataset(train_sequences)
    val_dataset = ReconstructionDataset(val_sequences)
    test_dataset = ReconstructionDataset(test_sequences)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Get dimensions
    sample = train_sequences[0]
    context_length = config["context_length"] * 2
    input_dim = sample["context_before"].shape[1]
    output_dim = input_dim
    output_length = config["gap_length"]
    
    print(f"\nModel configuration:")
    print(f"  Context length: {context_length}")
    print(f"  Input dim: {input_dim}")
    print(f"  Output length: {output_length}")
    print(f"  Output dim: {output_dim}")
    
    # Create model with smaller hidden dim to prevent overfitting
    model = LSTMReconstructor(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        output_length=output_length,
        dropout=0.3
    )
    
    # Train
    trainer = LSTMModelTrainer(model, learning_rate=0.001)
    trainer.train(train_loader, val_loader, epochs=20, patience=5)
    
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
    
    print(f"\nSaved metrics to {metrics_path}")
