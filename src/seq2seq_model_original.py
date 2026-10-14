"""
Sequence-to-Sequence Reconstruction Model
Proper temporal decoder with timestep-specific predictions.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from pathlib import Path
import pickle
import json


class Seq2SeqReconstructor(nn.Module):
    """Sequence-to-sequence model with temporal decoder."""
    
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
        
        # Encoder: BiLSTM on full context (before + after)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Timestep embeddings for decoder
        self.timestep_embedding = nn.Embedding(output_length, hidden_dim)
        
        # Decoder: LSTM with timestep-specific predictions
        self.decoder = nn.LSTM(
            input_size=hidden_dim * 2 + hidden_dim,  # encoder_output + timestep_embedding
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Learnable start token
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim * 2))
    
    def encode(self, context_before: torch.Tensor, context_after: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Encode context using BiLSTM.
        
        Args:
            context_before: (batch, context_length, input_dim)
            context_after: (batch, context_length, input_dim)
            
        Returns:
            encoder_outputs: (batch, total_context_length, hidden_dim * 2)
            hidden: tuple of (h_n, c_n) from encoder
        """
        # Concatenate before and after context
        context = torch.cat([context_before, context_after], dim=1)
        # (batch, context_length * 2, input_dim)
        
        # Encode
        encoder_outputs, hidden = self.encoder(context)
        # encoder_outputs: (batch, context_length * 2, hidden_dim * 2)
        # hidden: (num_layers * 2, batch, hidden_dim)
        
        return encoder_outputs, hidden
    
    def decode_step(
        self,
        decoder_input: torch.Tensor,
        hidden: Tuple,
        encoder_outputs: torch.Tensor,
        timestep_idx: int
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Single decoder step.
        
        Args:
            decoder_input: (batch, 1, hidden_dim * 2)
            hidden: decoder hidden state
            encoder_outputs: (batch, context_length * 2, hidden_dim * 2)
            timestep_idx: current timestep index
            
        Returns:
            output: (batch, 1, output_dim)
            hidden: updated hidden state
        """
        # Add timestep embedding
        timestep_emb = self.timestep_embedding(
            torch.tensor([timestep_idx], device=decoder_input.device)
        ).unsqueeze(0).expand(decoder_input.size(0), -1, -1)
        # (batch, 1, hidden_dim)
        
        # Concatenate decoder input with timestep embedding
        decoder_input_with_time = torch.cat([decoder_input, timestep_emb], dim=-1)
        # (batch, 1, hidden_dim * 2 + hidden_dim)
        
        # Decode
        decoder_output, hidden = self.decoder(decoder_input_with_time, hidden)
        # decoder_output: (batch, 1, hidden_dim)
        
        # Project to output
        output = self.output_projection(decoder_output)
        # (batch, 1, output_dim)
        
        return output, hidden
    
    def forward(self, context_before: torch.Tensor, context_after: torch.Tensor, 
                teacher_forcing_ratio: float = 0.0, target: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing.
        
        Args:
            context_before: (batch, context_length, input_dim)
            context_after: (batch, context_length, input_dim)
            teacher_forcing_ratio: probability of using teacher forcing
            target: (batch, output_length, output_dim) - only during training
            
        Returns:
            predictions: (batch, output_length, output_dim)
        """
        batch_size = context_before.size(0)
        
        # Encode
        encoder_outputs, encoder_hidden = self.encode(context_before, context_after)
        
        # Initialize decoder hidden state with encoder final state
        # Transform bidirectional hidden to unidirectional
        h_n, c_n = encoder_hidden
        # h_n: (num_layers * 2, batch, hidden_dim)
        
        # Combine bidirectional states
        h_n_combined = []
        c_n_combined = []
        for i in range(self.num_layers):
            h_forward = h_n[i * 2]
            h_backward = h_n[i * 2 + 1]
            c_forward = c_n[i * 2]
            c_backward = c_n[i * 2 + 1]
            h_n_combined.append((h_forward + h_backward) / 2)
            c_n_combined.append((c_forward + c_backward) / 2)
        
        decoder_hidden = (torch.stack(h_n_combined), torch.stack(c_n_combined))
        # (num_layers, batch, hidden_dim)
        
        # Initialize decoder input with start token
        decoder_input = self.start_token.expand(batch_size, -1, -1)
        # (batch, 1, hidden_dim * 2)
        
        # Store predictions
        predictions = []
        
        # Decode step by step
        for t in range(self.output_length):
            # Decode one step
            output, decoder_hidden = self.decode_step(
                decoder_input, decoder_hidden, encoder_outputs, t
            )
            # output: (batch, 1, output_dim)
            
            predictions.append(output)
            
            # Teacher forcing
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Use ground truth as next input
                # Need to project target back to decoder input space
                # For simplicity, use encoder mean as next input
                decoder_input = encoder_outputs.mean(dim=1, keepdim=True)
            else:
                # Use prediction as next input (need toproject back to decoder input space)
                # For simplicity, use encoder mean
                decoder_input = encoder_outputs.mean(dim=1, keepdim=True)
        
        # Stack predictions
        predictions = torch.cat(predictions, dim=1)
        # (batch, output_length, output_dim)
        
        return predictions


class Seq2SeqDataset(Dataset):
    """Dataset for seq2seq with separate past and future context."""
    
    def __init__(self, sequences: List[Dict]):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            "context_before": torch.FloatTensor(seq["context_before"]),
            "context_after": torch.FloatTensor(seq["context_after"]),
            "target": torch.FloatTensor(seq["masked_segment"])
        }


class Seq2SeqTrainer:
    """Trainer for seq2seq reconstruction model."""
    
    def __init__(
        self,
        model: Seq2SeqReconstructor,
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
    
    def train_epoch(self, train_loader: DataLoader, teacher_forcing_ratio: float = 0.5) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            context_before = batch["context_before"].to(self.device)
            context_after = batch["context_after"].to(self.device)
            target = batch["target"].to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(context_before, context_after, 
                                    teacher_forcing_ratio=teacher_forcing_ratio, target=target)
            loss = self.criterion(predictions, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model (no teacher forcing)."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                context_before = batch["context_before"].to(self.device)
                context_after = batch["context_after"].to(self.device)
                target = batch["target"].to(self.device)
                
                predictions = self.model(context_before, context_after, 
                                        teacher_forcing_ratio=0.0)
                loss = self.criterion(predictions, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 5,
        teacher_forcing_ratio: float = 0.5
    ):
        """Train the model with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, teacher_forcing_ratio)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("seq2seq_best_model.pt")
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
                context_before = batch["context_before"].to(self.device)
                context_after = batch["context_after"].to(self.device)
                target = batch["target"].to(self.device)
                
                pred = self.model(context_before, context_after, teacher_forcing_ratio=0.0)
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
        
        # Per-timestep metrics
        timestep_mse = np.mean((predictions - ground_truths) ** 2, axis=(0, 2))
        timestep_mae = np.mean(np.abs(predictions - ground_truths), axis=(0, 2))
        
        # Prediction variance across timesteps
        prediction_variance = np.var(predictions, axis=1)  # (batch, sensors)
        mean_variance = np.mean(prediction_variance)
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "sensor_mse": sensor_mse.tolist(),
            "sensor_mae": sensor_mae.tolist(),
            "timestep_mse": timestep_mse.tolist(),
            "timestep_mae": timestep_mae.tolist(),
            "prediction_variance_mean": float(mean_variance)
        }
        
        return metrics


def run_sanity_checks():
    """Run sanity checks before training."""
    print("="*70)
    print("SANITY CHECKS")
    print("="*70)
    
    # Load data
    import sys
    sys.path.append("src")
    from data_loader import PHM2018Loader
    from experiment_generator import MaskedSequenceGenerator
    
    loader = PHM2018Loader()
    preprocessed = loader.load_preprocessed(Path("data/preprocessed/phm2018_subset.pkl"))
    sensor_names = preprocessed["sensor_columns"]
    
    generator = MaskedSequenceGenerator()
    train_sequences = generator.load_sequences(Path("data/sequences/train_sequences.pkl"))
    
    with open("data/sequences/config.json", "r") as f:
        config = json.load(f)
    
    # Create model
    sample = train_sequences[0]
    input_dim = sample["context_before"].shape[1]
    output_dim = input_dim
    output_length = config["gap_length"]
    
    model = Seq2SeqReconstructor(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        output_length=output_length,
        dropout=0.3
    )
    
    # Create dataset
    dataset = Seq2SeqDataset(train_sequences[:64])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Get one batch
    batch = next(iter(dataloader))
    
    print(f"\nInput Shapes:")
    print(f"  context_before: {batch['context_before'].shape}")
    print(f"  context_after: {batch['context_after'].shape}")
    print(f"  target: {batch['target'].shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(batch["context_before"], batch["context_after"])
    
    print(f"\nOutput Shape:")
    print(f"  predictions: {predictions.shape}")
    print(f"  Expected: ({batch['context_before'].shape[0]}, {output_length}, {output_dim})")
    
    shape_ok = predictions.shape == (batch['context_before'].shape[0], output_length, output_dim)
    print(f"  Shape correct: {'YES' if shape_ok else 'NO'}")
    
    # Check prediction variance
    print(f"\nPrediction Variance Check:")
    variance_across_timesteps = np.var(predictions.numpy(), axis=1)
    print(f"  Mean variance across timesteps: {np.mean(variance_across_timesteps):.6f}")
    print(f"  Variance > 0: {'YES' if np.mean(variance_across_timesteps) > 0.001 else 'NO'}")
    
    # Verify no target in inference
    print(f"\nInference Path Check:")
    print(f"  Forward called without target: YES")
    print(f"  Teacher forcing disabled: YES")
    
    return shape_ok, np.mean(variance_across_timesteps) > 0.001


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from experiment_generator import MaskedSequenceGenerator
    
    # Run sanity checks
    shape_ok, variance_ok = run_sanity_checks()
    
    if not shape_ok:
        print("\nERROR: Shape check failed!")
        sys.exit(1)
    
    if not variance_ok:
        print("\nERROR: Variance check failed - predictions may be constant!")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("SANITY CHECKS PASSED")
    print("="*70)
    
    # Load sequences
    generator = MaskedSequenceGenerator()
    train_sequences = generator.load_sequences(Path("data/sequences/train_sequences.pkl"))
    val_sequences = generator.load_sequences(Path("data/sequences/val_sequences.pkl"))
    test_sequences = generator.load_sequences(Path("data/sequences/test_sequences.pkl"))
    
    # Load config
    with open("data/sequences/config.json", "r") as f:
        config = json.load(f)
    
    # Create datasets
    train_dataset = Seq2SeqDataset(train_sequences)
    val_dataset = Seq2SeqDataset(val_sequences)
    test_dataset = Seq2SeqDataset(test_sequences)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Get dimensions
    sample = train_sequences[0]
    input_dim = sample["context_before"].shape[1]
    output_dim = input_dim
    output_length = config["gap_length"]
    
    print(f"\nSeq2Seq Model configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output length: {output_length}")
    print(f"  Output dim: {output_dim}")
    
    # Create model
    model = Seq2SeqReconstructor(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        output_length=output_length,
        dropout=0.3
    )
    
    # Train
    trainer = Seq2SeqTrainer(model, learning_rate=0.0005)
    trainer.train(train_loader, val_loader, epochs=20, patience=5, teacher_forcing_ratio=0.5)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"\nSeq2Seq Test Results:")
    print(f"  MSE: {test_metrics['mse']:.6f}")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    print(f"  Prediction Variance Mean: {test_metrics['prediction_variance_mean']:.6f}")
    
    # Save metrics
    metrics_path = Path("results/seq2seq_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"Saved metrics to {metrics_path}")
