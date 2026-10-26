"""
Train Residual LSTM Model
Trains a simple LSTM to predict residuals from linear interpolation.
"""

import os
import sys
import json
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("src")
from residual_lstm_model import ResidualLSTM

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent
BENCHMARK_DIR = BASE_DIR / "data" / "benchmark_v2"
EXPERIMENT_DIR = BASE_DIR / "experiments" / "attention_seq2seq_v2_50epochs"
RESIDUAL_DIR = EXPERIMENT_DIR / "residual_experiment"

RESIDUAL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = RESIDUAL_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = RESIDUAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

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

# Training configuration
CONFIG = {
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 128,
    "learning_rate": 0.001,
    "max_epochs": 20,
    "patience": 5,
    "weight_decay": 1e-5
}

print("=" * 70)
print("RESIDUAL LSTM TRAINING")
print("=" * 70)
print(f"Output directory: {RESIDUAL_DIR}")
print(f"Device: {DEVICE}")
print(f"Config: {CONFIG}")
print()

# ============================================================
# DATASET CLASS
# ============================================================

class ResidualDataset(Dataset):
    """Dataset for residual prediction."""
    
    def __init__(self, before, after, target):
        self.before = torch.tensor(before, dtype=torch.float32)
        self.after = torch.tensor(after, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        before = self.before[idx]
        after = self.after[idx]
        target = self.target[idx]
        
        # Compute linear interpolation
        left = before[-1:, :]  # (1, 17)
        right = after[0:1, :]  # (1, 17)
        steps = torch.arange(1, 11, dtype=torch.float32).reshape(10, 1) / 11.0  # (10, 1)
        interpolation = left + steps * (right - left)  # (10, 17)
        
        # Compute residual
        residual = target - interpolation
        
        return before, after, interpolation, residual

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train_files = [
    "train_05_M02_DC_train.npz",
    "train_07_M02_DC_train.npz"
]

val_files = [
    "val_05_M02_DC_train.npz",
    "val_07_M02_DC_train.npz"
]

def load_data(file_list):
    """Load and concatenate data from multiple files."""
    all_before = []
    all_after = []
    all_target = []
    
    for file_name in file_list:
        data = np.load(BENCHMARK_DIR / file_name)
        all_before.append(data["context_before"].astype(np.float32))
        all_after.append(data["context_after"].astype(np.float32))
        all_target.append(data["masked_segment"].astype(np.float32))
    
    return (
        np.concatenate(all_before, axis=0),
        np.concatenate(all_after, axis=0),
        np.concatenate(all_target, axis=0)
    )

train_before, train_after, train_target = load_data(train_files)
val_before, val_after, val_target = load_data(val_files)

print(f"Training sequences: {len(train_target)}")
print(f"Validation sequences: {len(val_target)}")

# Create datasets
train_dataset = ResidualDataset(train_before, train_after, train_target)
val_dataset = ResidualDataset(val_before, val_after, val_target)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

print()

# ============================================================
# INITIALIZE MODEL
# ============================================================

print("=" * 70)
print("INITIALIZING MODEL")
print("=" * 70)

model = ResidualLSTM(
    input_dim=17,
    hidden_dim=CONFIG["hidden_dim"],
    num_layers=CONFIG["num_layers"],
    output_dim=17,
    output_length=10,
    dropout=CONFIG["dropout"]
)

model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# ============================================================
# TRAINING LOOP
# ============================================================

print("=" * 70)
print("TRAINING")
print("=" * 70)

train_losses = []
val_losses = []
best_val_loss = float('inf')
best_epoch = 0
epochs_no_improve = 0

for epoch in range(CONFIG["max_epochs"]):
    # Training
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['max_epochs']} [Train]")
    for before, after, interpolation, residual in pbar:
        before = before.to(DEVICE)
        after = after.to(DEVICE)
        interpolation = interpolation.to(DEVICE)
        residual = residual.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_residual = model(before, after, interpolation)
        
        # Loss
        loss = criterion(pred_residual, residual)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    avg_train_loss = train_loss / train_batches
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for before, after, interpolation, residual in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['max_epochs']} [Val]"):
            before = before.to(DEVICE)
            after = after.to(DEVICE)
            interpolation = interpolation.to(DEVICE)
            residual = residual.to(DEVICE)
            
            pred_residual = model(before, after, interpolation)
            loss = criterion(pred_residual, residual)
            
            val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss / val_batches
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        epochs_no_improve = 0
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "config": CONFIG
        }
        torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pt")
        print(f"  -> New best model saved (val_loss: {avg_val_loss:.6f})")
    else:
        epochs_no_improve += 1
        print(f"  -> No improvement for {epochs_no_improve} epoch(s)")
        
        if epochs_no_improve >= CONFIG["patience"]:
            print(f"  -> Early stopping triggered")
            break

print()

# ============================================================
# PLOT TRAINING CURVES
# ============================================================

print("=" * 70)
print("PLOTTING TRAINING CURVES")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(train_losses) + 1), train_losses, 'o-', label='Train Loss', linewidth=2, markersize=6)
ax.plot(range(1, len(val_losses) + 1), val_losses, 's-', label='Val Loss', linewidth=2, markersize=6)
ax.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Training and Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "training_curves.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path}")
plt.close()

print()

# ============================================================
# SAVE CONFIGURATION
# ============================================================

config_path = RESULTS_DIR / "config.json"
with open(config_path, "w") as f:
    json.dump(CONFIG, f, indent=2)
print(f"  Saved: {config_path}")

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
print(f"Results saved to: {RESULTS_DIR}")
print()
