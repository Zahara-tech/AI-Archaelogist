import os
import json
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from attention_seq2seq_model import AttentionSeq2SeqReconstructor


# ============================================================
# Configuration
# ============================================================

SEED = 42

SEQUENCE_FILE = "data/sequences/train_sequences.pkl"
VAL_SEQUENCE_FILE = "data/sequences/val_sequences.pkl"

MODEL_FILE = "models/attention_seq2seq_best_model.pt"
METRICS_FILE = "results/attention_seq2seq_training.json"

INPUT_DIM = 17
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 17
OUTPUT_LENGTH = 10

DROPOUT = 0.3
ATTENTION_DIM = 128

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5

EPOCHS = 15
PATIENCE = 5

TEACHER_FORCING_RATIO = 0.5
GRAD_CLIP = 1.0


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("ATTENTION SEQ2SEQ TRAINING")
print("=" * 70)

print(f"Device: {device}")

if device.type == "cuda":
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available - using CPU")


# ============================================================
# Dataset
# ============================================================

class ReconstructionDataset(Dataset):

    def __init__(self, sequences):

        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        item = self.sequences[idx]

        context_before = np.asarray(
            item["context_before"],
            dtype=np.float32,
        )

        context_after = np.asarray(
            item["context_after"],
            dtype=np.float32,
        )

        target = np.asarray(
            item["masked_segment"],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(context_before),
            torch.from_numpy(context_after),
            torch.from_numpy(target),
        )


# ============================================================
# Load sequences
# ============================================================

def load_sequences(path):

    print(f"\nLoading: {path}")

    with open(path, "rb") as f:
        sequences = pickle.load(f)

    print(f"Sequences loaded: {len(sequences):,}")

    return sequences


train_sequences = load_sequences(SEQUENCE_FILE)
val_sequences = load_sequences(VAL_SEQUENCE_FILE)


# ============================================================
# Inspect first sequence
# ============================================================

first = train_sequences[0]

print("\nSequence structure:")

print(
    "context_before:",
    np.asarray(first["context_before"]).shape,
)

print(
    "context_after:",
    np.asarray(first["context_after"]).shape,
)

print(
    "masked_segment:",
    np.asarray(first["masked_segment"]).shape,
)


# ============================================================
# DataLoaders
# ============================================================

train_dataset = ReconstructionDataset(
    train_sequences
)

val_dataset = ReconstructionDataset(
    val_sequences
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


# ============================================================
# Model
# ============================================================

model = AttentionSeq2SeqReconstructor(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    output_dim=OUTPUT_DIM,
    output_length=OUTPUT_LENGTH,
    dropout=DROPOUT,
    attention_dim=ATTENTION_DIM,
).to(device)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nModel:")
print(model)

print(
    f"\nTrainable parameters: {parameter_count:,}"
)


# ============================================================
# Loss / optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")
epochs_without_improvement = 0

history = []


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_count = 0

    for batch_idx, batch in enumerate(train_loader):

        context_before = batch[0].to(device)
        context_after = batch[1].to(device)
        target = batch[2].to(device)

        optimizer.zero_grad()

        prediction = model(
            context_before,
            context_after,
            target=target,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO,
        )

        loss = criterion(
            prediction,
            target,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        batch_size = context_before.size(0)

        train_loss_sum += (
            loss.item() * batch_size
        )

        train_count += batch_size

        if (batch_idx + 1) % 100 == 0:

            print(
                f"Epoch {epoch:02d} | "
                f"Batch {batch_idx + 1:04d}/"
                f"{len(train_loader):04d} | "
                f"Loss {loss.item():.6f}"
            )

    train_loss = (
        train_loss_sum / train_count
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():

        for batch in val_loader:

            context_before = batch[0].to(device)
            context_after = batch[1].to(device)
            target = batch[2].to(device)

            prediction = model(
                context_before,
                context_after,
                target=None,
                teacher_forcing_ratio=0.0,
            )

            loss = criterion(
                prediction,
                target,
            )

            batch_size = context_before.size(0)

            val_loss_sum += (
                loss.item() * batch_size
            )

            val_count += batch_size

    val_loss = (
        val_loss_sum / val_count
    )

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print()
    print("-" * 70)
    print(
        f"Epoch {epoch:02d}/{EPOCHS}"
    )
    print(
        f"Train MSE: {train_loss:.6f}"
    )
    print(
        f"Val MSE:   {val_loss:.6f}"
    )
    print(
        f"Learning rate: {current_lr:.8f}"
    )
    print("-" * 70)

    history.append(
        {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "learning_rate": current_lr,
        }
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "output_dim": OUTPUT_DIM,
            "output_length": OUTPUT_LENGTH,
            "dropout": DROPOUT,
            "attention_dim": ATTENTION_DIM,
            "best_val_loss": best_val_loss,
            "epoch": epoch,
        }

        torch.save(
            checkpoint,
            MODEL_FILE,
        )

        print(
            f"Saved best model -> {MODEL_FILE}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping."
            )

            break


# ============================================================
# Save training metrics
# ============================================================

os.makedirs(
    os.path.dirname(METRICS_FILE),
    exist_ok=True,
)

training_results = {
    "model": "attention_seq2seq",
    "device": str(device),
    "train_sequences": len(train_sequences),
    "validation_sequences": len(val_sequences),
    "best_validation_mse": best_val_loss,
    "epochs_completed": len(history),
    "parameter_count": parameter_count,
    "history": history,
}

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        training_results,
        f,
        indent=2,
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation MSE: {best_val_loss:.6f}"
)

print(
    f"Best model: {MODEL_FILE}"
)

print(
    f"Training metrics: {METRICS_FILE}"
)

print("=" * 70)
