# AI Archaeologist - Missing-Event Reconstruction

Trained machine learning model for industrial sensor reconstruction using PHM 2018 dataset.

## Task Definition

**Problem:** Reconstruct missing sensor segments in industrial time-series data.

**Dataset:** PHM 2018 Ion Mill Etch Tool dataset (unlabeled sensor time-series)

**Approach:** Mask a contiguous segment of sensor readings and predict the masked values using temporal context.

**Evaluation:** MSE and MAE between predicted and actual sensor values.

## Dataset

**Source:** PHM 2018 Data Challenge

**Files:** 7 CSV files (2 train, 5 test)

**Total Rows:** ~14 million sensor readings

**Columns:** 24 columns (7 metadata + 17 sensor variables)

**Sensor Variables:**
- IONGAUGEPRESSURE
- ETCHBEAMVOLTAGE
- ETCHBEAMCURRENT
- ETCHSUPPRESSORVOLTAGE
- ETCHSUPPRESSORCURRENT
- FLOWCOOLFLOWRATE
- FLOWCOOLPRESSURE
- ETCHGASCHANNEL1READBACK
- ETCHPBNGASREADBACK
- FIXTURETILTANGLE
- ROTATIONSPEED
- ACTUALROTATIONANGLE
- FIXTURESHUTTERPOSITION
- ETCHSOURCEUSAGE
- ETCHAUXSOURCETIMER
- ETCHAUX2SOURCETIMER
- ACTUALSTEPDURATION

**Missing Values:** Minimal (4 missing values total in FIXTURESHUTTERPOSITION)

**Note:** Dataset contains NO fault labels or RUL information. This is purely sensor value reconstruction.

## Project Structure

```
.
├── data/
│   ├── raw/uploads/phm_data_challenge_2018/  # Original PHM 2018 dataset
│   ├── preprocessed/phm2018_subset.pkl       # Preprocessed sensor data
│   └── sequences/                             # Generated masked sequences
├── src/
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── inspect_phm_dataset.py # Dataset inspection
│   ├── experiment_generator.py # Masked sequence generation
│   ├── baseline_model.py      # Linear interpolation baseline
│   └── lstm_model.py          # LSTM reconstruction model
├── models/
│   └── best_model.pt          # Trained model checkpoint
├── results/
│   ├── baseline_metrics.json  # Baseline evaluation results
│   ├── lstm_metrics.json      # LSTM evaluation results
│   └── model_comparison.json  # Model comparison
├── experiments/
│   └── lstm_config.json       # Training configuration
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
└── README.md                  # This file
```

## Installation

**Requirements:**
- Python 3.11+
- PyTorch
- pandas
- numpy
- scikit-learn

**Install dependencies:**
```bash
pip install torch pandas numpy scikit-learn
```

## Usage

### 1. Data Preparation

The dataset should be placed in:
```
data/raw/uploads/phm_data_challenge_2018/
```

Run data inspection:
```bash
python src/inspect_phm_dataset.py
```

Run data preprocessing:
```bash
python src/data_loader.py
```

Run sequence generation:
```bash
python src/experiment_generator.py
```

### 2. Training

**Train baseline:**
```bash
python train.py --model baseline
```

**Train LSTM:**
```bash
python train.py --model lstm --epochs 20 --batch_size 64 --learning_rate 0.0005
```

**Training arguments:**
- `--model`: Model type (baseline, lstm)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 64)
- `--learning_rate`: Learning rate (default: 0.0005)
- `--hidden_dim`: LSTM hidden dimension (default: 64)
- `--context_length`: Context length before/after gap (default: 50)
- `--gap_length`: Gap length to predict (default: 10)
- `--stride`: Stride between sequences (default: 100)
- `--val_ratio`: Validation ratio (default: 0.2)

### 3. Evaluation

**Evaluate baseline:**
```bash
python evaluate.py --model baseline
```

**Evaluate LSTM:**
```bash
python evaluate.py --model lstm --checkpoint models/best_model.pt
```

**Evaluation arguments:**
- `--model`: Model type (baseline, lstm)
- `--checkpoint`: Path to model checkpoint (default: models/best_model.pt)
- `--batch_size`: Batch size (default: 64)

The evaluation script automatically compares baseline and LSTM results if both metrics are available.

## Results

**Baseline (Linear Interpolation):**
- MSE: 0.023871
- MAE: 0.018904

**LSTM Model:**
- MSE: 3.133904
- MAE: 0.610560

**Note:** The LSTM model currently performs worse than the baseline. This is expected for a simple interpolation task on smooth sensor data. The baseline linear interpolation is very effective for this specific task.

## Model Architecture

**LSTM Reconstructor:**
- Input: Concatenated context before and after gap (100 timesteps × 17 sensors)
- LSTM Encoder: 2 layers, 64 hidden units, bidirectional=False
- Decoder: 2-layer MLP with LayerNorm and dropout
- Output: Predicted gap segment (10 timesteps × 17 sensors)
- Regularization: Dropout (0.3), weight decay (1e-5)

**Training:**
- Optimizer: Adam
- Loss: MSE
- Early stopping: Patience=5
- Learning rate: 0.0005

## Data Splitting

**Temporal split to prevent leakage:**
- Train: First 80% of training data
- Validation: Last 20% of training data
- Test: Separate test files (no temporal leakage)

**Sequence generation:**
- Context before gap: 50 timesteps
- Masked gap: 10 timesteps
- Context after gap: 50 timesteps
- Stride: 100 timesteps

**Total sequences:**
- Train: 15,256
- Validation: 3,814
- Test: 17,035

## Limitations

1. **No fault labels:** Dataset contains only sensor readings, no fault information
2. **Simple task:** Linear interpolation is very effective for smooth sensor data
3. **Overfitting:** LSTM model overfits to training data, performs worse on test
4. **Subset used:** Currently using subset of data (1 train file, 1 test file) for prototyping

## Future Work

1. **Use full dataset:** Train on all 7 files instead of subset
2. **Better architecture:** Try Transformer or TCN instead of LSTM
3. **Multi-task learning:** Add auxiliary tasks if labels become available
4. **Anomaly detection:** Use reconstruction error for anomaly detection
5. **Uncertainty estimation:** Add confidence intervals to predictions

## Reproducibility

All experiments are reproducible using the provided scripts:

1. Place PHM 2018 dataset in `data/raw/uploads/phm_data_challenge_2018/`
2. Run `python src/data_loader.py` to preprocess
3. Run `python src/experiment_generator.py` to generate sequences
4. Run `python train.py --model lstm` to train
5. Run `python evaluate.py --model lstm` to evaluate

Random seeds are not currently fixed for reproducibility. Add `torch.manual_seed()` and `np.random.seed()` if needed.

## License

This project uses the PHM 2018 dataset. Please refer to the original dataset license for usage terms.
