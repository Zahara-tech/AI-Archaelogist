"""
Write Final Report for Residual Correction Experiment
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RESIDUAL_DIR = BASE_DIR / "experiments" / "attention_seq2seq_v2_50epochs" / "residual_experiment"

# Load evaluation results
with open(RESIDUAL_DIR / "results" / "evaluation_results.json", "r") as f:
    results = json.load(f)

# Load config
with open(RESIDUAL_DIR / "results" / "config.json", "r") as f:
    config = json.load(f)

# Load training info
with open(RESIDUAL_DIR / "checkpoints" / "best_model.pt", "rb") as f:
    import torch
    checkpoint = torch.load(f, map_location="cpu")
    best_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["val_loss"]

report = f"""# Residual Correction Experiment Report

## Date
2024-10-25

## Objective

Improve generalization and reconstruction performance for industrial sensor time-series imputation by training a neural network to predict residuals from linear interpolation.

**Hypothesis**: Linear interpolation provides a strong baseline. A neural network can learn to correct systematic errors in interpolation, improving overall reconstruction quality.

## Experiment Design

### Architecture

**Residual LSTM Model**:
- Simple LSTM-based architecture
- Input: Concatenated context (50 timesteps before + 10 interpolation + 50 timesteps after)
- Hidden dimension: {config['hidden_dim']}
- Number of layers: {config['num_layers']}
- Dropout: {config['dropout']}
- Output: Residual (10 timesteps x 17 sensors)
- Total parameters: 82,922

**Final prediction**: `interpolation + predicted_residual`

### Training Configuration

- **Training data**: 38,545 sequences from train_05_M02_DC_train.npz and train_07_M02_DC_train.npz
- **Validation data**: 9,447 sequences from val_05_M02_DC_train.npz and val_07_M02_DC_train.npz
- **Test data**: 55,661 sequences from 5 held-out test files
- **Batch size**: {config['batch_size']}
- **Learning rate**: {config['learning_rate']}
- **Optimizer**: Adam with weight decay {config['weight_decay']}
- **Loss**: MSE on residuals
- **Max epochs**: {config['max_epochs']}
- **Early stopping patience**: {config['patience']}
- **Best epoch**: {best_epoch}
- **Best validation loss**: {best_val_loss:.6f}

### Data Usage

- **Scaler fitted on**: Training data only
- **Training gaps generated from**: Training files only
- **Validation gaps generated from**: Validation files only
- **Test gaps generated from**: Held-out test files only
- **No test targets used**: For model training or hyperparameter selection

## Results

### Overall Test Metrics (Normalized Space)

| Method | MSE | MAE | RMSE |
|--------|-----|-----|------|
| Linear Interpolation | {results['overall_metrics_normalized']['linear_interp']['mse']:.6f} | {results['overall_metrics_normalized']['linear_interp']['mae']:.6f} | {results['overall_metrics_normalized']['linear_interp']['rmse']:.6f} |
| Residual LSTM | {results['overall_metrics_normalized']['residual_lstm']['mse']:.6f} | {results['overall_metrics_normalized']['residual_lstm']['mae']:.6f} | {results['overall_metrics_normalized']['residual_lstm']['rmse']:.6f} |
| Seq2Seq | {results['overall_metrics_normalized']['seq2seq']['mse']:.6f} | {results['overall_metrics_normalized']['seq2seq']['mae']:.6f} | {results['overall_metrics_normalized']['seq2seq']['rmse']:.6f} |

### Overall Test Metrics (Original Units)

| Method | MSE | MAE | RMSE |
|--------|-----|-----|------|
| Linear Interpolation | {results['overall_metrics_original']['linear_interp']['mse']:.6f} | {results['overall_metrics_original']['linear_interp']['mae']:.6f} | {results['overall_metrics_original']['linear_interp']['rmse']:.6f} |
| Residual LSTM | {results['overall_metrics_original']['residual_lstm']['mse']:.6f} | {results['overall_metrics_original']['residual_lstm']['mae']:.6f} | {results['overall_metrics_original']['residual_lstm']['rmse']:.6f} |
| Seq2Seq | {results['overall_metrics_original']['seq2seq']['mse']:.6f} | {results['overall_metrics_original']['seq2seq']['mae']:.6f} | {results['overall_metrics_original']['seq2seq']['rmse']:.6f} |

### Performance Comparison

- **Residual LSTM vs Linear Interpolation**: {results['improvement_vs_interpolation']['residual_lstm']['mse_pct']:+.2f}% MSE
- **Seq2Seq vs Linear Interpolation**: {results['improvement_vs_interpolation']['seq2seq']['mse_pct']:+.2f}% MSE

**Best method overall**: {results['best_method']}

### Per-File Results

"""

for r in results['per_file_results']:
    report += f"""**{r['file']}** ({r['sequences']} sequences)
- Linear Interpolation: MSE={r['linear_interp']['mse']:.6f}, Time={r['linear_interp']['time']:.3f}s
- Residual LSTM: MSE={r['residual_lstm']['mse']:.6f}, Time={r['residual_lstm']['time']:.3f}s
- Seq2Seq: MSE={r['seq2seq']['mse']:.6f}, Time={r['seq2seq']['time']:.3f}s
- Best method: {r['best_method']}

"""

report += """### Per-Sensor Metrics (Best Method: Linear Interpolation)

| Sensor | MSE | MAE | RMSE |
|--------|-----|-----|------|
"""

for sensor in [
    "IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHBEAMCURRENT", "ETCHSUPPRESSORVOLTAGE",
    "ETCHSUPPRESSORCURRENT", "FLOWCOOLFLOWRATE", "FLOWCOOLPRESSURE", "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK", "FIXTURETILTANGLE", "ROTATIONSPEED", "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION", "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER", "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION"
]:
    report += f"| {sensor} | {results['per_sensor_metrics']['mse'][sensor]:.6f} | {results['per_sensor_metrics']['mae'][sensor]:.6f} | {results['per_sensor_metrics']['rmse'][sensor]:.6f} |\n"

report += """
### Per-Timestep Metrics (Best Method: Linear Interpolation)

| Timestep | MSE | MAE | RMSE |
|----------|-----|-----|------|
"""

for t in range(10):
    report += f"| {t+1} | {results['per_timestep_metrics']['mse'][t]:.6f} | {results['per_timestep_metrics']['mae'][t]:.6f} | {results['per_timestep_metrics']['rmse'][t]:.6f} |\n"

report += """
### Inference Time Comparison

Linear interpolation is extremely fast (<0.1s per file), while neural models are significantly slower:
- Linear Interpolation: ~0.02-0.06s per test file
- Residual LSTM: ~10-20s per test file
- Seq2Seq: ~47-90s per test file

## Analysis

### Did the Residual Network Learn Useful Corrections?

**No.** The residual LSTM performed slightly worse than linear interpolation (-0.82% MSE), indicating that it did not learn useful correction patterns.

### Why Did Residual Correction Fail?

1. **Linear interpolation is already near-optimal**: For this dataset and gap size (10 timesteps), linear interpolation between boundary values provides an excellent baseline. The residuals are small and noisy, making them difficult to learn.

2. **Training on limited distribution**: The model was trained only on M02 tool variants (05_M02_DC, 07_M02_DC) but tested on both M02 and M01 variants. The learned corrections may not generalize to unseen tool variants.

3. **Simple architecture insufficient**: The simple LSTM architecture may not have sufficient capacity to capture complex temporal dynamics beyond what linear interpolation already captures.

4. **Overfitting to training noise**: The model may have learned to fit noise in the training residuals rather than generalizable correction patterns.

### Comparison with Seq2Seq

The Residual LSTM performed significantly better than the existing Attention Seq2Seq V2 model:
- Residual LSTM MSE: 0.007763
- Seq2Seq MSE: 0.126779 (16.3x worse)

This suggests that the residual correction approach is more robust than the full Seq2Seq approach, but still cannot beat the simple linear interpolation baseline.

## Limitations

1. **Architecture simplicity**: The residual LSTM used a simple architecture. More sophisticated architectures might perform better.

2. **Limited training diversity**: Training only on M02 variants limits generalization to M01 variants.

3. **Small gap size**: For 10-timestep gaps, linear interpolation is very effective. Larger gaps might show more benefit from neural correction.

4. **No adaptive weighting**: The model always adds the full predicted residual. Adaptive weighting based on confidence might help.

## Conclusion

**The residual correction experiment did not improve upon linear interpolation.**

The Residual LSTM achieved an MSE of 0.007763 compared to linear interpolation's MSE of 0.007700, representing a -0.82% degradation in performance. This indicates that:

1. Linear interpolation is a very strong baseline for this reconstruction task
2. The residual network did not learn useful correction patterns
3. The neural model's predictions added noise rather than corrections

**Linear interpolation remains the best method for this task among those tested.**

## Recommended Next Steps

Based on these results, the recommended next steps are:

1. **Accept linear interpolation as the baseline**: Given its excellent performance and speed, linear interpolation should be the baseline for future experiments.

2. **Investigate larger gaps**: Test whether neural methods provide more benefit for larger gap sizes (e.g., 20-50 timesteps) where linear interpolation may be less accurate.

3. **Explore domain adaptation**: If neural methods are needed for generalization across tool variants, investigate domain adaptation techniques rather than residual correction.

4. **Ensemble methods**: Explore ensemble approaches that combine multiple interpolation methods (linear, spline, etc.) rather than neural correction.

5. **Focus on data diversity**: If neural methods are pursued, ensure training data includes diverse tool variants to improve generalization.

## Files Generated

- `audit_report.md` - Implementation audit report
- `checkpoints/best_model.pt` - Trained residual LSTM checkpoint
- `results/config.json` - Training configuration
- `results/training_curves.png` - Training and validation loss curves
- `results/evaluation_results.json` - Complete evaluation results
- `results/metrics_summary.csv` - Metrics summary CSV
- `results/overall_mse_comparison.png` - Overall MSE comparison plot
- `results/per_sensor_mse_comparison.png` - Per-sensor MSE comparison plot
- `results/per_timestep_mse_comparison.png` - Per-timestep MSE comparison plot
- `REPORT.md` - This report

## Reproducible Commands

```bash
# Phase 1: Audit implementation
python src/audit_implementation.py

# Phase 2-3: Train residual model
python src/train_residual_lstm.py

# Phase 4: Evaluate on test set
python src/evaluate_residual_lstm.py
```
"""

report_path = RESIDUAL_DIR / "REPORT.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Report saved to: {report_path}")
print("Done.")
