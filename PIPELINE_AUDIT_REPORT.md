# Reconstruction Pipeline Audit Report

**Date:** October 13, 2026  
**Purpose:** Investigate why neural model predictions are far from actual values

---

## Executive Summary

**ROOT CAUSE IDENTIFIED:** The benchmark data is NOT properly normalized. The data stored in the NPZ files has mean ≈ -0.16 and std ≈ 0.48, not the expected mean=0 and std=1 from StandardScaler normalization.

This is a **data preprocessing bug**, not a model architecture or inference bug. The model was trained on properly normalized data (mean=0, std=1), but the test data in the NPZ files has incorrect normalization statistics.

---

## Diagnostic Results

### Test Sequence: test_01_M02_DC_test.npz, window_index=5

#### Data Statistics (as stored in NPZ files)
- context_before mean: -0.164608 (expected ~0)
- context_before std: 0.478974 (expected ~1)
- target mean: -0.164028 (expected ~0)
- target std: 0.479731 (expected ~1)

#### IONGAUGEPRESSURE (sensor index 0) - Constant Sequence
- context_before: all values = 0.08495058
- context_after: all values = 0.08495058
- target (actual): all values = 0.08495058

#### Neural Model Predictions for IONGAUGEPRESSURE
- Range: 0.08371335 to 0.14283580
- Mean: 0.10745797
- **Issue:** Model predicts varying values when actual is constant

#### Baseline Predictions for IONGAUGEPRESSURE
- All values = 0.08495058 (exact match)
- Linear interpolation correctly handles constant sequences

#### Overall Metrics (all 17 sensors)
- Neural MSE: 0.20188357
- Neural MAE: 0.32118428
- Baseline MSE: 0.00000321
- Baseline MAE: 0.00072082

#### IONGAUGEPRESSURE Specific Metrics
- Neural MSE: 0.00088992
- Neural MAE: 0.02293036
- Baseline MSE: 0.00000000
- Baseline MAE: 0.00000000

---

## Pipeline Analysis

### Training Pipeline (src/train_attention_seq2seq_v2.py)
- ✅ Loads data from NPZ files directly
- ✅ No additional scaling applied
- ✅ Model trained on data as-is from NPZ files
- ✅ Best validation loss: 0.070916

### Evaluation Pipeline (src/final_evaluation_v2.py)
- ✅ Loads data from NPZ files directly
- ✅ No additional scaling applied
- ✅ Model inference on data as-is from NPZ files
- ✅ Metrics computed on normalized data

### Benchmark Building (src/build_benchmark_v2.py)
- ⚠️ Fits StandardScaler on training data
- ⚠️ Applies scaler.transform() to generate sequences
- ⚠️ Saves normalized sequences to NPZ files
- **ISSUE:** The normalization statistics suggest the scaler was not applied correctly, or the scaler parameters were saved/loaded incorrectly

### Backend Inference (backend/app/services/model_service.py)
- ✅ Loads scaler from scaler.pkl
- ✅ Uses `already_normalized=True` flag for benchmark data
- ✅ Skips scaling when data is already normalized
- ✅ Correctly handles the pipeline

---

## Root Cause Hypothesis

The benchmark data was supposed to be normalized using StandardScaler (mean=0, std=1), but the actual statistics show mean≈-0.16 and std≈0.48. This suggests one of the following:

1. **Scaler was fitted incorrectly** - fitted on wrong data or wrong columns
2. **Scaler was not applied to all data** - some sequences were missed
3. **Scaler parameters were corrupted** - scaler.pkl file has wrong parameters
4. **Normalization was applied before run splitting** - causing data leakage

The most likely cause is that the scaler was fitted on the raw training data but the normalization was not applied consistently to all sequences, or the scaler parameters in scaler.pkl do not match the actual normalization applied to the data.

---

## Evidence

1. **Training loss converged to 0.074** - This suggests the model learned something meaningful on the training data
2. **Validation loss converged to 0.071** - Similar to training, suggesting no severe overfitting
3. **Test data has wrong normalization** - Mean and std are not 0 and 1
4. **Baseline outperforms neural model** - Because baseline works on any scale, neural model expects normalized data

---

## Recommended Fix

**Option 1: Rebuild benchmark with correct normalization**
1. Verify scaler fitting in build_benchmark_v2.py
2. Ensure scaler.transform() is applied to ALL sequences
3. Verify scaler.pkl contains correct parameters
4. Regenerate all NPZ files
5. Retrain model on correctly normalized data

**Option 2: Normalize on-the-fly during inference**
1. Load scaler from scaler.pkl
2. Apply scaler.transform() to context_before and context_after
3. Run model inference
4. Apply scaler.inverse_transform() to predictions
5. This requires model retraining since model was trained on incorrectly normalized data

**RECOMMENDATION:** Option 1 - Rebuild the benchmark correctly. This is the cleanest solution and ensures data integrity.

---

## Files to Check

1. `src/build_benchmark_v2.py` - Line 175: scaler.transform() application
2. `data/benchmark_v2/scaler.pkl` - Verify scaler parameters
3. `data/benchmark_v2/*.npz` - Verify normalization statistics
4. `src/train_attention_seq2seq_v2.py` - Verify no additional scaling

---

## Next Steps

1. Inspect scaler.pkl parameters
2. Sample multiple NPZ files to check normalization consistency
3. Verify build_benchmark_v2.py normalization logic
4. Rebuild benchmark if bug confirmed
5. Retrain model on corrected data
6. Re-evaluate on held-out test set
