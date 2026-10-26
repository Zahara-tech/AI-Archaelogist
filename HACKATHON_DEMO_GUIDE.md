# AI Archaeologist - Hackathon Demonstration Guide

## Overview

This guide provides a step-by-step demonstration of the AI Archaeologist system for industrial sensor reconstruction. The system compares mathematical interpolation with neural-network approaches for reconstructing missing sensor observations.

## Prerequisites

- Backend server running on `http://localhost:8000`
- Frontend running on `http://localhost:5173`
- Required data and models available in the project structure

## Starting the Application

### Backend (FastAPI)
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend (React)
```bash
cd frontend
npm run dev
```

Access the dashboard at: `http://localhost:5173`

---

## 10-Step Demonstration Script

### Step 1: Introduce the Problem (2 minutes)

**What to show:**
- Navigate to the dashboard home page
- Point out the "Industrial AI Sensor Reconstruction" header
- Explain the context: Industrial sensors (17 features) from semiconductor manufacturing equipment
- The problem: Sensors sometimes have missing observations (gaps) that need reconstruction

**Key talking points:**
- "This is an experimental system for reconstructing missing industrial sensor data"
- "We have 17 sensor features from ion mill etch equipment"
- "When data is missing, we need to estimate it accurately"
- "We compare simple mathematical methods with neural network approaches"

---

### Step 2: Show Default Configuration (50-10-50) (1 minute)

**What to show:**
- Point to the "Reconstruction Configuration" section
- Show the default preset: "50-10-50"
- Explain: 50 observations before, 10 missing, 50 observations after
- Total window: 110 timesteps

**Key talking points:**
- "Our default configuration uses 50 timesteps of context before and after the gap"
- "The gap itself is 10 timesteps - this is a typical short missing interval"
- "The total analysis window is 110 timesteps"

---

### Step 3: Run Attention Seq2Seq (2 minutes)

**What to show:**
- Select "Attention Seq2Seq" from the method dropdown
- Select a test file (e.g., "test_03_M01_DC_test.npz")
- Select a sensor (e.g., "ETCHBEAMVOLTAGE")
- Click "Run Reconstruction"
- Wait for results to appear

**Key talking points:**
- "Attention Seq2Seq is a deep learning model with bidirectional LSTM encoder and attention mechanism"
- "It was trained specifically for the 50-10-50 configuration"
- "Let's run it and see the reconstruction quality"

**After results appear:**
- Point to the visualization showing ground truth vs prediction
- Show the metrics: MSE, MAE, RMSE
- Note the inference time

---

### Step 4: Compare with Linear Interpolation (2 minutes)

**What to show:**
- Change method to "Linear Interpolation"
- Click "Run Reconstruction" again
- Compare the metrics with Attention Seq2Seq

**Key talking points:**
- "Linear interpolation is a simple mathematical baseline"
- "It estimates missing values by drawing a straight line between the boundary observations"
- "It requires no training and works for any gap length"
- "Notice that for short gaps (10 timesteps), linear interpolation often performs very well"

**Honest observation:**
- "In our experiments, linear interpolation often matches or beats neural models for short gaps"
- "This is an important finding - simple methods can be effective"

---

### Step 5: Demonstrate Residual LSTM (2 minutes)

**What to show:**
- Change method to "Residual LSTM"
- Click "Run Reconstruction"
- Compare metrics with both previous methods

**Key talking points:**
- "Residual LSTM is a hybrid approach that learns corrections to linear interpolation"
- "Final prediction = linear interpolation + learned neural correction"
- "This approach can potentially capture non-linear patterns that pure interpolation misses"

**Honest observation:**
- "The improvement over linear interpolation is often modest"
- "The computational cost is higher than pure interpolation"
- "Whether the improvement justifies the cost depends on the application"

---

### Step 6: Demonstrate Configurable Gap (30-20-30) (2 minutes)

**What to show:**
- Select preset "30-20-30" from the dropdown
- Explain: 30 observations before, 20 missing, 30 observations after
- Note that Attention Seq2Seq is now disabled (not compatible)
- Run Linear Interpolation and Residual LSTM for comparison

**Key talking points:**
- "The system supports configurable gap configurations"
- "For longer gaps (20 timesteps), neural methods might show more advantage"
- "Attention Seq2Seq was only trained for gap length 10, so it's not available here"
- "Residual LSTM has separate models trained for different gap lengths"

**Important limitation:**
- "We don't claim every model works for every configuration"
- "The interface clearly shows which methods are compatible"

---

### Step 7: Demonstrate Longer Gap (40-50-40) (2 minutes)

**What to show:**
- Select preset "40-50-40"
- Explain: 40 observations before, 50 missing, 40 observations after
- This is a much longer missing interval
- Run Linear Interpolation and Residual LSTM

**Key talking points:**
- "For very long gaps (50 timesteps), the reconstruction problem becomes more challenging"
- "Linear interpolation may struggle with complex patterns over long intervals"
- "Neural methods might show more relative advantage here"
- "However, the absolute error will likely be higher for all methods"

---

### Step 8: Show Method Comparison Dashboard (2 minutes)

**What to show:**
- Scroll to the "Method Comparison" section at the bottom
- Point out the table comparing Linear Interpolation vs Residual LSTM
- Show the metrics for the current gap length
- Note the "Best" indicator

**Key talking points:**
- "This section shows aggregated results from our held-out test set"
- "We evaluated on thousands of test windows across multiple files"
- "The comparison is fair - same ground truth, same evaluation protocol"
- "The checkmark indicates which method achieved lower MSE"

**Honest interpretation:**
- "The differences are often modest"
- "Linear interpolation remains a strong baseline"
- "Neural methods show small improvements in some cases, but not dramatically"

---

### Step 9: Show Per-Sensor Analysis (1 minute)

**What to show:**
- In the "Performance Dashboard" section, scroll to the per-sensor table
- Point out that different sensors have different reconstruction errors
- Select different sensors from the dropdown to see variations

**Key talking points:**
- "Reconstruction quality varies by sensor"
- "Some sensors are more predictable than others"
- "This per-sensor analysis helps understand which features are harder to reconstruct"

---

### Step 10: Explain Findings and Practical Value (2 minutes)

**What to show:**
- Summarize the key findings from the demonstration
- Explain the practical implications for industrial applications

**Key talking points:**

**Findings:**
1. **Linear interpolation is a strong baseline** - For short gaps (10 timesteps), it often matches or beats neural methods
2. **Neural methods show modest improvements** - Residual LSTM can improve over interpolation, especially for longer gaps
3. **Gap length matters** - Longer gaps are more challenging for all methods
4. **Sensor variability** - Some sensors are reconstructed more accurately than others

**Practical Value:**
- **Simplicity first** - Start with linear interpolation; it's fast, requires no training, and works well for short gaps
- **Neural methods when justified** - Consider neural approaches when: (a) gaps are longer, (b) computational budget allows, (c) small improvements matter
- **Honest evaluation** - Don't assume neural is always better; test against baselines
- **Configurable approach** - The system allows selecting the right method for each use case

**Limitations to acknowledge:**
- "This is an experimental prototype, not a production system"
- "We don't claim to detect equipment faults - that's a different problem"
- "The models are trained on specific equipment and may not generalize"
- "The evaluation is on historical data; real-world performance may vary"

---

## Backup Demo Plan

If the main demonstration encounters issues:

### Alternative 1: Focus on Gap Length Comparison
- Skip the single-window reconstruction
- Go directly to the Method Comparison section
- Show how performance changes across gap lengths (10, 20, 50)
- Discuss the trend: longer gaps = higher error for all methods

### Alternative 2: Focus on Per-Sensor Analysis
- Select a specific sensor that shows interesting patterns
- Compare reconstruction quality across different sensors
- Discuss why some sensors are harder to reconstruct

### Alternative 3: Technical Deep Dive
- Explain the model architectures (Attention Seq2Seq, Residual LSTM)
- Show the training configuration and hyperparameters
- Discuss the evaluation protocol (train/validation/test splits)

---

## Common Questions and Answers

**Q: Why doesn't Attention Seq2Seq work for gap length 20 or 50?**
A: It was only trained for gap length 10. Retraining would be needed for other configurations. We don't resize or reshape predictions to make incompatible models appear to work.

**Q: Which method should I use in production?**
A: Start with linear interpolation - it's simple, fast, and effective for short gaps. Consider neural methods only if you have longer gaps and the computational budget justifies the modest improvement.

**Q: Can this system detect equipment faults?**
A: No. This system reconstructs missing data. Fault detection is a different problem that would require additional components.

**Q: How were the models trained?**
A: Using the PHM 2018 dataset with strict train/validation/test splits. The scaler was fitted only on training data to prevent data leakage.

**Q: What's the inference time?**
A: Linear interpolation is nearly instantaneous. Neural methods take 10-100ms depending on the model and hardware.

**Q: Can I use my own data?**
A: The system is designed for the PHM 2018 dataset structure. Using different data would require preprocessing and potentially retraining.

---

## Technical Details for Judges

### Model Architectures

**Attention Seq2Seq:**
- Bidirectional LSTM encoder (2 layers, 64 hidden units)
- Additive attention mechanism
- LSTM decoder (2 layers, 64 hidden units)
- Output projection to 17 sensor features
- ~331K parameters
- Trained for 50 epochs with early stopping

**Residual LSTM:**
- LSTM encoder (2 layers, 64 hidden units)
- Predicts correction to linear interpolation
- Final = interpolation + correction
- ~83K-127K parameters (varies by gap length)
- Separate model trained for each gap length (10, 20, 50)

### Evaluation Protocol

- **Data split:** Temporal split by runs (train/validation/test)
- **Scaler:** StandardScaler fitted on training data only
- **Metrics:** MSE, MAE, RMSE in normalized space
- **Test set:** Held-out files not used during training
- **Window count:** ~45K-55K test windows per gap length

### Files Modified/Created

**Backend:**
- `backend/app/services/model_service.py` - Added multi-method support, residual LSTM loading
- `backend/app/api/v1/sensor.py` - Added gap benchmark endpoints, method selection

**Frontend:**
- `frontend/src/pages/SensorReconstructionDashboard.tsx` - New comprehensive dashboard
- `frontend/src/App.tsx` - Added routing for new dashboard

**Existing Checkpoints Used:**
- `experiments/attention_seq2seq_v2_50epochs/checkpoints/best_model.pt`
- `experiments/gap_length_experiment/gap_10/checkpoints/best_model.pt`
- `experiments/gap_length_experiment/gap_20/checkpoints/best_model.pt`
- `experiments/gap_length_experiment/gap_50/checkpoints/best_model.pt`

**No retraining performed** - All models reused from previous experiments.

---

## Conclusion

This demonstration shows an honest, evidence-based comparison of reconstruction methods. The key takeaway is that simple baselines like linear interpolation are often competitive with neural approaches, especially for short gaps. Neural methods can provide modest improvements for longer gaps, but the benefit must be weighed against computational cost and complexity.

The system is designed to be configurable, allowing users to select the appropriate method for their specific use case and gap configuration.
