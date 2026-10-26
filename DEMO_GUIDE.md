# Industrial Sensor AI Demo Guide

## Quick Start

### 1. Start the Backend Server
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Frontend Server
```bash
cd frontend
npm run dev
```

### 3. Access the Application
Open your browser to: `http://localhost:5174` (or the port shown in the terminal)

The default page is the **Sensor Dashboard** with all reconstruction methods.

---

## 5-Minute Demonstration Sequence

### Step 1: Introduction (1 minute)
**Navigate to:** Sensor Dashboard (default page)

**Talking Points:**
- "This is an industrial sensor reconstruction system for semiconductor manufacturing equipment."
- "The system has 17 sensor features including pressure, voltage, current, flow rates, and rotation parameters."
- "We demonstrate four reconstruction methods: Linear Interpolation, Attention Seq2Seq, Hybrid Reconstruction, and Residual LSTM."

**What to Show:**
- The three gap configuration controls (left context, gap length, right context)
- The method selector dropdown showing all four methods
- The gap length selector (10, 20, 50)

---

### Step 2: Linear Interpolation Baseline (1 minute)
**Configuration:**
- Gap Length: 10
- Method: Linear Interpolation
- File: test_03_M01_DC_test.npz
- Window Index: 0

**Talking Points:**
- "Linear interpolation is the mathematical baseline - it draws a straight line between observed boundary values."
- "It requires no training and works for any gap length."
- "For short gaps like 10 timesteps, it often performs very well."

**What to Show:**
- Click "Run Reconstruction"
- Show the time-series chart with the reconstructed segment
- Point out the MSE, MAE, RMSE metrics
- Note the inference time (should be near-instantaneous)

---

### Step 3: Attention Seq2Seq (1 minute)
**Configuration:**
- Gap Length: 10
- Method: Attention Seq2Seq
- Same file and window

**Talking Points:**
- "Attention Seq2Seq is a deep learning model with bidirectional LSTM encoder and additive attention."
- "It was trained for 50 epochs on the PHM 2018 dataset."
- "It only supports gap length 10 due to training constraints."

**What to Show:**
- Click "Run Reconstruction"
- Compare the metrics with linear interpolation
- Note the inference time (typically 10-50ms)
- Explain that for gap 10, the improvement over interpolation is modest

---

### Step 4: Hybrid Reconstruction (30 seconds)
**Configuration:**
- Gap Length: 10
- Method: Hybrid Reconstruction
- Same file and window

**Talking Points:**
- "Hybrid reconstruction combines both methods using a fixed-weight blend."
- "It uses 10% neural model and 90% interpolation."
- "This aims to get the best of both approaches."

**What to Show:**
- Click "Run Reconstruction"
- Show that the fusion weight is 0.1
- Compare metrics with the other methods

---

### Step 5: Residual LSTM for Different Gap Lengths (1 minute)
**Configuration:**
- Gap Length: 20
- Method: Residual LSTM
- Same file and window

**Talking Points:**
- "Residual LSTM learns corrections to linear interpolation rather than predicting from scratch."
- "It supports multiple gap lengths: 10, 20, and 50."
- "For longer gaps, neural methods may show more advantage."

**What to Show:**
- Change gap length to 20
- Click "Run Reconstruction"
- Show the metrics
- Change gap length to 50 and run again
- Note that absolute error increases for longer gaps (expected behavior)

---

### Step 6: Method Comparison Dashboard (30 seconds)
**What to Show:**
- Scroll down to the "Method Comparison" section
- Show the table comparing Linear Interpolation and Residual LSTM
- Point out the total number of test sequences
- Explain that these are actual evaluation results from held-out test data

**Talking Points:**
- "This comparison is based on 55,661 test sequences for gap 10."
- "The metrics show the overall performance across all test data."
- "For gap 10, linear interpolation often performs very well."

---

### Step 7: Training Epoch Comparison (30 seconds)
**What to Show:**
- Scroll to the "Training Epoch Comparison" section
- Show the table with epochs 20, 30, 40, 50
- Show the training/validation loss chart

**Talking Points:**
- "The model was trained for 50 epochs with early stopping."
- "Best validation loss was achieved at epoch 48."
- "The chart shows how training and validation loss decreased over time."
- "These are actual loss values from the training history."

---

### Step 8: Methodology Page (30 seconds)
**Navigate to:** Methodology (sidebar menu)

**Talking Points:**
- "The methodology page explains all the technical details."
- "It covers the problem statement, artificial masking, each reconstruction method, and trade-offs."
- "It explains why performance changes with gap length."

**What to Show:**
- Scroll through the methodology sections
- Point out the architecture diagrams and formulas
- Show the trade-offs section

---

## Key Technical Details to Remember

### Dataset
- **Source:** PHM 2018 Ion Mill Etch Tool dataset
- **Features:** 17 sensor features
- **Data Split:** Train (2 files), Validation (2 files), Test (5 files) - no leakage

### Methods
1. **Linear Interpolation**
   - No training required
   - Supports any gap length
   - Near-instantaneous inference
   - Formula: linear blend between boundary values

2. **Attention Seq2Seq**
   - Trained for 50 epochs (best at epoch 48)
   - Only supports gap length 10
   - ~331K parameters
   - Inference time: ~10-50ms
   - Architecture: BiLSTM encoder + additive attention + LSTM decoder

3. **Hybrid Reconstruction**
   - Fixed-weight blend: 10% Seq2Seq + 90% interpolation
   - Only supports gap length 10
   - Inference time: sum of both methods
   - Fusion weight determined from training data (not test data)

4. **Residual LSTM**
   - Learns corrections to interpolation
   - Supports gaps 10, 20, 50
   - ~83K-127K parameters
   - Inference time: ~10-50ms
   - Separate models for each gap length

### Gap Length Experiments
- **Gap 10:** 55,661 test sequences
- **Gap 20:** 51,643 test sequences
- **Gap 50:** 45,483 test sequences

### Evaluation Metrics
- **MSE:** Mean Squared Error
- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- All metrics reported in normalized units

---

## Common Questions and Answers

**Q: Why does linear interpolation perform so well for short gaps?**
A: Industrial sensor data typically changes smoothly over short time intervals. A straight-line approximation is often very accurate for 10 timesteps.

**Q: Why don't the neural models show dramatic improvements?**
A: For this dataset and task, linear interpolation is a strong baseline. The neural models show modest improvements, but the computational cost may not be justified for short gaps.

**Q: Can I use Attention Seq2Seq for gap length 20?**
A: No, it was only trained for gap length 10. Using it for other gap lengths would require retraining.

**Q: What's the difference between normalized and original units?**
A: Normalized units have zero mean and unit variance (from StandardScaler). Original units are the actual sensor readings. Metrics are typically reported in normalized units for fair comparison.

**Q: How was the hybrid fusion weight determined?**
A: The fusion weight (α=0.1) was determined from training/validation data experiments to prevent data leakage. It was not selected using test ground truth.

---

## Troubleshooting

### Backend server won't start
- Check if port 8000 is already in use: `netstat -ano | findstr :8000`
- Kill the process: `taskkill /PID <PID> /F`

### Frontend server won't start
- Check if port 5173 is in use
- Vite will automatically try the next available port (e.g., 5174)

### API errors
- Verify backend is running on port 8000
- Check browser console for CORS errors
- Verify the API endpoint URLs are correct

### Model loading errors
- Verify checkpoint paths exist in the experiments directory
- Check that the model architecture matches the checkpoint

---

## Files Changed in This Implementation

### Backend
- `backend/app/api/v1/sensor.py` - Added gap benchmark endpoints, epoch comparison, hybrid method support
- `backend/app/services/model_service.py` - Added hybrid reconstruction method

### Frontend
- `frontend/src/pages/SensorReconstructionDashboard.tsx` - Main dashboard with all methods and comparisons
- `frontend/src/pages/Methodology.tsx` - Technical explanation page
- `frontend/src/App.tsx` - Added routing for new pages
- `frontend/src/components/Sidebar.tsx` - added Sensor Dashboard and Methodology menu items

### Tests
- `test_api_endpoints.py` - Automated API endpoint tests

---

## Running Automated Tests

```bash
python test_api_endpoints.py
```

This tests all critical endpoints including:
- Gap benchmark info and windows
- All four reconstruction methods
- Evaluation summary
- Epoch comparison
- Input validation

Expected result: 11 passed, 0 failed
