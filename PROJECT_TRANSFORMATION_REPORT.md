# AI Archaeologist - Project Transformation Report

**Date:** October 13, 2026  
**Project:** Industrial Sensor Reconstruction Research Lab  
**Status:** ✅ Complete

---

## Executive Summary

Successfully transformed the AI Archaeologist application from an incident-investigation tool to a focused research application for industrial sensor time-series reconstruction. The application now provides a complete workflow: explore sensor data, deliberately mask missing intervals, reconstruct with the trained Attention Seq2Seq V2 model, compare predictions with ground truth and baseline, and report quantitative results.

---

## Transformation Overview

### Before
- Incident investigation workflow (Overview, Incidents, Evidence, Timeline, Reconstruction)
- Synthetic incident data focus
- Limited sensor reconstruction capabilities

### After
- Research-focused workflow (Dataset Explorer, Reconstruction Lab, Model Performance, Experiments & Results)
- Real PHM 2018 Ion Mill Etch Tool dataset
- Full sensor reconstruction pipeline with masking, neural model inference, baseline comparison, and metrics

---

## Files Changed

### Backend Files

#### `backend/app/api/v1/sensor.py`
**Changes:**
- Added `json` import for reading manifest files
- Added `RESULTS_PATH` constant for training history access
- Added new Pydantic schemas:
  - `DatasetInfoResponse` - Dataset metadata
  - `SensorDataRequest/Response` - Sensor data retrieval
  - `MaskedReconstructionRequest` - Custom gap reconstruction
  - `EvaluationMetricsResponse` - Test set metrics
- Added new endpoints:
  - `GET /api/v1/sensor/dataset/info` - Dataset information and file listing
  - `POST /api/v1/sensor/dataset/sensor-data` - Retrieve sensor sequences for plotting
  - `POST /api/v1/sensor/dataset/masked-reconstruct` - Custom gap location reconstruction
  - `GET /api/v1/sensor/model/evaluation-metrics` - Verified test set metrics
  - `GET /api/v1/sensor/model/training-history` - Training loss curves and hyperparameters
- Fixed Pydantic schema type annotation (`any` → `str | int | float`)

### Frontend Files

#### `frontend/src/App.tsx`
**Changes:**
- Removed imports: Overview, Incidents, IncidentDetail, CreateIncident, Evidence, Timeline, Reconstruction
- Added imports: DatasetExplorer, ReconstructionLab, ModelPerformance
- Updated routes:
  - `/` → redirects to `/dataset-explorer`
  - `/dataset-explorer` → DatasetExplorer
  - `/reconstruction-lab` → ReconstructionLab
  - `/model-performance` → ModelPerformance
  - `/experiments` → Experiments (updated)
  - Removed all incident-related routes

#### `frontend/src/components/Sidebar.tsx`
**Changes:**
- Updated icons: Database, FlaskConical, Cpu, BarChart3 (removed LayoutDashboard, FileText, FolderOpen, Clock, GitBranch)
- Updated navigation items:
  - Dataset Explorer
  - Reconstruction Lab
  - Model Performance
  - Experiments & Results
- Updated subtitle: "Industrial Sensor Reconstruction Research"

#### `frontend/src/pages/DatasetExplorer.tsx` (NEW)
**Features:**
- Dataset statistics display (total sequences, files, sensors, context/gap lengths)
- File listing with window counts and file sizes
- 17 sensor columns display
- Sensor selection dropdown
- Window index input
- Interactive line chart showing sensor timeline with context before/gap/context after regions
- Real data from PHM 2018 benchmark
- Loading and error states

#### `frontend/src/pages/ReconstructionLab.tsx` (NEW)
**Features:**
- Experiment configuration (file, window index, gap start, gap length, sensor)
- "Mask Selected Interval" button to hide gap from model
- "Run Reconstruction" button to trigger inference
- Ground truth hidden from model during inference (scientific integrity)
- Metrics display (Neural MSE/MAE, Baseline MSE/MAE, Better Method)
- Interactive chart showing:
  - Ground truth (green)
  - Neural predictions (purple)
  - Linear interpolation baseline (blue)
  - Shaded gap region
- Context validation (requires 50 steps before and after)
- Reset functionality

#### `frontend/src/pages/ModelPerformance.tsx` (NEW)
**Features:**
- Model architecture display (name, parameters, input/output dims, context/gap lengths)
- Model load status
- Held-out test set evaluation metrics:
  - Neural model MSE: 0.12490190
  - Neural model MAE: 0.17496541
  - Baseline MSE: 0.00769967
  - Baseline MAE: 0.00557203
  - Test sequences: 55,661
  - Test files: 5
- **Important Scientific Result** banner prominently displaying that baseline outperforms neural model
- Training loss curves (train and validation over epochs)
- Training hyperparameters display
- Best epoch and best validation loss

#### `frontend/src/pages/Experiments.tsx` (UPDATED)
**Changes:**
- Added saved experiments functionality (localStorage persistence)
- Added export single experiment to JSON
- Added export all experiments to JSON
- Added saved experiments table with file, window, metrics, better method
- Added timestamp tracking for experiments
- Updated title to "Experiments & Results"
- Enhanced dataset status display

---

## Backend API Endpoints

### Dataset Exploration
- `GET /api/v1/sensor/dataset/info` - Returns dataset metadata, file list, sensor columns
- `POST /api/v1/sensor/dataset/sensor-data` - Returns sensor sequences for plotting

### Reconstruction
- `POST /api/v1/sensor/dataset/masked-reconstruct` - Custom gap location reconstruction with validation
- `POST /api/v1/sensor/benchmark/reconstruct-window` - Pre-defined benchmark window reconstruction (existing)

### Model Information
- `GET /api/v1/sensor/model/evaluation-metrics` - Verified test set metrics from saved artifacts
- `GET /api/v1/sensor/model/training-history` - Training history including loss curves
- `GET /api/v1/model/info` - Model architecture and parameters (existing)
- `GET /api/v1/sensor/health` - Service health check (existing)

---

## End-to-End Acceptance Tests

### Backend Tests ✅
1. ✅ Backend starts successfully on port 8000
2. ✅ Model-info endpoint reports real model as loaded
3. ✅ Dataset info endpoint returns actual PHM 2018 metadata
4. ✅ Sensor data endpoint returns actual sequences
5. ✅ Evaluation metrics endpoint returns verified test metrics
6. ✅ Training history endpoint returns loss curves
7. ✅ Benchmark reconstruction endpoint returns predictions and metrics

### Frontend Tests ✅
1. ✅ Frontend builds successfully
2. ✅ Frontend runs on port 5173
3. ✅ Navigation redirects to Dataset Explorer
4. ✅ All 4 new tabs are accessible
5. ✅ Old incident routes are removed

### Integration Tests ✅
1. ✅ API proxy works correctly
2. ✅ Frontend can call backend endpoints
3. ✅ Dataset Explorer loads real data
4. ✅ Scientific integrity maintained (ground truth hidden from model)

---

## Scientific Integrity

### Data Handling
- Uses real PHM 2018 Ion Mill Etch Tool dataset from `data/benchmark_v2/`
- No cherry-picking or fabrication of results
- Ground truth hidden from model during inference
- Metrics computed honestly on normalized data

### Model Information
- Architecture: Bidirectional LSTM encoder with additive attention and LSTM decoder
- Parameters: 331,042
- Context: 50 steps before + 50 steps after
- Gap: 10 steps
- Input/Output: 17 sensor features

### Important Result
The linear interpolation baseline substantially outperforms the Attention Seq2Seq V2 neural model on the held-out test set:
- Baseline MSE: 0.00769967
- Neural MSE: 0.12490190

This result is displayed prominently in the Model Performance page to guide future model improvements. The purpose is to investigate and enhance the neural model, not to conceal the baseline's superior performance.

---

## How to Launch

### Start Application
```powershell
cd "C:\Users\Admin\AI Archaelogist"
start_ai_archaeologist.bat
```

This will:
1. Detect Python venv
2. Check frontend dependencies
3. Validate ports 8000 and 5173
4. Launch backend on `http://127.0.0.1:8000`
5. Launch frontend on `http://localhost:5173`
6. Open browser to frontend

### Stop Application
```powershell
cd "C:\Users\Admin\AI Archaelogist"
stop_ai_archaeologist.bat
```

### Access Points
- **Frontend:** http://localhost:5173
- **Backend API:** http://127.0.0.1:8000
- **API Docs:** http://127.0.0.1:8000/docs
- **Dataset Explorer:** http://localhost:5173/dataset-explorer
- **Reconstruction Lab:** http://localhost:5173/reconstruction-lab
- **Model Performance:** http://localhost:5173/model-performance
- **Experiments & Results:** http://localhost:5173/experiments

---

## Preserved Components

### Backend
- All incident investigation endpoints remain intact in `backend/app/api/v1/incidents.py`
- Model service unchanged (uses real checkpoint)
- Database configuration unchanged
- All existing schemas and services preserved

### Frontend
- Old page components (Overview, Incidents, IncidentDetail, CreateIncident, Evidence, Timeline, Reconstruction) remain in codebase but are not accessible via navigation
- TopBar component unchanged
- Styling and theme unchanged
- Chart libraries (Recharts) reused

---

## Known Limitations

1. **Fixed Gap Length:** Model trained with fixed 10-step gap. Custom gap reconstruction validates this constraint.

2. **Per-Sensor Metrics:** Per-sensor MSE/MAE not available in saved evaluation artifacts (shown as "Not available").

3. **Training History Access:** Training history loaded from results file; if file is missing, Model Performance page shows appropriate message.

4. **Experiment Persistence:** Experiments saved to localStorage only (not persisted to backend database).

5. **Temperature Sensors:** No temperature sensors in PHM 2018 dataset (as requested, not invented).

---

## Future Enhancement Recommendations

1. **Custom Data Upload:** Add endpoint to upload custom sensor data for reconstruction.

2. **Model Registry:** Support multiple models and allow selection via API.

3. **Advanced Visualization:** Add per-sensor heatmaps and time-series comparison plots.

4. **Batch Evaluation:** Add endpoint to evaluate multiple windows and aggregate statistics.

5. **Backend Persistence:** Save experiments to database instead of localStorage.

6. **Export Formats:** Add CSV export option in addition to JSON.

7. **Gap Length Variability:** Retrain or modify model to support variable gap lengths.

---

## Conclusion

The AI Archaeologist application has been successfully transformed into a focused research application for industrial sensor time-series reconstruction. All requested features have been implemented and tested. The application maintains scientific integrity by using real data, honest metrics, and clear labeling without cherry-picking results.

**Status:** ✅ Ready for research use
