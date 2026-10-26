import React, { useState, useEffect, useMemo } from 'react';
import { 
  Play, 
  Settings, 
  BarChart3, 
  TrendingUp, 
  Clock, 
  AlertCircle, 
  CheckCircle, 
  Cpu,
  ChevronLeft,
  ChevronRight,
  Layers,
  Sparkles,
  Info,
  Check,
  AlertTriangle,
  RefreshCw,
  Activity
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ReferenceArea,
  ReferenceLine
} from 'recharts';

interface MethodDef {
  id: string;
  name: string;
  category: string;
  description: string;
  supportedGaps: number[];
  color: string;
  badge: string;
}

const METHODS: MethodDef[] = [
  {
    id: 'linear_interpolation',
    name: 'Linear Interpolation',
    category: 'Mathematical Baseline',
    description: 'Boundary interpolation prior between last observed and next observed values.',
    supportedGaps: [10, 20, 50],
    color: '#3b82f6',
    badge: 'Baseline'
  },
  {
    id: 'residual_lstm',
    name: 'Residual LSTM',
    category: 'Neural Residual Network',
    description: 'Deep LSTM that predicts residual delta on top of linear interpolation.',
    supportedGaps: [10, 20, 50],
    color: '#f59e0b',
    badge: 'Preferred AI'
  },
  {
    id: 'hybrid',
    name: 'Hybrid Reconstruction',
    category: 'Adaptive Blended Model',
    description: 'Fuses Attention Seq2Seq (1%) and Linear Interpolation (99%) with validated alpha.',
    supportedGaps: [10],
    color: '#10b981',
    badge: 'Gap 10 Only'
  },
  {
    id: 'attention_seq2seq',
    name: 'Attention Seq2Seq',
    category: 'Encoder-Decoder Network',
    description: 'Bidirectional LSTM encoder with Bahdanau attention and LSTM decoder.',
    supportedGaps: [10],
    color: '#8b5cf6',
    badge: 'Gap 10 Only'
  }
];

const PRESET_CONFIGS = [
  { name: '50-10-50 (Standard)', before: 50, gap: 10, after: 50, desc: 'Short gap baseline' },
  { name: '30-20-30', before: 30, gap: 20, after: 30, desc: 'Medium gap length' },
  { name: '30-30-30', before: 30, gap: 30, after: 30, desc: 'Custom gap length' },
  { name: '30-40-30', before: 30, gap: 40, after: 30, desc: 'Custom gap length' },
  { name: '40-50-40', before: 40, gap: 50, after: 40, desc: 'Large gap length' },
  { name: '50-50-50 (Full)', before: 50, gap: 50, after: 50, desc: 'Maximum evaluated gap' },
];

export const ReconstructionStudio: React.FC = () => {
  // State
  const [selectedFile, setSelectedFile] = useState<string>('test_01_M02_DC_test.npz');
  const [availableFiles, setAvailableFiles] = useState<Array<{ filename: string; window_count: number }>>([]);
  const [windowIndex, setWindowIndex] = useState<number>(0);
  const [maxWindows, setMaxWindows] = useState<number>(1000);
  
  const [contextBefore, setContextBefore] = useState<number>(50);
  const [gapLength, setGapLength] = useState<number>(10);
  const [contextAfter, setContextAfter] = useState<number>(50);
  
  const [selectedMethod, setSelectedMethod] = useState<string>('residual_lstm');
  const [selectedSensor, setSelectedSensor] = useState<string>('IONGAUGEPRESSURE');
  const [sensorColumns, setSensorColumns] = useState<string[]>([]);
  
  // Single-run results
  const [reconstructionData, setReconstructionData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Compare-all results
  const [comparisonData, setComparisonData] = useState<any>(null);
  const [comparingAll, setComparingAll] = useState<boolean>(false);
  const [showMultiChart, setShowMultiChart] = useState<boolean>(false);

  // Load available files and sensor list on mount
  useEffect(() => {
    fetch('/api/v1/sensor/gap-benchmark/info')
      .then(res => res.json())
      .then(() => {
        return fetch(`/api/v1/sensor/gap-benchmark/${gapLength === 20 ? 20 : gapLength === 50 ? 50 : 10}/windows`);
      })
      .then(res => res.json())
      .then(data => {
        if (data.windows && data.windows.length > 0) {
          setAvailableFiles(data.windows);
          const current = data.windows.find((w: any) => w.filename === selectedFile);
          if (current) setMaxWindows(current.window_count);
        }
      })
      .catch(err => console.error('Failed to load window metadata:', err));

    fetch('/api/v1/sensor/dataset/info')
      .then(res => res.json())
      .then(data => {
        if (data.sensor_columns) {
          setSensorColumns(data.sensor_columns);
          if (!selectedSensor && data.sensor_columns.length > 0) {
            setSelectedSensor(data.sensor_columns[0]);
          }
        }
      })
      .catch(err => console.error('Failed to load dataset info:', err));
  }, []);

  // Update window file list whenever gapLength changes
  useEffect(() => {
    const benchmarkGap = gapLength === 50 ? 50 : gapLength === 20 ? 20 : 10;
    fetch(`/api/v1/sensor/gap-benchmark/${benchmarkGap}/windows`)
      .then(res => res.json())
      .then(data => {
        if (data.windows && data.windows.length > 0) {
          setAvailableFiles(data.windows);
          const current = data.windows.find((w: any) => w.filename === selectedFile) || data.windows[0];
          if (current) {
            setSelectedFile(current.filename);
            setMaxWindows(current.window_count);
          }
        }
      })
      .catch(err => console.error('Failed to update gap windows:', err));
  }, [gapLength]);

  // Check compatibility of selected method
  const isMethodCompatible = useMemo(() => {
    const def = METHODS.find(m => m.id === selectedMethod);
    if (!def) return false;
    if (def.id === 'linear_interpolation') return true;
    if (def.id === 'attention_seq2seq' || def.id === 'hybrid') return gapLength === 10;
    if (def.id === 'residual_lstm') return [10, 20, 50].includes(gapLength);
    return false;
  }, [selectedMethod, gapLength]);

  // Execute Reconstruction
  const runReconstruction = async () => {
    if (!isMethodCompatible) return;
    setLoading(true);
    setError(null);

    const benchmarkGap = gapLength === 50 ? 50 : gapLength === 20 ? 20 : 10;

    try {
      const response = await fetch('/api/v1/sensor/gap-benchmark/reconstruct-window', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          gap_length: benchmarkGap,
          filename: selectedFile,
          window_index: windowIndex,
          method: selectedMethod,
          context_before: contextBefore,
          context_after: contextAfter
        })
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned ${response.status}`);
      }

      const data = await response.json();
      setReconstructionData(data);
    } catch (err: any) {
      setError(err.message || 'Failed to run reconstruction');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Compare All Methods on Current Sample
  const runCompareAll = async () => {
    setComparingAll(true);
    setError(null);
    const benchmarkGap = gapLength === 50 ? 50 : gapLength === 20 ? 20 : 10;

    try {
      const response = await fetch('/api/v1/sensor/gap-benchmark/compare-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          gap_length: benchmarkGap,
          filename: selectedFile,
          window_index: windowIndex,
          context_before: contextBefore,
          context_after: contextAfter
        })
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned ${response.status}`);
      }

      const data = await response.json();
      setComparisonData(data);
      // Also update single reconstruction data with the currently selected method from comparison
      if (data.methods && data.methods[selectedMethod] && data.methods[selectedMethod].available) {
        setReconstructionData({
          gap_length: data.gap_length,
          filename: data.filename,
          window_index: data.window_index,
          method: selectedMethod,
          predictions: data.methods[selectedMethod].predictions,
          ground_truth: data.ground_truth,
          metrics: data.methods[selectedMethod].metrics,
          sensor_columns: data.sensor_columns,
          context_before: data.context_before,
          context_after: data.context_after,
          inference_time_ms: data.methods[selectedMethod].inference_time_ms
        });
      }
    } catch (err: any) {
      setError(err.message || 'Failed to compare methods');
      console.error(err);
    } finally {
      setComparingAll(false);
    }
  };

  // Auto-run on first load once sensor columns are ready
  useEffect(() => {
    if (availableFiles.length > 0 && !reconstructionData && !loading) {
      runReconstruction();
    }
  }, [availableFiles]);

  // When user switches method, auto-trigger reconstruction if compatible
  useEffect(() => {
    if (isMethodCompatible && availableFiles.length > 0) {
      runReconstruction();
    }
  }, [selectedMethod]);

  // Prepare Chart Data
  const chartData = useMemo(() => {
    if (!reconstructionData) return { points: [], gapStart: 0, gapEnd: 0 };

    const sensorIdx = reconstructionData.sensor_columns?.indexOf(selectedSensor);
    if (sensorIdx === undefined || sensorIdx === -1) return { points: [], gapStart: 0, gapEnd: 0 };

    const cb = reconstructionData.context_before || [];
    const ca = reconstructionData.context_after || [];
    const gt = reconstructionData.ground_truth || [];
    const pred = reconstructionData.predictions || [];

    const points: any[] = [];
    let t = 0;

    // 1. Observed Left Context
    for (let i = 0; i < cb.length; i++) {
      points.push({
        timestep: t,
        label: `T${t}`,
        observed: cb[i][sensorIdx],
        groundTruth: null,
        reconstruction: null,
        region: 'Left Context (Observed)'
      });
      t++;
    }

    const gapStart = t;

    // 2. Missing Gap
    for (let i = 0; i < gt.length; i++) {
      const predVal = pred[i] ? pred[i][sensorIdx] : null;
      const gtVal = gt[i] ? gt[i][sensorIdx] : null;
      const err = predVal !== null && gtVal !== null ? Math.abs(predVal - gtVal) : null;

      const point: any = {
        timestep: t,
        label: `T${t} (Gap)`,
        observed: null,
        groundTruth: gtVal,
        reconstruction: predVal,
        residualError: err,
        region: 'Missing Telemetry Interval'
      };

      // Add multi-method predictions if comparison data is available
      if (comparisonData && comparisonData.methods) {
        Object.entries(comparisonData.methods).forEach(([mId, mRes]: [string, any]) => {
          if (mRes.available && mRes.predictions && mRes.predictions[i]) {
            point[`pred_${mId}`] = mRes.predictions[i][sensorIdx];
          }
        });
      }

      points.push(point);
      t++;
    }

    const gapEnd = t - 1;

    // 3. Observed Right Context
    for (let i = 0; i < ca.length; i++) {
      points.push({
        timestep: t,
        label: `T${t}`,
        observed: ca[i][sensorIdx],
        groundTruth: null,
        reconstruction: null,
        region: 'Right Context (Observed)'
      });
      t++;
    }

    return { points, gapStart, gapEnd };
  }, [reconstructionData, selectedSensor, comparisonData]);

  // Per-sensor metric for selected sensor
  const currentSensorMetrics = useMemo(() => {
    if (!reconstructionData || !reconstructionData.metrics) return null;
    const m = reconstructionData.metrics;
    const mse = m.per_sensor_mse ? m.per_sensor_mse[selectedSensor] : m.mse;
    const mae = m.per_sensor_mae ? m.per_sensor_mae[selectedSensor] : m.mae;
    return {
      mse: typeof mse === 'number' ? mse : m.mse,
      mae: typeof mae === 'number' ? mae : m.mae,
      overallMse: m.mse,
      overallMae: m.mae,
      overallRmse: m.rmse
    };
  }, [reconstructionData, selectedSensor]);

  return (
    <div className="main-content">
      {/* Page Header */}
      <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <span className="badge badge-cyan">Primary Evaluation</span>
            <span className="badge badge-emerald">Interactive Inference</span>
          </div>
          <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
            Reconstruction Studio
          </h1>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0 0' }}>
            Configure telemetry dropouts, execute real-time model inference, and audit reconstructed sensor waves against ground truth.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            onClick={runReconstruction}
            disabled={loading || !isMethodCompatible}
            className="btn btn-primary"
            style={{ padding: '0.65rem 1.25rem' }}
          >
            <Play size={16} />
            {loading ? 'Running...' : 'Run Single Inference'}
          </button>

          <button
            onClick={runCompareAll}
            disabled={comparingAll}
            className="btn btn-secondary"
            style={{ padding: '0.65rem 1.25rem' }}
          >
            <Layers size={16} color="var(--accent-cyan)" />
            {comparingAll ? 'Evaluating All...' : 'Compare All 4 Methods'}
          </button>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
        <div className="glass-panel-header">
          <div className="panel-title">
            <Settings size={18} color="var(--accent-cyan)" />
            Telemetry Gap & Window Configuration
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
              Presets:
            </span>
            {PRESET_CONFIGS.map(p => {
              const active = contextBefore === p.before && gapLength === p.gap && contextAfter === p.after;
              return (
                <button
                  key={p.name}
                  onClick={() => {
                    setContextBefore(p.before);
                    setGapLength(p.gap);
                    setContextAfter(p.after);
                  }}
                  style={{
                    padding: '0.25rem 0.6rem',
                    fontSize: '0.75rem',
                    borderRadius: '5px',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 600,
                    cursor: 'pointer',
                    background: active ? 'rgba(6, 182, 212, 0.2)' : 'var(--bg-secondary)',
                    border: `1px solid ${active ? 'var(--accent-cyan)' : 'var(--border-color)'}`,
                    color: active ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                    transition: 'all 0.15s ease'
                  }}
                >
                  {p.name.split(' ')[0]}
                </button>
              );
            })}
          </div>
        </div>

        {/* Controls Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '1rem',
          marginBottom: '1.25rem'
        }}>
          {/* File Selector */}
          <div>
            <label className="control-label">Test File Sample</label>
            <select
              className="industrial-select"
              value={selectedFile}
              onChange={e => setSelectedFile(e.target.value)}
            >
              {availableFiles.map(f => (
                <option key={f.filename} value={f.filename}>
                  {f.filename} ({f.window_count} windows)
                </option>
              ))}
            </select>
          </div>

          {/* Window Stepper */}
          <div>
            <label className="control-label">Window Index</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <button
                className="btn btn-secondary"
                style={{ padding: '0.65rem 0.75rem' }}
                disabled={windowIndex <= 0}
                onClick={() => setWindowIndex(Math.max(0, windowIndex - 1))}
              >
                <ChevronLeft size={16} />
              </button>
              <input
                type="number"
                className="industrial-input mono-cell"
                value={windowIndex}
                min={0}
                max={maxWindows - 1}
                onChange={e => setWindowIndex(Math.max(0, Math.min(maxWindows - 1, parseInt(e.target.value) || 0)))}
                style={{ textAlign: 'center', fontWeight: 600 }}
              />
              <button
                className="btn btn-secondary"
                style={{ padding: '0.65rem 0.75rem' }}
                disabled={windowIndex >= maxWindows - 1}
                onClick={() => setWindowIndex(Math.min(maxWindows - 1, windowIndex + 1))}
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>

          {/* Left Context */}
          <div>
            <label className="control-label">Left Context (Steps)</label>
            <input
              type="number"
              className="industrial-input mono-cell"
              value={contextBefore}
              min={10}
              max={50}
              onChange={e => setContextBefore(Math.max(10, Math.min(50, parseInt(e.target.value) || 50)))}
            />
          </div>

          {/* Missing Gap */}
          <div>
            <label className="control-label">Missing Gap Length</label>
            <select
              className="industrial-select mono-cell"
              value={gapLength}
              onChange={e => setGapLength(parseInt(e.target.value))}
            >
              <option value={10}>10 Steps (Standard Gap)</option>
              <option value={20}>20 Steps (Medium Gap)</option>
              <option value={30}>30 Steps (Custom Gap)</option>
              <option value={40}>40 Steps (Custom Gap)</option>
              <option value={50}>50 Steps (Large Gap)</option>
            </select>
          </div>

          {/* Right Context */}
          <div>
            <label className="control-label">Right Context (Steps)</label>
            <input
              type="number"
              className="industrial-input mono-cell"
              value={contextAfter}
              min={10}
              max={50}
              onChange={e => setContextAfter(Math.max(10, Math.min(50, parseInt(e.target.value) || 50)))}
            />
          </div>

          {/* Sensor Feature Selector */}
          <div>
            <label className="control-label">Sensor Telemetry Channel</label>
            <select
              className="industrial-select mono-cell"
              value={selectedSensor}
              onChange={e => setSelectedSensor(e.target.value)}
              style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}
            >
              {sensorColumns.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Method Selector Segmented Cards */}
        <div>
          <label className="control-label" style={{ marginBottom: '0.65rem' }}>
            Reconstruction Approach Selector
          </label>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '0.85rem'
          }}>
            {METHODS.map(m => {
              const selected = selectedMethod === m.id;
              const compatible = m.id === 'linear_interpolation' 
                ? true 
                : m.id === 'residual_lstm' 
                ? [10, 20, 50].includes(gapLength) 
                : gapLength === 10;

              return (
                <div
                  key={m.id}
                  onClick={() => {
                    if (compatible) setSelectedMethod(m.id);
                  }}
                  style={{
                    background: selected 
                      ? 'rgba(6, 182, 212, 0.12)' 
                      : compatible 
                      ? 'var(--bg-secondary)' 
                      : 'rgba(255, 255, 255, 0.02)',
                    border: `1px solid ${selected ? 'var(--accent-cyan)' : compatible ? 'var(--border-color)' : 'rgba(255, 255, 255, 0.06)'}`,
                    borderRadius: '8px',
                    padding: '0.9rem',
                    cursor: compatible ? 'pointer' : 'not-allowed',
                    opacity: compatible ? 1 : 0.45,
                    transition: 'all 0.15s ease',
                    position: 'relative'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                    <span style={{ fontSize: '0.88rem', fontWeight: 700, color: selected ? '#ffffff' : 'var(--text-primary)' }}>
                      {m.name}
                    </span>
                    <span className={`badge ${compatible ? (selected ? 'badge-cyan' : 'badge-slate') : 'badge-rose'}`}>
                      {compatible ? m.badge : 'Unavailable'}
                    </span>
                  </div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.4 }}>
                    {m.description}
                  </p>
                  {!compatible && (
                    <div style={{ fontSize: '0.7rem', color: 'var(--accent-rose)', marginTop: '0.35rem', fontWeight: 600 }}>
                      Requires Gap {m.supportedGaps.join(', ')}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Configuration Summary Badge Strip */}
        <div style={{
          marginTop: '1.15rem',
          padding: '0.75rem 1rem',
          borderRadius: '6px',
          background: 'rgba(6, 182, 212, 0.06)',
          border: '1px solid rgba(6, 182, 212, 0.2)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem'
        }}>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            <strong style={{ color: 'var(--accent-cyan)' }}>Current Telemetry Span:</strong>{' '}
            <span className="mono-cell" style={{ color: 'var(--text-primary)' }}>
              {contextBefore} steps observed + {gapLength} steps missing + {contextAfter} steps observed = {contextBefore + gapLength + contextAfter} total sequence
            </span>
          </div>

          {!isMethodCompatible && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              color: 'var(--accent-amber)',
              fontSize: '0.8rem',
              fontWeight: 600
            }}>
              <AlertTriangle size={15} />
              <span>{selectedMethod} is incompatible with gap length {gapLength}.</span>
              <button
                className="btn btn-primary"
                style={{ padding: '0.2rem 0.5rem', fontSize: '0.72rem' }}
                onClick={() => setGapLength(10)}
              >
                Switch to Gap 10
              </button>
            </div>
          )}
        </div>

        {error && (
          <div style={{
            marginTop: '1rem',
            padding: '0.75rem 1rem',
            background: 'var(--accent-rose-bg)',
            border: '1px solid var(--accent-rose-border)',
            borderRadius: '6px',
            color: 'var(--accent-rose)',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <AlertCircle size={16} />
            {error}
          </div>
        )}
      </div>

      {/* Main Reconstruction Chart */}
      <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
        <div className="glass-panel-header">
          <div>
            <div className="panel-title">
              <TrendingUp size={18} color="var(--accent-cyan)" />
              Continuous Telemetry Reconstruction Waveform
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              Feature: <strong style={{ color: 'var(--accent-cyan)' }}>{selectedSensor}</strong> | Model: <strong style={{ color: 'var(--text-primary)' }}>{selectedMethod}</strong> | Window: <span className="mono-cell">#{windowIndex}</span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            {comparisonData && (
              <button
                onClick={() => setShowMultiChart(!showMultiChart)}
                style={{
                  padding: '0.35rem 0.75rem',
                  fontSize: '0.75rem',
                  borderRadius: '5px',
                  background: showMultiChart ? 'rgba(6, 182, 212, 0.2)' : 'var(--bg-secondary)',
                  border: `1px solid ${showMultiChart ? 'var(--accent-cyan)' : 'var(--border-color)'}`,
                  color: showMultiChart ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontWeight: 600
                }}
              >
                {showMultiChart ? 'Hide Multi-Method Overlay' : 'Show Multi-Method Overlay'}
              </button>
            )}
            <span className="badge badge-emerald">Normalized Space</span>
          </div>
        </div>

        {/* Chart Viewport */}
        <div style={{ height: '460px', width: '100%', position: 'relative' }}>
          {loading ? (
            <div style={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.75rem',
              color: 'var(--text-muted)'
            }}>
              <RefreshCw size={28} className="animate-spin" color="var(--accent-cyan)" />
              <div style={{ fontSize: '0.9rem' }}>Executing reconstruction inference on sensor telemetry...</div>
            </div>
          ) : chartData.points && chartData.points.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData.points} margin={{ top: 20, right: 30, left: 15, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a253d" vertical={false} />
                
                <XAxis 
                  dataKey="timestep" 
                  stroke="#475569" 
                  tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  label={{ value: 'Continuous Timestep Index (1 Hz)', position: 'insideBottom', offset: -15, fill: '#64748b', fontSize: 12 }}
                />
                
                <YAxis 
                  stroke="#475569" 
                  domain={['auto', 'auto']}
                  tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  label={{ value: 'Normalized Sensor Value (z-score)', angle: -90, position: 'insideLeft', offset: -5, fill: '#64748b', fontSize: 12 }}
                />

                {/* Shaded Telemetry Gap Area */}
                {chartData.gapStart !== undefined && chartData.gapEnd !== undefined && (
                  <ReferenceArea 
                    x1={chartData.gapStart} 
                    x2={chartData.gapEnd} 
                    fill="rgba(245, 158, 11, 0.08)" 
                    stroke="rgba(245, 158, 11, 0.4)"
                    strokeDasharray="3 3"
                    label={{ 
                      value: `Missing Gap (${gapLength} steps)`, 
                      position: 'top', 
                      fill: '#f59e0b', 
                      fontSize: 11, 
                      fontWeight: 700,
                      fontFamily: 'var(--font-mono)'
                    }} 
                  />
                )}

                <Tooltip 
                  content={({ active, payload, label }) => {
                    if (!active || !payload || !payload.length) return null;
                    const p = payload[0].payload;
                    return (
                      <div className="custom-recharts-tooltip">
                        <div className="tooltip-title">{p.label} - {p.region}</div>
                        {p.observed !== null && (
                          <div className="tooltip-row" style={{ color: '#06b6d4' }}>
                            <span>Observed Value:</span>
                            <span className="tooltip-val">{Number(p.observed).toFixed(5)}</span>
                          </div>
                        )}
                        {p.groundTruth !== null && (
                          <div className="tooltip-row" style={{ color: '#f8fafc' }}>
                            <span>Hidden Ground Truth:</span>
                            <span className="tooltip-val">{Number(p.groundTruth).toFixed(5)}</span>
                          </div>
                        )}
                        {p.reconstruction !== null && (
                          <div className="tooltip-row" style={{ color: '#f59e0b' }}>
                            <span>Reconstructed ({selectedMethod}):</span>
                            <span className="tooltip-val">{Number(p.reconstruction).toFixed(5)}</span>
                          </div>
                        )}
                        {p.residualError !== null && (
                          <div className="tooltip-row" style={{ color: '#ec4899' }}>
                            <span>Absolute Residual Error:</span>
                            <span className="tooltip-val">{Number(p.residualError).toFixed(5)}</span>
                          </div>
                        )}
                      </div>
                    );
                  }}
                />

                <Legend 
                  verticalAlign="top" 
                  align="right"
                  wrapperStyle={{ paddingBottom: '10px', fontSize: '12px' }}
                />

                {/* Observed Sensor Telemetry (Before and After Gap) */}
                <Line 
                  type="monotone" 
                  dataKey="observed" 
                  stroke="#06b6d4" 
                  strokeWidth={2.5} 
                  dot={false}
                  activeDot={{ r: 5, fill: '#06b6d4' }}
                  name="Observed Telemetry" 
                  connectNulls={false} 
                />

                {/* Hidden Ground Truth (Only in Gap) */}
                <Line 
                  type="monotone" 
                  dataKey="groundTruth" 
                  stroke="#f8fafc" 
                  strokeWidth={2} 
                  strokeDasharray="4 4"
                  dot={{ r: 3, fill: '#f8fafc' }}
                  name="Hidden Ground Truth (Offline Benchmark)" 
                  connectNulls={false} 
                />

                {/* Reconstructed Prediction (Only in Gap) */}
                <Line 
                  type="monotone" 
                  dataKey="reconstruction" 
                  stroke="#f59e0b" 
                  strokeWidth={2.8} 
                  dot={{ r: 3, fill: '#f59e0b' }}
                  activeDot={{ r: 6, fill: '#f59e0b' }}
                  name={`Reconstruction (${METHODS.find(m => m.id === selectedMethod)?.name || selectedMethod})`} 
                  connectNulls={false} 
                />

                {/* Optional Multi-Method Overlay Lines */}
                {showMultiChart && comparisonData && (
                  <>
                    <Line type="monotone" dataKey="pred_linear_interpolation" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="Linear Baseline" connectNulls={false} />
                    <Line type="monotone" dataKey="pred_residual_lstm" stroke="#f59e0b" strokeWidth={2} dot={false} name="Residual LSTM" connectNulls={false} />
                    <Line type="monotone" dataKey="pred_hybrid" stroke="#10b981" strokeWidth={1.5} dot={false} name="Hybrid Blend" connectNulls={false} />
                    <Line type="monotone" dataKey="pred_attention_seq2seq" stroke="#8b5cf6" strokeWidth={1.5} dot={false} name="Attention Seq2Seq" connectNulls={false} />
                  </>
                )}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--text-muted)'
            }}>
              No reconstruction data loaded. Click "Run Single Inference" above.
            </div>
          )}
        </div>
      </div>

      {/* Selected Reconstruction Metrics Strip */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '1rem',
        marginBottom: '1.5rem'
      }}>
        {/* Metric 1: Selected Sensor MSE */}
        <div className="metric-card">
          <div className="metric-card-label">
            <TrendingUp size={14} color="var(--accent-cyan)" />
            Channel MSE ({selectedSensor.length > 12 ? selectedSensor.slice(0, 10) + '..' : selectedSensor})
          </div>
          <div className="metric-card-value" style={{ color: 'var(--accent-cyan)' }}>
            {currentSensorMetrics ? currentSensorMetrics.mse.toFixed(6) : 'Not evaluated'}
          </div>
          <div className="metric-card-sub">
            All 17-Ch Mean: <strong style={{ color: 'var(--text-primary)' }}>{currentSensorMetrics ? currentSensorMetrics.overallMse.toFixed(6) : '-'}</strong>
          </div>
        </div>

        {/* Metric 2: Selected Sensor MAE */}
        <div className="metric-card">
          <div className="metric-card-label">
            <BarChart3 size={14} color="var(--accent-emerald)" />
            Channel MAE
          </div>
          <div className="metric-card-value" style={{ color: 'var(--accent-emerald)' }}>
            {currentSensorMetrics ? currentSensorMetrics.mae.toFixed(6) : 'Not evaluated'}
          </div>
          <div className="metric-card-sub">
            All 17-Ch Mean: <strong style={{ color: 'var(--text-primary)' }}>{currentSensorMetrics ? currentSensorMetrics.overallMae.toFixed(6) : '-'}</strong>
          </div>
        </div>

        {/* Metric 3: Overall RMSE */}
        <div className="metric-card">
          <div className="metric-card-label">
            <Activity size={14} color="var(--accent-amber)" />
            Overall 17-Ch RMSE
          </div>
          <div className="metric-card-value" style={{ color: 'var(--accent-amber)' }}>
            {currentSensorMetrics ? currentSensorMetrics.overallRmse.toFixed(6) : 'Not evaluated'}
          </div>
          <div className="metric-card-sub">
            Standard deviation of error residuals
          </div>
        </div>

        {/* Metric 4: Inference Latency */}
        <div className="metric-card">
          <div className="metric-card-label">
            <Clock size={14} color="var(--accent-purple)" />
            Inference Latency
          </div>
          <div className="metric-card-value" style={{ color: 'var(--accent-purple)' }}>
            {reconstructionData?.inference_time_ms !== undefined ? `${reconstructionData.inference_time_ms} ms` : 'Not evaluated'}
          </div>
          <div className="metric-card-sub">
            Model: <strong style={{ color: 'var(--text-primary)' }}>{selectedMethod}</strong>
          </div>
        </div>
      </div>

      {/* Compare All Methods On Same Sample Section */}
      <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
        <div className="glass-panel-header">
          <div>
            <div className="panel-title">
              <Layers size={18} color="var(--accent-cyan)" />
              Fair Head-to-Head Comparison on Window #{windowIndex}
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: '0.2rem 0 0 0' }}>
              All compatible methods evaluated against the exact same sample window, gap mask, and ground-truth values.
            </p>
          </div>

          <button
            onClick={runCompareAll}
            disabled={comparingAll}
            className="btn btn-primary"
            style={{ padding: '0.45rem 1rem', fontSize: '0.8rem' }}
          >
            {comparingAll ? 'Evaluating...' : 'Refresh Comparison'}
          </button>
        </div>

        {comparisonData && comparisonData.methods ? (
          <div className="table-container">
            <table className="industrial-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Architecture</th>
                  <th>Availability</th>
                  <th>Selected Channel MSE ({selectedSensor})</th>
                  <th>Overall 17-Ch MSE</th>
                  <th>Overall MAE</th>
                  <th>Overall RMSE</th>
                  <th>Latency (ms)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparisonData.methods).map(([mId, mRes]: [string, any]) => {
                  const isBest = comparisonData.best_method === mId;
                  const def = METHODS.find(m => m.id === mId);
                  const sensorIdx = comparisonData.sensor_columns?.indexOf(selectedSensor);
                  const channelMse = (mRes.available && mRes.metrics?.per_sensor_mse && sensorIdx !== undefined) 
                    ? mRes.metrics.per_sensor_mse[selectedSensor] 
                    : mRes.metrics?.mse;

                  return (
                    <tr 
                      key={mId}
                      style={{
                        background: isBest ? 'rgba(16, 185, 129, 0.06)' : undefined
                      }}
                    >
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
                            {mRes.name}
                          </span>
                          {isBest && (
                            <span className="badge badge-emerald">Lowest Error</span>
                          )}
                        </div>
                      </td>
                      <td style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                        {def?.category || '-'}
                      </td>
                      <td>
                        <span className={`badge ${mRes.available ? 'badge-cyan' : 'badge-rose'}`}>
                          {mRes.available ? 'Evaluated' : mRes.status_message}
                        </span>
                      </td>
                      <td className="mono-cell" style={{ color: isBest ? 'var(--accent-emerald)' : 'var(--text-primary)' }}>
                        {mRes.available && typeof channelMse === 'number' ? channelMse.toFixed(6) : 'N/A'}
                      </td>
                      <td className="mono-cell" style={{ fontWeight: isBest ? 700 : 500, color: isBest ? 'var(--accent-emerald)' : 'var(--text-primary)' }}>
                        {mRes.available && mRes.metrics ? mRes.metrics.mse.toFixed(6) : 'N/A'}
                      </td>
                      <td className="mono-cell">
                        {mRes.available && mRes.metrics ? mRes.metrics.mae.toFixed(6) : 'N/A'}
                      </td>
                      <td className="mono-cell">
                        {mRes.available && mRes.metrics ? mRes.metrics.rmse.toFixed(6) : 'N/A'}
                      </td>
                      <td className="mono-cell">
                        {mRes.available && mRes.inference_time_ms !== undefined ? `${mRes.inference_time_ms} ms` : 'N/A'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{
            padding: '2rem',
            textAlign: 'center',
            color: 'var(--text-muted)',
            background: 'var(--bg-secondary)',
            borderRadius: '6px'
          }}>
            Click "Compare All 4 Methods" above to run all models simultaneously on this window.
          </div>
        )}
      </div>
    </div>
  );
};
