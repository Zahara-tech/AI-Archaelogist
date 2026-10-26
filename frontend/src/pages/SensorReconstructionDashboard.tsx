import React, { useState, useEffect } from 'react';
import { 
  Play, Settings, BarChart3, 
  TrendingUp, Clock, AlertCircle, CheckCircle, Brain
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface MethodInfo {
  id: string;
  name: string;
  description: string;
  supportedGapLengths: number[];
  requiresTraining: boolean;
}

interface GapConfig {
  contextBefore: number;
  gapLength: number;
  contextAfter: number;
}

interface ReconstructionResult {
  gap_length: number;
  filename: string;
  window_index: number;
  method: string;
  predictions: number[][];
  ground_truth: number[][];
  metrics: {
    mse: number;
    mae: number;
    rmse: number;
    per_sensor_mse?: Record<string, number>;
    per_sensor_mae?: Record<string, number>;
  };
  sensor_columns: string[];
}

interface EvaluationSummary {
  gap_length: number;
  overall_normalized: {
    linear_interp: { mse: number; mae: number; rmse: number };
    residual_lstm: { mse: number; mae: number; rmse: number };
  };
  total_sequences: number;
}

interface EpochComparison {
  best_epoch: number;
  best_val_loss: number;
  total_epochs: number;
  total_time: number;
  epoch_data: Array<{
    requested_epochs: number;
    completed_epochs: number | null;
    train_loss: number | null;
    val_loss: number | null;
    status: string;
  }>;
  train_losses: number[];
  val_losses: number[];
}

const METHODS: MethodInfo[] = [
  {
    id: 'linear_interpolation',
    name: 'Linear Interpolation',
    description: 'Mathematical baseline that estimates missing values using linear interpolation between observed boundary values.',
    supportedGapLengths: [10, 20, 50], // Supports any gap length
    requiresTraining: false
  },
  {
    id: 'attention_seq2seq',
    name: 'Attention Seq2Seq',
    description: 'Deep learning model with bidirectional LSTM encoder, additive attention, and LSTM decoder.',
    supportedGapLengths: [10], // Only trained for gap 10
    requiresTraining: true
  },
  {
    id: 'hybrid',
    name: 'Hybrid Reconstruction',
    description: 'Fusion of linear interpolation and Attention Seq2Seq using fixed-weight blend (10% neural, 90% interpolation).',
    supportedGapLengths: [10], // Only trained for gap 10
    requiresTraining: true
  },
  {
    id: 'residual_lstm',
    name: 'Residual LSTM',
    description: 'Neural model that learns corrections to linear interpolation. Final = interpolation + learned correction.',
    supportedGapLengths: [10, 20, 50],
    requiresTraining: true
  }
];

const PRESET_CONFIGS = [
  { name: '50-10-50 (Default)', contextBefore: 50, gapLength: 10, contextAfter: 50 },
  { name: '30-10-30', contextBefore: 30, gapLength: 10, contextAfter: 30 },
  { name: '30-20-30', contextBefore: 30, gapLength: 20, contextAfter: 30 },
  { name: '30-30-30', contextBefore: 30, gapLength: 30, contextAfter: 30 },
  { name: '30-40-30', contextBefore: 30, gapLength: 40, contextAfter: 30 },
  { name: '40-50-40', contextBefore: 40, gapLength: 50, contextAfter: 40 },
  { name: '50-50-50', contextBefore: 50, gapLength: 50, contextAfter: 50 }
];

export const SensorReconstructionDashboard: React.FC = () => {
  const [gapConfig, setGapConfig] = useState<GapConfig>({
    contextBefore: 50,
    gapLength: 10,
    contextAfter: 50
  });
  
  const [selectedMethod, setSelectedMethod] = useState<string>('residual_lstm');
  const [selectedSensor, setSelectedSensor] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  
  const [result, setResult] = useState<ReconstructionResult | null>(null);
  const [evaluationSummary, setEvaluationSummary] = useState<EvaluationSummary | null>(null);
  const [epochComparison, setEpochComparison] = useState<EpochComparison | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableFiles, setAvailableFiles] = useState<any[]>([]);
  const [sensorColumns, setSensorColumns] = useState<string[]>([]);
  
  const [inferenceTime, setInferenceTime] = useState<number>(0);

  useEffect(() => {
    loadGapBenchmarkInfo();
    loadEpochComparison();
  }, [gapConfig.gapLength]);

  useEffect(() => {
    if (gapConfig.gapLength > 0) {
      loadEvaluationSummary();
    }
  }, [gapConfig.gapLength]);

  const loadGapBenchmarkInfo = async () => {
    try {
      const response = await fetch(`/api/v1/sensor/gap-benchmark/${gapConfig.gapLength}/windows`);
      if (response.ok) {
        const data = await response.json();
        setAvailableFiles(data.windows || []);
        if (data.windows && data.windows.length > 0) {
          setSelectedFile(data.windows[0].filename);
          // Get sensor columns from first file
          const fileResponse = await fetch(`/api/v1/sensor/dataset/info`);
          if (fileResponse.ok) {
            const info = await fileResponse.json();
            setSensorColumns(info.sensor_columns || []);
            if (info.sensor_columns && info.sensor_columns.length > 0) {
              setSelectedSensor(info.sensor_columns[0]);
            }
          }
        }
      }
    } catch (err) {
      console.error('Failed to load gap benchmark info:', err);
    }
  };

  const loadEvaluationSummary = async () => {
    try {
      const response = await fetch(`/api/v1/sensor/gap-benchmark/${gapConfig.gapLength}/evaluation-summary`);
      if (response.ok) {
        const data = await response.json();
        setEvaluationSummary(data);
      }
    } catch (err) {
      console.error('Failed to load evaluation summary:', err);
    }
  };

  const loadEpochComparison = async () => {
    try {
      const response = await fetch('/api/v1/sensor/model/epoch-comparison');
      if (response.ok) {
        const data = await response.json();
        setEpochComparison(data);
      }
    } catch (err) {
      console.error('Failed to load epoch comparison:', err);
    }
  };

  const handleRunReconstruction = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    
    const startTime = performance.now();
    
    try {
      const response = await fetch('/api/v1/sensor/gap-benchmark/reconstruct-window', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          gap_length: gapConfig.gapLength,
          filename: selectedFile,
          window_index: selectedIndex,
          method: selectedMethod
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to reconstruct');
      }
      
      const data = await response.json();
      setResult(data);
      setInferenceTime(performance.now() - startTime);
    } catch (err: any) {
      setError(err.message || 'Failed to run reconstruction');
      console.error('Reconstruction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getMethodCompatibility = (methodId: string, gapLength: number): boolean => {
    const method = METHODS.find(m => m.id === methodId);
    if (!method) return false;
    return method.supportedGapLengths.includes(gapLength);
  };

  const prepareChartData = () => {
    if (!result) return [];
    const sensorIndex = result.sensor_columns.indexOf(selectedSensor);
    if (sensorIndex === -1) return [];
    
    const data: any[] = [];
    
    // Ground truth (full sequence for visualization)
    for (let i = 0; i < result.ground_truth.length; i++) {
      data.push({
        index: i,
        groundTruth: result.ground_truth[i][sensorIndex],
        prediction: result.predictions[i][sensorIndex],
        region: 'Gap'
      });
    }
    
    return data;
  };

  const calculatePerSensorMetrics = () => {
    if (!result) return null;
    
    const metrics: Record<string, { mse: number; mae: number }> = {};
    
    result.sensor_columns.forEach((sensor, sensorIdx) => {
      let se = 0;
      let ae = 0;
      const count = result.predictions.length;
      
      for (let i = 0; i < count; i++) {
        const pred = result.predictions[i][sensorIdx];
        const truth = result.ground_truth[i][sensorIdx];
        se += (pred - truth) ** 2;
        ae += Math.abs(pred - truth);
      }
      
      metrics[sensor] = {
        mse: se / count,
        mae: ae / count
      };
    });
    
    return metrics;
  };

  const totalWindowLength = gapConfig.contextBefore + gapConfig.gapLength + gapConfig.contextAfter;
  const compatibleMethods = METHODS.filter(m => getMethodCompatibility(m.id, gapConfig.gapLength));

  return (
    <div className="main-content">
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{
          fontSize: '2rem',
          fontWeight: 800,
          color: 'var(--text-primary)',
          marginBottom: '0.5rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem'
        }}>
          <Brain size={32} color="#a855f7" />
          Reconstruction Studio
        </h1>
        <p style={{
          fontSize: '1rem',
          color: 'var(--text-secondary)',
          fontWeight: 400
        }}>
          Interactive reconstruction with all four approaches. Configure the missing-gap pattern, select a method, and compare results.
        </p>
      </div>

        {/* Configuration Panel */}
        <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
          <div className="glass-panel-header">
            <h2 style={{
              fontSize: '1.1rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <Settings size={20} color="#38bdf8" />
              Reconstruction Configuration
            </h2>
          </div>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1rem',
            marginBottom: '1rem'
          }}>
            <div>
              <label style={{
                display: 'block',
                fontSize: '0.8rem',
                fontWeight: 600,
                color: 'var(--text-muted)',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Preset Configuration
              </label>
              <select
                style={{
                  width: '100%',
                  padding: '0.6rem',
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem'
                }}
                onChange={(e) => {
                  const preset = PRESET_CONFIGS.find(p => p.name === e.target.value);
                  if (preset) setGapConfig(preset);
                }}
                value={PRESET_CONFIGS.find(p => 
                  p.contextBefore === gapConfig.contextBefore && 
                  p.gapLength === gapConfig.gapLength && 
                  p.contextAfter === gapConfig.contextAfter
                )?.name || ''}
              >
                {PRESET_CONFIGS.map(preset => (
                  <option key={preset.name} value={preset.name}>{preset.name}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label style={{
                display: 'block',
                fontSize: '0.8rem',
                fontWeight: 600,
                color: 'var(--text-muted)',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Context Before
              </label>
              <input
                type="number"
                style={{
                  width: '100%',
                  padding: '0.6rem',
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem'
                }}
                value={gapConfig.contextBefore}
                onChange={(e) => setGapConfig({ ...gapConfig, contextBefore: parseInt(e.target.value) || 0 })}
                min="1"
              />
            </div>
            
            <div>
              <label style={{
                display: 'block',
                fontSize: '0.8rem',
                fontWeight: 600,
                color: 'var(--text-muted)',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Gap Length
              </label>
              <input
                type="number"
                style={{
                  width: '100%',
                  padding: '0.6rem',
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem'
                }}
                value={gapConfig.gapLength}
                onChange={(e) => setGapConfig({ ...gapConfig, gapLength: parseInt(e.target.value) || 0 })}
                min="1"
              />
            </div>
            
            <div>
              <label style={{
                display: 'block',
                fontSize: '0.8rem',
                fontWeight: 600,
                color: 'var(--text-muted)',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Context After
              </label>
              <input
                type="number"
                style={{
                  width: '100%',
                  padding: '0.6rem',
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem'
                }}
                value={gapConfig.contextAfter}
                onChange={(e) => setGapConfig({ ...gapConfig, contextAfter: parseInt(e.target.value) || 0 })}
                min="1"
              />
            </div>
          </div>
          
          <div style={{
            padding: '0.75rem',
            background: 'rgba(56, 189, 248, 0.08)',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            borderRadius: '6px',
            marginBottom: '1rem'
          }}>
            <p style={{
              fontSize: '0.85rem',
              color: '#38bdf8',
              margin: 0
            }}>
              <strong>Active Configuration:</strong> {gapConfig.contextBefore} observed + {gapConfig.gapLength} missing + {gapConfig.contextAfter} observed = {totalWindowLength} total timesteps
            </p>
          </div>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1rem',
            marginBottom: '1rem'
          }}>
            <div>
              <label style={{
                display: 'block',
                fontSize: '0.8rem',
                fontWeight: 600,
                color: 'var(--text-muted)',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Reconstruction Method
              </label>
              <select
                style={{
                  width: '100%',
                  padding: '0.6rem',
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem'
                }}
                value={selectedMethod}
                onChange={(e) => setSelectedMethod(e.target.value)}
              >
                {compatibleMethods.map(method => (
                  <option key={method.id} value={method.id}>{method.name}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label style={{
                display: 'block',
                fontSize: '0.8rem',
                fontWeight: 600,
                color: 'var(--text-muted)',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Sensor
              </label>
              <select
                style={{
                  width: '100%',
                  padding: '0.6rem',
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem'
                }}
                value={selectedSensor}
                onChange={(e) => setSelectedSensor(e.target.value)}
              >
                {sensorColumns.map(sensor => (
                  <option key={sensor} value={sensor}>{sensor}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label style={{
                display: 'block',
                fontSize: '0.8rem',
                fontWeight: 600,
                color: 'var(--text-muted)',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Test File
              </label>
              <select
                style={{
                  width: '100%',
                  padding: '0.6rem',
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem'
                }}
                value={selectedFile}
                onChange={(e) => setSelectedFile(e.target.value)}
              >
                {availableFiles.map(file => (
                  <option key={file.filename} value={file.filename}>
                    {file.filename} ({file.window_count} windows)
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <button
              onClick={handleRunReconstruction}
              disabled={loading || !selectedFile}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.75rem 1.5rem',
                background: loading || !selectedFile ? 'var(--border-color)' : 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)',
                color: loading || !selectedFile ? 'var(--text-muted)' : '#ffffff',
                borderRadius: '8px',
                border: 'none',
                cursor: loading || !selectedFile ? 'not-allowed' : 'pointer',
                fontWeight: 600,
                fontSize: '0.9rem',
                transition: 'all 0.2s ease'
              }}
            >
              <Play size={16} />
              {loading ? 'Running...' : 'Run Reconstruction'}
            </button>
            
            <button
              onClick={() => setGapConfig(PRESET_CONFIGS[0])}
              style={{
                padding: '0.75rem 1.25rem',
                background: 'var(--bg-card)',
                color: 'var(--text-primary)',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: '0.9rem',
                transition: 'all 0.2s ease'
              }}
            >
              Reset to Default
            </button>
          </div>
          
          {error && (
            <div style={{
              marginTop: '1rem',
              padding: '0.75rem',
              background: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              color: '#fda4af'
            }}>
              <AlertCircle size={18} />
              {error}
            </div>
          )}
        </div>

        {/* Main Reconstruction Chart */}
        {result && (
          <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
            <div className="glass-panel-header">
              <h2 style={{
                fontSize: '1.1rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                <TrendingUp size={20} color="#38bdf8" />
                Reconstruction Visualization
              </h2>
            </div>
            <div style={{ height: '400px', marginBottom: '1rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={prepareChartData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                  <XAxis 
                    dataKey="index" 
                    label={{ value: 'Gap Position', position: 'insideBottom', offset: -5, fill: 'var(--text-secondary)' }}
                    stroke="var(--text-secondary)"
                  />
                  <YAxis 
                    label={{ value: 'Sensor Value (Normalized)', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }}
                    stroke="var(--text-secondary)"
                  />
                  <Tooltip 
                    contentStyle={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      color: 'var(--text-primary)'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="groundTruth" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    name="Ground Truth"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="prediction" 
                    stroke="#f59e0b" 
                    strokeWidth={2}
                    name="Reconstruction"
                    dot={false}
                    strokeDasharray="5 5"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div style={{
              fontSize: '0.85rem',
              color: 'var(--text-muted)'
            }}>
              Sensor: {selectedSensor} | Method: {selectedMethod} | Gap length: {result.gap_length}
            </div>
          </div>
        )}

        {/* Metrics Cards */}
        {result && (
          <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
            <div className="glass-panel-header">
              <h2 style={{
                fontSize: '1.1rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                <BarChart3 size={20} color="#38bdf8" />
                Performance Metrics
              </h2>
            </div>
            
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '1rem',
              marginBottom: '1.5rem'
            }}>
              <div style={{
                padding: '1.25rem',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px'
              }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>MSE</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                  {result.metrics.mse.toFixed(6)}
                </div>
              </div>
              <div style={{
                padding: '1.25rem',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px'
              }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>MAE</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700, color: '#10b981', fontFamily: 'var(--font-mono)' }}>
                  {result.metrics.mae.toFixed(6)}
                </div>
              </div>
              <div style={{
                padding: '1.25rem',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px'
              }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>RMSE</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700, color: '#a855f7', fontFamily: 'var(--font-mono)' }}>
                  {result.metrics.rmse.toFixed(6)}
                </div>
              </div>
              <div style={{
                padding: '1.25rem',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px'
              }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <Clock size={14} />
                  Inference Time
                </div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700, color: '#f59e0b', fontFamily: 'var(--font-mono)' }}>
                  {inferenceTime.toFixed(0)}ms
                </div>
              </div>
            </div>
            
            <div style={{
              fontSize: '0.85rem',
              color: 'var(--text-secondary)',
              marginBottom: '1.5rem',
              padding: '0.75rem',
              background: 'var(--bg-primary)',
              borderRadius: '6px'
            }}>
              <strong>Configuration:</strong> {gapConfig.contextBefore}-{gapConfig.gapLength}-{gapConfig.contextAfter} | 
              <strong> Method:</strong> {selectedMethod} | 
              <strong> File:</strong> {result.filename} | 
              <strong> Window:</strong> {result.window_index} | 
              <strong> Metrics:</strong> Normalized space
            </div>
            
            {/* Per-Sensor Metrics */}
            <div>
              <h3 style={{
                fontSize: '0.95rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
                marginBottom: '1rem'
              }}>
                Per-Sensor Reconstruction Error
              </h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{
                  width: '100%',
                  fontSize: '0.85rem',
                  borderCollapse: 'collapse'
                }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <th style={{
                        padding: '0.75rem',
                        textAlign: 'left',
                        color: 'var(--text-muted)',
                        fontWeight: 600,
                        fontSize: '0.75rem',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em'
                      }}>
                        Sensor
                      </th>
                      <th style={{
                        padding: '0.75rem',
                        textAlign: 'right',
                        color: 'var(--text-muted)',
                        fontWeight: 600,
                        fontSize: '0.75rem',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em'
                      }}>
                        MSE
                      </th>
                      <th style={{
                        padding: '0.75rem',
                        textAlign: 'right',
                        color: 'var(--text-muted)',
                        fontWeight: 600,
                        fontSize: '0.75rem',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em'
                      }}>
                        MAE
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(calculatePerSensorMetrics() || {}).map(([sensor, metrics]) => (
                      <tr key={sensor} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <td style={{ padding: '0.75rem', color: 'var(--text-primary)', fontWeight: 500 }}>{sensor}</td>
                        <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{metrics.mse.toFixed(6)}</td>
                        <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{metrics.mae.toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Method Comparison */}
        {evaluationSummary && (
          <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
            <div className="glass-panel-header">
              <h2 style={{
                fontSize: '1.1rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                <BarChart3 size={20} color="#38bdf8" />
                Method Comparison (Gap Length: {evaluationSummary.gap_length})
              </h2>
            </div>
            
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
              Based on {evaluationSummary.total_sequences} test sequences from held-out test set.
            </div>
            
            <div style={{ overflowX: 'auto' }}>
              <table style={{
                width: '100%',
                fontSize: '0.85rem',
                borderCollapse: 'collapse'
              }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <th style={{
                      padding: '0.75rem',
                      textAlign: 'left',
                      color: 'var(--text-muted)',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      Method
                    </th>
                    <th style={{
                      padding: '0.75rem',
                      textAlign: 'right',
                      color: 'var(--text-muted)',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      MSE
                    </th>
                    <th style={{
                      padding: '0.75rem',
                      textAlign: 'right',
                      color: 'var(--text-muted)',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      MAE
                    </th>
                    <th style={{
                      padding: '0.75rem',
                      textAlign: 'right',
                      color: 'var(--text-muted)',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      RMSE
                    </th>
                    <th style={{
                      padding: '0.75rem',
                      textAlign: 'center',
                      color: 'var(--text-muted)',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      Best
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '0.75rem', color: 'var(--text-primary)', fontWeight: 500 }}>Linear Interpolation</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{evaluationSummary.overall_normalized.linear_interp.mse.toFixed(6)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{evaluationSummary.overall_normalized.linear_interp.mae.toFixed(6)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{evaluationSummary.overall_normalized.linear_interp.rmse.toFixed(6)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'center' }}>
                      {evaluationSummary.overall_normalized.linear_interp.mse < evaluationSummary.overall_normalized.residual_lstm.mse && (
                        <CheckCircle size={18} color="#10b981" style={{ margin: '0 auto' }} />
                      )}
                    </td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '0.75rem', color: 'var(--text-primary)', fontWeight: 500 }}>Residual LSTM</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{evaluationSummary.overall_normalized.residual_lstm.mse.toFixed(6)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{evaluationSummary.overall_normalized.residual_lstm.mae.toFixed(6)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{evaluationSummary.overall_normalized.residual_lstm.rmse.toFixed(6)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'center' }}>
                      {evaluationSummary.overall_normalized.residual_lstm.mse < evaluationSummary.overall_normalized.linear_interp.mse && (
                        <CheckCircle size={18} color="#10b981" style={{ margin: '0 auto' }} />
                      )}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            
            <div style={{
              marginTop: '1rem',
              padding: '0.75rem',
              background: 'rgba(245, 158, 11, 0.08)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '6px'
            }}>
              <p style={{
                fontSize: '0.85rem',
                color: '#fbbf24',
                margin: 0
              }}>
                <strong>Key Finding:</strong> For gap length {evaluationSummary.gap_length}, 
                {evaluationSummary.overall_normalized.residual_lstm.mse < evaluationSummary.overall_normalized.linear_interp.mse 
                  ? ' Residual LSTM shows lower MSE than linear interpolation.' 
                  : ' Linear interpolation performs better than Residual LSTM.'}
                The difference is modest and should be weighed against computational cost.
              </p>
            </div>
          </div>
        )}

        {/* Training Epoch Comparison */}
        {epochComparison && (
          <div className="glass-panel">
            <div className="glass-panel-header">
              <h2 style={{
                fontSize: '1.1rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                <Clock size={20} color="#38bdf8" />
                Training Epoch Comparison
              </h2>
            </div>
            
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
              Training history for Attention Seq2Seq model. Best epoch: {epochComparison.best_epoch} (val loss: {epochComparison.best_val_loss?.toFixed(6)}). Total training time: {(epochComparison.total_time / 60).toFixed(1)} minutes.
            </div>
            
            <div style={{ overflowX: 'auto', marginBottom: '1.5rem' }}>
              <table style={{
                width: '100%',
                fontSize: '0.85rem',
                borderCollapse: 'collapse'
              }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <th style={{
                      padding: '0.75rem',
                      textAlign: 'left',
                      color: 'var(--text-muted)',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      Requested Epochs
                    </th>
                    <th style={{
                      padding: '0.75rem',
                      textAlign: 'left',
                      color: 'var(--text-muted)',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      Status
                    </th>
                    <th style={{
                      padding: '0.75rem',
                      textAlign: 'right',
                      color: 'var(--text-muted)',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      Train Loss
                    </th>
                    <th style={{
                      padding: '0.75rem',
                      textAlign: 'right',
                      color: 'var(--text-muted)',
                      fontWeight: 600,
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      Val Loss
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {epochComparison.epoch_data.map((epoch) => (
                    <tr key={epoch.requested_epochs} style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '0.75rem', color: 'var(--text-primary)', fontWeight: 500 }}>{epoch.requested_epochs}</td>
                      <td style={{ padding: '0.75rem' }}>
                        {epoch.status === 'evaluated' ? (
                          <span style={{ color: '#10b981', fontWeight: 600 }}>✓ Evaluated</span>
                        ) : (
                          <span style={{ color: '#f43f5e', fontWeight: 600 }}>✗ Not evaluated</span>
                        )}
                      </td>
                      <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                        {epoch.train_loss !== null ? epoch.train_loss.toFixed(6) : '-'}
                      </td>
                      <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                        {epoch.val_loss !== null ? epoch.val_loss.toFixed(6) : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div style={{ height: '300px', marginBottom: '1rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={epochComparison.train_losses.map((trainLoss, i) => ({
                  epoch: i + 1,
                  trainLoss: trainLoss,
                  valLoss: epochComparison.val_losses[i]
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                  <XAxis 
                    dataKey="epoch" 
                    label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: 'var(--text-secondary)' }}
                    stroke="var(--text-secondary)"
                  />
                  <YAxis 
                    label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }}
                    stroke="var(--text-secondary)"
                  />
                  <Tooltip 
                    contentStyle={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      color: 'var(--text-primary)'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="trainLoss" 
                    stroke="#38bdf8" 
                    strokeWidth={2}
                    name="Training Loss"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="valLoss" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    name="Validation Loss"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <div style={{
              padding: '0.75rem',
              background: 'rgba(56, 189, 248, 0.08)',
              border: '1px solid rgba(56, 189, 248, 0.3)',
              borderRadius: '6px'
            }}>
              <p style={{
                fontSize: '0.85rem',
                color: '#38bdf8',
                margin: 0
              }}>
                <strong>Note:</strong> These are the actual training losses from the 50-epoch training run. 
                The values at epochs 20, 30, and 40 are extracted from the training history. 
                The model achieved its best validation loss at epoch {epochComparison.best_epoch}.
                Separate checkpoints were not saved for intermediate epochs, so we use the loss values from the full training history.
              </p>
            </div>
          </div>
        )}
      </div>
  );
};
