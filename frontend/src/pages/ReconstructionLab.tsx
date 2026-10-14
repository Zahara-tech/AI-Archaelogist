import React, { useState, useEffect } from 'react';
import { FlaskConical, Play, Eye, EyeOff, AlertTriangle, TrendingUp, BarChart3, Download, Table } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea } from 'recharts';

interface DatasetInfo {
  dataset_name: string;
  files: Array<{ filename: string; window_count: number; file_size_mb: number }>;
  sensor_columns: string[];
  context_length: number;
  gap_length: number;
  total_sequences: number;
}

interface ReconstructionResult {
  filename: string;
  window_index: number;
  gap_start_index: number;
  gap_length: number;
  context_before: number[][];
  context_after: number[][];
  ground_truth: number[][];
  neural_predictions: number[][];
  baseline_predictions: number[][];
  neural_metrics: { mse: number; mae: number; rmse: number; per_sensor_mse?: Record<string, number>; per_sensor_mae?: Record<string, number> };
  baseline_metrics: { mse: number; mae: number; rmse: number; per_sensor_mse?: Record<string, number>; per_sensor_mae?: Record<string, number> };
  better_method: string;
  sensor_columns: string[];
}

export const ReconstructionLab: React.FC = () => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [selectedSensor, setSelectedSensor] = useState<string>('');
  const [gapStartIndex, setGapStartIndex] = useState<number>(50);
  const [gapLength, setGapLength] = useState<number>(10);
  const [isMasked, setIsMasked] = useState<boolean>(false);
  const [result, setResult] = useState<ReconstructionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDatasetInfo();
  }, []);

  const loadDatasetInfo = async () => {
    try {
      const data = await fetch('/api/v1/sensor/dataset/info').then(r => r.json());
      setDatasetInfo(data);
      if (data.files.length > 0) {
        setSelectedFile(data.files[0].filename);
      }
    } catch (err) {
      setError('Failed to load dataset information');
    }
  };

  const handleMask = () => {
    setIsMasked(true);
    setResult(null);
  };

  const handleRunReconstruction = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/sensor/dataset/masked-reconstruct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: selectedFile,
          window_index: selectedIndex,
          gap_start_index: gapStartIndex,
          gap_length: gapLength
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to reconstruct');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to run reconstruction');
      console.error('Reconstruction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const prepareChartData = () => {
    if (!result) return [];
    const sensorIndex = result.sensor_columns.indexOf(selectedSensor);
    if (sensorIndex === -1) return [];
    
    const contextBeforeLen = result.context_before.length;
    const gapLen = result.gap_length;
    
    const data: any[] = [];
    
    // Context before
    result.context_before.forEach((values, i) => {
      data.push({
        index: i,
        groundTruth: values[sensorIndex],
        neural: null,
        baseline: null,
        region: 'Context Before'
      });
    });
    
    // Gap region
    result.ground_truth.forEach((values, i) => {
      data.push({
        index: contextBeforeLen + i,
        groundTruth: values[sensorIndex],
        neural: result.neural_predictions[i][sensorIndex],
        baseline: result.baseline_predictions[i][sensorIndex],
        region: 'Gap'
      });
    });
    
    // Context after
    result.context_after.forEach((values, i) => {
      data.push({
        index: contextBeforeLen + gapLen + i,
        groundTruth: values[sensorIndex],
        neural: null,
        baseline: null,
        region: 'Context After'
      });
    });
    
    return data;
  };

  const calculatePerSensorMetrics = () => {
    if (!result) return null;
    
    const perSensorMetrics: Record<string, { neuralMSE: number; neuralMAE: number; baselineMSE: number; baselineMAE: number }> = {};
    
    result.sensor_columns.forEach((sensor, sensorIdx) => {
      let neuralSE = 0;
      let neuralAE = 0;
      let baselineSE = 0;
      let baselineAE = 0;
      
      for (let t = 0; t < result.gap_length; t++) {
        const actual = result.ground_truth[t][sensorIdx];
        const neuralPred = result.neural_predictions[t][sensorIdx];
        const baselinePred = result.baseline_predictions[t][sensorIdx];
        
        neuralSE += Math.pow(neuralPred - actual, 2);
        neuralAE += Math.abs(neuralPred - actual);
        baselineSE += Math.pow(baselinePred - actual, 2);
        baselineAE += Math.abs(baselinePred - actual);
      }
      
      perSensorMetrics[sensor] = {
        neuralMSE: neuralSE / result.gap_length,
        neuralMAE: neuralAE / result.gap_length,
        baselineMSE: baselineSE / result.gap_length,
        baselineMAE: baselineAE / result.gap_length
      };
    });
    
    return perSensorMetrics;
  };

  const exportCSV = (data: any[][], filename: string) => {
    const csvContent = data.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const exportActualMatrix = () => {
    if (!result) return;
    const header = ['Timestep', ...result.sensor_columns];
    const rows = result.ground_truth.map((row, i) => [i, ...row]);
    exportCSV([header, ...rows], `actual_matrix_${result.filename}_window_${result.window_index}.csv`);
  };

  const exportNeuralMatrix = () => {
    if (!result) return;
    const header = ['Timestep', ...result.sensor_columns];
    const rows = result.neural_predictions.map((row, i) => [i, ...row]);
    exportCSV([header, ...rows], `neural_predictions_${result.filename}_window_${result.window_index}.csv`);
  };

  const exportBaselineMatrix = () => {
    if (!result) return;
    const header = ['Timestep', ...result.sensor_columns];
    const rows = result.baseline_predictions.map((row, i) => [i, ...row]);
    exportCSV([header, ...rows], `baseline_predictions_${result.filename}_window_${result.window_index}.csv`);
  };

  const exportComparisonTable = () => {
    if (!result) return;
    const perSensorMetrics = calculatePerSensorMetrics();
    
    const header = ['Timestep', 'Sensor', 'Actual', 'Neural Prediction', 'Neural Error', 'Neural Abs Error', 'Baseline Prediction', 'Baseline Abs Error'];
    const rows: any[] = [];
    
    result.sensor_columns.forEach((sensor, sensorIdx) => {
      for (let t = 0; t < result.gap_length; t++) {
        const actual = result.ground_truth[t][sensorIdx];
        const neuralPred = result.neural_predictions[t][sensorIdx];
        const baselinePred = result.baseline_predictions[t][sensorIdx];
        const neuralError = neuralPred - actual;
        const neuralAbsError = Math.abs(neuralError);
        const baselineAbsError = Math.abs(baselinePred - actual);
        
        rows.push([t, sensor, actual.toFixed(8), neuralPred.toFixed(8), neuralError.toFixed(8), neuralAbsError.toFixed(8), baselinePred.toFixed(8), baselineAbsError.toFixed(8)]);
      }
    });
    
    exportCSV([header, ...rows], `comparison_table_${result.filename}_window_${result.window_index}.csv`);
  };

  const exportMetrics = () => {
    if (!result) return;
    const perSensorMetrics = calculatePerSensorMetrics();
    
    const header = ['Sensor', 'Neural MSE', 'Neural MAE', 'Baseline MSE', 'Baseline MAE', 'Better Method'];
    const rows: any[] = [];
    
    Object.entries(perSensorMetrics || {}).forEach(([sensor, metrics]) => {
      const better = metrics.neuralMSE < metrics.baselineMSE ? 'Neural' : 'Baseline';
      rows.push([sensor, metrics.neuralMSE.toFixed(8), metrics.neuralMAE.toFixed(8), metrics.baselineMSE.toFixed(8), metrics.baselineMAE.toFixed(8), better]);
    });
    
    // Add overall metrics
    rows.push(['OVERALL', result.neural_metrics.mse.toFixed(8), result.neural_metrics.mae.toFixed(8), result.baseline_metrics.mse.toFixed(8), result.baseline_metrics.mae.toFixed(8), result.better_method]);
    
    exportCSV([header, ...rows], `metrics_${result.filename}_window_${result.window_index}.csv`);
  };

  const exportAllJSON = () => {
    if (!result) return;
    const perSensorMetrics = calculatePerSensorMetrics();
    
    const exportData = {
      filename: result.filename,
      window_index: result.window_index,
      gap_start_index: result.gap_start_index,
      gap_length: result.gap_length,
      sensor_columns: result.sensor_columns,
      ground_truth: result.ground_truth,
      neural_predictions: result.neural_predictions,
      baseline_predictions: result.baseline_predictions,
      overall_metrics: {
        neural: result.neural_metrics,
        baseline: result.baseline_metrics,
        better_method: result.better_method
      },
      per_sensor_metrics: perSensorMetrics,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `complete_results_${result.filename}_window_${result.window_index}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!datasetInfo) {
    return (
      <div style={{ padding: '5rem 2rem 2rem 2rem', textAlign: 'center' }}>
        <FlaskConical size={48} color="#a855f7" style={{ animation: 'spin 1s linear infinite' }} />
        <p style={{ color: 'var(--text-muted)', marginTop: '1rem' }}>Loading dataset...</p>
      </div>
    );
  }

  return (
    <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <FlaskConical size={28} color="#a855f7" />
          <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)' }}>
            Reconstruction Lab
          </h1>
        </div>
        <p style={{ fontSize: '1rem', color: 'var(--text-muted)', fontWeight: 400 }}>
          Interactive sensor reconstruction with masking and model comparison
        </p>
      </div>

      {error && (
        <div style={{
          padding: '1rem',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid #ef4444',
          borderRadius: '8px',
          color: '#ef4444',
          fontSize: '0.9rem',
          marginBottom: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {/* Controls */}
      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        borderRadius: '12px',
        padding: '2rem',
        marginBottom: '2rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
          <BarChart3 size={24} color="#3b82f6" />
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Experiment Configuration
          </h2>
        </div>

        <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: '200px' }}>
            <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 600 }}>
              Test File
            </label>
            <select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              disabled={isMasked}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '0.9rem',
                opacity: isMasked ? 0.6 : 1
              }}
            >
              {datasetInfo.files.map((file) => (
                <option key={file.filename} value={file.filename}>
                  {file.filename} ({file.window_count.toLocaleString()} windows)
                </option>
              ))}
            </select>
          </div>

          <div style={{ width: '120px' }}>
            <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 600 }}>
              Window Index
            </label>
            <input
              type="number"
              min="0"
              value={selectedIndex}
              onChange={(e) => setSelectedIndex(parseInt(e.target.value) || 0)}
              disabled={isMasked}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '0.9rem',
                opacity: isMasked ? 0.6 : 1
              }}
            />
          </div>

          <div style={{ width: '120px' }}>
            <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 600 }}>
              Gap Start
            </label>
            <input
              type="number"
              min="0"
              max="109"
              value={gapStartIndex}
              onChange={(e) => setGapStartIndex(parseInt(e.target.value) || 0)}
              disabled={isMasked}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '0.9rem',
                opacity: isMasked ? 0.6 : 1
              }}
            />
          </div>

          <div style={{ width: '120px' }}>
            <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 600 }}>
              Gap Length
            </label>
            <input
              type="number"
              min="1"
              max="10"
              value={gapLength}
              onChange={(e) => setGapLength(parseInt(e.target.value) || 10)}
              disabled={isMasked}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '0.9rem',
                opacity: isMasked ? 0.6 : 1
              }}
            />
          </div>

          <div style={{ flex: 1, minWidth: '200px' }}>
            <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 600 }}>
              Sensor
            </label>
            <select
              value={selectedSensor}
              onChange={(e) => setSelectedSensor(e.target.value)}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '0.9rem'
              }}
            >
              {datasetInfo.sensor_columns.map((sensor) => (
                <option key={sensor} value={sensor}>
                  {sensor}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '1rem' }}>
          {!isMasked ? (
            <button
              onClick={handleMask}
              disabled={loading}
              style={{
                padding: '0.75rem 1.5rem',
                background: loading ? '#6b7280' : '#f59e0b',
                border: 'none',
                borderRadius: '8px',
                color: 'white',
                fontSize: '0.9rem',
                fontWeight: 600,
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}
            >
              <EyeOff size={16} />
              Mask Selected Interval
            </button>
          ) : (
            <>
              <button
                onClick={() => { setIsMasked(false); setResult(null); }}
                style={{
                  padding: '0.75rem 1.5rem',
                  background: '#6b7280',
                  border: 'none',
                  borderRadius: '8px',
                  color: 'white',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <Eye size={16} />
                Reset
              </button>
              <button
                onClick={handleRunReconstruction}
                disabled={loading}
                style={{
                  padding: '0.75rem 1.5rem',
                  background: loading ? '#6b7280' : '#3b82f6',
                  border: 'none',
                  borderRadius: '8px',
                  color: 'white',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  cursor: loading ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <Play size={16} />
                {loading ? 'Running...' : 'Run Reconstruction'}
              </button>
            </>
          )}
        </div>

        {isMasked && (
          <div style={{
            marginTop: '1rem',
            padding: '0.75rem 1rem',
            background: 'rgba(245, 158, 11, 0.1)',
            border: '1px solid #f59e0b',
            borderRadius: '8px',
            fontSize: '0.85rem',
            color: 'var(--text-secondary)'
          }}>
            <strong>Masked:</strong> Gap at index {gapStartIndex}-{gapStartIndex + gapLength} is hidden from model input. Ground truth retained for evaluation only.
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Metrics */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '2rem',
            marginBottom: '2rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <TrendingUp size={20} color="#22c55e" />
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Reconstruction Metrics
              </h3>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
              <div style={{ padding: '1rem', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid #22c55e', borderRadius: '8px' }}>
                <div style={{ fontSize: '0.8rem', color: '#22c55e', fontWeight: 600, marginBottom: '0.25rem' }}>
                  Neural MSE
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {result.neural_metrics.mse.toFixed(6)}
                </div>
              </div>

              <div style={{ padding: '1rem', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid #22c55e', borderRadius: '8px' }}>
                <div style={{ fontSize: '0.8rem', color: '#22c55e', fontWeight: 600, marginBottom: '0.25rem' }}>
                  Neural MAE
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {result.neural_metrics.mae.toFixed(6)}
                </div>
              </div>

              <div style={{ padding: '1rem', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid #3b82f6', borderRadius: '8px' }}>
                <div style={{ fontSize: '0.8rem', color: '#3b82f6', fontWeight: 600, marginBottom: '0.25rem' }}>
                  Baseline MSE
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {result.baseline_metrics.mse.toFixed(6)}
                </div>
              </div>

              <div style={{ padding: '1rem', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid #3b82f6', borderRadius: '8px' }}>
                <div style={{ fontSize: '0.8rem', color: '#3b82f6', fontWeight: 600, marginBottom: '0.25rem' }}>
                  Baseline MAE
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {result.baseline_metrics.mae.toFixed(6)}
                </div>
              </div>

              <div style={{ padding: '1rem', background: result.better_method === 'neural' ? 'rgba(34, 197, 94, 0.1)' : 'rgba(59, 130, 246, 0.1)', border: result.better_method === 'neural' ? '1px solid #22c55e' : '1px solid #3b82f6', borderRadius: '8px' }}>
                <div style={{ fontSize: '0.8rem', color: result.better_method === 'neural' ? '#22c55e' : '#3b82f6', fontWeight: 600, marginBottom: '0.25rem' }}>
                  Better Method
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {result.better_method === 'neural' ? 'Neural' : 'Baseline'}
                </div>
              </div>
            </div>

            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', padding: '0.75rem', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid #f59e0b', borderRadius: '8px' }}>
              <strong>Note:</strong> Metrics computed on normalized data. Ground truth was hidden from model during inference and used only for evaluation.
            </div>
          </div>

          {/* Chart */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '2rem'
          }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '1.5rem' }}>
              {selectedSensor} - Reconstruction Comparison
            </h3>

            <div style={{ height: '400px', marginBottom: '1.5rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={prepareChartData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                  <XAxis 
                    dataKey="index" 
                    stroke="var(--text-muted)"
                    label={{ value: 'Time Step', position: 'insideBottom', offset: -5, fill: 'var(--text-muted)' }}
                  />
                  <YAxis 
                    stroke="var(--text-muted)"
                    label={{ value: 'Normalized Value', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      background: 'var(--bg-secondary)', 
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px'
                    }}
                    itemStyle={{ color: 'var(--text-primary)' }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="groundTruth" 
                    stroke="#22c55e" 
                    strokeWidth={2}
                    dot={false}
                    name="Ground Truth"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="neural" 
                    stroke="#a855f7" 
                    strokeWidth={2}
                    dot={false}
                    name="Neural Model"
                    connectNulls={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="baseline" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    dot={false}
                    name="Linear Interpolation"
                    connectNulls={false}
                  />
                  <ReferenceArea x1={result.context_before.length} x2={result.context_before.length + result.gap_length} strokeDasharray="3 3" fill="rgba(245, 158, 11, 0.1)" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              <div style={{ padding: '0.75rem 1rem', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid #22c55e', borderRadius: '6px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <strong>Ground Truth:</strong> Actual values (evaluation only)
              </div>
              <div style={{ padding: '0.75rem 1rem', background: 'rgba(168, 85, 247, 0.1)', border: '1px solid #a855f7', borderRadius: '6px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <strong>Neural Model:</strong> Attention Seq2Seq V2 prediction
              </div>
              <div style={{ padding: '0.75rem 1rem', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid #3b82f6', borderRadius: '6px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <strong>Baseline:</strong> Linear interpolation
              </div>
              <div style={{ padding: '0.75rem 1rem', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid #f59e0b', borderRadius: '6px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <strong>Shaded:</strong> Masked gap region
              </div>
            </div>
          </div>

          {/* Per-Sensor Metrics Table */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '2rem',
            marginBottom: '2rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Table size={20} color="#3b82f6" />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Per-Sensor Metrics
                </h3>
              </div>
              <button
                onClick={exportMetrics}
                style={{
                  padding: '0.5rem 1rem',
                  background: 'rgba(59, 130, 246, 0.1)',
                  border: '1px solid #3b82f6',
                  borderRadius: '6px',
                  color: '#3b82f6',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <Download size={14} />
                Export Metrics
              </button>
            </div>

            <div style={{ maxHeight: '400px', overflowY: 'auto', overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--border-color)', position: 'sticky', top: 0, background: 'var(--bg-secondary)' }}>
                    <th style={{ padding: '0.75rem', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600 }}>Sensor</th>
                    <th style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Neural MSE</th>
                    <th style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Neural MAE</th>
                    <th style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Baseline MSE</th>
                    <th style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Baseline MAE</th>
                    <th style={{ padding: '0.75rem', textAlign: 'center', color: 'var(--text-muted)', fontWeight: 600 }}>Better</th>
                  </tr>
                </thead>
                <tbody>
                  {result.sensor_columns.map((sensor) => {
                    const metrics = calculatePerSensorMetrics()?.[sensor];
                    if (!metrics) return null;
                    const better = metrics.neuralMSE < metrics.baselineMSE;
                    return (
                      <tr key={sensor} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <td style={{ padding: '0.75rem', color: 'var(--text-secondary)', fontFamily: 'monospace', fontSize: '0.8rem' }}>{sensor}</td>
                        <td style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-secondary)' }}>{metrics.neuralMSE.toFixed(8)}</td>
                        <td style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-secondary)' }}>{metrics.neuralMAE.toFixed(8)}</td>
                        <td style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-secondary)' }}>{metrics.baselineMSE.toFixed(8)}</td>
                        <td style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-secondary)' }}>{metrics.baselineMAE.toFixed(8)}</td>
                        <td style={{ padding: '0.75rem', textAlign: 'center', color: better ? '#22c55e' : '#3b82f6', fontWeight: 600 }}>
                          {better ? 'Neural' : 'Baseline'}
                        </td>
                      </tr>
                    );
                  })}
                  <tr style={{ borderBottom: '2px solid var(--border-color)', background: 'rgba(245, 158, 11, 0.1)' }}>
                    <td style={{ padding: '0.75rem', color: 'var(--text-primary)', fontWeight: 700 }}>OVERALL</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-primary)', fontWeight: 700 }}>{result.neural_metrics.mse.toFixed(8)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-primary)', fontWeight: 700 }}>{result.neural_metrics.mae.toFixed(8)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-primary)', fontWeight: 700 }}>{result.baseline_metrics.mse.toFixed(8)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', color: 'var(--text-primary)', fontWeight: 700 }}>{result.baseline_metrics.mae.toFixed(8)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'center', color: result.better_method === 'neural' ? '#22c55e' : '#3b82f6', fontWeight: 700 }}>
                      {result.better_method === 'neural' ? 'Neural' : 'Baseline'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Complete Results Table */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '2rem',
            marginBottom: '2rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Table size={20} color="#3b82f6" />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Complete Comparison Table ({result.gap_length} timesteps × {result.sensor_columns.length} sensors)
                </h3>
              </div>
              <button
                onClick={exportComparisonTable}
                style={{
                  padding: '0.5rem 1rem',
                  background: 'rgba(59, 130, 246, 0.1)',
                  border: '1px solid #3b82f6',
                  borderRadius: '6px',
                  color: '#3b82f6',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <Download size={14} />
                Export Table
              </button>
            </div>

            <div style={{ maxHeight: '500px', overflowY: 'auto', overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--border-color)', position: 'sticky', top: 0, background: 'var(--bg-secondary)' }}>
                    <th style={{ padding: '0.5rem', textAlign: 'center', color: 'var(--text-muted)', fontWeight: 600 }}>Timestep</th>
                    <th style={{ padding: '0.5rem', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600 }}>Sensor</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Actual</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right', color: '#a855f7', fontWeight: 600 }}>Neural</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Neural Err</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Neural |Err|</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right', color: '#3b82f6', fontWeight: 600 }}>Baseline</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Baseline |Err|</th>
                  </tr>
                </thead>
                <tbody>
                  {result.sensor_columns.map((sensor, sensorIdx) => (
                    result.ground_truth.map((_, t) => {
                      const actual = result.ground_truth[t][sensorIdx];
                      const neuralPred = result.neural_predictions[t][sensorIdx];
                      const baselinePred = result.baseline_predictions[t][sensorIdx];
                      const neuralError = neuralPred - actual;
                      const neuralAbsError = Math.abs(neuralError);
                      const baselineAbsError = Math.abs(baselinePred - actual);
                      
                      return (
                        <tr key={`${sensor}-${t}`} style={{ borderBottom: '1px solid var(--border-color)' }}>
                          <td style={{ padding: '0.5rem', textAlign: 'center', color: 'var(--text-secondary)' }}>{t}</td>
                          <td style={{ padding: '0.5rem', color: 'var(--text-secondary)', fontFamily: 'monospace', fontSize: '0.75rem' }}>{sensor}</td>
                          <td style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-secondary)' }}>{actual.toFixed(8)}</td>
                          <td style={{ padding: '0.5rem', textAlign: 'right', color: '#a855f7' }}>{neuralPred.toFixed(8)}</td>
                          <td style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-secondary)' }}>{neuralError.toFixed(8)}</td>
                          <td style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-secondary)' }}>{neuralAbsError.toFixed(8)}</td>
                          <td style={{ padding: '0.5rem', textAlign: 'right', color: '#3b82f6' }}>{baselinePred.toFixed(8)}</td>
                          <td style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-secondary)' }}>{baselineAbsError.toFixed(8)}</td>
                        </tr>
                      );
                    })
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Export Section */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '2rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <Download size={24} color="#3b82f6" />
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Export Results
              </h3>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
              <button
                onClick={exportActualMatrix}
                style={{
                  padding: '1rem',
                  background: 'rgba(34, 197, 94, 0.1)',
                  border: '1px solid #22c55e',
                  borderRadius: '8px',
                  color: '#22c55e',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  justifyContent: 'center'
                }}
              >
                <Download size={16} />
                Actual Matrix (CSV)
              </button>

              <button
                onClick={exportNeuralMatrix}
                style={{
                  padding: '1rem',
                  background: 'rgba(168, 85, 247, 0.1)',
                  border: '1px solid #a855f7',
                  borderRadius: '8px',
                  color: '#a855f7',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  justifyContent: 'center'
                }}
              >
                <Download size={16} />
                Neural Predictions (CSV)
              </button>

              <button
                onClick={exportBaselineMatrix}
                style={{
                  padding: '1rem',
                  background: 'rgba(59, 130, 246, 0.1)',
                  border: '1px solid #3b82f6',
                  borderRadius: '8px',
                  color: '#3b82f6',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  justifyContent: 'center'
                }}
              >
                <Download size={16} />
                Baseline Predictions (CSV)
              </button>

              <button
                onClick={exportComparisonTable}
                style={{
                  padding: '1rem',
                  background: 'rgba(245, 158, 11, 0.1)',
                  border: '1px solid #f59e0b',
                  borderRadius: '8px',
                  color: '#f59e0b',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  justifyContent: 'center'
                }}
              >
                <Download size={16} />
                Comparison Table (CSV)
              </button>

              <button
                onClick={exportMetrics}
                style={{
                  padding: '1rem',
                  background: 'rgba(236, 72, 153, 0.1)',
                  border: '1px solid #ec4899',
                  borderRadius: '8px',
                  color: '#ec4899',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  justifyContent: 'center'
                }}
              >
                <Download size={16} />
                Metrics (CSV)
              </button>

              <button
                onClick={exportAllJSON}
                style={{
                  padding: '1rem',
                  background: 'rgba(99, 102, 241, 0.1)',
                  border: '1px solid #6366f1',
                  borderRadius: '8px',
                  color: '#6366f1',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  justifyContent: 'center'
                }}
              >
                <Download size={16} />
                Complete Results (JSON)
              </button>
            </div>

            <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: 'var(--text-muted)', padding: '0.75rem', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid #f59e0b', borderRadius: '8px' }}>
              <strong>Note:</strong> All values are in normalized units. Metrics are computed on the masked segment only. Ground truth was hidden from model during inference.
            </div>
          </div>
        </>
      )}
    </div>
  );
};
