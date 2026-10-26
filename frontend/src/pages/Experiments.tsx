import React, { useState, useEffect } from 'react';
import { FlaskConical, Database, AlertTriangle, BarChart3, TrendingUp, Download, RefreshCw, Filter, Activity, Radar } from 'lucide-react';
import { api } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface EvaluationMetrics {
  model_name: string;
  test_mse: number | null;
  test_mae: number | null;
  baseline_mse: number | null;
  baseline_mae: number | null;
  test_sequences: number | null;
  test_files: number | null;
  per_sensor_mse: number[] | null;
  per_sensor_mae: number[] | null;
  training_history?: any;
}

interface EvaluationDetails {
  protocol: any;
  per_file: any[];
  overall: any;
}


type MetricType = 'mse' | 'mae' | 'rmse';
type ViewMode = 'aggregate' | 'single-gap';

export const Experiments: React.FC = () => {
  const [evaluationMetrics, setEvaluationMetrics] = useState<EvaluationMetrics | null>(null);
  const [evaluationDetails, setEvaluationDetails] = useState<EvaluationDetails | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('mse');
  const [viewMode, setViewMode] = useState<ViewMode>('aggregate');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadEvaluationData();
  }, []);

  const loadEvaluationData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [metrics, details] = await Promise.all([
        api.getEvaluationMetrics().catch(() => null),
        api.getEvaluationDetails().catch(() => null)
      ]);
      setEvaluationMetrics(metrics);
      setEvaluationDetails(details);
    } catch (err) {
      setError('Failed to load evaluation data');
    } finally {
      setLoading(false);
    }
  };

  const exportMetricsCSV = () => {
    if (!evaluationDetails) return;
    const header = ['File', 'Sequences', 'Baseline MSE', 'Baseline MAE', 'Attention V2 MSE', 'Attention V2 MAE', 'MSE Ratio', 'MAE Ratio'];
    const rows = evaluationDetails.per_file.map((f: any) => [
      f.file,
      f.sequences,
      f.baseline_mse?.toFixed(8) || 'N/A',
      f.baseline_mae?.toFixed(8) || 'N/A',
      f.attention_v2_mse?.toFixed(8) || 'N/A',
      f.attention_v2_mae?.toFixed(8) || 'N/A',
      f.mse_ratio_vs_baseline?.toFixed(4) || 'N/A',
      f.mae_ratio_vs_baseline?.toFixed(4) || 'N/A'
    ]);
    const csv = [header, ...rows].map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'evaluation_metrics.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const exportMetricsJSON = () => {
    if (!evaluationDetails) return;
    const blob = new Blob([JSON.stringify(evaluationDetails, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'evaluation_metrics.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const prepareBarChartData = () => {
    if (!evaluationDetails) return [];
    return evaluationDetails.per_file.map((file: any) => ({
      name: file.file.replace('test_', '').replace('.npz', ''),
      baseline: file.baseline_mse || 0,
      attention: file.attention_v2_mse || 0
    }));
  };


  const prepareRadarData = () => {
    if (!evaluationMetrics) return [];
    const metrics = evaluationMetrics;
    const normalizeError = (val: number | null, max: number) => {
      if (val === null) return 0;
      return Math.max(0, 100 - (val / max) * 100);
    };
    return [
      {
        metric: 'MSE',
        baseline: normalizeError(metrics.baseline_mse, 0.2),
        attention: normalizeError(metrics.test_mse, 0.2)
      },
      {
        metric: 'MAE',
        baseline: normalizeError(metrics.baseline_mae, 0.2),
        attention: normalizeError(metrics.test_mae, 0.2)
      }
    ];
  };


  if (loading) {
    return (
      <div style={{ padding: '5rem 2rem 2rem 2rem', textAlign: 'center' }}>
        <Activity size={48} color="#a855f7" style={{ animation: 'spin 1s linear infinite' }} />
        <p style={{ color: 'var(--text-muted)', marginTop: '1rem' }}>Loading evaluation data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
        <div style={{ padding: '1.5rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid #ef4444', borderRadius: '12px', color: '#ef4444' }}>
          <strong>Error:</strong> {error}
        </div>
      </div>
    );
  }

  const barChartData = prepareBarChartData();
  const radarChartData = prepareRadarData();

  return (
    <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <FlaskConical size={28} color="#a855f7" />
          <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)' }}>
            Experiments & Results
          </h1>
        </div>
        <p style={{ fontSize: '1rem', color: 'var(--text-muted)', fontWeight: 400 }}>
          Interactive scientific dashboard for model evaluation and comparison
        </p>
      </div>

      {/* Experiment Control Bar */}
      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        borderRadius: '12px',
        padding: '1.5rem',
        marginBottom: '2rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <Filter size={20} color="#3b82f6" />
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Experiment Controls
          </h3>
        </div>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600 }}>View Mode:</label>
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value as ViewMode)}
              style={{
                padding: '0.5rem 1rem',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                color: 'var(--text-primary)',
                fontSize: '0.85rem'
              }}
            >
              <option value="aggregate">Aggregate Test Results</option>
              <option value="single-gap">Single Gap Analysis</option>
            </select>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600 }}>Metric:</label>
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value as MetricType)}
              style={{
                padding: '0.5rem 1rem',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                color: 'var(--text-primary)',
                fontSize: '0.85rem'
              }}
            >
              <option value="mse">MSE</option>
              <option value="mae">MAE</option>
              <option value="rmse">RMSE</option>
            </select>
          </div>
          <button
            onClick={loadEvaluationData}
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
            <RefreshCw size={14} />
            Refresh
          </button>
        </div>
        {evaluationMetrics && (
          <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            <strong>Dataset:</strong> PHM 2018 | <strong>Test Sequences:</strong> {evaluationMetrics.test_sequences?.toLocaleString() || 'N/A'} | <strong>Test Files:</strong> {evaluationMetrics.test_files || 'N/A'}
          </div>
        )}
      </div>

      {/* Dashboard Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem', marginBottom: '2rem' }}>
        {/* Panel A: Model Performance Comparison */}
        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '1.5rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <BarChart3 size={20} color="#3b82f6" />
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Model Performance Comparison
            </h3>
          </div>
          <div style={{ height: '300px' }}>
            {barChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                  <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={12} />
                  <YAxis stroke="var(--text-muted)" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--bg-primary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      color: 'var(--text-primary)'
                    }}
                  />
                  <Legend />
                  <Bar dataKey="baseline" name="Linear Interpolation" fill="#3b82f6" />
                  <Bar dataKey="attention" name="Attention Seq2Seq V2" fill="#a855f7" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
                No evaluation data available
              </div>
            )}
          </div>
        </div>


        {/* Panel C: Model Performance Radar */}
        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '1.5rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <Radar size={20} color="#a855f7" />
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Performance Radar (Normalized Score)
            </h3>
          </div>
          <div style={{ height: '300px' }}>
            {radarChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={radarChartData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                  <XAxis type="number" stroke="var(--text-muted)" fontSize={12} domain={[0, 100]} />
                  <YAxis dataKey="metric" type="category" stroke="var(--text-muted)" fontSize={12} width={60} />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--bg-primary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      color: 'var(--text-primary)'
                    }}
                    formatter={(value: any) => `${Number(value || 0).toFixed(1)}%`}
                  />
                  <Legend />
                  <Bar dataKey="baseline" name="Linear Interpolation" fill="#3b82f6" />
                  <Bar dataKey="attention" name="Attention Seq2Seq V2" fill="#a855f7" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
                No evaluation data available
              </div>
            )}
          </div>
          <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Higher score = better performance (normalized to 0-100 scale)
          </div>
        </div>

        {/* Panel D: Results Table */}
        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '1.5rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <TrendingUp size={20} color="#22c55e" />
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Evaluation Results
              </h3>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                onClick={exportMetricsCSV}
                style={{
                  padding: '0.25rem 0.5rem',
                  background: 'rgba(59, 130, 246, 0.1)',
                  border: '1px solid #3b82f6',
                  borderRadius: '4px',
                  color: '#3b82f6',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.25rem'
                }}
              >
                <Download size={12} />
                CSV
              </button>
              <button
                onClick={exportMetricsJSON}
                style={{
                  padding: '0.25rem 0.5rem',
                  background: 'rgba(168, 85, 247, 0.1)',
                  border: '1px solid #a855f7',
                  borderRadius: '4px',
                  color: '#a855f7',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.25rem'
                }}
              >
                <Download size={12} />
                JSON
              </button>
            </div>
          </div>
          <div style={{ maxHeight: '250px', overflowY: 'auto' }}>
            {evaluationDetails ? (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)', position: 'sticky', top: 0, background: 'var(--bg-secondary)' }}>
                    <th style={{ padding: '0.5rem', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600 }}>File</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Baseline MSE</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Attention MSE</th>
                    <th style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-muted)', fontWeight: 600 }}>Ratio</th>
                  </tr>
                </thead>
                <tbody>
                  {evaluationDetails.per_file.map((file: any, i: number) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '0.5rem', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{file.file}</td>
                      <td style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-secondary)' }}>{file.baseline_mse?.toFixed(8) || 'N/A'}</td>
                      <td style={{ padding: '0.5rem', textAlign: 'right', color: file.attention_v2_mse < file.baseline_mse ? '#22c55e' : '#ef4444', fontWeight: 600 }}>{file.attention_v2_mse?.toFixed(8) || 'N/A'}</td>
                      <td style={{ padding: '0.5rem', textAlign: 'right', color: 'var(--text-secondary)' }}>{file.mse_ratio_vs_baseline?.toFixed(2) || 'N/A'}x</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
                No evaluation data available
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Overall Summary */}
      {evaluationMetrics && evaluationDetails && (
        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '2rem',
          marginBottom: '2rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <Database size={24} color="#3b82f6" />
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Overall Evaluation Summary
            </h2>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
            <div style={{ padding: '1rem', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid #3b82f6', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.8rem', color: '#3b82f6', fontWeight: 600, marginBottom: '0.25rem' }}>Baseline MSE</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>{evaluationMetrics.baseline_mse?.toFixed(8) || 'N/A'}</div>
            </div>
            <div style={{ padding: '1rem', background: 'rgba(168, 85, 247, 0.1)', border: '1px solid #a855f7', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.8rem', color: '#a855f7', fontWeight: 600, marginBottom: '0.25rem' }}>Attention V2 MSE</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>{evaluationMetrics.test_mse?.toFixed(8) || 'N/A'}</div>
            </div>
            <div style={{ padding: '1rem', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid #3b82f6', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.8rem', color: '#3b82f6', fontWeight: 600, marginBottom: '0.25rem' }}>Baseline MAE</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>{evaluationMetrics.baseline_mae?.toFixed(8) || 'N/A'}</div>
            </div>
            <div style={{ padding: '1rem', background: 'rgba(168, 85, 247, 0.1)', border: '1px solid #a855f7', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.8rem', color: '#a855f7', fontWeight: 600, marginBottom: '0.25rem' }}>Attention V2 MAE</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>{evaluationMetrics.test_mae?.toFixed(8) || 'N/A'}</div>
            </div>
          </div>
          <div style={{ marginTop: '1.5rem', padding: '1rem', background: evaluationDetails.overall.attention_v2_mse < evaluationDetails.overall.baseline_mse ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)', border: evaluationDetails.overall.attention_v2_mse < evaluationDetails.overall.baseline_mse ? '1px solid #22c55e' : '1px solid #ef4444', borderRadius: '8px', fontSize: '0.9rem', color: evaluationDetails.overall.attention_v2_mse < evaluationDetails.overall.baseline_mse ? '#22c55e' : '#ef4444' }}>
            <strong>Interpretation:</strong> {evaluationDetails.overall.attention_v2_mse < evaluationDetails.overall.baseline_mse ? 'Attention Seq2Seq V2 outperforms linear interpolation on overall test set.' : 'Linear interpolation outperforms Attention Seq2Seq V2 on overall test set.'}
          </div>
        </div>
      )}

      {/* Scientific Integrity Notice */}
      <div style={{ background: 'rgba(245, 158, 11, 0.1)', border: '1px solid #f59e0b', borderRadius: '12px', padding: '1.5rem', display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
        <AlertTriangle size={24} color="#f59e0b" style={{ flexShrink: 0 }} />
        <div>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f59e0b', marginBottom: '0.5rem' }}>
            Scientific Integrity Notice
          </h3>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', margin: 0 }}>
            These metrics are from held-out test set evaluation. The model may not always outperform the baseline linear interpolation. Results are computed honestly without cherry-picking. Current evaluation uses the old incorrectly-normalized benchmark data. New evaluation should be run after model retraining completes.
          </p>
        </div>
      </div>
    </div>
  );
};
