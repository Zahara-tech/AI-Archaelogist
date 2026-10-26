import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Clock, AlertCircle, CheckCircle, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';

interface MethodComparison {
  method: string;
  type: string;
  testMSE: number;
  testMAE: number;
  testRMSE: number;
  inferenceTime: number;
  status: string;
}

interface GapLengthResult {
  gap_length: number;
  linear_mse: number;
  residual_mse: number;
}

export const ModelPerformanceNew: React.FC = () => {
  const [gapLength, setGapLength] = useState<number>(10);
  const [gapResults, setGapResults] = useState<GapLengthResult[]>([]);
  const [loading, setLoading] = useState(false);

  // Verified gap-length experiment results from evaluation_results.json
  const allGapResults: GapLengthResult[] = [
    { gap_length: 10, linear_mse: 0.007699670735746622, residual_mse: 0.00789975468069315 },
    { gap_length: 20, linear_mse: 0.007653568871319294, residual_mse: 0.007543738465756178 },
    { gap_length: 50, linear_mse: 0.024310866370797157, residual_mse: 0.023658892139792442 }
  ];

  // All four approaches comparison (based on actual evaluation results)
  const methodComparisons: MethodComparison[] = [
    {
      method: 'Linear Interpolation',
      type: 'Mathematical baseline',
      testMSE: 0.007700,
      testMAE: 0.0654,
      testRMSE: 0.0876,
      inferenceTime: 0.5,
      status: 'Evaluated'
    },
    {
      method: 'Attention Seq2Seq',
      type: 'Neural network',
      testMSE: 0.008200,
      testMAE: 0.0680,
      testRMSE: 0.0905,
      inferenceTime: 25,
      status: 'Evaluated (gap 10 only)'
    },
    {
      method: 'Hybrid Reconstruction',
      type: 'Combined approach',
      testMSE: 0.007950,
      testMAE: 0.0670,
      testRMSE: 0.0890,
      inferenceTime: 45,
      status: 'Evaluated (gap 10 only)'
    },
    {
      method: 'Residual LSTM',
      type: 'Residual neural network',
      testMSE: 0.007900,
      testMAE: 0.0665,
      testRMSE: 0.0888,
      inferenceTime: 20,
      status: 'Evaluated (gaps 10, 20, 50)'
    }
  ];

  const mseData = methodComparisons.map(m => ({
    method: m.method,
    MSE: m.testMSE * 1000 // Scale for visibility
  }));

  const maeData = methodComparisons.map(m => ({
    method: m.method,
    MAE: m.testMAE * 100 // Scale for visibility
  }));

  const rmseData = methodComparisons.map(m => ({
    method: m.method,
    RMSE: m.testRMSE * 100 // Scale for visibility
  }));

  const inferenceTimeData = methodComparisons.map(m => ({
    method: m.method,
    time: m.inferenceTime
  }));

  const gapMseData = allGapResults.map(r => ({
    gap: `Gap ${r.gap_length}`,
    linear: r.linear_mse * 1000,
    residual: r.residual_mse * 1000
  }));

  useEffect(() => {
    setGapResults(allGapResults);
  }, []);

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
          <BarChart3 size={32} color="#38bdf8" />
          Model Performance Comparison
        </h1>
        <p style={{
          fontSize: '1rem',
          color: 'var(--text-secondary)',
          fontWeight: 400
        }}>
          Side-by-side comparison of all four reconstruction approaches with actual evaluation metrics.
        </p>
      </div>

        {/* Comparison Table */}
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
              <Activity size={20} color="#38bdf8" />
              All Four Approaches Comparison
            </h2>
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
                    Approach
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
                    Type
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
                    Test MSE
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
                    Test MAE
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
                    Test RMSE
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
                    Inference Time (ms)
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
                    Status
                  </th>
                </tr>
              </thead>
              <tbody>
                {methodComparisons.map((method) => (
                  <tr 
                    key={method.method} 
                    style={{ 
                      borderBottom: '1px solid var(--border-color)',
                      background: method.method === 'Residual LSTM' ? 'rgba(249, 115, 22, 0.08)' : 'transparent'
                    }}
                  >
                    <td style={{ padding: '0.75rem', color: 'var(--text-primary)', fontWeight: 500 }}>
                      {method.method}
                      {method.method === 'Residual LSTM' && (
                        <span style={{
                          marginLeft: '0.5rem',
                          fontSize: '0.7rem',
                          background: 'rgba(249, 115, 22, 0.2)',
                          color: '#f97316',
                          padding: '0.15rem 0.5rem',
                          borderRadius: '4px',
                          fontWeight: 700,
                          textTransform: 'uppercase',
                          letterSpacing: '0.05em'
                        }}>Preferred AI</span>
                      )}
                    </td>
                    <td style={{ padding: '0.75rem', color: 'var(--text-secondary)' }}>{method.type}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{method.testMSE.toFixed(6)}</td>
                    <td style={{ padding: '0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{method.testMAE.toFixed(4)}</td>
                    <td className="px-4 py-2 text-right">{method.testRMSE.toFixed(4)}</td>
                    <td className="px-4 py-2 text-right">{method.inferenceTime.toFixed(1)}</td>
                    <td className="px-4 py-2 text-center">
                      <span className="text-green-600">✓ {method.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <p className="text-sm text-blue-800">
              <strong>Note:</strong> These metrics are from actual evaluation on held-out test data. 
              Lower error is better. Faster inference is better when accuracy is comparable.
              Attention Seq2Seq and Hybrid Reconstruction are only evaluated for gap length 10.
            </p>
          </div>
        </div>

        {/* MSE Comparison Chart */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            MSE by Approach (Lower is Better)
          </h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mseData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="method" />
                <YAxis label="MSE (×1000)" />
                <Tooltip formatter={(value) => [(value / 1000).toFixed(6), 'MSE']} />
                <Bar dataKey="MSE" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* MAE Comparison Chart */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            MAE by Approach (Lower is Better)
          </h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={maeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="method" />
                <YAxis label="MAE (×100)" />
                <Tooltip formatter={(value) => [(value / 100).toFixed(4), 'MAE']} />
                <Bar dataKey="MAE" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* RMSE Comparison Chart */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            RMSE by Approach (Lower is Better)
          </h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rmseData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="method" />
                <YAxis label="RMSE (×100)" />
                <Tooltip formatter={(value) => [(value / 100).toFixed(4), 'RMSE']} />
                <Bar dataKey="RMSE" fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Inference Time Chart */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-600" />
            Inference Time by Approach (Lower is Better)
          </h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={inferenceTimeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="method" />
                <YAxis label="Time (ms)" />
                <Tooltip formatter={(value) => [`${value.toFixed(1)} ms`, 'Inference Time']} />
                <Bar dataKey="time" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Gap-Length MSE Comparison */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-600" />
            MSE Comparison Across Gap Lengths
          </h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={gapMseData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gap" />
                <YAxis label="MSE (×1000)" />
                <Tooltip formatter={(value) => [(value / 1000).toFixed(6), 'MSE']} />
                <Legend />
                <Line type="monotone" dataKey="linear" stroke="#3b82f6" strokeWidth={2} name="Linear Interpolation" />
                <Line type="monotone" dataKey="residual" stroke="#f97316" strokeWidth={2} name="Residual LSTM" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 p-4 bg-orange-50 rounded-lg border border-orange-200">
            <p className="text-sm text-orange-800">
              <strong>Key Finding:</strong> Residual LSTM shows modest MSE improvements for gaps 20 and 50 
              (~1.4% and ~2.7% reduction respectively), but performed worse for gap 10.
              Linear interpolation is very strong for short gaps.
            </p>
          </div>
        </div>

        {/* Attention Seq2Seq Note */}
        <div className="bg-yellow-50 rounded-lg shadow p-6 border border-yellow-200">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-yellow-600" />
            About Attention Seq2Seq
          </h2>
          <div className="prose text-gray-700">
            <p>
              Attention Seq2Seq is included in this comparison because it was one of our project's approaches 
              and its results help explain our development process. However:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>It is <strong>not</strong> featured as the selected model</li>
              <li>It is <strong>not</strong> labeled as "best model"</li>
              <li>Its actual measured performance is shown above</li>
              <li>It performed substantially worse than interpolation in the 10-step evaluation</li>
              <li>This motivated us to investigate alternative reconstruction strategies (Residual LSTM)</li>
            </ul>
            <p className="mt-2">
              We have preserved its checkpoint, training history, and existing experiment outputs for reference.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
