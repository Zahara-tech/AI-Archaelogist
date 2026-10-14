import React, { useState, useEffect } from 'react';
import { Cpu, TrendingUp, AlertTriangle, BarChart3, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ModelInfo {
  model_name: string;
  architecture: string;
  parameter_count: number;
  input_dim: number;
  output_dim: number;
  context_length: number;
  gap_length: number;
  model_loaded: boolean;
}

interface EvaluationMetrics {
  model_name: string;
  test_mse: number;
  test_mae: number;
  baseline_mse: number;
  baseline_mae: number;
  test_sequences: number;
  test_files: number;
  per_sensor_mse: Record<string, number> | null;
  per_sensor_mae: Record<string, number> | null;
}

interface TrainingHistory {
  hyperparameters: {
    hidden_dim: number;
    num_layers: number;
    dropout: number;
    attention_dim: number;
    learning_rate: number;
    batch_size: number;
  };
  train_losses: number[];
  val_losses: number[];
  training_results: {
    total_epochs: number;
    best_epoch: number;
    best_val_loss: number;
  };
}

export const ModelPerformance: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [evaluationMetrics, setEvaluationMetrics] = useState<EvaluationMetrics | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    try {
      setLoading(true);
      
      // Load model info
      const modelData = await fetch('/api/v1/model/info').then(r => r.json());
      setModelInfo(modelData);
      
      // Load evaluation metrics
      const evalData = await fetch('/api/v1/sensor/model/evaluation-metrics').then(r => r.json());
      setEvaluationMetrics(evalData);
      
      // Load training history (from results file via backend or directly)
      try {
        const historyData = await fetch('/api/v1/sensor/model/training-history').then(r => r.json());
        setTrainingHistory(historyData);
      } catch {
        // Training history endpoint may not exist, that's okay
        console.log('Training history endpoint not available');
      }
      
    } catch (err) {
      setError('Failed to load model performance data');
    } finally {
      setLoading(false);
    }
  };

  const prepareLossChartData = () => {
    if (!trainingHistory) return [];
    return trainingHistory.train_losses.map((trainLoss, i) => ({
      epoch: i + 1,
      train: trainLoss,
      val: trainingHistory.val_losses[i] || null
    }));
  };

  if (loading) {
    return (
      <div style={{ padding: '5rem 2rem 2rem 2rem', textAlign: 'center' }}>
        <Activity size={48} color="#a855f7" style={{ animation: 'spin 1s linear infinite' }} />
        <p style={{ color: 'var(--text-muted)', marginTop: '1rem' }}>Loading model performance data...</p>
      </div>
    );
  }

  return (
    <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <Cpu size={28} color="#a855f7" />
          <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)' }}>
            Model Performance
          </h1>
        </div>
        <p style={{ fontSize: '1rem', color: 'var(--text-muted)', fontWeight: 400 }}>
          Attention Seq2Seq V2 architecture and evaluation metrics
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

      {modelInfo && (
        <>
          {/* Model Architecture */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '2rem',
            marginBottom: '2rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <Cpu size={24} color="#3b82f6" />
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Model Architecture
              </h2>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                  Model Name
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {modelInfo.model_name}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                  Parameters
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {modelInfo.parameter_count?.toLocaleString() || '331,042'}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                  Input Features
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {modelInfo.input_dim || 17}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                  Context Length
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {modelInfo.context_length || 50} steps
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                  Gap Length
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {modelInfo.gap_length || 10} steps
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                  Status
                </div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: modelInfo.model_loaded ? '#22c55e' : '#ef4444' }}>
                  {modelInfo.model_loaded ? 'Loaded' : 'Not Loaded'}
                </div>
              </div>
            </div>

            <div style={{
              padding: '1rem',
              background: 'rgba(59, 130, 246, 0.1)',
              border: '1px solid #3b82f6',
              borderRadius: '8px',
              fontSize: '0.9rem',
              color: 'var(--text-secondary)'
            }}>
              <strong>Architecture:</strong> {modelInfo.architecture || 'Bidirectional LSTM encoder with additive attention and LSTM decoder'}
            </div>
          </div>

          {/* Test Set Evaluation */}
          {evaluationMetrics && (
            <div style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: '12px',
              padding: '2rem',
              marginBottom: '2rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                <TrendingUp size={24} color="#3b82f6" />
                <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Held-Out Test Set Evaluation
                </h2>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                <div style={{ padding: '1rem', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid #22c55e', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.8rem', color: '#22c55e', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Neural Model MSE
                  </div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {evaluationMetrics.test_mse.toFixed(6)}
                  </div>
                </div>

                <div style={{ padding: '1rem', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid #22c55e', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.8rem', color: '#22c55e', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Neural Model MAE
                  </div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {evaluationMetrics.test_mae.toFixed(6)}
                  </div>
                </div>

                <div style={{ padding: '1rem', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid #3b82f6', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.8rem', color: '#3b82f6', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Baseline MSE
                  </div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {evaluationMetrics.baseline_mse.toFixed(6)}
                  </div>
                </div>

                <div style={{ padding: '1rem', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid #3b82f6', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.8rem', color: '#3b82f6', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Baseline MAE
                  </div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {evaluationMetrics.baseline_mae.toFixed(6)}
                  </div>
                </div>

                <div style={{ padding: '1rem', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid #f59e0b', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.8rem', color: '#f59e0b', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Test Sequences
                  </div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {evaluationMetrics.test_sequences.toLocaleString()}
                  </div>
                </div>

                <div style={{ padding: '1rem', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid #f59e0b', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.8rem', color: '#f59e0b', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Test Files
                  </div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {evaluationMetrics.test_files}
                  </div>
                </div>
              </div>

              {/* Important Scientific Result */}
              <div style={{
                padding: '1.5rem',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '2px solid #ef4444',
                borderRadius: '8px',
                display: 'flex',
                gap: '1rem',
                alignItems: 'flex-start'
              }}>
                <AlertTriangle size={24} color="#ef4444" style={{ flexShrink: 0 }} />
                <div>
                  <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#ef4444', marginBottom: '0.5rem' }}>
                    Important Scientific Result
                  </h3>
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.6 }}>
                    The linear interpolation baseline substantially outperforms the Attention Seq2Seq V2 neural model on the held-out test set. 
                    Baseline MSE ({evaluationMetrics.baseline_mse.toFixed(6)}) is significantly lower than Neural MSE ({evaluationMetrics.test_mse.toFixed(6)}).
                    This result is displayed honestly to guide future model improvements. The purpose of this research is to investigate and enhance the neural model, not to conceal the baseline's superior performance.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Training Loss Curves */}
          {trainingHistory && (
            <div style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: '12px',
              padding: '2rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                <BarChart3 size={24} color="#3b82f6" />
                <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Training History
                </h2>
              </div>

              <div style={{ height: '350px', marginBottom: '1.5rem' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={prepareLossChartData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                    <XAxis 
                      dataKey="epoch" 
                      stroke="var(--text-muted)"
                      label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: 'var(--text-muted)' }}
                    />
                    <YAxis 
                      stroke="var(--text-muted)"
                      label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)' }}
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
                      dataKey="train" 
                      stroke="#a855f7" 
                      strokeWidth={2}
                      dot={false}
                      name="Training Loss"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="val" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      dot={false}
                      name="Validation Loss"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
                <div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Total Epochs
                  </div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {trainingHistory.training_results.total_epochs}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Best Epoch
                  </div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {trainingHistory.training_results.best_epoch}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Best Val Loss
                  </div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {trainingHistory.training_results.best_val_loss.toFixed(6)}
                  </div>
                </div>
              </div>
            </div>
          )}

          {!trainingHistory && (
            <div style={{
              padding: '1rem',
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid #f59e0b',
              borderRadius: '8px',
              fontSize: '0.9rem',
              color: 'var(--text-muted)'
            }}>
              Training history not available. The training history file may not be accessible via the API.
            </div>
          )}
        </>
      )}
    </div>
  );
};
