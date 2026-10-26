import React, { useState, useEffect } from 'react';
import { Clock, Activity, AlertCircle, CheckCircle, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface EpochData {
  requested_epochs: number;
  completed_epochs: number | null;
  train_loss: number | null;
  val_loss: number | null;
  status: string;
}

interface TrainingHistory {
  best_epoch: number;
  best_val_loss: number;
  total_epochs: number;
  total_time: number;
  epoch_data: EpochData[];
  train_losses: number[];
  val_losses: number[];
}

export const TrainingExperiments: React.FC = () => {
  const [selectedEpoch, setSelectedEpoch] = useState<number>(50);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadTrainingHistory();
  }, []);

  const loadTrainingHistory = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/sensor/model/epoch-comparison');
      if (response.ok) {
        const data = await response.json();
        setTrainingHistory(data);
      } else {
        setError('Failed to load training history');
      }
    } catch (err) {
      setError('Failed to load training history');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getEpochData = (epoch: number): EpochData | null => {
    if (!trainingHistory) return null;
    return trainingHistory.epoch_data.find(e => e.requested_epochs === epoch) || null;
  };

  const prepareChartData = () => {
    if (!trainingHistory) return [];
    return trainingHistory.train_losses.map((trainLoss, i) => ({
      epoch: i + 1,
      trainLoss: trainLoss,
      valLoss: trainingHistory.val_losses[i]
    }));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-gray-600">Loading training history...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-red-600">{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
            <Clock className="w-8 h-8 text-blue-600" />
            Training Experiments
          </h1>
          <p className="text-gray-600 mt-2">
            Examination of actual training history for Attention Seq2Seq model at different epoch counts.
          </p>
        </div>

        {/* Training Summary */}
        {trainingHistory && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-600" />
              Training Summary
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="text-sm text-gray-600">Best Epoch</div>
                <div className="text-2xl font-bold text-blue-900">{trainingHistory.best_epoch}</div>
              </div>
              <div className="p-4 bg-green-50 rounded-lg">
                <div className="text-sm text-gray-600">Best Val Loss</div>
                <div className="text-2xl font-bold text-green-900">{trainingHistory.best_val_loss?.toFixed(6)}</div>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <div className="text-sm text-gray-600">Total Epochs</div>
                <div className="text-2xl font-bold text-purple-900">{trainingHistory.total_epochs}</div>
              </div>
              <div className="p-4 bg-orange-50 rounded-lg">
                <div className="text-sm text-gray-600">Training Time</div>
                <div className="text-2xl font-bold text-orange-900">{(trainingHistory.total_time / 60).toFixed(1)}m</div>
              </div>
            </div>
          </div>
        )}

        {/* Epoch Comparison Table */}
        {trainingHistory && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              Epoch Comparison (20, 30, 40, 50)
            </h2>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left">Requested Epochs</th>
                    <th className="px-4 py-2 text-left">Status</th>
                    <th className="px-4 py-2 text-right">Train Loss</th>
                    <th className="px-4 py-2 text-right">Val Loss</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingHistory.epoch_data.map((epoch) => (
                    <tr key={epoch.requested_epochs} className="border-t">
                      <td className="px-4 py-2 font-medium">{epoch.requested_epochs}</td>
                      <td className="px-4 py-2">
                        {epoch.status === 'evaluated' ? (
                          <span className="text-green-600">✓ Evaluated</span>
                        ) : (
                          <span className="text-red-600">✗ Not evaluated</span>
                        )}
                      </td>
                      <td className="px-4 py-2 text-right">
                        {epoch.train_loss !== null ? epoch.train_loss.toFixed(6) : '-'}
                      </td>
                      <td className="px-4 py-2 text-right">
                        {epoch.val_loss !== null ? epoch.val_loss.toFixed(6) : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <p className="text-sm text-blue-800">
                <strong>Note:</strong> These are the actual training losses from the 50-epoch training run. 
                The values at epochs 20, 30, and 40 are extracted from the training history. 
                The model achieved its best validation loss at epoch {trainingHistory.best_epoch}.
                Separate checkpoints were not saved for intermediate epochs, so we use the loss values from the full training history.
              </p>
            </div>
          </div>
        )}

        {/* Training/Validation Loss Chart */}
        {trainingHistory && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-600" />
              Training and Validation Loss Over All Epochs
            </h2>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={prepareChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label="Epoch" />
                  <YAxis label="Loss" />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="trainLoss" 
                    stroke="#3b82f6" 
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
          </div>
        )}

        {/* Detailed Epoch Analysis */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-600" />
            Detailed Epoch Analysis
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[20, 30, 40, 50].map((epoch) => {
              const data = getEpochData(epoch);
              return (
                <div key={epoch} className="p-4 bg-gray-50 rounded-lg border">
                  <h3 className="font-semibold mb-2">{epoch} Epochs</h3>
                  {data ? (
                    <div className="text-sm space-y-1">
                      <div>Status: <span className={data.status === 'evaluated' ? 'text-green-600' : 'text-red-600'}>{data.status}</span></div>
                      <div>Train Loss: {data.train_loss !== null ? data.train_loss.toFixed(6) : 'N/A'}</div>
                      <div>Val Loss: {data.val_loss !== null ? data.val_loss.toFixed(6) : 'N/A'}</div>
                    </div>
                  ) : (
                    <div className="text-sm text-gray-500">No data available</div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Important Note */}
        <div className="bg-yellow-50 rounded-lg shadow p-6 border border-yellow-200">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-yellow-600" />
            Important Notes
          </h2>
          <div className="prose text-gray-700">
            <ul className="list-disc pl-6 space-y-2">
              <li>
                <strong>Intermediate Checkpoints:</strong> Separate checkpoints were not saved for epochs 20, 30, and 40. 
                The loss values shown are extracted from the full 50-epoch training history.
              </li>
              <li>
                <strong>Model Used:</strong> This training history is for the Attention Seq2Seq model. 
                Residual LSTM has separate training runs for each gap length (10, 20, 50).
              </li>
              <li>
                <strong>Early Stopping:</strong> Training stopped when validation loss stopped improving (at epoch 48). 
                This prevents overfitting.
              </li>
              <li>
                <strong>Gap Length Independence:</strong> Epoch setting is independent of missing-gap length. 
                Different gap lengths require separate model training.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
