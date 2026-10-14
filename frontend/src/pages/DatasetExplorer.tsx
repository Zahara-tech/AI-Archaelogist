import React, { useState, useEffect } from 'react';
import { Database, BarChart3, Activity, AlertTriangle, CheckCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface DatasetInfo {
  dataset_name: string;
  files: Array<{ filename: string; window_count: number; file_size_mb: number }>;
  sensor_columns: string[];
  context_length: number;
  gap_length: number;
  total_sequences: number;
}

interface SensorData {
  filename: string;
  window_index: number;
  context_before: number[][];
  context_after: number[][];
  masked_segment: number[][];
  sensor_columns: string[];
  full_sequence: number[][];
}

export const DatasetExplorer: React.FC = () => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [selectedSensor, setSelectedSensor] = useState<string>('');
  const [sensorData, setSensorData] = useState<SensorData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDatasetInfo();
  }, []);

  const loadDatasetInfo = async () => {
    try {
      setLoading(true);
      const data = await fetch('/api/v1/sensor/dataset/info').then(r => r.json());
      setDatasetInfo(data);
      if (data.files.length > 0) {
        setSelectedFile(data.files[0].filename);
      }
    } catch (err) {
      setError('Failed to load dataset information');
    } finally {
      setLoading(false);
    }
  };

  const loadSensorData = async () => {
    if (!selectedFile) return;
    try {
      setLoading(true);
      setError(null);
      const data = await fetch('/api/v1/sensor/dataset/sensor-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: selectedFile, window_index: selectedIndex, sensors: selectedSensor ? [selectedSensor] : null }),
      }).then(r => r.json());
      setSensorData(data);
    } catch (err) {
      setError('Failed to load sensor data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedFile && datasetInfo) {
      if (datasetInfo.sensor_columns.length > 0 && !selectedSensor) {
        setSelectedSensor(datasetInfo.sensor_columns[0]);
      }
      loadSensorData();
    }
  }, [selectedFile, selectedIndex, selectedSensor]);

  const prepareChartData = () => {
    if (!sensorData) return [];
    const sensorIndex = sensorData.sensor_columns.indexOf(selectedSensor);
    if (sensorIndex === -1) return [];
    
    return sensorData.full_sequence.map((values, index) => ({
      index,
      value: values[sensorIndex],
      region: index < 50 ? 'Context Before' : index < 60 ? 'Masked Gap' : 'Context After'
    }));
  };

  if (loading && !datasetInfo) {
    return (
      <div style={{ padding: '5rem 2rem 2rem 2rem', textAlign: 'center' }}>
        <Activity size={48} color="#a855f7" style={{ animation: 'spin 1s linear infinite' }} />
        <p style={{ color: 'var(--text-muted)', marginTop: '1rem' }}>Loading dataset information...</p>
      </div>
    );
  }

  return (
    <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <Database size={28} color="#a855f7" />
          <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)' }}>
            Dataset Explorer
          </h1>
        </div>
        <p style={{ fontSize: '1rem', color: 'var(--text-muted)', fontWeight: 400 }}>
          Explore PHM 2018 Ion Mill Etch Tool sensor data
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

      {datasetInfo && (
        <>
          {/* Dataset Statistics */}
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
                {datasetInfo.dataset_name}
              </h2>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                  Total Sequences
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {datasetInfo.total_sequences.toLocaleString()}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                  Test Files
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {datasetInfo.files.length}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                  Sensor Features
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {datasetInfo.sensor_columns.length}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                  Context Length
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {datasetInfo.context_length}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.5rem' }}>
                  Gap Length
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {datasetInfo.gap_length}
                </div>
              </div>
            </div>

            <div style={{
              padding: '1rem',
              background: 'rgba(34, 197, 94, 0.1)',
              border: '1px solid #22c55e',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <CheckCircle size={18} color="#22c55e" />
              <span style={{ fontSize: '0.9rem', color: '#22c55e', fontWeight: 600 }}>
                Dataset loaded successfully from data/benchmark_v2/
              </span>
            </div>
          </div>

          {/* Sensor Columns */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '2rem',
            marginBottom: '2rem'
          }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '1rem' }}>
              Sensor Columns (17 Features)
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0.5rem' }}>
              {datasetInfo.sensor_columns.map((sensor) => (
                <div key={sensor} style={{
                  padding: '0.5rem 0.75rem',
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  fontSize: '0.85rem',
                  color: 'var(--text-secondary)',
                  fontFamily: 'monospace'
                }}>
                  {sensor}
                </div>
              ))}
            </div>
          </div>

          {/* Data Visualization Controls */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '2rem',
            marginBottom: '2rem'
          }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '1.5rem' }}>
              Sensor Visualization
            </h3>

            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
              <div style={{ flex: 1, minWidth: '200px' }}>
                <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 600 }}>
                  Test File
                </label>
                <select
                  value={selectedFile}
                  onChange={(e) => setSelectedFile(e.target.value)}
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
                  {datasetInfo.files.map((file) => (
                    <option key={file.filename} value={file.filename}>
                      {file.filename} ({file.window_count.toLocaleString()} windows, {file.file_size_mb} MB)
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ width: '150px' }}>
                <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 600 }}>
                  Window Index
                </label>
                <input
                  type="number"
                  min="0"
                  value={selectedIndex}
                  onChange={(e) => setSelectedIndex(parseInt(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'var(--bg-primary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px',
                    color: 'var(--text-primary)',
                    fontSize: '0.9rem'
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

            {sensorData && (
              <>
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
                        dataKey="value" 
                        stroke="#a855f7" 
                        strokeWidth={2}
                        dot={false}
                        name={selectedSensor}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                  <div style={{
                    padding: '0.75rem 1rem',
                    background: 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid #3b82f6',
                    borderRadius: '6px',
                    fontSize: '0.85rem',
                    color: 'var(--text-secondary)'
                  }}>
                    <strong>Context Before:</strong> Steps 0-49
                  </div>
                  <div style={{
                    padding: '0.75rem 1rem',
                    background: 'rgba(245, 158, 11, 0.1)',
                    border: '1px solid #f59e0b',
                    borderRadius: '6px',
                    fontSize: '0.85rem',
                    color: 'var(--text-secondary)'
                  }}>
                    <strong>Masked Gap:</strong> Steps 50-59
                  </div>
                  <div style={{
                    padding: '0.75rem 1rem',
                    background: 'rgba(34, 197, 94, 0.1)',
                    border: '1px solid #22c55e',
                    borderRadius: '6px',
                    fontSize: '0.85rem',
                    color: 'var(--text-secondary)'
                  }}>
                    <strong>Context After:</strong> Steps 60-109
                  </div>
                </div>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
};
