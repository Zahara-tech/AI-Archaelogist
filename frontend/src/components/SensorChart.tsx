import React, { useState } from 'react';
import type { SensorDataResponse } from '../types';
import { Activity, Thermometer, Gauge, Zap, AlertTriangle } from 'lucide-react';

interface Props {
  data: SensorDataResponse | null;
  loading: boolean;
}

export const SensorChart: React.FC<Props> = ({ data, loading }) => {
  const [activeMetric, setActiveMetric] = useState<'temperature' | 'vibration' | 'coolant_pressure' | 'spindle_rpm'>('temperature');

  if (loading) {
    return (
      <div className="glass-panel" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
        <Activity size={32} className="spin" style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
        <p style={{ color: 'var(--text-muted)' }}>Loading 1Hz sensor telemetry series...</p>
      </div>
    );
  }

  if (!data || !data.records || data.records.length === 0) {
    return (
      <div className="glass-panel" style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
        <Activity size={32} style={{ margin: '0 auto 1rem', opacity: 0.4 }} />
        <p>No sensor telemetry data available for this incident.</p>
      </div>
    );
  }

  const records = data.records;
  const values = records.map(r => Number(r[activeMetric] || 0));
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const range = maxVal - minVal || 1;

  // Metric configs
  const configs = {
    temperature: {
      label: 'Spindle Bearing Temperature (°C)',
      color: '#ef4444',
      unit: '°C',
      warningThreshold: 65,
      criticalThreshold: 75,
      icon: <Thermometer size={16} color="#ef4444" />
    },
    vibration: {
      label: 'Spindle Accelerometer Vibration (mm/s)',
      color: '#f59e0b',
      unit: 'mm/s',
      warningThreshold: 3.5,
      criticalThreshold: 6.0,
      icon: <Activity size={16} color="#f59e0b" />
    },
    coolant_pressure: {
      label: 'Coolant Supply Line Pressure (bar)',
      color: '#06b6d4',
      unit: 'bar',
      warningThreshold: 2.5,
      criticalThreshold: 1.5,
      icon: <Gauge size={16} color="#06b6d4" />
    },
    spindle_rpm: {
      label: 'Spindle Speed (RPM)',
      color: '#3b82f6',
      unit: 'RPM',
      warningThreshold: 10000,
      criticalThreshold: 5000,
      icon: <Zap size={16} color="#3b82f6" />
    }
  };

  const currConfig = configs[activeMetric];

  // Generate SVG Points
  const svgWidth = 1000;
  const svgHeight = 260;
  const padding = 40;
  const plotWidth = svgWidth - padding * 2;
  const plotHeight = svgHeight - padding * 2;

  const points = records.map((r, i) => {
    const x = padding + (i / (records.length - 1 || 1)) * plotWidth;
    const normY = (Number(r[activeMetric] || 0) - minVal) / range;
    const y = svgHeight - padding - normY * plotHeight;
    return `${x},${y}`;
  }).join(' ');

  // Critical threshold Y
  let critY = null;
  if (currConfig.criticalThreshold >= minVal && currConfig.criticalThreshold <= maxVal) {
    const normCrit = (currConfig.criticalThreshold - minVal) / range;
    critY = svgHeight - padding - normCrit * plotHeight;
  }

  return (
    <div className="glass-panel">
      <div className="glass-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          {currConfig.icon}
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            High-Frequency Sensor Telemetry (1Hz Time Series)
          </h3>
        </div>

        {/* Metric Selector Tabs */}
        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
          {(['temperature', 'vibration', 'coolant_pressure', 'spindle_rpm'] as const).map(m => (
            <button
              key={m}
              className={`btn ${activeMetric === m ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
              onClick={() => setActiveMetric(m)}
            >
              {configs[m].label.split(' ')[0]}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
        <div>
          Active Metric: <strong style={{ color: currConfig.color }}>{currConfig.label}</strong>
        </div>
        <div style={{ display: 'flex', gap: '1.5rem', fontFamily: 'var(--font-mono)' }}>
          <span>Min: <strong style={{ color: 'var(--text-primary)' }}>{minVal.toFixed(1)} {currConfig.unit}</strong></span>
          <span>Max: <strong style={{ color: 'var(--text-primary)' }}>{maxVal.toFixed(1)} {currConfig.unit}</strong></span>
          <span>Data Points: <strong style={{ color: '#38bdf8' }}>{records.length}</strong></span>
        </div>
      </div>

      {/* SVG Interactive Chart */}
      <div style={{ width: '100%', overflowX: 'auto', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', padding: '0.5rem 0' }}>
        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
          {/* Grid lines */}
          <line x1={padding} y1={padding} x2={svgWidth - padding} y2={padding} stroke="var(--border-color)" strokeDasharray="3 3" />
          <line x1={padding} y1={svgHeight / 2} x2={svgWidth - padding} y2={svgHeight / 2} stroke="var(--border-color)" strokeDasharray="3 3" />
          <line x1={padding} y1={svgHeight - padding} x2={svgWidth - padding} y2={svgHeight - padding} stroke="var(--border-color)" />

          {/* Critical Threshold Line */}
          {critY !== null && (
            <g>
              <line x1={padding} y1={critY} x2={svgWidth - padding} y2={critY} stroke="#f43f5e" strokeDasharray="4 4" strokeWidth="1.5" />
              <text x={svgWidth - padding - 10} y={critY - 5} fill="#f43f5e" fontSize="10" textAnchor="end" fontFamily="var(--font-mono)">
                CRITICAL LIMIT ({currConfig.criticalThreshold} {currConfig.unit})
              </text>
            </g>
          )}

          {/* Telemetry Sparkline */}
          <polyline
            fill="none"
            stroke={currConfig.color}
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={points}
          />

          {/* Start & End labels */}
          <text x={padding} y={svgHeight - 12} fill="var(--text-muted)" fontSize="11" fontFamily="var(--font-mono)">
            {records[0]?.timestamp.split(' ')[1] || '10:00:00'}
          </text>
          <text x={svgWidth - padding} y={svgHeight - 12} fill="var(--text-muted)" fontSize="11" textAnchor="end" fontFamily="var(--font-mono)">
            {records[records.length - 1]?.timestamp.split(' ')[1] || '10:10:00'}
          </text>
        </svg>
      </div>

      <div style={{ marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
        <AlertTriangle size={13} color="#f59e0b" />
        <span>Telemetry gap detected between 10:05:10 and 10:06:15 during emergency control reset.</span>
      </div>
    </div>
  );
};
