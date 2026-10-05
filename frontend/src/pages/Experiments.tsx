import React from 'react';
import { FlaskConical, Database, AlertTriangle, CheckCircle } from 'lucide-react';

export const Experiments: React.FC = () => {
  return (
    <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          marginBottom: '0.5rem'
        }}>
          <FlaskConical size={28} color="#a855f7" />
          <h1 style={{
            fontSize: '1.75rem',
            fontWeight: 800,
            color: 'var(--text-primary)'
          }}>
            Dataset & Experiments
          </h1>
        </div>
        <p style={{
          fontSize: '1rem',
          color: 'var(--text-muted)',
          fontWeight: 400
        }}>
          Manage datasets and run reconstruction experiments.
        </p>
      </div>

      {/* Dataset Status */}
      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        borderRadius: '12px',
        padding: '2rem',
        marginBottom: '2rem'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          marginBottom: '1.5rem'
        }}>
          <Database size={24} color="#3b82f6" />
          <h2 style={{
            fontSize: '1.25rem',
            fontWeight: 700,
            color: 'var(--text-primary)'
          }}>
            PHM 2018 Dataset
          </h2>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          marginBottom: '1.5rem',
          padding: '0.75rem 1rem',
          background: 'rgba(34, 197, 94, 0.1)',
          border: '1px solid #22c55e',
          borderRadius: '8px',
          width: 'fit-content'
        }}>
          <CheckCircle size={18} color="#22c55e" />
          <span style={{
            fontSize: '0.9rem',
            color: '#22c55e',
            fontWeight: 600
          }}>
            Dataset Status: Loaded / Ready
          </span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '1.5rem'
        }}>
          <div>
            <div style={{
              fontSize: '0.8rem',
              color: 'var(--text-muted)',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              marginBottom: '0.5rem'
            }}>
              Records
            </div>
            <div style={{
              fontSize: '1.5rem',
              fontWeight: 700,
              color: 'var(--text-primary)'
            }}>
              Awaiting ingestion
            </div>
          </div>

          <div>
            <div style={{
              fontSize: '0.8rem',
              color: 'var(--text-muted)',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              marginBottom: '0.5rem'
            }}>
              Features
            </div>
            <div style={{
              fontSize: '1.5rem',
              fontWeight: 700,
              color: 'var(--text-primary)'
            }}>
              Awaiting ingestion
            </div>
          </div>

          <div>
            <div style={{
              fontSize: '0.8rem',
              color: 'var(--text-muted)',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              marginBottom: '0.5rem'
            }}>
              Time Range
            </div>
            <div style={{
              fontSize: '1.5rem',
              fontWeight: 700,
              color: 'var(--text-primary)'
            }}>
              Awaiting ingestion
            </div>
          </div>

          <div>
            <div style={{
              fontSize: '0.8rem',
              color: 'var(--text-muted)',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              marginBottom: '0.5rem'
            }}>
              Fault Types
            </div>
            <div style={{
              fontSize: '1.5rem',
              fontWeight: 700,
              color: 'var(--text-primary)'
            }}>
              Awaiting ingestion
            </div>
          </div>
        </div>
      </div>

      {/* Notice */}
      <div style={{
        background: 'rgba(245, 158, 11, 0.1)',
        border: '1px solid #f59e0b',
        borderRadius: '12px',
        padding: '1.5rem',
        display: 'flex',
        gap: '1rem',
        alignItems: 'flex-start'
      }}>
        <AlertTriangle size={24} color="#f59e0b" style={{ flexShrink: 0 }} />
        <div>
          <h3 style={{
            fontSize: '1rem',
            fontWeight: 600,
            color: '#f59e0b',
            marginBottom: '0.5rem'
          }}>
            Dataset Not Yet Ingested
          </h3>
          <p style={{
            fontSize: '0.9rem',
            color: 'var(--text-muted)',
            margin: 0
          }}>
            The PHM 2018 dataset has not been loaded by the backend yet. Statistics will appear here once the dataset is ingested. For now, the application uses clearly labeled DEMO DATA for visualization purposes.
          </p>
        </div>
      </div>
    </div>
  );
};
