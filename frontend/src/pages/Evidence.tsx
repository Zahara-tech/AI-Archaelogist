import React from 'react';
import { FolderOpen, FileText, Database, UploadCloud } from 'lucide-react';

export const Evidence: React.FC = () => {
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
          <FolderOpen size={28} color="#a855f7" />
          <h1 style={{
            fontSize: '1.75rem',
            fontWeight: 800,
            color: 'var(--text-primary)'
          }}>
            Evidence Management
          </h1>
        </div>
        <p style={{
          fontSize: '1rem',
          color: 'var(--text-muted)',
          fontWeight: 400
        }}>
          Upload and manage evidence sources for incident reconstruction.
        </p>
      </div>

      {/* Upload Area */}
      <div style={{
        background: 'var(--bg-secondary)',
        border: '2px dashed var(--border-color)',
        borderRadius: '12px',
        padding: '3rem',
        textAlign: 'center',
        marginBottom: '2rem',
        cursor: 'pointer'
      }}>
        <UploadCloud size={48} color="#64748b" style={{ marginBottom: '1rem' }} />
        <h3 style={{
          fontSize: '1.1rem',
          fontWeight: 600,
          color: 'var(--text-primary)',
          marginBottom: '0.5rem'
        }}>
          Drop evidence files here
        </h3>
        <p style={{
          fontSize: '0.9rem',
          color: 'var(--text-muted)'
        }}>
          Supported formats: CSV, LOG, JSON, PDF, XLSX
        </p>
      </div>

      {/* Evidence Types */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '1rem'
      }}>
        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '1.5rem'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '1rem'
          }}>
            <Database size={24} color="#06b6d4" />
            <h3 style={{
              fontSize: '1rem',
              fontWeight: 600,
              color: 'var(--text-primary)'
            }}>
              Sensor Data
            </h3>
          </div>
          <p style={{
            fontSize: '0.85rem',
            color: 'var(--text-muted)',
            marginBottom: '0.5rem'
          }}>
            Time-series sensor readings from industrial equipment
          </p>
          <div style={{
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)'
          }}>
            CSV, JSON
          </div>
        </div>

        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '1.5rem'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '1rem'
          }}>
            <FileText size={24} color="#60a5fa" />
            <h3 style={{
              fontSize: '1rem',
              fontWeight: 600,
              color: 'var(--text-primary)'
            }}>
              Machine Logs
            </h3>
          </div>
          <p style={{
            fontSize: '0.85rem',
            color: 'var(--text-muted)',
            marginBottom: '0.5rem'
          }}>
              System logs and event records from machines
          </p>
          <div style={{
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)'
          }}>
            LOG, TXT
          </div>
        </div>

        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '1.5rem'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '1rem'
          }}>
            <FileText size={24} color="#f59e0b" />
            <h3 style={{
              fontSize: '1rem',
              fontWeight: 600,
              color: 'var(--text-primary)'
            }}>
              Reports
            </h3>
          </div>
          <p style={{
            fontSize: '0.85rem',
            color: 'var(--text-muted)',
            marginBottom: '0.5rem'
          }}>
            Operator reports and maintenance documentation
          </p>
          <div style={{
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)'
          }}>
            PDF, XLSX
          </div>
        </div>
      </div>
    </div>
  );
};
