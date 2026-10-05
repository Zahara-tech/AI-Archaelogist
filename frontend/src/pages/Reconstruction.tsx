import React from 'react';
import { GitBranch } from 'lucide-react';

export const Reconstruction: React.FC = () => {
  return (
    <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
      <div style={{ marginBottom: '2rem' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          marginBottom: '0.5rem'
        }}>
          <GitBranch size={28} color="#a855f7" />
          <h1 style={{
            fontSize: '1.75rem',
            fontWeight: 800,
            color: 'var(--text-primary)'
          }}>
            Event Reconstruction
          </h1>
        </div>
        <p style={{
          fontSize: '1rem',
          color: 'var(--text-muted)',
          fontWeight: 400
        }}>
          Reconstruct missing events from evidence gaps.
        </p>
      </div>

      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        borderRadius: '12px',
        padding: '3rem',
        textAlign: 'center',
        color: 'var(--text-muted)'
      }}>
        <GitBranch size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
        <p>Select an incident to view reconstruction options.</p>
      </div>
    </div>
  );
};
