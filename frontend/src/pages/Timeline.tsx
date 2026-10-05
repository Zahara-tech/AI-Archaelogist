import React from 'react';
import { Clock } from 'lucide-react';

export const Timeline: React.FC = () => {
  return (
    <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
      <div style={{ marginBottom: '2rem' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          marginBottom: '0.5rem'
        }}>
          <Clock size={28} color="#a855f7" />
          <h1 style={{
            fontSize: '1.75rem',
            fontWeight: 800,
            color: 'var(--text-primary)'
          }}>
            Unified Timeline View
          </h1>
        </div>
        <p style={{
          fontSize: '1rem',
          color: 'var(--text-muted)',
          fontWeight: 400
        }}>
          View all incident events in a unified timeline.
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
        <Clock size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
        <p>Select an incident to view its timeline.</p>
      </div>
    </div>
  );
};
