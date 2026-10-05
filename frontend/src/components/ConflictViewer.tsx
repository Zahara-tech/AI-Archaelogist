import React from 'react';
import type { ConflictItem } from '../types';
import { AlertOctagon, GitCompare, ShieldAlert } from 'lucide-react';

interface Props {
  conflicts: ConflictItem[];
}

export const ConflictViewer: React.FC<Props> = ({ conflicts }) => {
  return (
    <div className="glass-panel" style={{ border: conflicts.length > 0 ? '1px solid var(--color-conflict-border)' : '1px solid var(--border-color)' }}>
      <div className="glass-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <ShieldAlert size={18} color="#f43f5e" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Multi-Source Evidence Contradictions ({conflicts.length})
          </h3>
        </div>
        <span className="badge badge-conflict">
          CONFLICT DETECTION ENGINE
        </span>
      </div>

      <div style={{ marginBottom: '1.25rem', padding: '0.6rem 0.8rem', background: 'rgba(244, 63, 94, 0.1)', borderLeft: '3px solid #f43f5e', borderRadius: '4px', fontSize: '0.82rem', color: '#fecdd3' }}>
        <strong>Forensic Principle:</strong> Heterogeneous industrial sources (PLC clocks, operator recall, visual timestamps) frequently exhibit asynchronous drift or logging delays. AI Archaeologist preserves both opposing pieces of evidence rather than silently picking one.
      </div>

      {conflicts.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '2.5rem 1rem', color: 'var(--text-muted)' }}>
          <GitCompare size={32} style={{ margin: '0 auto 0.75rem', opacity: 0.4 }} />
          <p>No multi-source discrepancies detected across modalities.</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {conflicts.map((conf, idx) => (
            <div 
              key={conf.id}
              style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--color-conflict-border)',
                borderRadius: '10px',
                padding: '1.25rem'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <span className="badge badge-conflict">
                    <AlertOctagon size={12} /> CONFLICT #{idx + 1}: {conf.conflict_type.toUpperCase()}
                  </span>
                  <span style={{ fontSize: '0.82rem', color: '#f43f5e', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                    Δ {conf.time_discrepancy_seconds.toFixed(1)}s Discrepancy
                  </span>
                </div>
              </div>

              <p style={{ fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: '1rem', lineHeight: 1.5 }}>
                {conf.description}
              </p>

              {/* Side-by-side Evidence Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
                {/* Source A */}
                <div 
                  style={{
                    background: 'rgba(0,0,0,0.3)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px',
                    padding: '0.9rem'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                    <span className="badge badge-source" style={{ textTransform: 'capitalize' }}>
                      SOURCE A: {conf.source_a.replace('_', ' ')}
                    </span>
                    {conf.evidence_a?.timestamp && (
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: '#38bdf8' }}>
                        {new Date(conf.evidence_a.timestamp).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: '0.85rem', color: '#e2e8f0' }}>
                    {conf.evidence_a?.description || JSON.stringify(conf.evidence_a)}
                  </div>
                </div>

                {/* Source B */}
                <div 
                  style={{
                    background: 'rgba(0,0,0,0.3)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px',
                    padding: '0.9rem'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                    <span className="badge badge-source" style={{ textTransform: 'capitalize' }}>
                      SOURCE B: {conf.source_b.replace('_', ' ')}
                    </span>
                    {conf.evidence_b?.timestamp && (
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: '#38bdf8' }}>
                        {new Date(conf.evidence_b.timestamp).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: '0.85rem', color: '#e2e8f0' }}>
                    {conf.evidence_b?.description || JSON.stringify(conf.evidence_b)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
