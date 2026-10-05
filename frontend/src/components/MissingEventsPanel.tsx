import React from 'react';
import type { TemporalGap, ReconstructedEvent } from '../types';
import { Sparkles, Clock } from 'lucide-react';

interface Props {
  temporalGaps: TemporalGap[];
  reconstructions: ReconstructedEvent[];
  onTriggerReconstruction: (gapId?: string) => void;
  loading: boolean;
}

export const MissingEventsPanel: React.FC<Props> = ({
  temporalGaps,
  reconstructions,
  onTriggerReconstruction,
  loading
}) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Gaps List */}
      <div className="glass-panel">
        <div className="glass-panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Clock size={18} color="#f59e0b" />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Detected Temporal Gaps ({temporalGaps.length})
            </h3>
          </div>
          <button 
            className="btn btn-accent" 
            onClick={() => onTriggerReconstruction()}
            disabled={loading || temporalGaps.length === 0}
            style={{ fontSize: '0.78rem', padding: '0.35rem 0.75rem' }}
          >
            <Sparkles size={14} /> Reconstruct All Gaps
          </button>
        </div>

        {temporalGaps.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No unexplained temporal gaps detected.</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {temporalGaps.map(gap => (
              <div 
                key={gap.gap_id}
                style={{
                  background: 'var(--bg-secondary)',
                  border: '1px dashed #f59e0b',
                  borderRadius: '8px',
                  padding: '1rem',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: '1rem'
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem' }}>
                    <span className="badge badge-reported">
                      GAP DURATION: {gap.duration_seconds.toFixed(0)} SECONDS
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: '#38bdf8' }}>
                      {new Date(gap.gap_start).toLocaleTimeString()} → {new Date(gap.gap_end).toLocaleTimeString()}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-primary)', maxWidth: '900px' }}>
                    {gap.description}
                  </div>
                </div>

                <button 
                  className="btn btn-secondary" 
                  onClick={() => onTriggerReconstruction(gap.gap_id)}
                  disabled={loading}
                  style={{ fontSize: '0.78rem' }}
                >
                  <Sparkles size={14} color="#a855f7" /> Infer Candidate Activity
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Candidate Reconstructions */}
      <div className="glass-panel" style={{ border: '1px solid var(--color-inferred-border)' }}>
        <div className="glass-panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Sparkles size={18} color="#a855f7" />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Inferred Candidate Event Reconstructions ({reconstructions.length})
            </h3>
          </div>
          <span className="badge badge-inferred">
            PROBABILISTIC INFERENCE
          </span>
        </div>

        <div style={{ marginBottom: '1rem', padding: '0.6rem 0.8rem', background: 'rgba(168, 85, 247, 0.1)', borderLeft: '3px solid #a855f7', borderRadius: '4px', fontSize: '0.82rem', color: '#e2e8f0' }}>
          <strong>Forensic Integrity Notice:</strong> Inferred events represent high-probability candidate hypotheses synthesized from surrounding telemetry, visual detections, and operator notes. They are clearly tagged as <span className="badge badge-inferred" style={{ fontSize: '0.65rem' }}>INFERRED</span> and must not be treated as confirmed physical sensor records.
        </div>

        {reconstructions.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2.5rem 1rem', color: 'var(--text-muted)' }}>
            <Sparkles size={32} style={{ margin: '0 auto 0.75rem', opacity: 0.4 }} />
            <p>No candidate reconstructions generated yet. Click "Reconstruct Gaps" above.</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {reconstructions.map(rec => (
              <div 
                key={rec.id}
                style={{
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--color-inferred-border)',
                  borderRadius: '10px',
                  padding: '1.1rem'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <span className="badge badge-inferred" style={{ fontSize: '0.78rem' }}>
                      <Sparkles size={12} /> {rec.candidate_event_type}
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: '#38bdf8' }}>
                      Window: {new Date(rec.gap_start).toLocaleTimeString()} — {new Date(rec.gap_end).toLocaleTimeString()}
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Inference Confidence:</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#a855f7', fontSize: '0.9rem' }}>
                      {(rec.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div style={{ fontSize: '0.92rem', color: '#ffffff', fontWeight: 600, marginBottom: '0.5rem' }}>
                  {rec.description}
                </div>

                <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', background: 'rgba(0,0,0,0.25)', padding: '0.75rem', borderRadius: '6px', marginBottom: '0.75rem' }}>
                  <strong style={{ color: '#cbd5e1' }}>Causal Reasoning:</strong> {rec.reasoning}
                </div>

                {/* Supporting evidence pills */}
                {rec.supporting_evidence && rec.supporting_evidence.length > 0 && (
                  <div>
                    <h5 style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.35rem' }}>
                      Corroborating Evidence:
                    </h5>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                      {rec.supporting_evidence.map((sup, sIdx) => (
                        <span 
                          key={sIdx}
                          style={{
                            fontSize: '0.75rem',
                            background: 'rgba(59, 130, 246, 0.15)',
                            color: '#93c5fd',
                            padding: '0.25rem 0.55rem',
                            borderRadius: '4px',
                            border: '1px solid rgba(59, 130, 246, 0.3)',
                            fontFamily: 'var(--font-mono)'
                          }}
                        >
                          {sup.source || 'evidence'}: {sup.event || sup.quote || sup.detail || 'corroborated'}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
