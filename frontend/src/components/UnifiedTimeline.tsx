import React, { useState } from 'react';
import type { EventItem, TemporalGap, ConflictItem } from '../types';
import { 
  Activity, 
  FileText, 
  Camera, 
  Server, 
  Sparkles, 
  ChevronDown, 
  ChevronUp, 
  CheckCircle
} from 'lucide-react';

interface Props {
  events: EventItem[];
  temporalGaps: TemporalGap[];
  conflicts: ConflictItem[];
  onSelectGap?: (gap: TemporalGap) => void;
}

export const UnifiedTimeline: React.FC<Props> = ({
  events,
  temporalGaps: _temporalGaps,
  conflicts: _conflicts,
  onSelectGap: _onSelectGap
}) => {
  const [filter, setFilter] = useState<'all' | 'observed' | 'inferred' | 'conflicts'>('all');
  const [expandedEvents, setExpandedEvents] = useState<Record<string, boolean>>({});

  const toggleExpand = (id: string) => {
    setExpandedEvents(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'sensor':
        return <Activity size={14} color="#06b6d4" />;
      case 'machine_log':
        return <Server size={14} color="#60a5fa" />;
      case 'cctv_video':
        return <Camera size={14} color="#a855f7" />;
      case 'operator_report':
      case 'maintenance_report':
        return <FileText size={14} color="#f59e0b" />;
      case 'inferred_reconstruction':
        return <Sparkles size={14} color="#ec4899" />;
      default:
        return <CheckCircle size={14} color="#94a3b8" />;
    }
  };

  const filteredEvents = events.filter(e => {
    if (filter === 'observed') return e.observed_or_inferred === 'observed';
    if (filter === 'inferred') return e.observed_or_inferred === 'inferred';
    if (filter === 'conflicts') return e.certainty_type === 'conflicted';
    return true;
  });

  return (
    <div className="glass-panel">
      <div className="glass-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Unified Evidence Timeline
          </h3>
          <span className="badge badge-source" style={{ fontSize: '0.72rem' }}>
            {filteredEvents.length} Events
          </span>
        </div>

        {/* Filter buttons */}
        <div style={{ display: 'flex', gap: '0.4rem' }}>
          <button 
            className={`btn ${filter === 'all' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem' }}
            onClick={() => setFilter('all')}
          >
            All Events
          </button>
          <button 
            className={`btn ${filter === 'observed' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem' }}
            onClick={() => setFilter('observed')}
          >
            Observed ({events.filter(e => e.observed_or_inferred === 'observed').length})
          </button>
          <button 
            className={`btn ${filter === 'inferred' ? 'btn-accent' : 'btn-secondary'}`}
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem' }}
            onClick={() => setFilter('inferred')}
          >
            Inferred ({events.filter(e => e.observed_or_inferred === 'inferred').length})
          </button>
        </div>
      </div>

      {events.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
          <Activity size={36} style={{ margin: '0 auto 1rem', opacity: 0.4 }} />
          <p>No processed timeline events yet. Click "Process Evidence" above to analyze attached sources.</p>
        </div>
      ) : (
        <div className="timeline-container">
          <div className="timeline-line" />

          {filteredEvents.map((evt) => {
            const isExpanded = !!expandedEvents[evt.id];
            const isInferred = evt.observed_or_inferred === 'inferred';
            const isReported = evt.certainty_type === 'reported';

            let nodeClass = 'node-observed';
            let cardBorder = 'var(--border-color)';
            let cardBg = 'var(--bg-secondary)';

            if (isInferred) {
              nodeClass = 'node-inferred';
              cardBorder = 'var(--color-inferred-border)';
              cardBg = 'var(--color-inferred-bg)';
            } else if (isReported) {
              nodeClass = 'node-reported';
              cardBorder = 'var(--color-reported-border)';
            }

            return (
              <div key={evt.id} className="timeline-item">
                <div className={`timeline-node ${nodeClass}`} />

                <div 
                  style={{
                    background: cardBg,
                    border: `1px solid ${cardBorder}`,
                    borderRadius: '10px',
                    padding: '0.9rem 1.1rem',
                    transition: 'all 0.15s ease'
                  }}
                >
                  {/* Top row: Timestamp, Badges, Confidence */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.4rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.85rem', color: '#38bdf8' }}>
                        {new Date(evt.timestamp_start).toLocaleTimeString()}
                      </span>

                      {isInferred ? (
                        <span className="badge badge-inferred">
                          <Sparkles size={11} /> INFERRED EVENT
                        </span>
                      ) : (
                        <span className="badge badge-observed">
                          <CheckCircle size={11} /> OBSERVED
                        </span>
                      )}

                      <span className="badge badge-source" style={{ textTransform: 'capitalize' }}>
                        {getSourceIcon(evt.source)} {evt.source.replace('_', ' ')}
                      </span>

                      {isReported && (
                        <span className="badge badge-reported">REPORTED (HUMAN)</span>
                      )}
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                        Conf: <strong>{(evt.confidence * 100).toFixed(0)}%</strong>
                      </span>
                      <button 
                        onClick={() => toggleExpand(evt.id)}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: 'var(--text-secondary)',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.25rem',
                          fontSize: '0.75rem'
                        }}
                      >
                        {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />} Evidence ({evt.evidence_refs.length})
                      </button>
                    </div>
                  </div>

                  {/* Description */}
                  <div style={{ fontSize: '0.9rem', color: 'var(--text-primary)', fontWeight: 500, marginBottom: '0.4rem' }}>
                    {evt.description}
                  </div>

                  {/* Expandable Traceability Evidence Panel */}
                  {isExpanded && evt.evidence_refs && evt.evidence_refs.length > 0 && (
                    <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '6px' }}>
                      <h5 style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.4rem' }}>
                        Supporting Evidence Traceability:
                      </h5>
                      {evt.evidence_refs.map((ref, rIdx) => (
                        <div key={rIdx} style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem', fontFamily: 'var(--font-mono)' }}>
                          • <strong>{ref.file_name || ref.source_type}</strong>
                          {ref.row_index !== undefined && ` [Row #${ref.row_index}]`}
                          {ref.line_number !== undefined && ` [Line #${ref.line_number}]`}
                          {ref.metric_name && ` | Metric: ${ref.metric_name} = ${ref.metric_value}`}
                          {ref.quote && <div style={{ color: '#cbd5e1', fontStyle: 'italic', marginTop: '0.2rem' }}>"{ref.quote}"</div>}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
