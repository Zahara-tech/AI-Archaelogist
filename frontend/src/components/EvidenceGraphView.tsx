import React, { useState } from 'react';
import type { EvidenceGraphResponse, GraphNode } from '../types';
import { Network, Info } from 'lucide-react';

interface Props {
  graph: EvidenceGraphResponse | null;
  loading: boolean;
}

export const EvidenceGraphView: React.FC<Props> = ({ graph, loading }) => {
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [filterType, setFilterType] = useState<string>('all');

  if (loading) {
    return (
      <div className="glass-panel" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
        <Network size={32} className="spin" style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
        <p style={{ color: 'var(--text-muted)' }}>Generating NetworkX Multi-Relational Evidence Graph...</p>
      </div>
    );
  }

  if (!graph || !graph.nodes || graph.nodes.length === 0) {
    return (
      <div className="glass-panel" style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
        <Network size={32} style={{ margin: '0 auto 1rem', opacity: 0.4 }} />
        <p>No graph relationships generated. Process evidence first.</p>
      </div>
    );
  }

  const nodes = graph.nodes;

  const filteredNodes = nodes.filter(n => {
    if (filterType === 'events_observed') return n.type === 'event_observed';
    if (filterType === 'events_inferred') return n.type === 'event_inferred';
    if (filterType === 'evidence') return n.type.startsWith('evidence');
    return true;
  });

  const getNodeColor = (type: string) => {
    if (type === 'event_observed') return '#10b981';
    if (type === 'event_inferred') return '#a855f7';
    if (type.includes('sensor')) return '#06b6d4';
    if (type.includes('log')) return '#3b82f6';
    if (type.includes('video')) return '#ec4899';
    if (type.includes('report')) return '#f59e0b';
    return '#64748b';
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 340px', gap: '1.25rem' }}>
      {/* Graph Area */}
      <div className="glass-panel">
        <div className="glass-panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Network size={18} color="#38bdf8" />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Forensic Evidence Graph (NetworkX Topology)
            </h3>
          </div>

          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <button 
              className={`btn ${filterType === 'all' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
              onClick={() => setFilterType('all')}
            >
              All ({nodes.length})
            </button>
            <button 
              className={`btn ${filterType === 'events_observed' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
              onClick={() => setFilterType('events_observed')}
            >
              Observed
            </button>
            <button 
              className={`btn ${filterType === 'events_inferred' ? 'btn-accent' : 'btn-secondary'}`}
              style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem' }}
              onClick={() => setFilterType('events_inferred')}
            >
              Inferred
            </button>
          </div>
        </div>

        {/* Interactive Node Matrix Visualizer */}
        <div 
          style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '10px',
            padding: '1.25rem',
            minHeight: '420px',
            maxHeight: '520px',
            overflowY: 'auto'
          }}
        >
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.65rem' }}>
            {filteredNodes.map(node => {
              const color = getNodeColor(node.type);
              const isSelected = selectedNode?.id === node.id;

              return (
                <div
                  key={node.id}
                  onClick={() => setSelectedNode(node)}
                  style={{
                    background: isSelected ? 'rgba(56, 189, 248, 0.15)' : 'var(--bg-card)',
                    border: `1.5px solid ${isSelected ? '#38bdf8' : color}`,
                    borderRadius: '8px',
                    padding: '0.6rem 0.85rem',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    maxWidth: '280px',
                    boxShadow: isSelected ? '0 0 12px rgba(56, 189, 248, 0.3)' : 'none'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.25rem' }}>
                    <span 
                      style={{ 
                        width: '8px', 
                        height: '8px', 
                        borderRadius: '50%', 
                        background: color,
                        display: 'inline-block' 
                      }} 
                    />
                    <span style={{ fontSize: '0.72rem', fontWeight: 700, color: color, textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                      {node.type.replace('_', ' ')}
                    </span>
                  </div>

                  <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {node.label}
                  </div>

                  {node.timestamp && (
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: '0.2rem' }}>
                      {node.timestamp.includes('T') ? new Date(node.timestamp).toLocaleTimeString() : node.timestamp}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            <strong>Network Topology Stats:</strong> {graph.stats.total_nodes} Nodes, {graph.stats.total_edges} Directed Relational Edges, DAG Verified: {graph.stats.is_dag ? 'True' : 'False'}, Density: {graph.stats.density?.toFixed(4)}
          </div>
        </div>
      </div>

      {/* Node Inspector Panel */}
      <div className="glass-panel">
        <div className="glass-panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Info size={16} color="#60a5fa" />
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Node Inspector
            </h4>
          </div>
        </div>

        {selectedNode ? (
          <div>
            <div style={{ marginBottom: '0.75rem' }}>
              <span 
                className="badge"
                style={{ 
                  background: 'rgba(56, 189, 248, 0.1)', 
                  color: getNodeColor(selectedNode.type),
                  border: `1px solid ${getNodeColor(selectedNode.type)}`
                }}
              >
                {selectedNode.type.toUpperCase()}
              </span>
            </div>

            <h5 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.5rem' }}>
              {selectedNode.label}
            </h5>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              <div>ID: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{selectedNode.id}</span></div>
              {selectedNode.source && (
                <div>Source: <strong style={{ color: 'var(--text-primary)', textTransform: 'capitalize' }}>{selectedNode.source}</strong></div>
              )}
              {selectedNode.confidence !== undefined && (
                <div>Confidence: <strong style={{ color: '#38bdf8' }}>{(selectedNode.confidence * 100).toFixed(0)}%</strong></div>
              )}
              {selectedNode.certainty && (
                <div>Certainty: <strong style={{ color: '#a855f7' }}>{selectedNode.certainty.toUpperCase()}</strong></div>
              )}
            </div>

            {selectedNode.details && Object.keys(selectedNode.details).length > 0 && (
              <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-color)' }}>
                <h6 style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.4rem' }}>
                  Node Properties:
                </h6>
                <pre style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: '#cbd5e1', background: 'rgba(0,0,0,0.3)', padding: '0.6rem', borderRadius: '6px', overflowX: 'auto' }}>
                  {JSON.stringify(selectedNode.details, null, 2)}
                </pre>
              </div>
            )}
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Click any node in the topology map to inspect its forensic properties and causal edges.
          </div>
        )}
      </div>
    </div>
  );
};
