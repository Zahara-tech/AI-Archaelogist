import React from 'react';
import { Link } from 'react-router-dom';
import type { IncidentSummary } from '../types';
import { AlertTriangle, CheckCircle, Clock, FileText } from 'lucide-react';

interface Props {
  incident: IncidentSummary;
}

export const IncidentCard: React.FC<Props> = ({ incident }) => {
  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'resolved':
        return '#22c55e';
      case 'under investigation':
        return '#f59e0b';
      default:
        return '#64748b';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'resolved':
        return <CheckCircle size={14} />;
      case 'under investigation':
        return <Clock size={14} />;
      default:
        return <AlertTriangle size={14} />;
    }
  };

  return (
    <Link
      to={`/incidents/${incident.id}`}
      style={{
        textDecoration: 'none',
        display: 'block'
      }}
    >
      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        borderRadius: '12px',
        padding: '1.25rem',
        transition: 'all 0.2s ease',
        cursor: 'pointer'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = 'var(--accent-color)';
        e.currentTarget.style.transform = 'translateY(-2px)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'var(--border-color)';
        e.currentTarget.style.transform = 'translateY(0)';
      }}
      >
        {/* Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: '0.75rem'
        }}>
          <div>
            <div style={{
              fontSize: '0.75rem',
              fontWeight: 700,
              color: '#38bdf8',
              fontFamily: 'var(--font-mono)',
              marginBottom: '0.25rem'
            }}>
              {incident.id}
            </div>
            <h3 style={{
              fontSize: '1rem',
              fontWeight: 600,
              color: 'var(--text-primary)',
              marginBottom: '0.25rem'
            }}>
              {incident.title}
            </h3>
            <div style={{
              fontSize: '0.8rem',
              color: 'var(--text-muted)'
            }}>
              Machine: {incident.machine_id}
            </div>
          </div>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.4rem 0.75rem',
            background: `${getStatusColor(incident.status)}20`,
            borderRadius: '6px',
            fontSize: '0.75rem',
            fontWeight: 600,
            color: getStatusColor(incident.status)
          }}>
            {getStatusIcon(incident.status)}
            {incident.status}
          </div>
        </div>

        {/* Stats */}
        <div style={{
          display: 'flex',
          gap: '1.5rem',
          paddingTop: '0.75rem',
          borderTop: '1px solid var(--border-color)'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)'
          }}>
            <FileText size={14} />
            <span>{incident.evidence_count} sources</span>
          </div>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)'
          }}>
            <Clock size={14} />
            <span>{incident.event_count} events</span>
          </div>
          {incident.conflict_count > 0 && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              fontSize: '0.8rem',
              color: '#ef4444'
            }}>
              <AlertTriangle size={14} />
              <span>{incident.conflict_count} conflicts</span>
            </div>
          )}
        </div>
      </div>
    </Link>
  );
};
