import React, { useState, useEffect } from 'react';
import { IncidentCard } from '../components/IncidentCard';
import { api } from '../services/api';
import { FileText } from 'lucide-react';

export const Incidents: React.FC = () => {
  const [incidents, setIncidents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadIncidents();
  }, []);

  const loadIncidents = async () => {
    try {
      const data = await api.listIncidents();
      setIncidents(data);
    } catch (err) {
      setError('Failed to load incidents from backend');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
        <div style={{ color: 'var(--text-muted)' }}>Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
        <div style={{ color: '#ef4444' }}>{error}</div>
      </div>
    );
  }

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
          <FileText size={28} color="#a855f7" />
          <h1 style={{
            fontSize: '1.75rem',
            fontWeight: 800,
            color: 'var(--text-primary)'
          }}>
            All Incidents
          </h1>
        </div>
        <p style={{
          fontSize: '1rem',
          color: 'var(--text-muted)',
          fontWeight: 400
        }}>
          Browse and investigate industrial incidents.
        </p>
      </div>

      {/* Incidents Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
        gap: '1rem'
      }}>
        {incidents.map((incident) => (
          <IncidentCard key={incident.id} incident={incident} />
        ))}
      </div>
    </div>
  );
};
