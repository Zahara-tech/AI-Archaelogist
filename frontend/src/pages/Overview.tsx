import React, { useState, useEffect } from 'react';
import { KPICard } from '../components/KPICard';
import { IncidentCard } from '../components/IncidentCard';
import { api } from '../services/api';
import { Activity, FileText, Clock, AlertTriangle, LayoutDashboard } from 'lucide-react';

export const Overview: React.FC = () => {
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

  const kpiData = {
    activeInvestigations: incidents.filter(i => i.status !== 'closed').length,
    evidenceSources: incidents.reduce((sum, i) => sum + (i.evidence_count || 0), 0),
    reconstructedEvents: incidents.reduce((sum, i) => sum + (i.gap_count || 0), 0),
    unresolvedConflicts: incidents.reduce((sum, i) => sum + (i.conflict_count || 0), 0),
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
        <h1 style={{
          fontSize: '1.75rem',
          fontWeight: 800,
          color: 'var(--text-primary)',
          marginBottom: '0.5rem'
        }}>
          Industrial Investigation Center
        </h1>
        <p style={{
          fontSize: '1rem',
          color: 'var(--text-muted)',
          fontWeight: 400
        }}>
          Reconstruct incident timelines from incomplete and conflicting evidence.
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '1rem',
        marginBottom: '2.5rem'
      }}>
        <KPICard 
          icon={LayoutDashboard}
          label="Active Investigations"
          value={kpiData.activeInvestigations}
          color="#a855f7"
        />
        <KPICard 
          icon={FileText}
          label="Evidence Sources"
          value={kpiData.evidenceSources}
          color="#3b82f6"
        />
        <KPICard 
          icon={Clock}
          label="Reconstructed Events"
          value={kpiData.reconstructedEvents}
          color="#22c55e"
        />
        <KPICard 
          icon={AlertTriangle}
          label="Unresolved Conflicts"
          value={kpiData.unresolvedConflicts}
          color="#ef4444"
        />
      </div>

      {/* Recent Investigations */}
      <div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          marginBottom: '1.5rem'
        }}>
          <Activity size={24} color="#a855f7" />
          <h2 style={{
            fontSize: '1.25rem',
            fontWeight: 700,
            color: 'var(--text-primary)'
          }}>
            Recent Investigations
          </h2>
        </div>

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
    </div>
  );
};
