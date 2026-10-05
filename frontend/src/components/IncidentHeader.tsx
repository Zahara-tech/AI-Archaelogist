import React from 'react';
import type { IncidentDetail } from '../types';
import { 
  AlertTriangle, 
  Cpu, 
  Layers, 
  Clock, 
  Play, 
  Sparkles, 
  UploadCloud, 
  FolderPlus,
  CheckCircle2,
  RefreshCw
} from 'lucide-react';

interface Props {
  incident: IncidentDetail | null;
  loading: boolean;
  onProcess: () => void;
  onReconstruct: () => void;
  onLoadSynthetic: () => void;
  onOpenUpload: () => void;
}

export const IncidentHeader: React.FC<Props> = ({
  incident,
  loading,
  onProcess,
  onReconstruct,
  onLoadSynthetic,
  onOpenUpload,
}) => {
  if (!incident) {
    return (
      <div className="glass-panel" style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>AI Archaeologist</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Industrial Multimodal Incident Reconstruction & Forensics</p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button className="btn btn-secondary" onClick={onLoadSynthetic}>
            <FolderPlus size={16} /> Load CNC Demo Scenario
          </button>
          <button className="btn btn-primary" onClick={onOpenUpload}>
            <UploadCloud size={16} /> New Incident
          </button>
        </div>
      </div>
    );
  }

  const getSeverityBadge = (sev: string) => {
    switch (sev.toUpperCase()) {
      case 'CRITICAL':
        return <span className="badge badge-conflict"><AlertTriangle size={12} /> CRITICAL SEVERITY</span>;
      case 'HIGH':
        return <span className="badge badge-reported"><AlertTriangle size={12} /> HIGH SEVERITY</span>;
      default:
        return <span className="badge badge-observed">{sev}</span>;
    }
  };

  const getStatusBadge = (st: string) => {
    switch (st) {
      case 'analyzed':
        return <span className="badge badge-observed"><CheckCircle2 size={12} /> EVIDENCE ANALYZED</span>;
      case 'processing':
        return <span className="badge badge-reported"><RefreshCw size={12} className="spin" /> PROCESSING</span>;
      default:
        return <span className="badge badge-inferred">{st.toUpperCase()}</span>;
    }
  };

  return (
    <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.35rem' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: '#38bdf8', fontWeight: 700 }}>
              {incident.id}
            </span>
            {getSeverityBadge(incident.severity)}
            {getStatusBadge(incident.status)}
          </div>
          <h1 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.02em', marginBottom: '0.4rem' }}>
            {incident.title}
          </h1>
          <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
              <Cpu size={14} color="#60a5fa" /> Machine: <strong style={{ color: 'var(--text-primary)' }}>{incident.machine_id || 'N/A'}</strong>
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
              <Layers size={14} color="#a78bfa" /> Part / Batch: <strong style={{ color: 'var(--text-primary)' }}>{incident.component_id || 'N/A'}</strong>
            </span>
            {incident.start_time && (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                <Clock size={14} color="#34d399" /> Window: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                  {new Date(incident.start_time).toLocaleTimeString()} — {incident.end_time ? new Date(incident.end_time).toLocaleTimeString() : 'Active'}
                </span>
              </span>
            )}
          </div>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
          <button 
            className="btn btn-secondary" 
            onClick={onLoadSynthetic}
            title="Reload realistic CNC incident dataset package"
          >
            <FolderPlus size={15} /> Demo Scenario
          </button>
          
          <button 
            className="btn btn-primary" 
            onClick={onProcess}
            disabled={loading}
          >
            <Play size={15} /> {loading ? 'Processing...' : 'Process Evidence'}
          </button>

          <button 
            className="btn btn-accent" 
            onClick={onReconstruct}
            disabled={loading}
          >
            <Sparkles size={15} /> Reconstruct Gaps
          </button>

          <button 
            className="btn btn-secondary" 
            onClick={onOpenUpload}
          >
            <UploadCloud size={15} /> Attach File
          </button>
        </div>
      </div>
    </div>
  );
};
