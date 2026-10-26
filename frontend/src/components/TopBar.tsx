import React, { useState, useEffect } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { 
  Database, 
  CheckCircle, 
  Cpu, 
  Layers,
  ExternalLink
} from 'lucide-react';

export const TopBar: React.FC = () => {
  const location = useLocation();
  const [backendOnline, setBackendOnline] = useState<boolean>(true);

  useEffect(() => {
    // Quick health probe
    fetch('/api/v1/sensor/health')
      .then(res => setBackendOnline(res.ok))
      .catch(() => setBackendOnline(false));
  }, [location.pathname]);

  const getPageMeta = () => {
    switch (location.pathname) {
      case '/overview':
        return {
          title: 'Platform Overview',
          subtitle: 'AI-assisted reconstruction of missing industrial sensor telemetry'
        };
      case '/reconstruction-studio':
        return {
          title: 'Reconstruction Studio',
          subtitle: 'Primary interactive telemetry missing-gap recovery demo'
        };
      case '/model-performance':
        return {
          title: 'Model Performance & Comparative Evaluation',
          subtitle: 'Verified benchmarks and trade-off analysis across all four approaches'
        };
      case '/training-experiments':
        return {
          title: 'Training Experiments & Loss Curves',
          subtitle: 'Historical training trajectories, epoch checkpoints, and model parameters'
        };
      case '/about':
        return {
          title: 'About the Project & Methodology',
          subtitle: 'Problem formulation, 17 sensor channels, mathematical priors, and limitations'
        };
      default:
        return {
          title: 'AI Archaeologist',
          subtitle: 'Industrial Sensor Telemetry Reconstruction Engine'
        };
    }
  };

  const meta = getPageMeta();

  return (
    <header style={{
      height: '68px',
      background: 'var(--bg-header)',
      borderBottom: '1px solid var(--border-color)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 2.5rem',
      position: 'fixed',
      top: 0,
      left: '260px',
      right: 0,
      zIndex: 40
    }}>
      {/* Dynamic Page Header */}
      <div>
        <h2 style={{
          fontSize: '1.15rem',
          fontWeight: 800,
          color: 'var(--text-primary)',
          letterSpacing: '-0.3px',
          margin: 0,
          lineHeight: 1.2
        }}>
          {meta.title}
        </h2>
        <p style={{
          fontSize: '0.78rem',
          color: 'var(--text-muted)',
          fontWeight: 500,
          margin: 0,
          marginTop: '0.15rem'
        }}>
          {meta.subtitle}
        </p>
      </div>

      {/* Industrial Status Badges */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.85rem'
      }}>
        {/* Backend Connectivity Status */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.45rem',
          padding: '0.35rem 0.75rem',
          borderRadius: '6px',
          background: backendOnline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)',
          border: `1px solid ${backendOnline ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
          fontSize: '0.75rem',
          fontFamily: 'var(--font-mono)',
          fontWeight: 600,
          color: backendOnline ? 'var(--accent-emerald)' : 'var(--accent-rose)'
        }}>
          <span style={{
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            background: backendOnline ? 'var(--accent-emerald)' : 'var(--accent-rose)'
          }} />
          {backendOnline ? 'API CONNECTED' : 'API DISCONNECTED'}
        </div>

        {/* Dataset Badge */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.35rem 0.85rem',
          borderRadius: '6px',
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          fontSize: '0.78rem',
          color: 'var(--text-secondary)'
        }}>
          <Database size={14} color="var(--accent-cyan)" />
          <span>Dataset:</span>
          <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>PHM 2018 (17 Ch)</strong>
        </div>

        {/* Evaluation Status Badge */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.35rem 0.85rem',
          borderRadius: '6px',
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          fontSize: '0.78rem',
          color: 'var(--text-secondary)'
        }}>
          <CheckCircle size={14} color="var(--accent-cyan)" />
          <span>Evaluation:</span>
          <span style={{ color: 'var(--accent-cyan)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>Verified v2</span>
        </div>

        {/* Quick Studio CTA if not on studio */}
        {location.pathname !== '/reconstruction-studio' && (
          <Link
            to="/reconstruction-studio"
            className="btn btn-primary"
            style={{
              padding: '0.4rem 0.9rem',
              fontSize: '0.78rem',
              borderRadius: '6px',
              textDecoration: 'none'
            }}
          >
            <Cpu size={14} />
            Live Demo
          </Link>
        )}
      </div>
    </header>
  );
};
