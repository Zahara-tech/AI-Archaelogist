import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Database, 
  FlaskConical, 
  Cpu, 
  BarChart3,
  Activity
} from 'lucide-react';

export const Sidebar: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/dataset-explorer', label: 'Dataset Explorer', icon: Database },
    { path: '/reconstruction-lab', label: 'Reconstruction Lab', icon: FlaskConical },
    { path: '/model-performance', label: 'Model Performance', icon: Cpu },
    { path: '/experiments', label: 'Experiments & Results', icon: BarChart3 },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <aside style={{
      width: '260px',
      background: 'var(--bg-sidebar)',
      borderRight: '1px solid var(--border-color)',
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      position: 'fixed',
      left: 0,
      top: 0
    }}>
      {/* Logo */}
      <div style={{
        padding: '1.5rem',
        borderBottom: '1px solid var(--border-color)'
      }}>
        <h1 style={{
          fontSize: '1.1rem',
          fontWeight: 800,
          color: 'var(--text-primary)',
          letterSpacing: '0.5px',
          marginBottom: '0.25rem'
        }}>
          AI ARCHAEOLOGIST
        </h1>
        <p style={{
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
          fontWeight: 500
        }}>
          Industrial Sensor Reconstruction Research
        </p>
      </div>

      {/* Navigation */}
      <nav style={{
        flex: 1,
        padding: '1rem 0.75rem'
      }}>
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <Link
              key={item.path}
              to={item.path}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                padding: '0.75rem 1rem',
                borderRadius: '8px',
                textDecoration: 'none',
                color: isActive(item.path) ? 'var(--text-primary)' : 'var(--text-secondary)',
                background: isActive(item.path) ? 'var(--accent-bg)' : 'transparent',
                marginBottom: '0.25rem',
                transition: 'all 0.15s ease',
                fontSize: '0.9rem',
                fontWeight: isActive(item.path) ? 600 : 500
              }}
            >
              <Icon size={18} color={isActive(item.path) ? '#a855f7' : '#64748b'} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* System Status */}
      <div style={{
        padding: '1rem 0.75rem',
        borderTop: '1px solid var(--border-color)',
        background: 'rgba(0,0,0,0.2)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          marginBottom: '0.5rem',
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
          fontWeight: 600
        }}>
          <Activity size={14} color="#22c55e" />
          SYSTEM STATUS
        </div>
        <div style={{
          fontSize: '0.8rem',
          color: 'var(--text-secondary)',
          marginBottom: '0.25rem'
        }}>
          Dataset: <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>PHM 2018</span>
        </div>
        <div style={{
          fontSize: '0.8rem',
          color: 'var(--text-secondary)'
        }}>
          Processing: <span style={{ color: '#22c55e', fontWeight: 500 }}>Ready</span>
        </div>
      </div>
    </aside>
  );
};
