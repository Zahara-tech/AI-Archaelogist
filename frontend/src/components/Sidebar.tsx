import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard,
  Cpu,
  BarChart3,
  GitBranch,
  BookOpen,
  Activity,
  CheckCircle2,
  Layers
} from 'lucide-react';

export const Sidebar: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/overview', label: 'Overview', icon: LayoutDashboard, badge: null },
    { path: '/reconstruction-studio', label: 'Reconstruction Studio', icon: Cpu, badge: 'Demo' },
    { path: '/model-performance', label: 'Model Performance', icon: BarChart3, badge: null },
    { path: '/training-experiments', label: 'Training Experiments', icon: GitBranch, badge: '50ep' },
    { path: '/about', label: 'About Project', icon: BookOpen, badge: null },
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
      top: 0,
      zIndex: 50
    }}>
      {/* Brand Header */}
      <div style={{
        padding: '1.4rem 1.5rem',
        borderBottom: '1px solid var(--border-color)',
        display: 'flex',
        alignItems: 'center',
        gap: '0.85rem'
      }}>
        <div style={{
          width: '38px',
          height: '38px',
          borderRadius: '8px',
          background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%)',
          border: '1px solid rgba(6, 182, 212, 0.4)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <Activity size={20} color="var(--accent-cyan)" />
        </div>
        <div>
          <h1 style={{
            fontSize: '1rem',
            fontWeight: 800,
            color: 'var(--text-primary)',
            letterSpacing: '0.04em',
            margin: 0,
            lineHeight: 1.2
          }}>
            AI ARCHAEOLOGIST
          </h1>
          <p style={{
            fontSize: '0.72rem',
            color: 'var(--text-muted)',
            fontWeight: 500,
            margin: 0,
            marginTop: '0.2rem'
          }}>
            Telemetry Reconstruction
          </p>
        </div>
      </div>

      {/* Navigation */}
      <nav style={{
        flex: 1,
        padding: '1.25rem 0.85rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.35rem'
      }}>
        <div style={{
          fontSize: '0.68rem',
          fontWeight: 700,
          color: 'var(--text-muted)',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          padding: '0 0.75rem 0.5rem 0.75rem'
        }}>
          Navigation
        </div>

        {navItems.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.path);
          return (
            <Link
              key={item.path}
              to={item.path}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0.75rem 0.9rem',
                borderRadius: '7px',
                textDecoration: 'none',
                color: active ? '#ffffff' : 'var(--text-secondary)',
                background: active ? 'rgba(6, 182, 212, 0.14)' : 'transparent',
                borderLeft: active ? '3px solid var(--accent-cyan)' : '3px solid transparent',
                transition: 'all 0.15s ease',
                fontSize: '0.88rem',
                fontWeight: active ? 600 : 500
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <Icon size={18} color={active ? 'var(--accent-cyan)' : 'var(--text-muted)'} />
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span style={{
                  fontSize: '0.65rem',
                  padding: '0.15rem 0.45rem',
                  borderRadius: '4px',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: 700,
                  background: active ? 'rgba(6, 182, 212, 0.3)' : 'rgba(255, 255, 255, 0.08)',
                  color: active ? '#38bdf8' : 'var(--text-muted)'
                }}>
                  {item.badge}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Industrial Status Card */}
      <div style={{
        padding: '1rem',
        borderTop: '1px solid var(--border-color)',
        background: 'rgba(9, 13, 22, 0.6)'
      }}>
        <div style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '8px',
          padding: '0.85rem'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.45rem',
            fontSize: '0.72rem',
            fontWeight: 700,
            color: 'var(--accent-emerald)',
            letterSpacing: '0.04em',
            marginBottom: '0.5rem'
          }}>
            <span style={{
              width: '7px',
              height: '7px',
              borderRadius: '50%',
              background: 'var(--accent-emerald)',
              boxShadow: '0 0 8px rgba(16, 185, 129, 0.8)'
            }} />
            SYSTEM OPERATIONAL
          </div>
          <div style={{
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
            marginBottom: '0.25rem'
          }}>
            Telemetry: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>PHM 2018 (17 Ch)</span>
          </div>
          <div style={{
            fontSize: '0.75rem',
            color: 'var(--text-secondary)'
          }}>
            Checkpoints: <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>Active (10, 20, 50)</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
