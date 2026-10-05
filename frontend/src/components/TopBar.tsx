import React from 'react';
import { Search, Bell, User } from 'lucide-react';

export const TopBar: React.FC = () => {
  return (
    <header style={{
      height: '64px',
      background: 'var(--bg-header)',
      borderBottom: '1px solid var(--border-color)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 2rem',
      position: 'fixed',
      top: 0,
      left: '260px',
      right: 0,
      zIndex: 100
    }}>
      {/* Left: Title */}
      <div>
        <h2 style={{
          fontSize: '1.25rem',
          fontWeight: 700,
          color: 'var(--text-primary)',
          marginBottom: '0.1rem'
        }}>
          AI Archaeologist
        </h2>
        <p style={{
          fontSize: '0.8rem',
          color: 'var(--text-muted)',
          fontWeight: 500
        }}>
          Industrial Incident Investigation
        </p>
      </div>

      {/* Center: Search */}
      <div style={{
        flex: 1,
        maxWidth: '400px',
        margin: '0 2rem'
      }}>
        <div style={{
          position: 'relative',
          display: 'flex',
          alignItems: 'center'
        }}>
          <Search 
            size={16} 
            color="#64748b" 
            style={{
              position: 'absolute',
              left: '12px',
              top: '50%',
              transform: 'translateY(-50%)'
            }}
          />
          <input
            type="text"
            placeholder="Search incidents..."
            style={{
              width: '100%',
              padding: '0.6rem 1rem 0.6rem 2.5rem',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              color: 'var(--text-primary)',
              fontSize: '0.9rem',
              outline: 'none',
              transition: 'border-color 0.15s ease'
            }}
          />
        </div>
      </div>

      {/* Right: Actions */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '1rem'
      }}>
        <button
          style={{
            background: 'transparent',
            border: 'none',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            padding: '0.5rem',
            borderRadius: '8px',
            transition: 'background 0.15s ease',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
          title="Notifications"
        >
          <Bell size={20} />
        </button>
        
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          padding: '0.5rem 1rem',
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          cursor: 'pointer'
        }}>
          <User size={18} color="var(--text-secondary)" />
          <span style={{
            fontSize: '0.85rem',
            color: 'var(--text-primary)',
            fontWeight: 500
          }}>
            Engineer
          </span>
        </div>
      </div>
    </header>
  );
};
