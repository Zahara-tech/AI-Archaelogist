import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface Props {
  icon: LucideIcon;
  label: string;
  value: string | number;
  color?: string;
}

export const KPICard: React.FC<Props> = ({ icon: Icon, label, value, color = '#a855f7' }) => {
  return (
    <div style={{
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border-color)',
      borderRadius: '12px',
      padding: '1.5rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '0.75rem'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem'
      }}>
        <div style={{
          background: `${color}20`,
          padding: '0.6rem',
          borderRadius: '8px'
        }}>
          <Icon size={20} color={color} />
        </div>
        <span style={{
          fontSize: '0.85rem',
          color: 'var(--text-muted)',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }}>
          {label}
        </span>
      </div>
      <div style={{
        fontSize: '2rem',
        fontWeight: 800,
        color: 'var(--text-primary)',
        lineHeight: 1
      }}>
        {value}
      </div>
    </div>
  );
};
