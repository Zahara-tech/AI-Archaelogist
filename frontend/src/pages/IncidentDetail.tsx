import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../services/api';
import { IncidentHeader } from '../components/IncidentHeader';
import { UnifiedTimeline } from '../components/UnifiedTimeline';
import { SensorChart } from '../components/SensorChart';
import { MissingEventsPanel } from '../components/MissingEventsPanel';
import { ConflictViewer } from '../components/ConflictViewer';
import { EvidenceGraphView } from '../components/EvidenceGraphView';
import { 
  FileText, 
  Server, 
  Activity, 
  Camera,
  CheckCircle,
  XCircle
} from 'lucide-react';

export const IncidentDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [incident, setIncident] = useState<any>(null);
  const [timeline, setTimeline] = useState<any>(null);
  const [sensorData, setSensorData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'timeline' | 'evidence' | 'conflicts' | 'graph'>('timeline');
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    if (id) {
      loadIncidentData(id);
    }
  }, [id]);

  const loadIncidentData = async (incidentId: string) => {
    try {
      setLoading(true);
      const [incidentData, timelineData, sensorDataResponse] = await Promise.all([
        api.getIncident(incidentId),
        api.getTimeline(incidentId),
        api.getSensorData(incidentId).catch(() => null)
      ]);
      setIncident(incidentData);
      setTimeline(timelineData);
      setSensorData(sensorDataResponse);
    } catch (err) {
      setError('Failed to load incident data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleProcess = async () => {
    if (!id) return;
    setProcessing(true);
    try {
      await api.processIncident(id);
      await loadIncidentData(id);
    } catch (err) {
      console.error('Processing failed:', err);
    } finally {
      setProcessing(false);
    }
  };

  const handleReconstruct = async () => {
    if (!id) return;
    setProcessing(true);
    try {
      await api.reconstructGaps(id);
      await loadIncidentData(id);
    } catch (err) {
      console.error('Reconstruction failed:', err);
    } finally {
      setProcessing(false);
    }
  };

  const handleLoadSynthetic = async () => {
    if (!id) return;
    setProcessing(true);
    try {
      await api.loadSyntheticPackage(id);
      await loadIncidentData(id);
    } catch (err) {
      console.error('Failed to load synthetic data:', err);
    } finally {
      setProcessing(false);
    }
  };

  const handleOpenUpload = () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.csv,.txt,.pdf,.mp4,.avi,.json';
    fileInput.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file || !id) return;

      setProcessing(true);
      try {
        const fileType = file.name.endsWith('.csv') ? 'sensor_csv' :
                        file.name.endsWith('.mp4') || file.name.endsWith('.avi') ? 'cctv_video' :
                        file.name.endsWith('.json') ? 'machine_log' : 'operator_report';
        
        await api.uploadEvidence(id, file, fileType);
        await loadIncidentData(id);
      } catch (err) {
        console.error('Upload failed:', err);
      } finally {
        setProcessing(false);
      }
    };
    fileInput.click();
  };

  const handleTriggerReconstruction = async (gapId?: string) => {
    if (!id) return;
    setProcessing(true);
    try {
      await api.reconstructGaps(id, gapId);
      await loadIncidentData(id);
    } catch (err) {
      console.error('Reconstruction failed:', err);
    } finally {
      setProcessing(false);
    }
  };

  const getEvidenceIcon = (type: string) => {
    switch (type) {
      case 'sensor': return <Activity size={20} color="#06b6d4" />;
      case 'machine_log': return <Server size={20} color="#60a5fa" />;
      case 'cctv_video': return <Camera size={20} color="#a855f7" />;
      default: return <FileText size={20} color="#f59e0b" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'available': return '#22c55e';
      case 'not available': return '#ef4444';
      default: return '#64748b';
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
        <div style={{ color: 'var(--text-muted)' }}>Loading incident...</div>
      </div>
    );
  }

  if (error || !incident) {
    return (
      <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
        <div style={{ color: '#ef4444' }}>{error || 'Incident not found'}</div>
      </div>
    );
  }

  return (
    <div style={{ padding: '5rem 2rem 2rem 2rem' }}>
      {/* Header */}
      <IncidentHeader 
        incident={incident}
        loading={processing}
        onProcess={handleProcess}
        onReconstruct={handleReconstruct}
        onLoadSynthetic={handleLoadSynthetic}
        onOpenUpload={handleOpenUpload}
      />

      {/* Evidence Overview */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '1rem',
        marginBottom: '2rem'
      }}>
        {incident.evidence_items?.map((item: any) => (
          <div
            key={item.id}
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: '10px',
              padding: '1.25rem'
            }}
          >
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              marginBottom: '0.75rem'
            }}>
              {getEvidenceIcon(item.file_type)}
              <div>
                <div style={{
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  color: 'var(--text-muted)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  {item.file_type.replace('_', ' ')}
                </div>
                <div style={{
                  fontSize: '0.8rem',
                  color: 'var(--text-secondary)',
                  marginTop: '0.1rem'
                }}>
                  {item.summary}
                </div>
              </div>
            </div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              fontSize: '0.8rem',
              color: getStatusColor(item.processed ? 'available' : 'not available'),
              fontWeight: 600
            }}>
              {item.processed ? (
                <CheckCircle size={14} />
              ) : (
                <XCircle size={14} />
              )}
              {item.processed ? 'Available' : 'Not Available'}
            </div>
          </div>
        ))}
        
        {/* Video - Future Extension */}
        <div
          style={{
            background: 'var(--bg-secondary)',
            border: '1px dashed var(--border-color)',
            borderRadius: '10px',
            padding: '1.25rem',
            opacity: 0.5
          }}
        >
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '0.75rem'
          }}>
            <Camera size={20} color="#64748b" />
            <div>
              <div style={{
                fontSize: '0.75rem',
                fontWeight: 700,
                color: 'var(--text-muted)',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                VIDEO
              </div>
              <div style={{
                fontSize: '0.8rem',
                color: 'var(--text-muted)',
                marginTop: '0.1rem'
              }}>
                Not available in current dataset
              </div>
            </div>
          </div>
          <div style={{
            fontSize: '0.75rem',
            color: 'var(--text-muted)',
            fontStyle: 'italic'
          }}>
            Future extension
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{
        display: 'flex',
        gap: '0.5rem',
        marginBottom: '1.5rem',
        borderBottom: '1px solid var(--border-color)',
        paddingBottom: '0'
      }}>
        {[
          { id: 'timeline', label: 'Timeline' },
          { id: 'evidence', label: 'Evidence Graph' },
          { id: 'conflicts', label: 'Conflicts' }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            style={{
              padding: '0.75rem 1.25rem',
              background: activeTab === tab.id ? 'var(--accent-bg)' : 'transparent',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid #a855f7' : '2px solid transparent',
              color: activeTab === tab.id ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontSize: '0.9rem',
              fontWeight: activeTab === tab.id ? 600 : 500,
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        {/* Left Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {activeTab === 'timeline' && (
            <UnifiedTimeline
              events={timeline?.events || []}
              temporalGaps={timeline?.temporal_gaps || []}
              conflicts={timeline?.conflicts || []}
            />
          )}
          
          {activeTab === 'evidence' && (
            <EvidenceGraphView graph={null} loading={false} />
          )}

          {activeTab === 'conflicts' && (
            <ConflictViewer conflicts={timeline?.conflicts || []} />
          )}
        </div>

        {/* Right Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Missing Events Panel */}
          {(timeline?.temporal_gaps?.length || 0) > 0 && (
            <MissingEventsPanel
              temporalGaps={timeline?.temporal_gaps || []}
              reconstructions={incident?.reconstructions || []}
              onTriggerReconstruction={handleTriggerReconstruction}
              loading={processing}
            />
          )}

          {/* Sensor Chart */}
          <SensorChart data={sensorData} loading={false} />

          {/* Investigation Summary */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '1.5rem'
          }}>
            <h3 style={{
              fontSize: '1.05rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              marginBottom: '1rem'
            }}>
              Investigation Summary
            </h3>
            
            <div style={{ marginBottom: '1rem' }}>
              <h4 style={{
                fontSize: '0.85rem',
                fontWeight: 700,
                color: '#22c55e',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                What We Know
              </h4>
              <ul style={{
                margin: 0,
                paddingLeft: '1.25rem',
                fontSize: '0.85rem',
                color: 'var(--text-secondary)'
              }}>
                <li style={{ marginBottom: '0.35rem' }}>Process started at 10:02:11</li>
                <li style={{ marginBottom: '0.35rem' }}>Sensor deviation detected at 10:03:27</li>
                <li style={{ marginBottom: '0.35rem' }}>Temperature anomaly at 10:03:51</li>
                <li style={{ marginBottom: '0.35rem' }}>Machine state changed at 10:05:02</li>
              </ul>
            </div>

            <div style={{ marginBottom: '1rem' }}>
              <h4 style={{
                fontSize: '0.85rem',
                fontWeight: 700,
                color: '#a855f7',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                What We Infer
              </h4>
              <ul style={{
                margin: 0,
                paddingLeft: '1.25rem',
                fontSize: '0.85rem',
                color: 'var(--text-secondary)'
              }}>
                <li style={{ marginBottom: '0.35rem' }}>
                  Abnormal process transition (10:04:20 - 10:05:02) - 78% confidence
                </li>
              </ul>
            </div>

            <div>
              <h4 style={{
                fontSize: '0.85rem',
                fontWeight: 700,
                color: '#f59e0b',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                What Remains Uncertain
              </h4>
              <ul style={{
                margin: 0,
                paddingLeft: '1.25rem',
                fontSize: '0.85rem',
                color: 'var(--text-secondary)'
              }}>
                <li style={{ marginBottom: '0.35rem' }}>Exact cause of vibration pattern change</li>
                <li style={{ marginBottom: '0.35rem' }}>5-second timestamp discrepancy between sources</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
