import type {
  IncidentSummary,
  IncidentDetail,
  UnifiedTimelineResponse,
  EventItem,
  ConflictItem,
  ReconstructedEvent,
  EvidenceItem,
  EvidenceGraphResponse,
  SensorDataResponse,
  InvestigationSummaryResponse
} from '../types';

const API_BASE = 'http://localhost:8000/api/v1';

export const api = {
  // Incidents
  listIncidents: async (): Promise<IncidentSummary[]> => {
    const res = await fetch(`${API_BASE}/incidents`);
    if (!res.ok) throw new Error('Failed to fetch incidents');
    return res.json();
  },

  getIncident: async (id: string): Promise<IncidentDetail> => {
    const res = await fetch(`${API_BASE}/incidents/${id}`);
    if (!res.ok) throw new Error(`Failed to fetch incident ${id}`);
    return res.json();
  },

  createIncident: async (data: {
    id?: string;
    title: string;
    machine_id?: string;
    component_id?: string;
    severity?: string;
    description?: string;
  }): Promise<IncidentDetail> => {
    const res = await fetch(`${API_BASE}/incidents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to create incident');
    return res.json();
  },

  loadSyntheticPackage: async (incidentId: string): Promise<{ loaded_files: string[]; status: string }> => {
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/load-synthetic`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to load synthetic package');
    return res.json();
  },

  processIncident: async (incidentId: string): Promise<{
    incident_id: string;
    status: string;
    total_events: number;
    gaps_detected: number;
    conflicts_detected: number;
    start_time?: string;
    end_time?: string;
  }> => {
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/process`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to process incident evidence');
    return res.json();
  },

  // Timeline & Events
  getTimeline: async (incidentId: string): Promise<UnifiedTimelineResponse> => {
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/timeline`);
    if (!res.ok) throw new Error('Failed to fetch unified timeline');
    return res.json();
  },

  getEvents: async (incidentId: string, filter?: { source?: string; observed_or_inferred?: string }): Promise<EventItem[]> => {
    const params = new URLSearchParams();
    if (filter?.source) params.append('source', filter.source);
    if (filter?.observed_or_inferred) params.append('observed_or_inferred', filter.observed_or_inferred);
    
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/events?${params.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch events');
    return res.json();
  },

  getConflicts: async (incidentId: string): Promise<ConflictItem[]> => {
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/conflicts`);
    if (!res.ok) throw new Error('Failed to fetch conflicts');
    return res.json();
  },

  // Missing-Event Reconstruction
  reconstructGaps: async (incidentId: string, gapId?: string): Promise<{
    incident_id: string;
    gaps_analyzed: number;
    candidates: ReconstructedEvent[];
    summary: string;
  }> => {
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/reconstruct`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gap_id: gapId, use_llm: true }),
    });
    if (!res.ok) throw new Error('Failed to run reconstruction');
    return res.json();
  },

  getReconstructions: async (incidentId: string): Promise<ReconstructedEvent[]> => {
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/reconstructions`);
    if (!res.ok) throw new Error('Failed to fetch reconstructions');
    return res.json();
  },

  // Evidence & Telemetry
  getEvidence: async (incidentId: string): Promise<EvidenceItem[]> => {
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/evidence`);
    if (!res.ok) throw new Error('Failed to fetch evidence files');
    return res.json();
  },

  getSensorData: async (incidentId: string): Promise<SensorDataResponse> => {
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/sensor-data`);
    if (!res.ok) throw new Error('Failed to fetch sensor data');
    return res.json();
  },

  getSummaryReport: async (incidentId: string): Promise<InvestigationSummaryResponse> => {
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/summary`);
    if (!res.ok) throw new Error('Failed to fetch investigation summary');
    return res.json();
  },

  // Evidence Graph
  getGraph: async (incidentId: string): Promise<EvidenceGraphResponse> => {
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/graph`);
    if (!res.ok) throw new Error('Failed to fetch evidence graph');
    return res.json();
  },

  uploadEvidence: async (incidentId: string, file: File, fileType: string, summary?: string): Promise<EvidenceItem> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('file_type', fileType);
    if (summary) formData.append('summary', summary);

    const res = await fetch(`${API_BASE}/incidents/${incidentId}/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Failed to upload evidence');
    return res.json();
  },
};
