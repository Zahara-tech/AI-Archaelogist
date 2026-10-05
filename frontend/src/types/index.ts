export type CertaintyType = 'observed' | 'reported' | 'inferred' | 'conflicted';
export type ObservedOrInferred = 'observed' | 'inferred';
export type SourceType = 
  | 'sensor' 
  | 'machine_log' 
  | 'operator_report' 
  | 'maintenance_report' 
  | 'cctv_video' 
  | 'production_record' 
  | 'inferred_reconstruction';

export interface EvidenceReference {
  source_type: string;
  file_name?: string;
  line_number?: number;
  row_index?: number;
  timestamp_raw?: string;
  metric_name?: string;
  metric_value?: any;
  bounding_box?: number[];
  quote?: string;
  details?: Record<string, any>;
}

export interface EventItem {
  id: string;
  incident_id: string;
  timestamp_start: string;
  timestamp_end?: string;
  event_type: string;
  description: string;
  source: SourceType;
  confidence: number;
  certainty_type: CertaintyType;
  observed_or_inferred: ObservedOrInferred;
  machine_id?: string;
  component_id?: string;
  location?: string;
  evidence_refs: EvidenceReference[];
  extra_metadata?: Record<string, any>;
}

export interface TemporalGap {
  gap_id: string;
  gap_start: string;
  gap_end: string;
  duration_seconds: number;
  previous_event_id?: string;
  previous_event_desc?: string;
  next_event_id?: string;
  next_event_desc?: string;
  machine_id?: string;
  description: string;
  reconstruction_candidates_count: number;
}

export interface ConflictItem {
  id: string;
  incident_id: string;
  conflict_type: string;
  description: string;
  source_a: string;
  source_b: string;
  evidence_a: Record<string, any>;
  evidence_b: Record<string, any>;
  time_discrepancy_seconds: number;
  status: string;
  created_at: string;
}

export interface ReconstructedEvent {
  id: string;
  incident_id: string;
  gap_start: string;
  gap_end: string;
  candidate_event_type: string;
  description: string;
  confidence: number;
  supporting_evidence: Record<string, any>[];
  conflicting_evidence: Record<string, any>[];
  reasoning: string;
  status: string;
  created_at: string;
}

export interface EvidenceItem {
  id: string;
  incident_id: string;
  file_name: string;
  file_type: string;
  file_path: string;
  uploaded_at: string;
  processed: number;
  summary?: string;
  extra_metadata?: Record<string, any>;
}

export interface IncidentSummary {
  id: string;
  title: string;
  machine_id?: string;
  component_id?: string;
  status: string;
  severity: string;
  created_at: string;
  event_count: number;
  evidence_count: number;
  gap_count: number;
  conflict_count: number;
}

export interface IncidentDetail {
  id: string;
  title: string;
  machine_id?: string;
  component_id?: string;
  severity: string;
  status: string;
  created_at: string;
  start_time?: string;
  end_time?: string;
  description?: string;
  evidence_items: EvidenceItem[];
  events: EventItem[];
}

export interface UnifiedTimelineResponse {
  incident_id: string;
  total_events: number;
  observed_events_count: number;
  inferred_events_count: number;
  time_range_start?: string;
  time_range_end?: string;
  events: EventItem[];
  temporal_gaps: TemporalGap[];
  conflicts: ConflictItem[];
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  source?: string;
  timestamp?: string;
  confidence?: number;
  certainty?: string;
  details: Record<string, any>;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  label?: string;
  weight: number;
}

export interface EvidenceGraphResponse {
  incident_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: Record<string, any>;
}

export interface SensorDataResponse {
  incident_id: string;
  total_points: number;
  columns: string[];
  records: Array<{
    timestamp: string;
    machine_id: string;
    temperature: number;
    vibration: number;
    coolant_pressure: number;
    spindle_rpm: number;
    power_draw_kw: number;
    machine_state: string;
  }>;
}

export interface InvestigationSummaryResponse {
  incident_id: string;
  title: string;
  machine_id?: string;
  component_id?: string;
  severity: string;
  status: string;
  time_window: {
    start?: string;
    end?: string;
  };
  what_happened: string;
  what_is_known: string[];
  what_is_inferred: Array<{
    event_type: string;
    time_window: string;
    confidence: number;
    reasoning: string;
  }>;
  chronology_highlights: Array<{
    time: string;
    event: string;
    source: string;
    confidence: number;
  }>;
  evidence_breakdown: Record<string, number>;
  conflicts_detected: Array<{
    id: string;
    type: string;
    description: string;
    discrepancy_seconds: number;
    source_a: string;
    source_b: string;
  }>;
  residual_uncertainties: string[];
}
