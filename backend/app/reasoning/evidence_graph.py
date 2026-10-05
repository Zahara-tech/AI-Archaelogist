"""
Evidence Graph Engine utilizing NetworkX.
Constructs multi-relational graphs linking observed events, raw evidence nodes,
causal chains, contradictions, and inferred reconstruction candidates.
"""

import networkx as nx
from typing import List, Dict, Any
from app.schemas.event import EventResponse
from app.schemas.timeline import ConflictItem
from app.schemas.reconstruction import ReconstructedEventResponse
from app.schemas.graph import GraphNode, GraphEdge, EvidenceGraphResponse

class EvidenceGraphBuilder:
    def __init__(self):
        pass

    def build_graph(
        self,
        incident_id: str,
        events: List[EventResponse],
        conflicts: List[ConflictItem],
        reconstructions: List[ReconstructedEventResponse]
    ) -> EvidenceGraphResponse:
        G = nx.DiGraph()

        # 1. Add Event Nodes
        for e in events:
            node_type = "event_inferred" if e.observed_or_inferred == "inferred" else "event_observed"
            G.add_node(
                e.id,
                label=f"[{e.source.upper()}] {e.event_type}",
                type=node_type,
                source=e.source,
                timestamp=e.timestamp_start.isoformat(),
                confidence=e.confidence,
                certainty=e.certainty_type,
                details={
                    "description": e.description,
                    "location": e.location,
                    "machine_id": e.machine_id,
                    "component_id": e.component_id,
                    "evidence_refs_count": len(e.evidence_refs)
                }
            )

        # 2. Add Temporal Ordering Edges (occurred_before)
        sorted_events = sorted(events, key=lambda x: x.timestamp_start)
        for i in range(len(sorted_events) - 1):
            G.add_edge(
                sorted_events[i].id,
                sorted_events[i + 1].id,
                relation="occurred_before",
                label="then",
                weight=1.0
            )

        # 3. Add Raw Evidence Nodes and 'supported_by' Edges
        evidence_seen = set()
        for e in events:
            for ref in e.evidence_refs:
                evidence_node_id = f"RAW_{ref.source_type}_{ref.file_name}_{ref.line_number or ref.row_index or ref.timestamp_raw}"
                if evidence_node_id not in evidence_seen:
                    evidence_seen.add(evidence_node_id)
                    G.add_node(
                        evidence_node_id,
                        label=f"Evidence: {ref.file_name or ref.source_type}",
                        type=f"evidence_{ref.source_type}",
                        source=ref.source_type,
                        timestamp=ref.timestamp_raw,
                        confidence=1.0,
                        certainty="raw_data",
                        details=ref.details or {}
                    )
                G.add_edge(
                    e.id,
                    evidence_node_id,
                    relation="supported_by",
                    label="supported by",
                    weight=0.8
                )

        # 4. Add Conflict Edges (conflicts_with)
        for c in conflicts:
            ev_a_id = c.evidence_a.get("event_id")
            ev_b_id = c.evidence_b.get("event_id")
            if ev_a_id and ev_b_id and G.has_node(ev_a_id) and G.has_node(ev_b_id):
                G.add_edge(
                    ev_a_id,
                    ev_b_id,
                    relation="conflicts_with",
                    label=f"conflicts ({c.time_discrepancy_seconds:.1f}s)",
                    weight=1.5
                )

        # 5. Add Reconstructed Nodes & 'inferred_from' Edges
        for rec in reconstructions:
            G.add_node(
                rec.id,
                label=f"[INFERRED] {rec.candidate_event_type}",
                type="event_inferred",
                source="inferred_reconstruction",
                timestamp=rec.gap_start.isoformat(),
                confidence=rec.confidence,
                certainty="inferred",
                details={
                    "description": rec.description,
                    "reasoning": rec.reasoning,
                    "status": rec.status
                }
            )
            # Connect inferred node to surrounding events
            for e in events:
                if rec.gap_start <= e.timestamp_start <= rec.gap_end:
                    G.add_edge(
                        rec.id,
                        e.id,
                        relation="inferred_from",
                        label="corroborated by",
                        weight=1.2
                    )

        # Convert NetworkX graph to API response format
        nodes: List[GraphNode] = []
        for n_id, n_data in G.nodes(data=True):
            nodes.append(GraphNode(
                id=str(n_id),
                label=str(n_data.get("label", n_id)),
                type=str(n_data.get("type", "event_observed")),
                source=n_data.get("source"),
                timestamp=n_data.get("timestamp"),
                confidence=n_data.get("confidence"),
                certainty=n_data.get("certainty"),
                details=n_data.get("details", {})
            ))

        edges: List[GraphEdge] = []
        for u, v, e_data in G.edges(data=True):
            edges.append(GraphEdge(
                source=str(u),
                target=str(v),
                relation=str(e_data.get("relation", "relates_to")),
                label=e_data.get("label"),
                weight=float(e_data.get("weight", 1.0))
            ))

        stats = {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(G),
            "density": float(nx.density(G))
        }

        return EvidenceGraphResponse(
            incident_id=incident_id,
            nodes=nodes,
            edges=edges,
            stats=stats
        )
