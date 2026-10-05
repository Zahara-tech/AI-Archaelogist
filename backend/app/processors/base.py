"""
Abstract Base Evidence Processor Interface.
All modality processors (sensor, log, text, video, ERP) inherit from this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.schemas.event import EventBase, EvidenceReference

class BaseProcessor(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(
        self, 
        file_path: str, 
        incident_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EventBase]:
        """
        Extract standardized events from the raw evidence file.
        Must return a list of EventBase objects with concrete evidence references.
        """
        pass
