"""
Base PostProcessor interface for Verifier decision adjustments.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from app.pipeline.models import Evidence, FactLabel, ParsedClaim


@dataclass
class Decision:
    """
    Decision structure passed between kernel and postprocessors.
    
    Attributes:
        label: Current FactLabel decision
        confidence: Confidence score (0.0 to 1.0)
        best_evidence: Best evidence object
        best_entailment: Best entailment score
        best_contradiction: Best contradiction score
        threshold: Current threshold used for decision
        evidence_scores: List of all evidence score dicts
        notes: Optional debug notes
        metadata: Optional metadata dict for passing data between processors
    """
    label: FactLabel
    confidence: float
    best_evidence: Optional[Evidence]
    best_entailment: float
    best_contradiction: float
    threshold: float
    evidence_scores: List[Dict[str, Any]]
    notes: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def add_note(self, note: str) -> None:
        """Add a debug note."""
        if self.notes is None:
            self.notes = []
        self.notes.append(note)


class PostProcessor(ABC):
    """
    Base class for post-processors that adjust Verifier decisions.
    
    Each processor can modify the Decision object based on domain-specific heuristics.
    Processors are applied sequentially after the NLI kernel decision.
    """
    
    @abstractmethod
    def process(
        self,
        claim: str,
        evidences: List[Evidence],
        decision: Decision,
        parsed_claim: Optional[ParsedClaim] = None,
    ) -> Decision:
        """
        Process and potentially adjust the decision.
        
        Args:
            claim: The original claim text
            evidences: List of all evidence objects
            decision: Current decision from kernel or previous processors
            parsed_claim: Optional parsed claim for context
            
        Returns:
            Updated Decision object (may be the same or modified)
        """
        pass
    
    @property
    def name(self) -> str:
        """Return processor name for logging."""
        return self.__class__.__name__

