"""
Temporal reasoning post-processor.

Adjusts NLI scores based on tense mismatches between claim and evidence.
"""
import logging
from typing import Optional, List

from app.pipeline.postprocessors.base import PostProcessor, Decision
from app.pipeline.models import Evidence, ParsedClaim, FactLabel
from app.pipeline.utils import infer_tense

logger = logging.getLogger(__name__)


class TemporalPostProcessor(PostProcessor):
    """
    Post-processor that adjusts scores based on tense matching.
    
    Reduces confidence if claim and evidence have mismatched tenses
    (e.g., past claim with present evidence, or vice versa).
    """
    
    def process(
        self,
        claim: str,
        evidences: List[Evidence],
        decision: Decision,
        parsed_claim: Optional[ParsedClaim] = None,
    ) -> Decision:
        """
        Apply temporal reasoning adjustments.
        
        If claim and evidence have mismatched tenses, reduce confidence.
        """
        if not parsed_claim or parsed_claim.tense == "unknown":
            return decision
        
        if not decision.best_evidence:
            return decision
        
        # Get evidence text
        evidence_text = (
            decision.best_evidence.parsed.raw
            if decision.best_evidence.parsed and decision.best_evidence.parsed.raw
            else decision.best_evidence.snippet
        )
        
        if not evidence_text:
            return decision
        
        evidence_tense = infer_tense(evidence_text)
        
        # Adjust scores based on tense mismatch
        if parsed_claim.tense == "past" and evidence_tense == "present":
            # Past claim but present evidence - reduce confidence
            decision.best_entailment *= 0.85
            decision.confidence *= 0.85
            decision.add_note(f"TEMPORAL: Past claim with present evidence, reduced entailment to {decision.best_entailment:.3f}")
            logger.debug(f"TEMPORAL: Past claim with present evidence, reduced entailment to {decision.best_entailment:.3f}")
            
        elif parsed_claim.tense == "present" and evidence_tense == "past":
            # Present claim but only past evidence - reduce confidence
            decision.best_entailment *= 0.80
            decision.confidence *= 0.80
            decision.add_note(f"TEMPORAL: Present claim with past evidence, reduced entailment to {decision.best_entailment:.3f}")
            logger.debug(f"TEMPORAL: Present claim with past evidence, reduced entailment to {decision.best_entailment:.3f}")
        
        # Re-evaluate label if confidence dropped significantly
        if decision.best_entailment < decision.threshold and decision.label == FactLabel.SUPPORTED:
            decision.label = FactLabel.NOT_ENOUGH_INFO
            decision.add_note("TEMPORAL: Label downgraded due to tense mismatch")
        
        return decision

