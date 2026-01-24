"""
Negation handling post-processor.

Adjusts NLI scores based on negation in claim or evidence.
"""
import logging
from typing import Optional, List

from app.pipeline.postprocessors.base import PostProcessor, Decision
from app.pipeline.models import Evidence, ParsedClaim

logger = logging.getLogger(__name__)


class NegationPostProcessor(PostProcessor):
    """
    Post-processor that adjusts scores based on negation handling.
    
    Handles cases where claim or evidence contains negation words.
    """
    
    NEGATION_WORDS = ["nicht", "nie", "kein", "keine", "keinen", "keiner", "no", "not", "never"]
    
    def process(
        self,
        claim: str,
        evidences: List[Evidence],
        decision: Decision,
        parsed_claim: Optional[ParsedClaim] = None,
    ) -> Decision:
        """
        Apply negation handling adjustments.
        
        Adjusts scores based on negation in claim or evidence.
        """
        if not decision.best_evidence:
            return decision
        
        claim_lower = claim.lower()
        
        # Get evidence text
        evidence_text = (
            decision.best_evidence.parsed.raw
            if decision.best_evidence.parsed and decision.best_evidence.parsed.raw
            else decision.best_evidence.snippet
        )
        
        if not evidence_text:
            return decision
        
        evidence_lower = evidence_text.lower()
        
        # Check for negation in claim
        claim_has_negation = any(
            f" {neg} " in claim_lower or claim_lower.startswith(f"{neg} ") or claim_lower.endswith(f" {neg}")
            for neg in self.NEGATION_WORDS
        )
        
        # Check for negation in evidence
        evidence_has_negation = any(
            f" {neg} " in evidence_lower or evidence_lower.startswith(f"{neg} ") or evidence_lower.endswith(f" {neg}")
            for neg in self.NEGATION_WORDS
        )
        
        # Adjust scores based on negation patterns
        if claim_has_negation and not evidence_has_negation:
            # Negated claim with positive evidence -> likely contradiction
            decision.best_contradiction = min(1.0, decision.best_contradiction * 1.2)
            decision.best_entailment *= 0.8
            decision.confidence *= 0.8
            decision.add_note(
                f"NEGATION: Negated claim with positive evidence, "
                f"boosted contradiction to {decision.best_contradiction:.3f}"
            )
            logger.debug(
                f"NEGATION: Negated claim with positive evidence, "
                f"boosted contradiction to {decision.best_contradiction:.3f}"
            )
            
        elif not claim_has_negation and evidence_has_negation:
            # Positive claim with negated evidence -> likely contradiction
            decision.best_contradiction = min(1.0, decision.best_contradiction * 1.2)
            decision.best_entailment *= 0.8
            decision.confidence *= 0.8
            decision.add_note(
                f"NEGATION: Positive claim with negated evidence, "
                f"boosted contradiction to {decision.best_contradiction:.3f}"
            )
            logger.debug(
                f"NEGATION: Positive claim with negated evidence, "
                f"boosted contradiction to {decision.best_contradiction:.3f}"
            )
            
        elif claim_has_negation and evidence_has_negation:
            # Both negated -> double negation might mean positive
            if decision.best_entailment > 0.5:
                decision.best_entailment = min(1.0, decision.best_entailment * 1.1)
                decision.confidence = min(1.0, decision.confidence * 1.1)
                decision.add_note(
                    f"NEGATION: Both claim and evidence negated, "
                    f"boosted entailment to {decision.best_entailment:.3f}"
                )
                logger.debug(
                    f"NEGATION: Both claim and evidence negated, "
                    f"boosted entailment to {decision.best_entailment:.3f}"
                )
        
        return decision

