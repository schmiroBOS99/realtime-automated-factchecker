"""
Year matching post-processor.

Adjusts NLI scores based on year presence/mismatch in evidence.
"""
import re
import logging
from typing import Optional, List

from app.pipeline.postprocessors.base import PostProcessor, Decision
from app.pipeline.models import Evidence, ParsedClaim

logger = logging.getLogger(__name__)


class YearMatchPostProcessor(PostProcessor):
    """
    Post-processor that adjusts scores based on year matching.
    
    Boosts confidence if claim year matches evidence year.
    Reduces confidence if years are significantly mismatched.
    """
    
    def process(
        self,
        claim: str,
        evidences: List[Evidence],
        decision: Decision,
        parsed_claim: Optional[ParsedClaim] = None,
    ) -> Decision:
        """
        Apply year matching adjustments.
        
        If claim has a year, check if it matches evidence years.
        """
        if not parsed_claim or not parsed_claim.year:
            return decision
        
        if not decision.best_evidence:
            return decision
        
        claim_year = parsed_claim.year
        
        # Get evidence text
        evidence_text = (
            decision.best_evidence.parsed.raw
            if decision.best_evidence.parsed and decision.best_evidence.parsed.raw
            else decision.best_evidence.snippet
        )
        
        if not evidence_text:
            return decision
        
        # Extract years from evidence (1900-2099)
        year_matches = re.findall(r'\b((?:19|20)\d{2})\b', evidence_text)
        evidence_years = [int(y) for y in year_matches]
        
        if not evidence_years:
            return decision  # No years found, no adjustment
        
        # Check if claim year matches any evidence year
        if claim_year in evidence_years:
            # Year matches - boost confidence
            decision.best_entailment = min(1.0, decision.best_entailment * 1.15)
            decision.confidence = min(1.0, decision.confidence * 1.15)
            decision.add_note(f"YEAR_MATCH: Year {claim_year} found in evidence, boosted entailment to {decision.best_entailment:.3f}")
            logger.debug(f"YEAR_MATCH: Year {claim_year} found in evidence, boosted entailment to {decision.best_entailment:.3f}")
        else:
            # Check if years are close (within 2 years)
            closest_year = min(evidence_years, key=lambda x: abs(x - claim_year))
            year_diff = abs(closest_year - claim_year)
            
            if year_diff > 2:
                # Significant year mismatch - reduce confidence
                decision.best_entailment *= 0.80
                decision.best_contradiction = min(1.0, decision.best_contradiction * 1.2)
                decision.confidence *= 0.80
                decision.add_note(
                    f"YEAR_MISMATCH: Claim year {claim_year} vs evidence year {closest_year} "
                    f"(diff={year_diff}), reduced entailment to {decision.best_entailment:.3f}"
                )
                logger.debug(
                    f"YEAR_MISMATCH: Claim year {claim_year} vs evidence year {closest_year} "
                    f"(diff={year_diff}), reduced entailment to {decision.best_entailment:.3f}"
                )
        
        return decision

