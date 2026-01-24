"""
Sports year heuristic post-processor.

Handles year mismatches for sports claims.
"""
import logging
from typing import Optional, List

from app.pipeline.postprocessors.base import PostProcessor, Decision
from app.pipeline.models import Evidence, ParsedClaim, FactLabel
from app.pipeline.utils import extract_years

logger = logging.getLogger(__name__)


class SportsYearPostProcessor(PostProcessor):
    """
    Post-processor for sports year heuristics.
    
    Upgrades NOT_ENOUGH_INFO to REFUTED for sports claims with year mismatches
    if the event is also mentioned.
    """
    
    def process(
        self,
        claim: str,
        evidences: List[Evidence],
        decision: Decision,
        parsed_claim: Optional[ParsedClaim] = None,
    ) -> Decision:
        """
        Apply sports year heuristic.
        
        For sports_title/sports_title_year claims with year mismatches,
        upgrade NOT_ENOUGH_INFO to REFUTED if event is mentioned.
        """
        if not parsed_claim or not parsed_claim.year:
            return decision
        
        if parsed_claim.category not in ["sports_title", "sports_title_year"]:
            return decision
        
        if decision.label != FactLabel.NOT_ENOUGH_INFO:
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
        
        # Extract years from evidence
        evidence_years = extract_years(evidence_text)
        
        if not evidence_years:
            return decision
        
        claim_year = parsed_claim.year
        
        # Check if claim year matches any evidence year
        if claim_year in evidence_years:
            return decision  # Year matches, no adjustment needed
        
        # Year mismatch: check if event is mentioned in evidence
        claim_lower = claim.lower()
        evidence_lower = evidence_text.lower()
        
        # Check for sports event keywords
        sports_keywords = ["weltmeisterschaft", "wm", "champions league", "bundesliga", "olympia", "tour de france"]
        event_mentioned = any(keyword in evidence_lower for keyword in sports_keywords if keyword in claim_lower)
        
        if event_mentioned:
            # Event is mentioned but year doesn't match -> likely REFUTED
            decision.label = FactLabel.REFUTED
            decision.add_note(
                f"SPORTS_YEAR: Year mismatch (claim={claim_year}, evidence={evidence_years}) "
                f"with event mentioned -> REFUTED"
            )
            logger.info(
                f"SPORTS_YEAR: Year mismatch (claim={claim_year}, evidence={evidence_years}) "
                f"with event mentioned -> REFUTED"
            )
        
        return decision

