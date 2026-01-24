"""
Subject-based voting post-processor.

Uses subject-matching evidence for early strong decisions.
"""
import logging
from typing import Optional, List

from app.pipeline.postprocessors.base import PostProcessor, Decision
from app.pipeline.models import Evidence, ParsedClaim, FactLabel

logger = logging.getLogger(__name__)


class SubjectVotingPostProcessor(PostProcessor):
    """
    Post-processor that uses subject-matching evidence for early decisions.
    
    If strong entailment or contradiction is found from subject-relevant evidence,
    makes an early decision before other processors.
    """
    
    STRONG_CONTRA_THRESHOLD = 0.70
    STRONG_ENTAIL_THRESHOLD = 0.70
    
    def process(
        self,
        claim: str,
        evidences: List[Evidence],
        decision: Decision,
        parsed_claim: Optional[ParsedClaim] = None,
    ) -> Decision:
        """
        Apply subject-based voting.
        
        If we have subject matches, check for strong entailment/contradiction
        and make early decision.
        """
        if not parsed_claim or not parsed_claim.subject:
            return decision
        
        # Skip subject voting for generic/common nouns (they appear everywhere)
        # Subject voting is designed for specific entities (people, places, organizations)
        if self._is_generic_subject(parsed_claim.subject):
            return decision
        
        # Filter evidence scores by subject match
        subject_results = [
            entry for entry in decision.evidence_scores
            if self._subject_matches_evidence(parsed_claim.subject, entry["evidence"])
        ]
        
        if not subject_results:
            return decision
        
        # Check for strong contradiction from subject evidence
        worst_subject = min(subject_results, key=lambda x: x["entailment"] - x["contradiction"])
        
        # CRITICAL: Only set REFUTED if:
        # 1. Contradiction is high (>= threshold)
        # 2. Contradiction is significantly higher than entailment (difference >= 0.3)
        # 3. We have multiple evidences OR high confidence in the evidence quality
        # This prevents false negatives from single, low-quality evidences
        
        contradiction_score = worst_subject["contradiction"]
        entailment_score = worst_subject["entailment"]
        score_difference = contradiction_score - entailment_score
        
        # CRITICAL: Check if this is a biological/definition claim (e.g., "Erdbeere ist eine Nuss")
        # These claims are complex and should be more lenient - prefer NOT_ENOUGH_INFO over REFUTED
        claim_lower = claim.lower()
        is_biological_claim = any(kw in claim_lower for kw in [
            "ist eine", "ist ein", "gehört zu", "ist eine art", "ist eine sorte",
            "ist eine nuss", "ist eine frucht", "ist eine beere", "ist eine sammelnussfrucht"
        ])
        
        # Require both high contradiction AND significant difference to avoid false negatives
        if (contradiction_score >= self.STRONG_CONTRA_THRESHOLD and 
            contradiction_score > entailment_score and
            score_difference >= 0.3):  # At least 30% difference between contradiction and entailment
            
            # CRITICAL: For biological claims, be more lenient - require higher confidence
            # "Erdbeere ist eine Nuss" is technically a Sammelnussfrucht, so it's not clearly wrong
            if is_biological_claim:
                # For biological claims, require even higher confidence (>= 0.95) to set REFUTED
                # This prevents false negatives for claims that are technically correct but nuanced
                if contradiction_score < 0.95 or score_difference < 0.5:
                    decision.add_note(
                        f"SUBJECT_VOTE: Biological claim with contradiction={contradiction_score:.3f}, "
                        f"diff={score_difference:.3f} (below threshold for biological claims) -> keeping NOT_ENOUGH_INFO"
                    )
                    logger.info(
                        f"SUBJECT_VOTE: Biological claim with insufficient confidence "
                        f"(contradiction={contradiction_score:.3f} < 0.95 or diff={score_difference:.3f} < 0.5) -> NOT_ENOUGH_INFO"
                    )
                    # Don't change label, keep NOT_ENOUGH_INFO
                    return decision
            
            # Additional check: if we only have 1 evidence and it's not highly relevant, be more cautious
            evidence_count = len(subject_results)
            if evidence_count == 1:
                # For single evidence, require even higher confidence (>= 0.85) to avoid false negatives
                if contradiction_score >= 0.85 and score_difference >= 0.4:
                    decision.label = FactLabel.REFUTED
                    decision.best_evidence = worst_subject["evidence"]
                    decision.best_entailment = worst_subject["entailment"]
                    decision.best_contradiction = worst_subject["contradiction"]
                    decision.confidence = worst_subject["contradiction"]
                    decision.add_note(
                        f"SUBJECT_VOTE: Strong contradiction from subject evidence (score={contradiction_score:.3f}, "
                        f"diff={score_difference:.3f}, single_evidence) -> REFUTED"
                    )
                    logger.info(
                        f"SUBJECT_VOTE: Strong contradiction from subject evidence (single_evidence, "
                        f"score={contradiction_score:.3f}) -> REFUTED"
                    )
                    return decision
                else:
                    # Single evidence but not high enough confidence -> NOT_ENOUGH_INFO
                    decision.add_note(
                        f"SUBJECT_VOTE: Single evidence with contradiction={contradiction_score:.3f}, "
                        f"diff={score_difference:.3f} (below threshold for single evidence) -> keeping NOT_ENOUGH_INFO"
                    )
                    logger.info(
                        f"SUBJECT_VOTE: Single evidence with insufficient confidence "
                        f"(contradiction={contradiction_score:.3f} < 0.85 or diff={score_difference:.3f} < 0.4) -> NOT_ENOUGH_INFO"
                    )
                    # Don't change label, keep NOT_ENOUGH_INFO
                    return decision
            else:
                # Multiple evidences - can be more confident
                decision.label = FactLabel.REFUTED
                decision.best_evidence = worst_subject["evidence"]
                decision.best_entailment = worst_subject["entailment"]
                decision.best_contradiction = worst_subject["contradiction"]
                decision.confidence = worst_subject["contradiction"]
                decision.add_note(
                    f"SUBJECT_VOTE: Strong contradiction from subject evidence (score={contradiction_score:.3f}, "
                    f"diff={score_difference:.3f}, {evidence_count} evidences) -> REFUTED"
                )
                logger.info(
                    f"SUBJECT_VOTE: Strong contradiction from subject evidence "
                    f"({evidence_count} evidences, score={contradiction_score:.3f}) -> REFUTED"
                )
                return decision
        
        # Check for strong entailment from subject evidence
        best_subject = max(subject_results, key=lambda x: x["entailment"] - x["contradiction"])
        if best_subject["entailment"] >= self.STRONG_ENTAIL_THRESHOLD and \
           best_subject["entailment"] > best_subject["contradiction"]:
            decision.label = FactLabel.SUPPORTED
            decision.best_evidence = best_subject["evidence"]
            decision.best_entailment = best_subject["entailment"]
            decision.best_contradiction = best_subject["contradiction"]
            decision.confidence = best_subject["entailment"]
            decision.add_note("SUBJECT_VOTE: Strong entailment from subject evidence -> SUPPORTED")
            logger.info("SUBJECT_VOTE: Strong entailment from subject evidence -> SUPPORTED")
            return decision
        
        return decision
    
    def _is_generic_subject(self, subject: str) -> bool:
        """
        Check if subject is a generic/common noun (not a specific entity).
        
        Generic subjects like "Ein Jahr", "Der Tag", "Die Woche" shouldn't trigger
        subject voting because they appear in almost every related article.
        
        Args:
            subject: Subject string from parsed claim
            
        Returns:
            True if subject is generic/common
        """
        if not subject:
            return True
        
        subject_lower = subject.lower().strip()
        
        # Remove leading articles (der, die, das, ein, eine)
        articles = ["der ", "die ", "das ", "ein ", "eine ", "einen ", "einem ", "einer "]
        for article in articles:
            if subject_lower.startswith(article):
                subject_lower = subject_lower[len(article):].strip()
        
        # List of generic/common nouns (time, abstract concepts, generic objects)
        generic_nouns = {
            # Time concepts
            "jahr", "monat", "tag", "woche", "stunde", "minute", "sekunde",
            "kalenderjahr", "schaltjahr",
            # Abstract concepts
            "welt", "erde", "universum", "leben", "tod", "zeit", "raum",
            # Generic objects/concepts
            "mensch", "menschen", "person", "leute", "tier", "pflanze",
            "ding", "sache", "objekt", "gegenstand",
        }
        
        return subject_lower in generic_nouns
    
    def _subject_matches_evidence(self, subject: str, evidence: Evidence) -> bool:
        """
        Check if evidence mentions the subject.
        
        Args:
            subject: Subject string from parsed claim
            evidence: Evidence object
            
        Returns:
            True if subject appears in evidence title or snippet
        """
        if not subject or not evidence:
            return False
        
        subject_lower = subject.lower()
        subject_tokens = set(subject_lower.split())
        surname = list(subject_tokens)[-1] if subject_tokens else ""
        
        title_lower = evidence.title.lower()
        snippet_lower = evidence.snippet.lower()
        
        # Check if subject matches (exact match, surname match, or token overlap)
        return (
            subject_lower in title_lower or
            surname in title_lower or
            subject_lower in snippet_lower or
            surname in snippet_lower
        )

