"""
Population-specific post-processor.

Uses Wikidata P1082 as primary source, falls back to Wikipedia extraction.
Implements robust comparison logic with dynamic thresholds.
"""
import logging
from typing import Optional, List, Dict, Any

from app.pipeline.postprocessors.base import PostProcessor, Decision
from app.pipeline.models import Evidence, ParsedClaim, FactLabel
from app.pipeline.population_parser import parse_population_number

logger = logging.getLogger(__name__)

def format_population(n: int) -> str:
    """
    Format population for UI (German style).
    - If n >= 1_000_000: show one decimal in millions with comma: "48,6 Millionen"
    - Else: show as integer string
    """
    if n is None:
        return ""
    if n >= 1_000_000:
        m = n / 1_000_000.0
        s = f"{m:.1f}".replace(".", ",")
        return f"{s} Millionen"
    return str(n)


class PopulationVerifierPostProcessor(PostProcessor):
    """
    Post-processor for population claims.
    
    Flow:
    1. Try Wikidata P1082 (if wikidata_client available)
    2. Fallback to Wikipedia extraction using unified parser
    3. Compare with dynamic thresholds based on population size
    4. Handle year matching if claim includes year
    """
    
    name = "population_verifier"
    
    def __init__(self):
        """Initialize population verifier."""
        pass
    
    def process(
        self,
        claim: str,
        evidences: List[Evidence],
        decision: Decision,
        parsed_claim: Optional[ParsedClaim] = None,
    ) -> Decision:
        """
        Process population claims with Wikidata fast-path and robust comparison.
        """
        claim_lower = claim.lower()
        is_population_claim = any(kw in claim_lower for kw in ["einwohner", "bevölkerung", "population"])
        
        if not is_population_claim:
            return decision
        
        if not parsed_claim or not parsed_claim.subject:
            return decision
        
        logger.info(f"POPULATION_VERIFIER: Processing population claim for '{parsed_claim.subject}'")
        
        # Extract claimed population from claim
        claim_values = parse_population_number(claim)
        if not claim_values:
            logger.warning(f"POPULATION_VERIFIER: Could not extract population from claim '{claim}'")
            return decision
        
        # Use largest value from claim (in case multiple matches)
        claim_value = max(claim_values, key=lambda v: v.normalized_value_int)
        claim_pop_int = claim_value.normalized_value_int
        claim_year = parsed_claim.year or claim_value.year
        
        logger.info(f"POPULATION_VERIFIER: Claimed population={claim_pop_int:,} (year={claim_year})")
        
        # Step 1: Try Wikidata fast-path (pre-fetched data from factchecker.py)
        evidence_pop_int = None
        evidence_year = None
        evidence_source = None
        evidence_sentence = None
        evidence_title = None
        
        wikidata_result = self._get_wikidata_from_metadata(decision)
        if wikidata_result:
            evidence_pop_int = wikidata_result["value_int"]
            evidence_year = wikidata_result.get("point_in_time")
            evidence_source = "wikidata"
            evidence_title = f"Wikidata: {parsed_claim.subject}"
            # Use a safe, population-only sentence to avoid random numbers in UI
            evidence_sentence = f"{parsed_claim.subject} hat {format_population(evidence_pop_int)} Einwohner"
            if evidence_year:
                evidence_sentence += f" (Stand: {evidence_year})"
            logger.info(f"POPULATION_VERIFIER: Wikidata result: {evidence_pop_int:,} (year={evidence_year})")

            # Persist structured correction fields for formatter/UI
            decision.metadata = decision.metadata or {}
            decision.metadata["population_correction"] = {
                "correct_value": evidence_pop_int,
                "correct_value_year": evidence_year,
                "correct_value_source": "wikidata",
                "correct_value_str": format_population(evidence_pop_int),
            }
        
        # Step 2: Fallback to Wikipedia extraction
        if evidence_pop_int is None:
            wikipedia_result = self._extract_from_wikipedia_evidences(
                evidences, parsed_claim.subject, claim_year
            )
            if wikipedia_result:
                evidence_pop_int = wikipedia_result["value_int"]
                evidence_year = wikipedia_result.get("year")
                evidence_source = "wikipedia"
                evidence_sentence = wikipedia_result.get("sentence")
                evidence_title = wikipedia_result.get("title")
                logger.info(f"POPULATION_VERIFIER: Wikipedia result: {evidence_pop_int:,} from '{evidence_title}'")

                # Persist correction fields for fallback too (used only if wikidata missing)
                decision.metadata = decision.metadata or {}
                decision.metadata["population_correction"] = {
                    "correct_value": evidence_pop_int,
                    "correct_value_year": evidence_year,
                    "correct_value_source": "wikipedia",
                    "correct_value_str": format_population(evidence_pop_int),
                }
        
        if evidence_pop_int is None:
            logger.warning(f"POPULATION_VERIFIER: No population data found for '{parsed_claim.subject}'")
            decision.label = FactLabel.NOT_ENOUGH_INFO
            decision.add_note(f"POPULATION_VERIFIER: No population data found (searched Wikidata and {len(evidences)} Wikipedia evidences)")
            return decision
        
        # Step 3: Compare with dynamic thresholds
        diff_abs = abs(claim_pop_int - evidence_pop_int)
        diff_rel = diff_abs / max(claim_pop_int, evidence_pop_int) if max(claim_pop_int, evidence_pop_int) > 0 else 1.0
        
        # Dynamic tolerance based on population size
        if evidence_pop_int >= 50_000_000:
            max_rel_diff = 0.03  # 3%
            max_abs_diff = 2_000_000  # 2M
        elif evidence_pop_int >= 5_000_000:
            max_rel_diff = 0.04  # 4%
            max_abs_diff = 500_000  # 500k
        else:
            max_rel_diff = 0.05  # 5%
            max_abs_diff = 100_000  # 100k
        
        # Year matching: if claim has year and evidence year differs significantly, be stricter
        if claim_year and evidence_year:
            year_diff = abs(claim_year - evidence_year)
            if year_diff > 5:  # More than 5 years difference
                logger.info(f"POPULATION_VERIFIER: Year mismatch (claim={claim_year}, evidence={evidence_year}), requiring exact match")
                if year_diff > 10:
                    decision.label = FactLabel.NOT_ENOUGH_INFO
                    decision.add_note(
                        f"POPULATION_VERIFIER: Year mismatch too large (claim={claim_year}, evidence={evidence_year}) -> NOT_ENOUGH_INFO"
                    )
                    return decision
                # Stricter tolerance for year mismatches
                max_rel_diff *= 0.5
                max_abs_diff *= 0.5
        
        # Determine label
        if diff_rel <= max_rel_diff and diff_abs <= max_abs_diff:
            decision.label = FactLabel.SUPPORTED
            decision.add_note(
                f"POPULATION_VERIFIER: SUPPORTED (claim={claim_pop_int:,}, evidence={evidence_pop_int:,}, "
                f"diff={diff_abs:,} ({diff_rel:.1%}), source={evidence_source})"
            )
        else:
            decision.label = FactLabel.REFUTED
            decision.add_note(
                f"POPULATION_VERIFIER: REFUTED (claim={claim_pop_int:,}, evidence={evidence_pop_int:,}, "
                f"diff={diff_abs:,} ({diff_rel:.1%}), threshold={max_rel_diff:.1%}/{max_abs_diff:,}, source={evidence_source})"
            )
        
        # Update best_evidence:
        # - If wikidata: force a synthetic population-only evidence sentence (prevents UI extracting wrong numbers)
        # - If wikipedia: keep matching evidence or synthesize
        if evidence_source == "wikidata" and evidence_sentence and evidence_title:
            from app.pipeline.models import Evidence as EvidenceModel
            url = ""
            if wikidata_result and wikidata_result.get("entity_qid"):
                url = f"https://www.wikidata.org/wiki/{wikidata_result['entity_qid']}"
            decision.best_evidence = EvidenceModel(
                source="wikidata",
                title=evidence_title,
                snippet=evidence_sentence,
                url=url,
            )
        elif evidence_source == "wikipedia" and evidence_sentence and evidence_title:
            # Find matching evidence object
            for ev in evidences:
                if ev.title == evidence_title and evidence_sentence in ev.snippet:
                    decision.best_evidence = ev
                    break
            # If not found, create synthetic evidence
            if decision.best_evidence is None or decision.best_evidence.title != evidence_title:
                from app.pipeline.models import Evidence
                decision.best_evidence = Evidence(
                    source="wikipedia",
                    title=evidence_title,
                    snippet=evidence_sentence,
                    url=f"https://de.wikipedia.org/wiki/{evidence_title.replace(' ', '_')}",
                )
        
        return decision
    
    def _get_wikidata_from_metadata(
        self, decision: Decision
    ) -> Optional[Dict[str, Any]]:
        """
        Get pre-fetched Wikidata population data from Decision metadata.
        
        Metadata key: "wikidata_population"
        """
        if decision.metadata and "wikidata_population" in decision.metadata:
            return decision.metadata["wikidata_population"]
        return None
    
    def _extract_from_wikipedia_evidences(
        self,
        evidences: List[Evidence],
        subject: str,
        claim_year: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract population from Wikipedia evidences.
        
        Filters evidences to:
        - Main article (title == subject)
        - Contains population keywords
        - Contains parseable population number
        
        Returns best candidate by score.
        """
        subject_lower = subject.lower()
        candidates = []
        
        for ev in evidences:
            title_lower = ev.title.lower()
            snippet_lower = ev.snippet.lower()
            
            # Filter: must be main article or contain subject
            is_main_article = title_lower.strip() == subject_lower
            subject_in_title = any(token in title_lower for token in subject_lower.split() if len(token) > 3)
            subject_in_snippet = any(token in snippet_lower for token in subject_lower.split() if len(token) > 3)
            
            if not (is_main_article or subject_in_title or subject_in_snippet):
                continue
            
            # Filter: exclude disambiguation/list pages
            if any(kw in title_lower for kw in ["begriffsklärung", "liste", "geschichte", "krieg", "chronik"]):
                continue
            
            # Filter: must contain population keywords
            pop_keywords = ["einwohner", "bevölkerung", "population", "habitantes", "inhabitants"]
            has_pop_keyword = any(kw in snippet_lower for kw in pop_keywords)
            if not has_pop_keyword:
                continue

            # Reject delta/change cues (fallback only): avoid “+7 Millionen”, migration deltas, etc.
            delta_cues = ["stieg", "sank", "zuwachs", "anstieg", "rückgang", "um ", "migration", "einwander"]
            if any(cue in snippet_lower for cue in delta_cues):
                continue
            
            # Extract population numbers
            pop_values = parse_population_number(ev.snippet)
            if not pop_values:
                continue
            
            # Filter out unrealistic values
            valid_values = [v for v in pop_values if 100_000 <= v.normalized_value_int <= 10_000_000_000]
            if not valid_values:
                continue
            
            # Use largest valid value
            best_value = max(valid_values, key=lambda v: v.normalized_value_int)
            
            # Score candidate
            score = 0.0
            if is_main_article:
                score += 50.0  # Very strong bonus for main article
            elif subject_in_title:
                score += 10.0
            
            # Bonus for demographic keywords
            if any(kw in snippet_lower for kw in ["bevölkerung", "demografie", "demographie"]):
                score += 5.0
            
            # Bonus for recent years
            recent_years = ["2024", "2023", "2022", "2021", "2020"]
            if any(year in ev.snippet for year in recent_years):
                score += 8.0
            
            # Bonus for current indicators
            current_indicators = ["aktuell", "heute", "laut", "nach", "stand", "zählung", "volkszählung"]
            if any(ind in snippet_lower for ind in current_indicators):
                score += 4.0
            
            # Year match bonus
            if claim_year and best_value.year:
                if best_value.year == claim_year:
                    score += 15.0
                elif abs(best_value.year - claim_year) <= 2:
                    score += 5.0
            
            candidates.append({
                "evidence": ev,
                "value_int": best_value.normalized_value_int,
                "year": best_value.year,
                "score": score,
            })
        
        if not candidates:
            return None
        
        # Return best candidate
        best = max(candidates, key=lambda c: c["score"])
        return {
            "value_int": best["value_int"],
            "year": best["year"],
            "title": best["evidence"].title,
            "sentence": best["evidence"].snippet,
        }

