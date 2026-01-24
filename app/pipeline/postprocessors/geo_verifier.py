"""
Geo-specific post-processor for location claims.

Uses Wikidata P17 (country) as primary source for location verification.
Handles claims like "Berlin liegt in Frankreich" -> REFUTED with correction "Deutschland".
"""
import logging
import re
from typing import Optional, List, Dict, Any

from app.pipeline.postprocessors.base import PostProcessor, Decision
from app.pipeline.models import Evidence, ParsedClaim, FactLabel

logger = logging.getLogger(__name__)


class GeoVerifierPostProcessor(PostProcessor):
    """
    Post-processor for geographic location claims.
    
    Flow:
    1. Detect location claims (liegt in / befindet sich in / gehört zu / ist in)
    2. Try Wikidata P17 (country) for subject entity
    3. Compare with claimed location
    4. Set correction data for response formatter
    """
    
    name = "geo_verifier"
    
    def __init__(self):
        """Initialize geo verifier."""
        pass
    
    def process(
        self,
        claim: str,
        evidences: List[Evidence],
        decision: Decision,
        parsed_claim: Optional[ParsedClaim] = None,
    ) -> Decision:
        """
        Process geographic location claims with Wikidata fast-path.
        """
        claim_lower = claim.lower()
        
        # Detect location claim patterns (including "Hauptstadt" claims)
        location_patterns = [
            "liegt in", "befindet sich in", "gehört zu", "ist in",
            "fließt durch", "umkreist",
        ]
        is_location_claim = any(pattern in claim_lower for pattern in location_patterns)
        is_hauptstadt_claim = "hauptstadt" in claim_lower and "ist" in claim_lower
        
        if not (is_location_claim or is_hauptstadt_claim):
            return decision
        
        if not parsed_claim or not parsed_claim.subject:
            return decision
        
        # Extract claimed location from object or claim
        claimed_location = parsed_claim.object or ""
        if not claimed_location:
            # For "Hauptstadt" claims, extract location from "von Y" or "Y"
            # CRITICAL: Allow hyphens and multiple words for compound names like "Baden-Württemberg", "Rheinland-Pfalz"
            if is_hauptstadt_claim:
                # Pattern: "Hauptstadt von Baden-Württemberg" or "Hauptstadt Baden-Württemberg"
                # Allow: word characters, hyphens, spaces, uppercase/lowercase
                hauptstadt_match = re.search(
                    r'hauptstadt\s+(?:von|der|des)?\s+([A-ZÄÖÜ][a-zäöüß\-]+(?:\s+[A-ZÄÖÜ][a-zäöüß\-]+)*)',
                    claim,
                    re.IGNORECASE
                )
                if hauptstadt_match:
                    claimed_location = hauptstadt_match.group(1).strip()
            else:
                # Try to extract from claim pattern (also allow hyphens)
                location_match = re.search(
                    r'(?:liegt|befindet\s+sich|gehört|ist)\s+in\s+([A-ZÄÖÜ][a-zäöüß\-]+(?:\s+[A-ZÄÖÜ][a-zäöüß\-]+)*)',
                    claim,
                    re.IGNORECASE
                )
                if location_match:
                    claimed_location = location_match.group(1).strip()
        
        # CRITICAL: Ensure claimed_location is a string, not None
        if not claimed_location or not isinstance(claimed_location, str):
            return decision
        
        logger.info(f"GEO_VERIFIER: Processing location claim for '{parsed_claim.subject}' in '{claimed_location}'")
        
        # CRITICAL: For "Hauptstadt" claims, we need to check if the claimed location is a German state (Bundesland)
        # If it is, we should NOT use P17 (country) comparison, but instead check P1376 (capital of) or P131 (administrative entity)
        # For now, we skip the geo verifier for Hauptstadt claims of German states to avoid false REFUTED
        german_states = {
            "baden-württemberg", "bayern", "berlin", "brandenburg", "bremen",
            "hamburg", "hessen", "mecklenburg-vorpommern", "niedersachsen",
            "nordrhein-westfalen", "rheinland-pfalz", "saarland", "sachsen",
            "sachsen-anhalt", "schleswig-holstein", "thüringen",
        }
        
        # Normalize claimed location (handles None case)
        normalized_result = self._normalize_location_name(claimed_location)
        if not normalized_result:
            # If normalization returns empty string, skip geo verification
            logger.info(f"GEO_VERIFIER: Skipping geo verification (could not normalize location '{claimed_location}')")
            return decision
        
        claimed_location_normalized_lower = normalized_result.lower()
        
        # CRITICAL: If the claimed location is a German state (Bundesland), we should NOT use P17 (country) comparison
        # P17 returns the country (Deutschland), not the state. For state-level claims, we need P131 (administrative entity).
        # For now, skip geo verification entirely for German state claims and let evidence-based NLI handle it.
        # This applies to BOTH Hauptstadt claims AND normal location claims ("liegt in Bayern").
        if claimed_location_normalized_lower in german_states:
            logger.info(
                f"GEO_VERIFIER: Skipping geo verification for German state claim '{claimed_location}' "
                f"(P17 country comparison would be incorrect - should use P131 for state-level verification, "
                f"but falling back to evidence-based NLI instead)"
            )
            return decision
        
        # Step 1: Try Wikidata P17 (country) from metadata (if pre-fetched)
        wikidata_result = self._get_wikidata_from_metadata(decision)
        if wikidata_result:
            country_qid = wikidata_result.get("country_qid")
            country_label = wikidata_result.get("country_label")
            
            # CRITICAL: If country_label is an administrative entity (e.g., "Mark Brandenburg"), 
            # we need to get the actual country. For now, use known mappings.
            # TODO: This should ideally use recursive lookup, but for now use mappings
            administrative_to_country = {
                "mark brandenburg": "Deutschland",
                "brandenburg": "Deutschland",
                "bayern": "Deutschland",
                "baden-württemberg": "Deutschland",
                "niedersachsen": "Deutschland",
                "nordrhein-westfalen": "Deutschland",
                "hessen": "Deutschland",
                "sachsen": "Deutschland",
                "thüringen": "Deutschland",
                "rheinland-pfalz": "Deutschland",
                "saarland": "Deutschland",
                "schleswig-holstein": "Deutschland",
                "mecklenburg-vorpommern": "Deutschland",
                "sachsen-anhalt": "Deutschland",
                "bremen": "Deutschland",
                "hamburg": "Deutschland",
                "berlin": "Deutschland",
                # UK and constituent countries
                "england": "Vereinigtes Königreich",
                "united kingdom": "Vereinigtes Königreich",
                "vereinigtes königreich": "Vereinigtes Königreich",
                "großbritannien": "Vereinigtes Königreich",
                "great britain": "Vereinigtes Königreich",
            }
            
            # Check if country_label is an administrative entity or needs mapping
            country_label_lower = country_label.lower() if country_label else ""
            if country_label_lower in administrative_to_country:
                original_label = country_label
                country_label = administrative_to_country[country_label_lower]
                logger.info(
                    f"GEO_VERIFIER: Mapped '{original_label}' to country '{country_label}' "
                    f"(original wikidata: '{wikidata_result.get('country_label')}')"
                )
            
            if country_qid and country_label:
                # Normalize location names for comparison
                claimed_location_normalized = self._normalize_location_name(claimed_location)
                country_label_normalized = self._normalize_location_name(country_label)
                
                # Check if claimed location matches actual country
                if claimed_location_normalized == country_label_normalized:
                    decision.label = FactLabel.SUPPORTED
                    decision.add_note(
                        f"GEO_VERIFIER: SUPPORTED (subject='{parsed_claim.subject}' in '{country_label}', "
                        f"matches claimed '{claimed_location}')"
                    )
                else:
                    decision.label = FactLabel.REFUTED
                    decision.add_note(
                        f"GEO_VERIFIER: REFUTED (subject='{parsed_claim.subject}' in '{country_label}', "
                        f"not in claimed '{claimed_location}')"
                    )
                    
                    # Set correction data for response formatter
                    decision.metadata = decision.metadata or {}
                    decision.metadata["geo_correction"] = {
                        "type": "geo",
                        "subject": parsed_claim.subject,
                        "true_value": country_label,
                        "claimed_value": claimed_location,
                        "relation": "located_in_country",
                        "source": "wikidata",
                    }
                
                return decision
        
        # Step 2: Fallback to evidence extraction (if Wikidata not available)
        # This is a fallback - ideally Wikidata should be used
        if decision.best_evidence:
            snippet = decision.best_evidence.snippet
            snippet_lower = snippet.lower()
            
            # Try to extract location from evidence
            location_patterns = [
                r'(?:liegt|befindet\s+sich|ist)\s+in\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                r'(?:in|auf)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s+(?:liegt|befindet)',
            ]
            
            for pattern in location_patterns:
                match = re.search(pattern, snippet, re.IGNORECASE)
                if match:
                    evidence_location = match.group(1).strip()
                    claimed_location_normalized = self._normalize_location_name(claimed_location)
                    evidence_location_normalized = self._normalize_location_name(evidence_location)
                    
                    if evidence_location_normalized != claimed_location_normalized:
                        decision.label = FactLabel.REFUTED
                        decision.add_note(
                            f"GEO_VERIFIER: REFUTED (evidence says '{evidence_location}', "
                            f"not claimed '{claimed_location}')"
                        )
                        
                        # Set correction data
                        decision.metadata = decision.metadata or {}
                        decision.metadata["geo_correction"] = {
                            "type": "geo",
                            "subject": parsed_claim.subject,
                            "true_value": evidence_location,
                            "claimed_value": claimed_location,
                            "relation": "located_in_country",
                            "source": "wikipedia",
                        }
                        return decision
        
        return decision
    
    def _get_wikidata_from_metadata(
        self, decision: Decision
    ) -> Optional[Dict[str, Any]]:
        """
        Get pre-fetched Wikidata geo data from Decision metadata.
        
        Metadata key: "wikidata_geo"
        """
        if decision.metadata and "wikidata_geo" in decision.metadata:
            return decision.metadata["wikidata_geo"]
        return None
    
    @staticmethod
    def _normalize_location_name(location: str) -> str:
        """
        Normalize location name for comparison.
        
        Removes articles, converts to lowercase, handles common variants.
        Preserves hyphens for compound names like "Baden-Württemberg".
        """
        if not location:
            return ""
        
        normalized = location.lower().strip()
        
        # Remove articles
        articles = ["der", "die", "das", "ein", "eine"]
        words = normalized.split()
        if words and words[0] in articles:
            normalized = " ".join(words[1:])
        
        # Common location name mappings (apply after article removal)
        location_mappings = {
            "deutschland": "deutschland",
            "germany": "deutschland",
            "frankreich": "frankreich",
            "france": "frankreich",
            "italien": "italien",
            "italy": "italien",
            "spanien": "spanien",
            "spain": "spanien",
            "portugal": "portugal",
            "kroatien": "kroatien",
            "croatia": "kroatien",
            # UK and constituent countries
            "england": "vereinigtes königreich",  # England -> UK
            "united kingdom": "vereinigtes königreich",
            "vereinigtes königreich": "vereinigtes königreich",
            "großbritannien": "vereinigtes königreich",
            "great britain": "vereinigtes königreich",
            # German states (preserve hyphenated names)
            "baden-württemberg": "baden-württemberg",
            "rheinland-pfalz": "rheinland-pfalz",
            "mecklenburg-vorpommern": "mecklenburg-vorpommern",
            "nordrhein-westfalen": "nordrhein-westfalen",
            "sachsen-anhalt": "sachsen-anhalt",
            "schleswig-holstein": "schleswig-holstein",
        }
        
        return location_mappings.get(normalized, normalized)
