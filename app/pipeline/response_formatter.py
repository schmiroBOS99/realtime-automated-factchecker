"""
Response formatter for natural language answers.

Generates user-friendly answers from fact-check results.
"""
import html
import re
import logging
from typing import Dict, Any, Optional, Tuple
from app.pipeline.models import FactLabel

logger = logging.getLogger(__name__)

# Profession keywords (from claim_parser.py)
SPORT_PROFESSION_KEYWORDS = {
    "tennis": ["tennisspieler", "tennisspielerin", "tennisprofi", "tennis-spieler", "tennis"],
    "cycling": ["radrennfahrer", "radrennfahrerin", "radprofi", "radsportler", "radsportlerin", "radfahrer", "radsport"],
    "football": ["fußballer", "fussballer", "fußballspieler", "fussballspieler", "fußballerin", "fussballerin", "soccer", "footballer"],
    "skiing": ["skifahrer", "skifahrerin", "skiläufer", "skiläuferin", "ski-fahrer"],
}

ROLE_KEYWORDS = [
    "bundeskanzler", "bundeskanzlerin", "bundespräsident", "bundespräsidentin",
    "präsident", "präsidentin", "kanzler", "kanzlerin", "minister", "ministerin",
]


class ResponseFormatter:
    """Formats fact-check results into natural language answers."""
    
    def format_answer(
        self,
        claim: str,
        label: str,
        parsed_claim: Dict[str, Any],
        best_evidence: Optional[Dict[str, Any]] = None,
        comparison_details: Optional[Dict[str, Any]] = None,
        population_correction: Optional[Dict[str, Any]] = None,
        geo_correction: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format fact-check result into a natural language answer.
        
        Args:
            claim: Original claim text
            label: FactLabel value (SUPPORTED, REFUTED, NOT_ENOUGH_INFO)
            parsed_claim: Parsed claim dictionary
            best_evidence: Best evidence dictionary (optional)
            comparison_details: Comparison details dictionary (optional)
            population_correction: Population correction data (optional)
            geo_correction: Geo location correction data (optional)
            
        Returns:
            Natural language answer string
        """
        if label == FactLabel.SUPPORTED.value:
            return self._format_supported(claim, parsed_claim, best_evidence, comparison_details, population_correction, geo_correction)
        elif label == FactLabel.REFUTED.value:
            return self._format_refuted(claim, parsed_claim, best_evidence, comparison_details, population_correction, geo_correction)
        else:
            return self._format_not_enough_info(claim, parsed_claim)
    
    def _format_supported(
        self,
        claim: str,
        parsed_claim: Dict[str, Any],
        best_evidence: Optional[Dict[str, Any]],
        comparison_details: Optional[Dict[str, Any]],
        population_correction: Optional[Dict[str, Any]] = None,
        geo_correction: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format SUPPORTED claim into natural language answer."""
        # Special handling for comparison claims
        if comparison_details and "x_value" in comparison_details and "y_value" in comparison_details:
            x_entity = comparison_details.get("x_entity", "")
            y_entity = comparison_details.get("y_entity", "")
            x_value = comparison_details.get("x_value")
            y_value = comparison_details.get("y_value")
            comparison_word = comparison_details.get("comparison_word", "")
            attribute = comparison_details.get("attribute", "")
            
            if x_value is not None and y_value is not None:
                # Format values (remove decimal if .0, add units if available)
                x_value_str = self._format_number(x_value, attribute)
                y_value_str = self._format_number(y_value, attribute)
                
                # Extract articles from original claim
                x_entity_with_article = self._add_article_to_entity(claim, x_entity, is_first=True)
                y_entity_with_article = self._add_article_to_entity(claim, y_entity, is_first=False)
                
                return f"Ja, {x_entity_with_article} ist mit {x_value_str} {comparison_word} als {y_entity_with_article} mit {y_value_str}."
        
        # Default SUPPORTED answer
        # Try to extract subject and predicate/object from parsed_claim
        subject = parsed_claim.get("subject", "")
        object_claim = parsed_claim.get("object", "")
        role = parsed_claim.get("role", "")
        
        # Build answer from claim structure
        if subject and role:
            # Political office claim
            # CRITICAL: Capitalize role (e.g., "präsident" -> "Präsident")
            role_capitalized = role.capitalize() if role else ""
            
            # Extract verb from claim to preserve tense ("ist" vs "war")
            claim_lower = claim.lower()
            verb = "ist"
            if " war " in claim_lower or claim_lower.startswith("war "):
                verb = "war"
            elif " ist " in claim_lower or claim_lower.startswith("ist "):
                verb = "ist"
            
            if object_claim:
                # Handle genitive forms (e.g., "Frankreichs" -> "Frankreich")
                object_clean = object_claim.rstrip("s") if object_claim.endswith("s") else object_claim
                return f"Ja, {subject} {verb} {role_capitalized} von {object_clean}."
            else:
                return f"Ja, {subject} {verb} {role_capitalized}."
        elif subject and object_claim:
            # Subject-object claim
            # Extract verb from claim (simple: look for "ist", "war", "hat", etc.)
            claim_lower = claim.lower()
            if "ist" in claim_lower or "war" in claim_lower:
                verb = "ist" if "ist" in claim_lower else "war"
                return f"Ja, {subject} {verb} {object_claim}."
            else:
                return f"Ja, {subject} {object_claim}."
        else:
            # Fallback: simple confirmation
            return f"Ja, {claim}."
    
    def _format_refuted(
        self,
        claim: str,
        parsed_claim: Dict[str, Any],
        best_evidence: Optional[Dict[str, Any]],
        comparison_details: Optional[Dict[str, Any]],
        population_correction: Optional[Dict[str, Any]] = None,
        geo_correction: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format REFUTED claim into natural language answer."""
        subject = parsed_claim.get("subject", "")
        claim_lower = claim.lower()
        
        # CRITICAL: If subject is just an article (e.g., "die"), extract the actual noun from claim
        # This happens when the parser extracts only the article as subject
        if subject and subject.lower() in ["der", "die", "das", "ein", "eine"]:
            # Try to find the noun after the article in the claim
            # Pattern: "die Erdbeere" -> extract "Erdbeere"
            claim_words = claim.split()
            for i, word in enumerate(claim_words):
                if word.lower() == subject.lower() and i < len(claim_words) - 1:
                    next_word = claim_words[i + 1]
                    if next_word[0].isupper():
                        # Found the actual noun after the article
                        subject = f"{word} {next_word}"
                        break
        
        # Try to preserve article from original claim if subject doesn't have one
        # Extract article from claim (e.g., "Die EU" -> "Die", "Der Rhein" -> "Der")
        if subject and not subject.lower().startswith(("der ", "die ", "das ", "ein ", "eine ")):
            # Try to find article in original claim before subject
            subject_lower = subject.lower()
            claim_words = claim.split()
            for i, word in enumerate(claim_words):
                if word.lower() == subject_lower or word.lower().startswith(subject_lower.split()[0].lower()):
                    # Found subject, check previous word for article
                    if i > 0:
                        prev_word = claim_words[i-1].lower()
                        if prev_word in ["der", "die", "das", "ein", "eine", "einen", "einem", "einer"]:
                            subject = f"{claim_words[i-1]} {subject}"
                            break
        
        # Special handling for geo location claims: prefer structured correction (Wikidata) over snippet extraction
        is_geo_claim = any(kw in claim_lower for kw in ["liegt in", "befindet sich in", "gehört zu", "ist in", "fließt durch", "umkreist"])
        is_hauptstadt_claim = "hauptstadt" in claim_lower and "ist" in claim_lower
        
        # CRITICAL: Handle "Hauptstadt" claims FIRST using geo_correction (before other patterns)
        if is_hauptstadt_claim and geo_correction and geo_correction.get("true_value"):
            true_location = geo_correction.get("true_value")
            claimed_location = geo_correction.get("claimed_value", "")
            subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
            
            if claimed_location:
                return f"Nein, {subject_with_article} ist nicht die Hauptstadt von {claimed_location}, sondern die Hauptstadt von {true_location}."
            return f"Nein, {subject_with_article} ist nicht die Hauptstadt von {claimed_location}."
        
        # CRITICAL: Handle "Hauptstadt" claims even without geo_correction (for Bundesländer cases)
        # Extract claimed location from claim and provide grammatically correct negation
        if is_hauptstadt_claim and subject:
            # Try to extract claimed location from claim pattern: "Hauptstadt von X"
            hauptstadt_match = re.search(
                r'hauptstadt\s+(?:von|der|des)?\s+([A-ZÄÖÜ][a-zäöüß\-]+(?:\s+[A-ZÄÖÜ][a-zäöüß\-]+)*)',
                claim,
                re.IGNORECASE
            )
            if hauptstadt_match:
                claimed_location = hauptstadt_match.group(1).strip()
                subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                # CRITICAL: Grammatically correct negation - "Hauptstadt" is feminine, use "nicht die Hauptstadt"
                return f"Nein, {subject_with_article} ist nicht die Hauptstadt von {claimed_location}."
            else:
                # No location found in claim, provide simple grammatically correct negation
                subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                # CRITICAL: Use "nicht die Hauptstadt" (not "keine Hauptstadt" or "kein Hauptstadt")
                return f"Nein, {subject_with_article} ist nicht die Hauptstadt."
        
        # Handle geo location claims (liegt in, fließt durch, etc.)
        if is_geo_claim and geo_correction and geo_correction.get("true_value"):
            true_location = geo_correction.get("true_value")
            claimed_location = geo_correction.get("claimed_value", "")
            subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
            
            if claimed_location:
                return f"Nein, {subject_with_article} liegt nicht in {claimed_location}, sondern in {true_location}."
            return f"Nein, {subject_with_article} liegt in {true_location}."
        
        # Special handling for population claims: prefer structured correction (Wikidata) over snippet extraction
        is_population_claim = any(kw in claim_lower for kw in ["einwohner", "bevölkerung", "population"])
        
        # CRITICAL: Always use structured correction if available (from Wikidata or Wikipedia extraction)
        # This prevents extracting wrong numbers from random snippets
        if is_population_claim and population_correction and population_correction.get("correct_value") is not None:
            correct_value = population_correction.get("correct_value")
            correct_str = population_correction.get("correct_value_str")
            year = population_correction.get("correct_value_year")
            source = population_correction.get("correct_value_source", "unknown")
            
            # Format correct_str if not provided
            if not correct_str and correct_value:
                from app.pipeline.postprocessors.population_verifier import format_population
                correct_str = format_population(correct_value)
            
            logger.info(f"USING_CORRECTION: source={source} value={correct_value} year={year} correct_str={correct_str}")

            # Try to extract claimed number from claim (optional)
            claimed_number = self._extract_population_from_claim(claim)
            if claimed_number:
                claimed_str = self._format_millions_de(claimed_number)
                if year:
                    return f"Nein, {subject} hat nicht {claimed_str} Millionen Einwohner, sondern etwa {correct_str} (Stand: {year})."
                return f"Nein, {subject} hat nicht {claimed_str} Millionen Einwohner, sondern etwa {correct_str}."
            if year:
                return f"Nein, {subject} hat etwa {correct_str} Einwohner (Stand: {year})."
            return f"Nein, {subject} hat etwa {correct_str} Einwohner."
        
        # Fallback: Only try snippet extraction if no structured correction is available
        # This should rarely happen if population_verifier is working correctly
        if is_population_claim and best_evidence:
            snippet = best_evidence.get("snippet", "")
            title = best_evidence.get("title", "")
            if snippet:
                # Try to extract correct population number from evidence
                pop_details = self._extract_population_details_from_evidence(snippet, subject, title)
                if pop_details:
                    # Extract claimed number from claim
                    claimed_number = self._extract_population_from_claim(claim)
                    if claimed_number:
                        claimed_str = self._format_millions_de(claimed_number)
                        correct_str = self._format_millions_de(pop_details["millions"])
                        raw_str = pop_details.get("raw_str")
                        if raw_str:
                            return f"Nein, {subject} hat nicht {claimed_str} Millionen Einwohner, sondern etwa {correct_str} Millionen ({raw_str})."
                        return f"Nein, {subject} hat nicht {claimed_str} Millionen Einwohner, sondern etwa {correct_str} Millionen."
                    else:
                        correct_str = self._format_millions_de(pop_details["millions"])
                        raw_str = pop_details.get("raw_str")
                        if raw_str:
                            return f"Nein, {subject} hat etwa {correct_str} Millionen Einwohner ({raw_str})."
                        return f"Nein, {subject} hat etwa {correct_str} Millionen Einwohner."
        
        # Special handling for comparison claims
        if comparison_details and "x_value" in comparison_details and "y_value" in comparison_details:
            x_entity = comparison_details.get("x_entity", "")
            y_entity = comparison_details.get("y_entity", "")
            x_value = comparison_details.get("x_value")
            y_value = comparison_details.get("y_value")
            comparison_word = comparison_details.get("comparison_word", "")
            attribute = comparison_details.get("attribute", "")
            
            if x_value is not None and y_value is not None:
                # Format values
                x_value_str = self._format_number(x_value, attribute)
                y_value_str = self._format_number(y_value, attribute)
                
                # Extract articles from original claim
                x_entity_with_article = self._add_article_to_entity(claim, x_entity, is_first=True)
                y_entity_with_article = self._add_article_to_entity(claim, y_entity, is_first=False)
                
                # Determine correct comparison word based on actual values
                # If claim says "X ist höher als Y" but x_value < y_value (REFUTED),
                # then Y is actually höher, not niedriger
                comparison_word_lower = comparison_word.lower()
                if comparison_word_lower in ["höher", "größer", "länger", "mehr"]:
                    # Claim says X is more, but it's actually less → Y is more
                    if x_value < y_value:
                        correct_comparison = comparison_word  # Y is höher/größer/länger/mehr
                    else:
                        # This shouldn't happen (REFUTED but x > y?), use opposite
                        opposite_map = {
                            "höher": "niedriger",
                            "größer": "kleiner",
                            "länger": "kürzer",
                            "mehr": "weniger",
                        }
                        correct_comparison = opposite_map.get(comparison_word_lower, comparison_word)
                elif comparison_word_lower in ["niedriger", "kleiner", "kürzer", "weniger"]:
                    # Claim says X is less, but it's actually more → Y is less
                    if x_value > y_value:
                        correct_comparison = comparison_word  # Y is niedriger/kleiner/kürzer/weniger
                    else:
                        # Use opposite
                        opposite_map = {
                            "niedriger": "höher",
                            "kleiner": "größer",
                            "kürzer": "länger",
                            "weniger": "mehr",
                        }
                        correct_comparison = opposite_map.get(comparison_word_lower, comparison_word)
                else:
                    # Fallback: use comparison_word as-is
                    correct_comparison = comparison_word
                
                return f"Nein, {x_entity_with_article} ist mit {x_value_str} nicht {comparison_word} als {y_entity_with_article} mit {y_value_str}, sondern {y_entity_with_article} ist {correct_comparison}."
        
        # Try to extract correct information from evidence
        if subject and best_evidence:
            snippet = best_evidence.get("snippet", "")
            if snippet:
                # Extract claimed role/profession from claim
                claimed_role = parsed_claim.get("role", "")
                claimed_sport = parsed_claim.get("sport", "")
                
                # Try to extract correct profession/role from evidence
                correct_profession = self._extract_correct_profession(
                    snippet, subject, claimed_role, claimed_sport
                )
                
                if correct_profession:
                    # Determine tense from claim
                    claim_lower = claim.lower()
                    verb = "war" if "war" in claim_lower else "ist"
                    return f"Nein, {subject} {verb} {correct_profession}."
                
                # If we have a claimed role/profession, use negation
                if claimed_role or claimed_sport:
                    claim_lower = claim.lower()
                    verb = "war" if "war" in claim_lower else "ist"
                    
                    # Extract the profession/role word from claim
                    if claimed_role:
                        profession_word = claimed_role
                    elif claimed_sport:
                        # Try to find profession keyword from sport
                        profession_word = self._get_profession_from_sport(claimed_sport, claim_lower)
                    else:
                        profession_word = None
                    
                    if profession_word:
                        return f"Nein, {subject} {verb} kein {profession_word}."
        
        # Default REFUTED answer with negation if subject available
        if subject:
            claim_lower = claim.lower()
            # Determine verb from claim (fallback to "ist" if not found)
            verb = "war" if "war" in claim_lower else ("ist" if "ist" in claim_lower else ("hat" if "hat" in claim_lower else "ist"))
            
            # Special handling for population claims FIRST - they need correct formatting
            is_population_claim = any(kw in claim_lower for kw in ["einwohner", "bevölkerung", "population"])
            if is_population_claim and population_correction and population_correction.get("correct_value") is not None:
                correct_str = population_correction.get("correct_value_str") or str(population_correction.get("correct_value"))
                year = population_correction.get("correct_value_year")
                source = population_correction.get("correct_value_source")
                logger.info(f"USING_CORRECTION: source={source} value={population_correction.get('correct_value')} year={year}")
                claimed_number = self._extract_population_from_claim(claim)
                if claimed_number:
                    claimed_str = self._format_millions_de(claimed_number)
                    if year:
                        return f"Nein, {subject} hat nicht {claimed_str} Millionen Einwohner, sondern etwa {correct_str} (Stand: {year})."
                    return f"Nein, {subject} hat nicht {claimed_str} Millionen Einwohner, sondern etwa {correct_str}."
                if year:
                    return f"Nein, {subject} hat etwa {correct_str} Einwohner (Stand: {year})."
                return f"Nein, {subject} hat etwa {correct_str} Einwohner."
            if is_population_claim and best_evidence:
                snippet = best_evidence.get("snippet", "")
                title = best_evidence.get("title", "")
                if snippet:
                    # Try to extract correct population number from evidence
                    pop_details = self._extract_population_details_from_evidence(snippet, subject, title)
                    if pop_details:
                        # Extract claimed number from claim
                        claimed_number = self._extract_population_from_claim(claim)
                        if claimed_number:
                            claimed_str = self._format_millions_de(claimed_number)
                            correct_str = self._format_millions_de(pop_details["millions"])
                            raw_str = pop_details.get("raw_str")
                            if raw_str:
                                return f"Nein, {subject} hat nicht {claimed_str} Millionen Einwohner, sondern etwa {correct_str} Millionen ({raw_str})."
                            return f"Nein, {subject} hat nicht {claimed_str} Millionen Einwohner, sondern etwa {correct_str} Millionen."
                        else:
                            correct_str = self._format_millions_de(pop_details["millions"])
                            raw_str = pop_details.get("raw_str")
                            if raw_str:
                                return f"Nein, {subject} hat etwa {correct_str} Millionen Einwohner ({raw_str})."
                            return f"Nein, {subject} hat etwa {correct_str} Millionen Einwohner."
                    else:
                        # Population not found in evidence, but we can still provide a refuted answer with the claimed number
                        claimed_number = self._extract_population_from_claim(claim)
                        if claimed_number:
                            claimed_str = self._format_millions_de(claimed_number)
                            return f"Nein, {subject} hat nicht {claimed_str} Millionen Einwohner."
            
            # Special handling for specific claim patterns
            object_claim = parsed_claim.get("object", "")
            
            # Pattern 1: "X liegt in Y" / "X fließt durch Y" / "X umkreist Y"
            if "liegt in" in claim_lower or "fließt durch" in claim_lower or "umkreist" in claim_lower:
                if object_claim:
                    # Extract correct location/object from evidence if available
                    if best_evidence:
                        snippet = best_evidence.get("snippet", "")
                        title = best_evidence.get("title", "")
                        snippet_lower = snippet.lower()
                        title_lower = title.lower()
                        
                        # Try to extract correct information from evidence
                        # For "Rom liegt in Deutschland" -> evidence might say "Rom liegt in Italien" or "Rom ist die Hauptstadt Italiens"
                        if "liegt in" in claim_lower:
                            # Multiple patterns to find location information
                            location_patterns = [
                                r'liegt\s+in\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                                r'\bin\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s+(?:liegt|befindet)',
                                r'(?:liegt|befindet\s+sich)\s+in\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                                r'ist\s+(?:die|der|das)?\s*hauptstadt\s+(?:von|der|des)?\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',  # "Rom ist die Hauptstadt Italiens"
                                r'hauptstadt\s+(?:von|der|des)?\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',  # "Hauptstadt Italiens"
                            ]
                            
                            correct_location = None
                            for pattern in location_patterns:
                                match = re.search(pattern, snippet, re.IGNORECASE)
                                if match:
                                    potential_location = match.group(1).strip().rstrip('.,!?')
                                    # Normalize common country names
                                    country_mappings = {
                                        "italien": "Italien",
                                        "italy": "Italien",
                                        "deutschland": "Deutschland",
                                        "germany": "Deutschland",
                                        "frankreich": "Frankreich",
                                        "france": "Frankreich",
                                    }
                                    potential_location_normalized = country_mappings.get(potential_location.lower(), potential_location)
                                    object_normalized = country_mappings.get(object_claim.lower(), object_claim)
                                    
                                    # Only use if it's different from claimed location and is a known country
                                    if (potential_location_normalized.lower() != object_normalized.lower() and 
                                        potential_location_normalized.lower() in [k.lower() for k in country_mappings.values()]):
                                        correct_location = potential_location_normalized
                                        break
                            
                            # If no location found in snippet, try to infer from known facts
                            if not correct_location:
                                # Known city-country mappings
                                city_country_mappings = {
                                    "rom": "Italien",
                                    "rome": "Italien",
                                    "berlin": "Deutschland",
                                    "paris": "Frankreich",
                                    "madrid": "Spanien",
                                    "london": "Großbritannien",
                                    "wien": "Österreich",
                                }
                                subject_lower = subject.lower()
                                if subject_lower in city_country_mappings:
                                    inferred_country = city_country_mappings[subject_lower]
                                    object_normalized = country_mappings.get(object_claim.lower(), object_claim)
                                    if inferred_country.lower() != object_normalized.lower():
                                        correct_location = inferred_country
                            
                            if correct_location:
                                subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                                object_normalized = country_mappings.get(object_claim.lower(), object_claim)
                                return f"Nein, {subject_with_article} liegt nicht in {object_normalized}, sondern in {correct_location}."
                            
                            # Fallback: simple negation with object
                            subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                            return f"Nein, {subject_with_article} liegt nicht in {object_claim}."
                            
                        elif "fließt durch" in claim_lower or "durchfließt" in claim_lower:
                            # Extract location after "durch" or "durchfließt"
                            location_patterns = [
                                r'(?:fließt\s+durch|durchfließt)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                                r'\bdurch\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\b',
                            ]
                            
                            for pattern in location_patterns:
                                match = re.search(pattern, snippet, re.IGNORECASE)
                                if match:
                                    correct_location = match.group(1).strip().rstrip('.,!?')
                                    if correct_location.lower() != object_claim.lower():
                                        subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                                        return f"Nein, {subject_with_article} fließt nicht durch {object_claim}, sondern durch {correct_location}."
                            
                            # Fallback: simple negation with object
                            subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                            return f"Nein, {subject_with_article} fließt nicht durch {object_claim}."
                            
                        elif "umkreist" in claim_lower or "kreist um" in claim_lower:
                            # Extract subject and object from evidence snippet
                            # Pattern: "X umkreist Y" or "X kreist um Y"
                            umkreist_match = re.search(r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s+(?:umkreist|kreist\s+um)\s+(?:der|die|das|ein|eine)?\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)', snippet, re.IGNORECASE)
                            if umkreist_match:
                                evidence_subject = umkreist_match.group(1).strip()
                                evidence_object = umkreist_match.group(2).strip()
                                
                                # Check if the relationship is reversed
                                # Claim: "Sonne umkreist Erde" (subject=Sonne, object=Erde)
                                # Evidence: "Erde umkreist Sonne" (evidence_subject=Erde, evidence_object=Sonne)
                                if "umkreist" in claim_lower:
                                    if (evidence_subject.lower() == object_claim.lower() and 
                                        evidence_object.lower() == subject.lower()):
                                        # Relationship is reversed
                                        subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                                        object_with_article = self._add_article_to_entity(claim, object_claim, is_first=False)
                                        return f"Nein, {subject_with_article} umkreist nicht {object_with_article}, sondern {object_with_article} umkreist {subject_with_article}."
                                    elif evidence_object.lower() != object_claim.lower():
                                        # Different object mentioned
                                        subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                                        evidence_object_with_article = self._add_article_to_entity(claim, evidence_object, is_first=False)
                                        return f"Nein, {subject_with_article} umkreist nicht {object_claim}, sondern {evidence_object_with_article}."
                            
                            # Fallback: simple negation with object
                            subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                            object_with_article = self._add_article_to_entity(claim, object_claim, is_first=False)
                            return f"Nein, {subject_with_article} umkreist nicht {object_with_article}."
                    
                    # Fallback: simple negation with object (no evidence available)
                    subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                    if "liegt in" in claim_lower:
                        return f"Nein, {subject_with_article} liegt nicht in {object_claim}."
                    elif "fließt durch" in claim_lower:
                        return f"Nein, {subject_with_article} fließt nicht durch {object_claim}."
                    elif "umkreist" in claim_lower:
                        object_with_article = self._add_article_to_entity(claim, object_claim, is_first=False)
                        return f"Nein, {subject_with_article} umkreist nicht {object_with_article}."
            
            # Pattern 2: "X hat Y Mitgliedstaaten" / "X hat Y Einwohner"
            if "hat" in claim_lower and ("mitgliedstaaten" in claim_lower or "mitglieder" in claim_lower):
                # Extract number from claim
                number_match = re.search(r'(\d+)\s+mitgliedstaaten', claim_lower)
                if number_match:
                    claimed_number = number_match.group(1)
                    # Try to extract correct number from evidence
                    if best_evidence:
                        snippet = best_evidence.get("snippet", "")
                        # Look for "X Mitgliedstaaten" or "X Mitglieder" in evidence
                        correct_match = re.search(r'(\d+)\s+mitgliedstaaten', snippet.lower())
                        if correct_match:
                            correct_number = correct_match.group(1)
                            return f"Nein, {subject} hat nicht {claimed_number} Mitgliedstaaten, sondern {correct_number}."
                    return f"Nein, {subject} hat nicht {claimed_number} Mitgliedstaaten."
            
            # Pattern 3: "X ist die Hauptstadt von Y" / "X ist die Hauptstadt Y"
            # CRITICAL: Skip this pattern if geo_correction is already used (handled above)
            if (not is_hauptstadt_claim or not geo_correction) and "ist" in claim_lower and "hauptstadt" in claim_lower and object_claim:
                # Normalize common country names
                country_mappings = {
                    "deutschland": "Deutschland",
                    "germany": "Deutschland",
                    "frankreich": "Frankreich",
                    "france": "Frankreich",
                    "italien": "Italien",
                    "italy": "Italien",
                }
                
                correct_country = None
                
                if best_evidence:
                    snippet = best_evidence.get("snippet", "")
                    snippet_lower = snippet.lower()
                    
                    # Look for "Hauptstadt von" or "Hauptstadt" in evidence
                    # Pattern: "X ist die Hauptstadt von Y" or "X ist Hauptstadt von Y"
                    hauptstadt_patterns = [
                        r'(?:ist|war)\s+(?:die|der|das)?\s*hauptstadt\s+(?:von|der|des)?\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                        r',\s*(?:die|der|das)?\s*hauptstadt\s+(?:von|der|des)?\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                        r'hauptstadt\s+(?:von|der|des)?\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                        r'\b([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s+(?:ist|war)\s+(?:die|der|das)?\s*hauptstadt',  # "Deutschland ist die Hauptstadt" (reversed)
                    ]
                    
                    for pattern in hauptstadt_patterns:
                        hauptstadt_match = re.search(pattern, snippet, re.IGNORECASE)
                        if hauptstadt_match:
                            potential_country = hauptstadt_match.group(1).strip().rstrip('.,!?')
                            potential_country_normalized = country_mappings.get(potential_country.lower(), potential_country)
                            object_normalized = country_mappings.get(object_claim.lower(), object_claim)
                            
                            # Only use if it's different from claimed country and is a known country
                            if (potential_country_normalized.lower() != object_normalized.lower() and 
                                potential_country_normalized.lower() in [k.lower() for k in country_mappings.values()]):
                                correct_country = potential_country_normalized
                                break
                
                # If no country found in evidence, try to infer from known facts
                if not correct_country:
                    # Known city-country mappings for capitals
                    city_country_mappings = {
                        "berlin": "Deutschland",
                        "paris": "Frankreich",
                        "rom": "Italien",
                        "rome": "Italien",
                        "madrid": "Spanien",
                        "london": "Großbritannien",
                        "wien": "Österreich",
                    }
                    subject_lower = subject.lower()
                    if subject_lower in city_country_mappings:
                        inferred_country = city_country_mappings[subject_lower]
                        object_normalized = country_mappings.get(object_claim.lower(), object_claim)
                        if inferred_country.lower() != object_normalized.lower():
                            correct_country = inferred_country
                
                if correct_country:
                    subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                    object_normalized = country_mappings.get(object_claim.lower(), object_claim)
                    return f"Nein, {subject_with_article} ist nicht die Hauptstadt von {object_normalized}, sondern die Hauptstadt von {correct_country}."
                
                # Fallback: simple negation
                subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                return f"Nein, {subject_with_article} ist nicht die Hauptstadt von {object_claim}."
            
            # Pattern 4: "X ist der höchste Berg in Y" / "X ist der höchste Berg Y"
            if "ist" in claim_lower and "höchste berg" in claim_lower and "in" in claim_lower:
                # Extract location from claim (after "in")
                location_match = re.search(r'in\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)', claim, re.IGNORECASE)
                claimed_location = location_match.group(1).strip() if location_match else ""
                
                if best_evidence and claimed_location:
                    snippet = best_evidence.get("snippet", "")
                    snippet_lower = snippet.lower()
                    
                    # Common location name mappings
                    location_mappings = {
                        "asien": "Asien",
                        "asia": "Asien",
                        "europa": "Europa",
                        "europe": "Europa",
                        "afrika": "Afrika",
                        "africa": "Afrika",
                        "südamerika": "Südamerika",
                        "south america": "Südamerika",
                        "nordamerika": "Nordamerika",
                        "north america": "Nordamerika",
                    }
                    
                    # Look for location information in evidence
                    # Pattern: "in Y" or "liegt in Y" or "befindet sich in Y" or "ist in Y"
                    location_patterns = [
                        r'(?:liegt|befindet\s+sich|ist|fließt)\s+in\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                        r'(?:in|auf)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s+(?:liegt|befindet)',
                        r'\bin\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\b',  # Simple "in Y" pattern
                    ]
                    
                    for pattern in location_patterns:
                        location_match = re.search(pattern, snippet, re.IGNORECASE)
                        if location_match:
                            correct_location = location_match.group(1).strip().rstrip('.,!?')
                            correct_location_normalized = location_mappings.get(correct_location.lower(), correct_location)
                            
                            if correct_location_normalized.lower() != claimed_location.lower():
                                subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                                return f"Nein, {subject_with_article} ist nicht der höchste Berg in {claimed_location}, sondern liegt in {correct_location_normalized}."
                
                # Fallback: simple negation
                if claimed_location:
                    subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                    return f"Nein, {subject_with_article} ist nicht der höchste Berg in {claimed_location}."
            
            # Pattern 5: "X ist ein Fluss in Y" / "X ist ein Fluss Y"
            if "ist" in claim_lower and ("fluss" in claim_lower or "river" in claim_lower) and "in" in claim_lower:
                # Extract location from claim (after "in")
                location_match = re.search(r'in\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)', claim, re.IGNORECASE)
                claimed_location = location_match.group(1).strip() if location_match else ""
                
                if best_evidence and claimed_location:
                    snippet = best_evidence.get("snippet", "")
                    snippet_lower = snippet.lower()
                    
                    # Common location name mappings
                    location_mappings = {
                        "asien": "Asien",
                        "asia": "Asien",
                        "europa": "Europa",
                        "europe": "Europa",
                        "afrika": "Afrika",
                        "africa": "Afrika",
                        "südamerika": "Südamerika",
                        "south america": "Südamerika",
                        "nordamerika": "Nordamerika",
                        "north america": "Nordamerika",
                    }
                    
                    # Look for location information in evidence
                    # Pattern: "in Y" or "liegt in Y" or "durchfließt Y" or "fließt durch Y"
                    location_patterns = [
                        r'(?:liegt|befindet\s+sich|ist|fließt)\s+in\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                        r'(?:durchfließt|fließt\s+durch)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                        r'(?:in|auf)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s+(?:liegt|befindet|fließt)',
                        r'\bin\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\b',  # Simple "in Y" pattern
                    ]
                    
                    for pattern in location_patterns:
                        location_match = re.search(pattern, snippet, re.IGNORECASE)
                        if location_match:
                            correct_location = location_match.group(1).strip().rstrip('.,!?')
                            correct_location_normalized = location_mappings.get(correct_location.lower(), correct_location)
                            
                            claimed_location_normalized = location_mappings.get(claimed_location.lower(), claimed_location)
                            
                            if correct_location_normalized.lower() != claimed_location_normalized.lower():
                                subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                                return f"Nein, {subject_with_article} ist kein Fluss in {claimed_location_normalized}, sondern ein Fluss in {correct_location_normalized}."
                
                # Fallback: simple negation
                if claimed_location:
                    subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                    return f"Nein, {subject_with_article} ist kein Fluss in {claimed_location}."
            
            # Pattern 6: "X ist Y" (e.g., "Mond ist Planet", "Erdbeere ist Nuss")
            if "ist" in claim_lower and object_claim:
                # CRITICAL: Ensure subject is complete (with article if present in claim)
                subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                
                # CRITICAL: Capitalize object_claim properly (first letter uppercase)
                object_claim_capitalized = object_claim[0].upper() + object_claim[1:] if object_claim else ""
                
                # CRITICAL: Determine correct negation word based on object gender
                # Common feminine nouns that need "keine"
                feminine_nouns = ["nuss", "frucht", "beere", "sammelnussfrucht", "hauptstadt", "stadt", "land", "republik"]
                is_feminine = any(fnoun in object_claim.lower() for fnoun in feminine_nouns)
                negation_word = "keine" if is_feminine else "kein"
                
                # Try to extract correct information from evidence
                if best_evidence:
                    snippet = best_evidence.get("snippet", "")
                    # For "Mond ist Planet" -> evidence might say "Mond ist ein Satellit"
                    # Look for "ist ein" or "ist eine" in evidence
                    match = re.search(r'ist\s+(?:ein|eine|der|die|das)\s+([a-zäöüß]+)', snippet.lower())
                    if match:
                        correct_type = match.group(1)
                        # Capitalize correct_type
                        correct_type_capitalized = correct_type[0].upper() + correct_type[1:] if correct_type else ""
                        if correct_type.lower() != object_claim.lower():
                            # Determine article for correct_type (ein/eine)
                            correct_type_is_feminine = any(fnoun in correct_type.lower() for fnoun in feminine_nouns)
                            correct_article = "eine" if correct_type_is_feminine else "ein"
                            return f"Nein, {subject_with_article} ist {negation_word} {object_claim_capitalized}, sondern {correct_article} {correct_type_capitalized}."
                
                # Fallback: simple negation with grammatically correct form
                return f"Nein, {subject_with_article} ist {negation_word} {object_claim_capitalized}."
            
            # Pattern 4: "X besteht aus Y" / "X enthält Y"
            if "besteht aus" in claim_lower or "enthält" in claim_lower:
                if object_claim:
                    # Try to extract correct composition from evidence
                    if best_evidence:
                        snippet = best_evidence.get("snippet", "")
                        # Look for "besteht aus" in evidence
                        if "besteht aus" in snippet.lower():
                            # Extract what it actually consists of
                            match = re.search(r'besteht\s+aus\s+([^.,]+)', snippet, re.IGNORECASE)
                            if match:
                                correct_composition = match.group(1).strip()
                                return f"Nein, {subject} besteht nicht aus {object_claim}, sondern aus {correct_composition}."
                    return f"Nein, {subject} besteht nicht aus {object_claim}."
            
            # Try to extract the main predicate/profession for negation
            # Simple approach: extract words after "war"/"ist"/"hat" for profession claims
            if "war" in claim_lower or "ist" in claim_lower or "hat" in claim_lower:
                # Extract profession/role from claim (words after verb, before end)
                parts = claim_lower.split(verb, 1)
                if len(parts) > 1:
                    predicate_part = parts[1].strip().rstrip(".,!?")
                    # Remove subject if present
                    predicate_part = re.sub(rf'\b{re.escape(subject.lower())}\b', '', predicate_part).strip()
                    # Remove articles and prepositions
                    predicate_part = re.sub(r'^\s*(der|die|das|ein|eine|einen|einem|einer|von|in|durch)\s+', '', predicate_part, flags=re.IGNORECASE).strip()
                    # Remove "kein"/"keine" if already present to avoid double negation
                    predicate_part = re.sub(r'^\s*(kein|keine|keinen|keinem|keiner)\s+', '', predicate_part, flags=re.IGNORECASE).strip()
                    
                    if predicate_part:
                        # CRITICAL: Ensure subject is complete (with article if present in claim)
                        subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                        
                        # CRITICAL: Capitalize predicate_part properly (first letter uppercase)
                        predicate_part_capitalized = predicate_part[0].upper() + predicate_part[1:] if predicate_part else ""
                        
                        # CRITICAL: Check for feminine nouns that need "keine" instead of "kein"
                        # "Hauptstadt", "Nuss", "Frucht", "Beere" are feminine
                        feminine_nouns = ["hauptstadt", "stadt", "land", "republik", "demokratie", "monarchie", 
                                         "nuss", "frucht", "beere", "sammelnussfrucht"]
                        is_feminine = any(fnoun in predicate_part.lower() for fnoun in feminine_nouns)
                        
                        # Check if this is a population claim - use "keine" for plural "Millionen Einwohner"
                        is_population_claim = any(kw in predicate_part for kw in ["millionen einwohner", "millionen bevölkerung", "million einwohner", "million bevölkerung"])
                        
                        # Determine correct negation word
                        if is_population_claim:
                            negation_word = "keine"
                        elif is_feminine:
                            negation_word = "keine"
                        else:
                            negation_word = "kein"
                        
                        return f"Nein, {subject_with_article} {verb} {negation_word} {predicate_part_capitalized}."
            
            # Last resort: generic but grammatically correct answer
            if object_claim:
                # CRITICAL: Ensure subject is complete (with article if present in claim)
                subject_with_article = self._add_article_to_entity(claim, subject, is_first=True)
                # Capitalize object_claim
                object_claim_capitalized = object_claim[0].upper() + object_claim[1:] if object_claim else ""
                return f"Nein, {subject_with_article} {verb} nicht {object_claim_capitalized}."
            return f"Nein, das ist nicht korrekt."
        else:
            return f"Nein, das ist nicht korrekt."
    
    def _extract_correct_profession(
        self,
        snippet: str,
        subject: str,
        claimed_role: Optional[str],
        claimed_sport: Optional[str],
    ) -> Optional[str]:
        """
        Extract correct profession/role from evidence snippet.
        
        Args:
            snippet: Evidence snippet text
            subject: Subject name (e.g., "Lothar Matthäus")
            claimed_role: Claimed role (e.g., "Skifahrer")
            claimed_sport: Claimed sport category
            
        Returns:
            Correct profession/role string if found, None otherwise
        """
        snippet_lower = snippet.lower()
        subject_lower = subject.lower()
        
        # Check if subject is mentioned in snippet
        if subject_lower not in snippet_lower:
            return None
        
        # Get claimed profession keywords for filtering
        claimed_professions = set()
        if claimed_role:
            claimed_professions.add(claimed_role.lower())
        if claimed_sport:
            claimed_profession = self._get_profession_from_sport(claimed_sport, "")
            if claimed_profession:
                claimed_professions.add(claimed_profession.lower())
        
        # Look for profession keywords directly in snippet (more reliable than regex)
        all_professions = []
        for sport_keywords in SPORT_PROFESSION_KEYWORDS.values():
            all_professions.extend(sport_keywords)
        all_professions.extend(ROLE_KEYWORDS)
        
        # Sort by length (longer matches first, e.g., "fußballspieler" before "spieler")
        all_professions_sorted = sorted(all_professions, key=len, reverse=True)
        
        for profession in all_professions_sorted:
            profession_lower = profession.lower()
            if profession_lower in snippet_lower:
                # Skip if it's the claimed profession
                if profession_lower in claimed_professions:
                    continue
                # Return the profession (capitalize first letter)
                return profession.capitalize()
        
        # Fallback: try regex patterns for roles (political roles)
        role_patterns = [
            r'(?:war|ist|war der|ist der|war die|ist die)\s+(bundeskanzler|bundeskanzlerin|bundespräsident|präsident|kanzler|minister)[a-zäöüß]*',
        ]
        
        for pattern in role_patterns:
            matches = re.findall(pattern, snippet_lower)
            if matches:
                for match in matches:
                    profession = match.strip()
                    if profession not in claimed_professions:
                        return profession.capitalize()
        
        return None
    
    def _get_profession_from_sport(self, sport: str, claim_text: str) -> Optional[str]:
        """
        Get profession keyword from sport category.
        
        Args:
            sport: Sport category (e.g., "football", "tennis")
            claim_text: Claim text for additional context
            
        Returns:
            Profession keyword if found, None otherwise
        """
        sport_lower = sport.lower()
        keywords = SPORT_PROFESSION_KEYWORDS.get(sport_lower, [])
        
        if keywords:
            # Return first keyword (most common)
            return keywords[0]
        
        # Fallback: try to extract from claim text
        if claim_text:
            for sport_key, sport_keywords in SPORT_PROFESSION_KEYWORDS.items():
                for keyword in sport_keywords:
                    if keyword in claim_text.lower():
                        return keyword
        
        return None
    
    def _format_not_enough_info(
        self,
        claim: str,
        parsed_claim: Dict[str, Any],
    ) -> str:
        """Format NOT_ENOUGH_INFO claim into natural language answer."""
        return f"Es konnte keine ausreichende Evidenz für '{claim}' gefunden werden."
    
    def _add_article_to_entity(self, claim: str, entity: str, is_first: bool = True) -> str:
        """
        Add article (der/die/das) to entity based on original claim.
        
        Args:
            claim: Original claim text
            entity: Entity name (with or without article, e.g., "nil", "rhein", "die Erdbeere", or "die")
            is_first: True if this is the first entity (X), False if second entity (Y)
            
        Returns:
            Entity with article (e.g., "der Nil", "der Rhein", "die Erdbeere")
        """
        if not entity:
            return ""
        
        entity_lower = entity.lower().strip()
        claim_lower = claim.lower()
        
        # CRITICAL: Check if entity already has an article (e.g., "die Erdbeere", "der Rhein")
        # If so, return it with proper capitalization
        if entity_lower.startswith(("der ", "die ", "das ", "ein ", "eine ")):
            # Entity already has article - just ensure proper capitalization
            entity_words = entity.split()
            if len(entity_words) >= 2:
                article = entity_words[0]
                noun = entity_words[1]
                # Capitalize noun properly
                noun_capitalized = noun[0].upper() + noun[1:] if noun else ""
                return f"{article} {noun_capitalized}"
            # If only one word after article, return as-is with capitalization
            return entity
        
        # CRITICAL: If entity is just an article (e.g., "die"), extract the actual noun from claim
        if entity_lower in ["der", "die", "das", "ein", "eine"]:
            # Try to find the noun after the article in the claim
            # Pattern: "die Erdbeere" -> extract "Erdbeere"
            article_pattern = rf'\b{re.escape(entity_lower)}\s+([A-ZÄÖÜ][a-zäöüß]+)\b'
            match = re.search(article_pattern, claim, re.IGNORECASE)
            if match:
                actual_entity = match.group(1)
                # Return with the article + actual entity (already capitalized)
                return f"{entity} {actual_entity}"
            # If no noun found, try to find it from claim structure
            claim_words = claim.split()
            for i, word in enumerate(claim_words):
                if word.lower() == entity_lower and i < len(claim_words) - 1:
                    next_word = claim_words[i + 1]
                    if next_word[0].isupper():
                        return f"{word} {next_word}"
        
        # Try to find article from original claim
        # Pattern for first entity: "der/die/das ENTITY"
        # Pattern for second entity: "als der/die/das ENTITY"
        if is_first:
            pattern = rf'\b((?:der|die|das)\s+){re.escape(entity_lower)}\b'
        else:
            pattern = rf'als\s+((?:der|die|das)\s+){re.escape(entity_lower)}\b'
        
        match = re.search(pattern, claim_lower, re.IGNORECASE)
        if match:
            article = match.group(1).strip()
            entity_cap = entity[0].upper() + entity[1:] if entity else ""
            return f"{article} {entity_cap}"
        else:
            # Fallback: use "der" for most entities (default for rivers, countries, etc.)
            entity_cap = entity[0].upper() + entity[1:] if entity else ""
            return f"der {entity_cap}"
    
    def _format_number(self, value: float, attribute: str) -> str:
        """
        Format numeric value with appropriate unit.
        
        Args:
            value: Numeric value
            attribute: Attribute name (e.g., "länge", "größe")
            
        Returns:
            Formatted string (e.g., "6650 km", "1233 km")
        """
        # Remove decimal if .0
        if value == int(value):
            value_str = str(int(value))
        else:
            value_str = f"{value:.1f}"
        
        # Add unit based on attribute
        unit_map = {
            "länge": "km",
            "größe": "km²",
            "anzahl": "",
            "höhe": "m",
        }
        unit = unit_map.get(attribute.lower(), "")
        
        if unit:
            return f"{value_str} {unit}"
        else:
            return value_str
    
    def _extract_population_from_evidence(self, snippet: str, subject: str, title: Optional[str] = None) -> Optional[float]:
        """
        Extract population number from evidence snippet using unified parser.
        
        Args:
            snippet: Evidence snippet text (may contain HTML entities)
            subject: Subject name (e.g., "Portugal", "Kroatien")
            title: Optional evidence title (for subject matching)
            
        Returns:
            Population number in millions if found, None otherwise
        """
        from app.pipeline.population_parser import parse_population_number
        
        snippet_lower = snippet.lower()
        subject_lower = subject.lower()
        title_lower = (title or "").lower()
        
        logger.debug(f"POPULATION_EXTRACT_DEBUG: Extracting from snippet='{snippet[:200]}...' subject='{subject}' title='{title}'")
        
        # Check if snippet or title mentions the subject (more lenient check)
        subject_tokens = [token for token in subject_lower.split() if len(token) > 3]
        if subject_tokens:
            subject_in_snippet = any(token in snippet_lower for token in subject_tokens)
            subject_in_title = any(token in title_lower for token in subject_tokens) if title else False
            logger.debug(f"POPULATION_EXTRACT_DEBUG: subject_in_snippet={subject_in_snippet} subject_in_title={subject_in_title} subject_tokens={subject_tokens}")
            if not (subject_in_snippet or subject_in_title):
                logger.warning(f"POPULATION_EXTRACT_DEBUG: Subject '{subject}' not found in snippet or title - returning None")
                return None
        
        # Use unified parser
        pop_values = parse_population_number(snippet)
        if not pop_values:
            logger.warning(f"POPULATION_EXTRACT_DEBUG: No population number found! Snippet='{snippet[:300]}...'")
            return None
        
        # Use largest value and convert to millions
        best_value = max(pop_values, key=lambda v: v.normalized_value_int)
        number_millions = best_value.normalized_value_int / 1_000_000.0
        logger.info(f"POPULATION_EXTRACT_DEBUG: Extracted number={number_millions:.1f}M (absolute={best_value.normalized_value_int:,})")
        return number_millions

    def _extract_population_details_from_evidence(
        self, snippet: str, subject: str, title: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Like _extract_population_from_evidence, but also returns the exact raw integer if found.
        Supports formats like "10.344.802 Einwohner" and Portuguese "habitantes".
        """
        # Clean HTML entities first (e.g., &#160; -> space, &#91; -> [)
        snippet_clean = html.unescape(snippet or "")
        # Also handle numeric entities manually (fallback)
        snippet_clean = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), snippet_clean)
        snippet_clean = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1, 16))), snippet_clean) if False else re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), snippet_clean)
        # Normalize non-breaking spaces and whitespace to help regex matching
        try:
            snippet_clean = snippet_clean.replace("\u00a0", " ").replace("\u202f", " ")
            snippet_clean = re.sub(r"\s+", " ", snippet_clean)
        except Exception:
            pass
        
        snippet_lower = snippet_clean.lower()
        subject_lower = (subject or "").lower()
        title_lower = (title or "").lower()

        # Subject gating (same idea as _extract_population_from_evidence, but a bit more tolerant)
        subject_tokens = [t for t in subject_lower.split() if len(t) > 3]
        if subject_tokens:
            subject_in_snippet = any(t in snippet_lower for t in subject_tokens)
            subject_in_title = any(t in title_lower for t in subject_tokens) if title else False
            if not (subject_in_snippet or subject_in_title):
                return None

        # 1) Try to extract an exact raw integer with thousand separators (common in infoboxes / citations)
        # Examples: "10.344.802 Einwohner", "Portugal hat 10 344 802 Einwohner", "10,344,802 inhabitants"
        pop_keywords = r"(?:einwohner|einwohnern|einwohnerzahl|bevölkerung|bevölkerungszahl|bewohner|menschen|habitantes|inhabitants)"
        raw_patterns = [
            rf"(\d{{1,3}}(?:[.\s]\d{{3}})+)\s*{pop_keywords}\b",
            rf"{pop_keywords}\D{{0,25}}(\d{{1,3}}(?:[.\s]\d{{3}})+)\b",
            rf"(\d{{1,3}}(?:,\d{{3}})+)\s*{pop_keywords}\b",
            rf"{pop_keywords}\D{{0,25}}(\d{{1,3}}(?:,\d{{3}})+)\b",
        ]

        for pat in raw_patterns:
            m = re.search(pat, snippet_lower, re.IGNORECASE)
            if not m:
                continue
            raw_str = m.group(1)
            cleaned = raw_str.replace(".", "").replace(" ", "").replace(",", "")
            try:
                raw_int = int(cleaned)
                millions = raw_int / 1_000_000.0
                return {
                    "millions": float(millions),
                    "raw": raw_int,
                    "raw_str": f"{raw_int:,}".replace(",", "."),
                }
            except ValueError:
                continue

        # 2) Fallback to the existing "X Millionen" extraction
        millions = self._extract_population_from_evidence(snippet, subject, title)
        if millions:
            return {"millions": float(millions), "raw": None, "raw_str": None}
        return None

    @staticmethod
    def _format_millions_de(value: float) -> str:
        """Format a float like 10.344802 -> '10,3' (German decimal comma, one decimal)."""
        try:
            return f"{float(value):.1f}".replace(".", ",")
        except Exception:
            return str(value)
    
    def _extract_population_from_claim(self, claim: str) -> Optional[float]:
        """
        Extract population number from claim text.
        
        Args:
            claim: Claim text (e.g., "Portugal hat 83 Millionen Einwohner")
            
        Returns:
            Population number in millions if found, None otherwise
        """
        import re
        claim_lower = claim.lower()
        
        # Pattern: number + "Millionen" + "Einwohner"
        patterns = [
            r"(\d+)[.,](\d+)\s*(?:millionen|million|mio)\s*(?:einwohner|bevölkerung)",
            r"(\d+)\s*(?:millionen|million|mio)\s*(?:einwohner|bevölkerung)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, claim_lower, re.IGNORECASE)
            if match:
                if match.lastindex == 2:
                    # Decimal number
                    integer_part = match.group(1)
                    decimal_part = match.group(2)
                    number = float(f"{integer_part}.{decimal_part}")
                    return number
                elif match.lastindex == 1:
                    # Integer number
                    number = float(match.group(1))
                    return number
        
        return None
