"""
Dynamic threshold post-processor.

Adjusts decision threshold based on evidence count, quality, and relevance signals.
"""
import html
import logging
import re
from typing import Optional, List, Dict, Any

from app.pipeline.postprocessors.base import PostProcessor, Decision
from app.pipeline.models import Evidence, ParsedClaim, FactLabel
from app.pipeline.utils import tokenize_text, COMMON_STOPWORDS
from app.pipeline.population_parser import parse_population_number

logger = logging.getLogger(__name__)


class DynamicThresholdPostProcessor(PostProcessor):
    """
    Post-processor that adjusts threshold based on evidence quality and relevance.
    
    Lowers threshold for highly relevant evidence (subject in title, year match, etc.)
    to handle paraphrasing issues.
    """
    
    def process(
        self,
        claim: str,
        evidences: List[Evidence],
        decision: Decision,
        parsed_claim: Optional[ParsedClaim] = None,
    ) -> Decision:
        """
        Adjust threshold based on evidence count and quality.
        
        Then re-evaluate label with adjusted threshold.
        """
        if not decision.best_evidence:
            return decision
        
        # CRITICAL: Check for contradiction patterns FIRST, before any threshold adjustments
        # This ensures REFUTED is set correctly and not overwritten
        claim_lower = claim.lower()
        contradiction_patterns = [
            ("hauptstadt", "von"),  # "X ist die Hauptstadt von Y" - check if evidence says different country
            ("höchste berg", "in"),  # "X ist der höchste Berg in Y" - check if evidence says different location
            ("fluss", "in"),  # "X ist ein Fluss in Y" - check if evidence says different location
        ]
        
        should_check_contradiction = False
        for pattern_keyword, pattern_prep in contradiction_patterns:
            if pattern_keyword in claim_lower and pattern_prep in claim_lower:
                should_check_contradiction = True
                break
        
        # If this is a contradiction pattern claim, check if evidence explicitly contradicts it
        if should_check_contradiction and parsed_claim and parsed_claim.object:
            evidence_lower = decision.best_evidence.snippet.lower()
            object_lower = parsed_claim.object.lower()
            
            # Check if evidence mentions the object but contradicts it
            # For "Berlin ist die Hauptstadt von Frankreich", evidence might say "Berlin ist die Hauptstadt von Deutschland"
            if "hauptstadt" in claim_lower:
                # Look for "Hauptstadt von X" in evidence where X != object
                hauptstadt_patterns = [
                    r'(?:ist|war)\s+(?:die|der|das)?\s*hauptstadt\s+(?:von|der|des)?\s+([a-zäöüß]+(?:\s+[a-zäöüß]+)?)',
                    r',\s*(?:die|der|das)?\s*hauptstadt\s+(?:von|der|des)?\s+([a-zäöüß]+(?:\s+[a-zäöüß]+)?)',
                    r'hauptstadt\s+(?:von|der|des)?\s+([a-zäöüß]+(?:\s+[a-zäöüß]+)?)',
                ]
                for pattern in hauptstadt_patterns:
                    hauptstadt_match = re.search(pattern, evidence_lower, re.IGNORECASE)
                    if hauptstadt_match:
                        evidence_country = hauptstadt_match.group(1).strip().rstrip('.,!?')
                        if evidence_country.lower() != object_lower:
                            # Evidence contradicts: claim says "von Frankreich" but evidence says "von Deutschland"
                            # This should be REFUTED regardless of NLI scores or threshold
                            decision.label = FactLabel.REFUTED
                            decision.add_note(f"CONTRADICTION_PATTERN: Evidence says '{evidence_country}' but claim says '{object_lower}' -> REFUTED")
                            logger.info(f"CONTRADICTION_PATTERN: Setting REFUTED for Hauptstadt claim (evidence='{evidence_country}', claim='{object_lower}')")
                            return decision
            
            # For "höchste Berg in Y" or "Fluss in Y"
            if ("höchste berg" in claim_lower or "fluss" in claim_lower or "river" in claim_lower) and "in" in claim_lower:
                # Look for location information in evidence (use original snippet to preserve capitalization)
                evidence_original = decision.best_evidence.snippet
                evidence_lower = evidence_original.lower()
                
                # Normalize location names
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
                
                # Extract location from claim (object)
                object_normalized = location_mappings.get(object_lower, object_lower)
                
                # Try multiple patterns to find location in evidence
                location_patterns = [
                    r'(?:liegt|befindet\s+sich|ist|fließt)\s+in\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                    r'(?:durchfließt|fließt\s+durch)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
                    r'\bin\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\b',  # Simple "in Y" pattern
                    r'(?:in|auf)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s+(?:liegt|befindet)',  # Reversed pattern
                ]
                
                evidence_location = None
                for pattern in location_patterns:
                    location_match = re.search(pattern, evidence_original, re.IGNORECASE)
                    if location_match:
                        potential_location = location_match.group(1).strip().rstrip('.,!?')
                        evidence_location_normalized = location_mappings.get(potential_location.lower(), potential_location)
                        
                        # Check if evidence location contradicts claim location
                        if evidence_location_normalized.lower() != object_normalized.lower():
                            evidence_location = evidence_location_normalized
                            break
                
                # If no location found in evidence patterns, try direct keyword search
                if not evidence_location:
                    # Check if evidence mentions known continents/countries
                    for location_key, location_normalized in location_mappings.items():
                        if location_key in evidence_lower and location_key != object_lower:
                            evidence_location = location_normalized
                            break
                
                # If we found a contradicting location, set REFUTED
                if evidence_location and evidence_location.lower() != object_normalized.lower():
                    # This is a clear contradiction - set REFUTED regardless of NLI scores
                    decision.label = FactLabel.REFUTED
                    decision.add_note(f"CONTRADICTION_PATTERN: Evidence says location '{evidence_location}' but claim says '{object_normalized}' -> REFUTED")
                    logger.info(f"CONTRADICTION_PATTERN: Setting REFUTED for location claim (evidence='{evidence_location}', claim='{object_normalized}')")
            return decision
        
        # Start with base threshold
        base_threshold = 0.70
        threshold = base_threshold
        
        # Adjust based on evidence count
        if len(evidences) >= 3:
            threshold = base_threshold - 0.05  # Lower threshold with more evidence
        elif len(evidences) == 1:
            threshold = base_threshold + 0.05  # Higher threshold with single evidence
        
        # Adjust based on evidence quality (title match)
        if decision.best_evidence.title.lower() in claim.lower():
            threshold -= 0.03  # Title match = more reliable
        
        # Special handling for low scores: if evidence is highly relevant (title/subject match) but NLI score is low,
        # it might be due to paraphrasing. Lower threshold significantly.
        if parsed_claim and parsed_claim.subject:
            subject_lower = parsed_claim.subject.lower()
            subject_tokens = set(subject_lower.split())
            evidence_lower = decision.best_evidence.snippet.lower()
            title_lower = decision.best_evidence.title.lower()
            
            # Check if subject tokens appear in evidence
            subject_in_evidence = any(token in evidence_lower for token in subject_tokens if len(token) > 3)
            subject_in_title = any(token in title_lower for token in subject_tokens if len(token) > 3)
            
            if subject_in_title:
                # Subject in title = very relevant, lower threshold significantly
                threshold -= 0.15
            elif subject_in_evidence:
                # Subject in snippet = relevant, lower threshold
                threshold -= 0.10
        
        # For year-based claims, if year matches in evidence, lower threshold significantly
        if parsed_claim and parsed_claim.year:
            year_str = str(parsed_claim.year)
            if year_str in decision.best_evidence.snippet or year_str in decision.best_evidence.title:
                threshold -= 0.10
        
        # Check for key action words in evidence that match claim
        claim_lower = claim.lower()
        evidence_lower = decision.best_evidence.snippet.lower()
        action_words = ["gewann", "gewonnen", "siegte", "weltmeister", "titel", "endete", "endet"]
        claim_has_action = any(word in claim_lower for word in action_words)
        evidence_has_action = any(word in evidence_lower for word in action_words)
        
        if claim_has_action and evidence_has_action:
            # Both claim and evidence mention action words -> very relevant
            threshold -= 0.10
        
        # Special handling for relational claims (umkreist, besteht aus, fließt durch, hat X)
        # These claims have low NLI scores due to paraphrasing, but are often correct if evidence is relevant
        relational_markers = [
            "umkreist", "kreist um",  # Orbital relationships
            "besteht aus", "enthält", "enthalten",  # Composition
            "fließt durch", "durchfließt", "durchquert",  # Geographic flow
            "hat", "hatte", "haben", "hatten",  # Has/contains
        ]
        is_relational_claim = any(marker in claim_lower for marker in relational_markers)
        
        if is_relational_claim:
            # For relational claims, if evidence is relevant (subject/object in title or snippet), be very lenient
            if parsed_claim and parsed_claim.subject:
                subject_lower = parsed_claim.subject.lower()
                title_lower = decision.best_evidence.title.lower()
                snippet_lower = decision.best_evidence.snippet.lower()
                
                subject_in_title = any(token in title_lower for token in subject_lower.split() if len(token) > 3)
                subject_in_snippet = any(token in snippet_lower for token in subject_lower.split() if len(token) > 3)
                
                # Check if object is also mentioned (for claims like "Erde umkreist Sonne")
                object_in_title = False
                object_in_snippet = False
                if parsed_claim.object:
                    object_lower = parsed_claim.object.lower()
                    object_in_title = any(token in title_lower for token in object_lower.split() if len(token) > 3)
                    object_in_snippet = any(token in snippet_lower for token in object_lower.split() if len(token) > 3)
                
                # CRITICAL: Only lower threshold if evidence doesn't contradict the relationship
                # For claims like "Berlin ist die Hauptstadt von Frankreich", if evidence says "von Deutschland",
                # we should NOT lower the threshold (it's a clear contradiction)
                evidence_contradicts = False
                if parsed_claim.object and should_check_contradiction:
                    object_lower = parsed_claim.object.lower()
                    # Check if evidence explicitly contradicts the object
                    if "hauptstadt" in claim_lower:
                        hauptstadt_match = re.search(
                            r'hauptstadt\s+(?:von|der|des)?\s+([a-zäöüß]+(?:\s+[a-zäöüß]+)?)',
                            snippet_lower,
                            re.IGNORECASE
                        )
                        if hauptstadt_match:
                            evidence_country = hauptstadt_match.group(1).strip()
                            if evidence_country.lower() != object_lower:
                                evidence_contradicts = True
                
                if not evidence_contradicts:
                    if subject_in_title or subject_in_snippet:
                        # Very relevant evidence for relational claim - lower threshold significantly
                        threshold -= 0.25  # Much larger reduction for relational claims
                        if object_in_title or object_in_snippet:
                            # Both subject and object mentioned - even more relevant
                            threshold -= 0.10
        
        threshold = max(0.20, min(0.85, threshold))  # Clamp between 0.20-0.85 (lowered min from 0.30 to 0.20)
        
        # CRITICAL: If REFUTED was already set by contradiction pattern check, don't overwrite it
        if decision.label == FactLabel.REFUTED:
            decision.threshold = threshold
            decision.add_note(f"DYNAMIC_THRESHOLD: Keeping REFUTED label from contradiction pattern check (threshold={threshold:.2f})")
            return decision
        
        # Update threshold in decision
        decision.threshold = threshold
        
        # Re-evaluate label with adjusted threshold
        if decision.best_entailment >= threshold and decision.best_entailment > decision.best_contradiction:
            # Check for negation in claim (handled by NegationPostProcessor, but we need to check here for threshold logic)
            claim_lower = claim.lower()
            negation_words = ["nicht", "nie", "kein", "keine", "keinen", "keiner", "no", "not", "never"]
            claim_has_negation = any(
                f" {neg} " in claim_lower or claim_lower.startswith(f"{neg} ") or claim_lower.endswith(f" {neg}")
                for neg in negation_words
            )
            
            if claim_has_negation:
                # Negated claims: if evidence contradicts the negation, it supports the claim
                if decision.best_contradiction > 0.6:
                    decision.label = FactLabel.SUPPORTED
                else:
                    decision.label = FactLabel.NOT_ENOUGH_INFO
            else:
                decision.label = FactLabel.SUPPORTED
            decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
            
        elif decision.best_contradiction >= 0.70 and decision.best_contradiction > decision.best_entailment:
            # High contradiction - but check if it's a false positive
            # CRITICAL: Check evidence quality and relevance BEFORE setting REFUTED
            # If evidence is low quality or irrelevant, prefer NOT_ENOUGH_INFO to avoid false negatives
            
            evidence_count = len(evidences)
            claim_lower = claim.lower()
            evidence_lower = decision.best_evidence.snippet.lower() if decision.best_evidence else ""
            
            # Check 1: Evidence relevance - is subject mentioned in evidence?
            subject_in_evidence = False
            subject_in_title = False
            if parsed_claim and parsed_claim.subject:
                subject_lower = parsed_claim.subject.lower()
                subject_tokens = set(subject_lower.split())
                if decision.best_evidence:
                    title_lower = decision.best_evidence.title.lower()
                    snippet_lower = decision.best_evidence.snippet.lower()
                    subject_in_title = any(token in title_lower for token in subject_tokens if len(token) > 3)
                    subject_in_evidence = any(token in snippet_lower for token in subject_tokens if len(token) > 3)
            
            # Check 2: Evidence quality - low evidence count OR low lexical overlap
            # For single evidence with low relevance, be more cautious
            evidence_quality_low = False
            if evidence_count == 1:
                # Single evidence: require high relevance (subject in title or high lexical overlap)
                if not subject_in_title:
                    # Check lexical overlap as fallback
                    claim_tokens = set(tokenize_text(claim_lower, COMMON_STOPWORDS, min_length=3))
                    evidence_tokens = set(tokenize_text(evidence_lower, COMMON_STOPWORDS, min_length=3))
                    overlap = len(claim_tokens & evidence_tokens)
                    if overlap < 3:  # Less than 3 key terms overlap
                        evidence_quality_low = True
                        logger.info(
                            f"DYNAMIC_THRESHOLD: Single evidence with low relevance "
                            f"(subject_in_title={subject_in_title}, overlap={overlap}) -> preferring NOT_ENOUGH_INFO"
                        )
            
            # Check 3: Claim type - some claims should be more lenient (e.g., biological/definition claims)
            # "die Erdbeere ist eine Nuss" - biological claims are complex and might need NOT_ENOUGH_INFO
            # The Erdbeere is technically a Sammelnussfrucht, so "ist eine Nuss" is not clearly wrong
            is_biological_claim = any(kw in claim_lower for kw in [
                "ist eine", "ist ein", "gehört zu", "ist eine art", "ist eine sorte",
                "ist eine nuss", "ist eine frucht", "ist eine beere", "ist eine sammelnussfrucht"
            ])
            is_definition_claim = any(kw in claim_lower for kw in ["besteht aus", "enthält", "ist", "sind"])
            
            # CRITICAL: For biological claims, be more lenient - even if evidence quality is OK,
            # biological classification is complex and nuanced, so prefer NOT_ENOUGH_INFO
            # unless we have very high confidence (contradiction >= 0.95)
            if is_biological_claim:
                # For biological claims, require very high contradiction (>= 0.95) to set REFUTED
                # This prevents false negatives for claims that are technically correct but nuanced
                if decision.best_contradiction < 0.95:
                    decision.label = FactLabel.NOT_ENOUGH_INFO
                    decision.add_note(
                        f"DYNAMIC_THRESHOLD: Biological claim with contradiction={decision.best_contradiction:.3f} "
                        f"(below 0.95 threshold for biological claims) -> NOT_ENOUGH_INFO"
                    )
                    logger.info(
                        f"DYNAMIC_THRESHOLD: Biological claim with insufficient confidence "
                        f"(contra={decision.best_contradiction:.3f} < 0.95) -> NOT_ENOUGH_INFO"
                    )
                    decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                    return decision
            
            # If evidence quality is low OR claim is definition type, prefer NOT_ENOUGH_INFO
            if evidence_quality_low or (is_definition_claim and not subject_in_title and not subject_in_evidence):
                decision.label = FactLabel.NOT_ENOUGH_INFO
                decision.add_note(
                    f"DYNAMIC_THRESHOLD: High contradiction (contra={decision.best_contradiction:.3f}) but "
                    f"low evidence quality (count={evidence_count}, subject_in_title={subject_in_title}, "
                    f"subject_in_evidence={subject_in_evidence}, biological={is_biological_claim}) -> NOT_ENOUGH_INFO"
                )
                logger.info(
                    f"DYNAMIC_THRESHOLD: High contradiction but low evidence quality -> NOT_ENOUGH_INFO "
                    f"(contra={decision.best_contradiction:.3f}, evidence_count={evidence_count}, "
                    f"subject_in_title={subject_in_title})"
                )
                decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                return decision
            
            # Special handling for numerical claims: "X hat Y Millionen Einwohner" vs "X hat Y,Z Millionen Einwohner"
            # NLI models often mark these as contradiction even though numbers are approximately equal
            is_numerical_claim = self._is_numerical_claim(claim_lower, evidence_lower)
            
            # CRITICAL: Check for population claims FIRST - they need special handling
            is_population_claim = any(kw in claim_lower for kw in ["einwohner", "bevölkerung", "population"])
            if is_population_claim:
                logger.info(f"POPULATION_CLAIM_DEBUG: Processing population claim '{claim[:60]}...'")
                logger.info(f"POPULATION_CLAIM_DEBUG: Current best_evidence title='{decision.best_evidence.title}' snippet='{decision.best_evidence.snippet[:100]}...'")
                
                # For population claims, ALWAYS prioritize main article (exact title match with subject)
                # This ensures we use "Portugal" not "Lissabon", "Kroatien" not "Skradin", etc.
                # Use unified population parser
                claim_values = parse_population_number(claim)
                claim_numbers = [v.normalized_value_int / 1_000_000.0 for v in claim_values] if claim_values else []
                evidence_values = parse_population_number(decision.best_evidence.snippet)
                evidence_numbers = [v.normalized_value_int / 1_000_000.0 for v in evidence_values] if evidence_values else []
                
                logger.info(f"POPULATION_CLAIM_DEBUG: claim_numbers={claim_numbers}, evidence_numbers={evidence_numbers}")
                logger.info(f"POPULATION_CLAIM_DEBUG: evidence_lower contains 'bevölkerung'={('bevölkerung' in evidence_lower)}, 'demografie'={('demografie' in evidence_lower or 'demographie' in evidence_lower)}")
                
                # CRITICAL: For population claims, ALWAYS search for main article first
                # Even if current evidence has numbers, we should prefer main article if it exists
                main_article_evidence = None
                subject_lower = None
                if parsed_claim and parsed_claim.subject:
                    subject_lower = parsed_claim.subject.lower()
                    
                    # Common country name mappings for better matching
                    country_mappings = {
                        "usa": ["vereinigte", "staaten", "amerika"],
                        "vereinigte staaten": ["vereinigte", "staaten"],
                        "us": ["vereinigte", "staaten"],
                        "u.s.a": ["vereinigte", "staaten"],
                        "u.s.a.": ["vereinigte", "staaten"],
                    }
                    
                    # Find main article: exact title match first, then check for country mappings
                    for ev in evidences:
                        ev_title_lower = ev.title.lower().strip()
                        ev_title_tokens = set(ev_title_lower.split())
                        
                        # Exact match: title equals subject (e.g., "Portugal" == "portugal")
                        if ev_title_lower == subject_lower:
                            main_article_evidence = ev
                            logger.info(f"POPULATION_CLAIM_DEBUG: Found main article '{ev.title}' (exact match with subject '{parsed_claim.subject}')")
                            break
                        
                        # Check country mappings (e.g., "usa" -> "Vereinigte Staaten")
                        if subject_lower in country_mappings:
                            expected_tokens = set(country_mappings[subject_lower])
                            if expected_tokens.issubset(ev_title_tokens):
                                # Additional check: should not be a disambiguation page or list
                                if not any(kw in ev_title_lower for kw in ["begriffsklärung", "liste", "geschichte", "krieg"]):
                                    main_article_evidence = ev
                                    logger.info(f"POPULATION_CLAIM_DEBUG: Found main article '{ev.title}' (subject '{parsed_claim.subject}' maps to title)")
                                    break
                        
                        # Check if subject tokens appear in title (e.g., "kroatien" in "Kroatien")
                        subject_tokens = set(subject_lower.split())
                        if any(token in ev_title_tokens for token in subject_tokens if len(token) > 2):
                            # Additional check: should not be a disambiguation page or list
                            if not any(kw in ev_title_lower for kw in ["begriffsklärung", "liste", "geschichte", "krieg"]):
                                # Prefer exact matches, but if we haven't found one, use this
                                if not main_article_evidence:
                                    main_article_evidence = ev
                                    logger.info(f"POPULATION_CLAIM_DEBUG: Found main article '{ev.title}' (subject '{parsed_claim.subject}' tokens match title)")
                    
                    # Fallback: canonical 'Vereinigte Staaten' for USA mapping
                    if not main_article_evidence:
                        for ev in evidences:
                            if ev.title.lower().strip() == "vereinigte staaten":
                                main_article_evidence = ev
                                logger.info("POPULATION_CLAIM_DEBUG: Fallback main article set to 'Vereinigte Staaten'")
                                break
                    
                    # If still no match, use first evidence with subject in title (most relevant)
                    if not main_article_evidence:
                        for ev in evidences:
                            ev_title_lower = ev.title.lower().strip()
                            # Check if title looks like a country article (not disambiguation, not list)
                            if not any(kw in ev_title_lower for kw in ["begriffsklärung", "liste", "geschichte", "krieg", "spanische", "königreich"]):
                                # Check if subject appears in title
                                if subject_lower in ev_title_lower or any(token in ev_title_lower for token in subject_lower.split() if len(token) > 2):
                                    main_article_evidence = ev
                                    logger.info(f"POPULATION_CLAIM_DEBUG: Using first plausible article '{ev.title}' as main article (subject='{parsed_claim.subject}')")
                                    break
                
                # If we found main article, prioritize it (even if it doesn't have numbers yet)
                if main_article_evidence and subject_lower:
                    # Try to extract numbers from main article (with HTML entity decoding)
                    main_article_snippet_clean = self._clean_html_entities(main_article_evidence.snippet)
                    main_article_lower = main_article_snippet_clean.lower()
                    main_article_numbers = self._extract_numbers(main_article_lower, debug=True)
                    main_article_numbers_filtered = [n for n in main_article_numbers if n >= 1.0]
                    
                    has_demographic_kw = any(kw in main_article_lower for kw in ["bevölkerung", "demografie", "demographie"])
                    has_population_kw = any(kw in main_article_lower for kw in ["einwohner", "bevölkerungszahl", "einwohnerzahl", "habitantes"])
                    recent_years = ["2024", "2023", "2022", "2021", "2020"]
                    has_recent_year = any(year in main_article_evidence.snippet for year in recent_years)
                    current_indicators = ["aktuell", "heute", "laut", "nach", "stand", "zählung", "volkszählung"]
                    has_current_indicator = any(ind in main_article_lower for ind in current_indicators)
                    
                    logger.info(f"POPULATION_CLAIM_DEBUG: Main article '{main_article_evidence.title}' numbers={main_article_numbers_filtered} demographic_kw={has_demographic_kw} population_kw={has_population_kw} recent_year={has_recent_year}")
                    
                    # CRITICAL: If main article has numbers, use them directly
                    if main_article_numbers_filtered:
                        decision.best_evidence = main_article_evidence
                        evidence_lower = main_article_lower
                        evidence_numbers = main_article_numbers_filtered
                        logger.info(f"POPULATION_CLAIM_DEBUG: Using main article '{main_article_evidence.title}' with numbers={main_article_numbers_filtered}")
                    else:
                        # Main article has no numbers - search ONLY within main article evidences
                        # Find all evidences from the main article
                        main_article_evidences = [ev for ev in evidences if ev.title.lower().strip() == subject_lower]
                        logger.info(f"POPULATION_CLAIM_DEBUG: Main article has no numbers, searching {len(main_article_evidences)} evidences from main article '{main_article_evidence.title}'...")
                        
                        best_evidence_with_numbers = None
                        best_evidence_numbers = None
                        best_evidence_score = -1.0
                        
                        for idx, ev in enumerate(main_article_evidences):
                            ev_snippet_clean = self._clean_html_entities(ev.snippet)
                            ev_lower = ev_snippet_clean.lower()
                            
                            ev_numbers = self._extract_numbers(ev_lower, debug=True)
                            ev_numbers_filtered = [n for n in ev_numbers if n >= 1.0]
                            
                            if ev_numbers_filtered:
                                has_demo_kw = any(kw in ev_lower for kw in ["bevölkerung", "demografie", "demographie"])
                                has_pop_kw = any(kw in ev_lower for kw in ["einwohner", "bevölkerungszahl", "einwohnerzahl", "habitantes"])
                                has_recent = any(year in ev.snippet for year in recent_years)
                                has_current = any(ind in ev_lower for ind in current_indicators)
                                
                                score = 0.0
                                if has_demo_kw:
                                    score += 5.0
                                if has_pop_kw:
                                    score += 3.0
                                if has_recent:
                                    score += 8.0
                                if has_current:
                                    score += 4.0
                                
                                max_number = max(ev_numbers_filtered)
                                if max_number > 2000.0:
                                    continue
                                
                                logger.info(f"POPULATION_CLAIM_DEBUG: Main article evidence[{idx}] scored={score:.1f} numbers={ev_numbers_filtered}")
                                
                                if score > best_evidence_score:
                                    best_evidence_with_numbers = ev
                                    best_evidence_numbers = ev_numbers_filtered
                                    best_evidence_score = score
                        
                        if best_evidence_with_numbers and best_evidence_numbers:
                            decision.best_evidence = best_evidence_with_numbers
                            evidence_lower = self._clean_html_entities(best_evidence_with_numbers.snippet).lower()
                            evidence_numbers = best_evidence_numbers
                            logger.info(f"POPULATION_CLAIM_DEBUG: Found evidence with numbers in main article '{best_evidence_with_numbers.title}' numbers={best_evidence_numbers}")
                        else:
                            # Still use main article even if no numbers found
                            decision.best_evidence = main_article_evidence
                            evidence_lower = main_article_lower
                            evidence_numbers = []
                            logger.info(f"POPULATION_CLAIM_DEBUG: No numbers found in main article, but keeping main article '{main_article_evidence.title}' as best_evidence")
                
                # If current evidence doesn't have numbers AND we don't have a main article, try to find better evidence from all evidences
                if claim_numbers and not evidence_numbers and not main_article_evidence:
                    logger.info(f"POPULATION_CLAIM_DEBUG: Current evidence has no numbers, searching {len(evidences)} evidences for better match...")
                    # Search all evidences for one with numbers and subject match
                    claim_max = max(claim_numbers)
                    if parsed_claim and parsed_claim.subject:
                        subject_lower = parsed_claim.subject.lower()
                        best_evidence_with_numbers = None
                        best_evidence_numbers = None
                        best_evidence_score = -1.0
                        
                        for idx, ev in enumerate(evidences):
                            # Clean HTML entities before extraction
                            ev_snippet_clean = self._clean_html_entities(ev.snippet)
                            ev_lower = ev_snippet_clean.lower()
                            ev_title_lower = ev.title.lower()
                            
                            # Check if evidence is about the subject
                            # For main article, use exact match
                            is_main_article = (ev_title_lower.strip() == subject_lower)
                            subject_in_title = is_main_article or (any(token in ev_title_lower for token in subject_lower.split() if len(token) > 3))
                            subject_in_snippet = any(token in ev_lower for token in subject_lower.split() if len(token) > 3)
                            
                            logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] title='{ev.title}' is_main_article={is_main_article} subject_in_title={subject_in_title} subject_in_snippet={subject_in_snippet}")
                            logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] snippet='{ev.snippet[:150]}...'")
                            
                            if subject_in_title or subject_in_snippet:
                                # Try to extract numbers from this evidence (already cleaned)
                                ev_numbers = self._extract_numbers(ev_lower, debug=True)
                                
                                # Filter out small numbers (< 1 million) - these are likely percentages or other metrics
                                # Exception: very small countries (e.g., Monaco, Liechtenstein) might have < 1M, but we'll be conservative
                                ev_numbers_filtered = [n for n in ev_numbers if n >= 1.0]
                                
                                has_demographic_kw = any(kw in ev_lower for kw in ["bevölkerung", "demografie", "demographie"])
                                has_population_kw = any(kw in ev_lower for kw in ["einwohner", "bevölkerungszahl", "einwohnerzahl"])
                                
                                # Check for recent years (2020-2025) - prioritize current data
                                recent_years = ["2024", "2023", "2022", "2021", "2020"]
                                has_recent_year = any(year in ev.snippet for year in recent_years)
                                
                                # Check for current indicators
                                current_indicators = ["aktuell", "heute", "laut", "nach", "stand", "zählung", "volkszählung"]
                                has_current_indicator = any(ind in ev_lower for ind in current_indicators)
                                
                                logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] ev_numbers={ev_numbers} filtered={ev_numbers_filtered} has_demographic_kw={has_demographic_kw} has_population_kw={has_population_kw} has_recent_year={has_recent_year} has_current_indicator={has_current_indicator}")
                                
                                if ev_numbers_filtered:
                                    # Score evidence: prioritize evidence with current data and demographic keywords
                                    score = 0.0
                                    # CRITICAL: Main article (exact title match) gets highest priority
                                    if is_main_article:
                                        score += 50.0  # Very strong bonus for main article (exact match)
                                    elif subject_in_title:
                                        score += 10.0  # Subject in title is important
                                    if has_demographic_kw:
                                        score += 5.0  # Bonus for demographic keywords
                                    if has_population_kw:
                                        score += 3.0  # Bonus for population keywords
                                    if has_recent_year:
                                        score += 8.0  # Strong bonus for recent year (2020-2024)
                                    if has_current_indicator:
                                        score += 4.0  # Bonus for current indicators
                                    
                                    # Prefer larger numbers (they're more likely to be the main population figure)
                                    # But only if they're reasonable (< 2000 million for any country)
                                    max_number = max(ev_numbers_filtered)
                                    if max_number > 2000.0:
                                        # Unrealistically large - likely an error
                                        logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] has unrealistic number {max_number:.1f}M - skipping")
                                        continue
                                    
                                    logger.info(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] scored={score:.1f} (subject_in_title={subject_in_title}, demographic_kw={has_demographic_kw}, population_kw={has_population_kw}, recent_year={has_recent_year}, current_indicator={has_current_indicator}, max_number={max_number:.1f}M)")
                                    
                                    if score > best_evidence_score:
                                        best_evidence_with_numbers = ev
                                        best_evidence_numbers = ev_numbers_filtered
                                        best_evidence_score = score
                                elif ev_numbers:
                                    logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] has numbers but all filtered out (< 1M): {ev_numbers}")
                                else:
                                    logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] has no numbers extracted (snippet length={len(ev.snippet)})")
                        
                        # If we found better evidence, use it
                        if best_evidence_with_numbers and best_evidence_numbers:
                            evidence_max = max(best_evidence_numbers)
                            decision.best_evidence = best_evidence_with_numbers
                            evidence_lower = best_evidence_with_numbers.snippet.lower()
                            evidence_numbers = best_evidence_numbers
                            logger.info(f"NUMERICAL_CLAIM: Found better evidence with numbers (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, score={best_evidence_score:.1f})")
                            logger.info(f"POPULATION_CLAIM_DEBUG: New best_evidence title='{best_evidence_with_numbers.title}' snippet='{best_evidence_with_numbers.snippet[:200]}...'")
                        else:
                            logger.warning(f"POPULATION_CLAIM_DEBUG: No evidence with numbers found! Searched {len(evidences)} evidences. Subject='{subject_lower}'")
                            # Log first few evidences for debugging
                            for idx, ev in enumerate(evidences[:5]):
                                logger.warning(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] title='{ev.title}' snippet='{ev.snippet[:100]}...'")
                
                if claim_numbers and evidence_numbers:
                    # Both have numbers - compare them
                    claim_max = max(claim_numbers)
                    evidence_max = max(evidence_numbers)
                    diff = abs(claim_max - evidence_max)
                    relative_diff = diff / max(claim_max, evidence_max) if max(claim_max, evidence_max) > 0 else 1.0
                    
                    # Check if numbers match approximately
                    if relative_diff < 0.15 and diff < 5.0:
                        # Numbers match approximately - SUPPORTED despite high contradiction
                        decision.label = FactLabel.SUPPORTED
                        decision.add_note(f"NUMERICAL_CLAIM: High contradiction but numbers match approximately (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, diff={diff:.1f}M) -> SUPPORTED")
                        logger.info(f"NUMERICAL_CLAIM: Accepting population claim despite high contradiction (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M)")
                    else:
                        # Numbers differ significantly - REFUTED
                        decision.label = FactLabel.REFUTED
                        decision.add_note(f"NUMERICAL_CLAIM: Population claim numbers differ significantly (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, diff={diff:.1f}M, relative={relative_diff:.1%}) -> REFUTED")
                        logger.info(f"NUMERICAL_CLAIM: Population claim REFUTED due to number difference (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M)")
                    decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                    return decision
                elif claim_numbers and not evidence_numbers:
                    # Claim has numbers but evidence doesn't - check if evidence is about the entity
                    logger.warning(f"POPULATION_CLAIM_DEBUG: Claim has numbers ({claim_numbers}) but current evidence has no numbers!")
                    logger.warning(f"POPULATION_CLAIM_DEBUG: Current evidence title='{decision.best_evidence.title}' snippet='{decision.best_evidence.snippet[:200]}...'")
                    
                    if parsed_claim and parsed_claim.subject:
                        subject_lower = parsed_claim.subject.lower()
                        title_lower = decision.best_evidence.title.lower()
                        snippet_lower = evidence_lower
                        
                        subject_in_title = any(token in title_lower for token in subject_lower.split() if len(token) > 3)
                        subject_in_snippet = any(token in snippet_lower for token in subject_lower.split() if len(token) > 3)
                        
                        logger.info(f"POPULATION_CLAIM_DEBUG: subject_in_title={subject_in_title} subject_in_snippet={subject_in_snippet}")
                        
                        if subject_in_title or subject_in_snippet:
                            # Evidence is about the entity but doesn't contain numbers - likely REFUTED
                            claim_max = max(claim_numbers)
                            decision.label = FactLabel.REFUTED
                            decision.add_note(f"NUMERICAL_CLAIM: Population claim with number ({claim_max:.1f}M) but evidence about entity doesn't contain numbers -> REFUTED")
                            logger.info(f"NUMERICAL_CLAIM: Population claim REFUTED - evidence about entity but no numbers (claim={claim_max:.1f}M)")
                            decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                            return decision
                        else:
                            logger.warning(f"POPULATION_CLAIM_DEBUG: Evidence is NOT about subject '{subject_lower}' - this might be wrong evidence!")
                    # Fall through to standard REFUTED logic
            
            # Standard handling for non-population numerical claims or when numbers can't be extracted
            # CRITICAL: Re-check subject_in_title and subject_in_evidence here (they might not be set from earlier checks)
            subject_in_title_recheck = False
            subject_in_evidence_recheck = False
            if parsed_claim and parsed_claim.subject and decision.best_evidence:
                subject_lower = parsed_claim.subject.lower()
                title_lower = decision.best_evidence.title.lower()
                snippet_lower = decision.best_evidence.snippet.lower()
                subject_tokens = set(subject_lower.split())
                subject_in_title_recheck = any(token in title_lower for token in subject_tokens if len(token) > 3)
                subject_in_evidence_recheck = any(token in snippet_lower for token in subject_tokens if len(token) > 3)
                
                if subject_in_title_recheck:
                    # Subject matches title but high contradiction - might be missing info or numerical false positive
                    from app.pipeline.utils import tokenize_text, COMMON_STOPWORDS
                    claim_key_terms = set(tokenize_text(claim, stopwords=COMMON_STOPWORDS, min_length=3))
                    evidence_key_terms = set(tokenize_text(decision.best_evidence.snippet, stopwords=COMMON_STOPWORDS, min_length=3))
                    overlap = len(claim_key_terms & evidence_key_terms)
                    
                    if is_numerical_claim and overlap >= 3:
                        # Numerical claim with high overlap - likely a false positive contradiction
                        # Check if numbers are approximately equal
                        claim_numbers = self._extract_numbers(claim_lower)
                        evidence_numbers = self._extract_numbers(evidence_lower)
                        if claim_numbers and evidence_numbers:
                            # Check if the largest numbers are approximately equal (within 15% or 5 million)
                            claim_max = max(claim_numbers)
                            evidence_max = max(evidence_numbers)
                            diff = abs(claim_max - evidence_max)
                            if diff / max(claim_max, evidence_max) < 0.15 or diff < 5.0:  # Within 15% or 5 million
                                # Numbers are approximately equal - this is likely SUPPORTED, not REFUTED
                                decision.label = FactLabel.SUPPORTED
                                decision.add_note(f"NUMERICAL_CLAIM: High contradiction score but numbers are approximately equal (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, diff={diff:.1f}M) -> SUPPORTED")
                                logger.info(f"NUMERICAL_CLAIM: Accepting numerical claim despite high contradiction (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M)")
                            else:
                                # Numbers differ significantly - but evidence is relevant, so REFUTED
                                decision.label = FactLabel.REFUTED
                                decision.add_note(
                                    f"NUMERICAL_CLAIM: Numbers differ significantly (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, diff={diff:.1f}M) and evidence is relevant -> REFUTED"
                                )
                        else:
                            decision.label = FactLabel.NOT_ENOUGH_INFO
                    elif overlap >= 2:  # At least 2 key terms overlap
                        # Some overlap but not enough for numerical claim - prefer NOT_ENOUGH_INFO
                        decision.label = FactLabel.NOT_ENOUGH_INFO
                        decision.add_note(
                            f"NUMERICAL_CLAIM: Moderate overlap ({overlap} terms) -> NOT_ENOUGH_INFO"
                        )
                    else:
                        # Low overlap - prefer NOT_ENOUGH_INFO
                        decision.label = FactLabel.NOT_ENOUGH_INFO
                        decision.add_note(
                            f"NUMERICAL_CLAIM: Low overlap ({overlap} terms) -> NOT_ENOUGH_INFO"
                        )
                elif subject_in_evidence_recheck:
                    # Subject in snippet but not in title - still relevant, but prefer NOT_ENOUGH_INFO for ambiguous cases
                    decision.label = FactLabel.NOT_ENOUGH_INFO
                    decision.add_note(
                        f"DYNAMIC_THRESHOLD: Subject in evidence snippet but not in title -> NOT_ENOUGH_INFO (ambiguous evidence)"
                    )
                else:
                    # Subject not mentioned in evidence at all - definitely NOT_ENOUGH_INFO
                    decision.label = FactLabel.NOT_ENOUGH_INFO
                    decision.add_note(
                        f"DYNAMIC_THRESHOLD: High contradiction but evidence doesn't mention subject -> NOT_ENOUGH_INFO"
                    )
            elif parsed_claim and parsed_claim.subject:
                # Subject exists but no evidence - NOT_ENOUGH_INFO
                decision.label = FactLabel.NOT_ENOUGH_INFO
                decision.add_note(
                    f"DYNAMIC_THRESHOLD: High contradiction but no evidence available -> NOT_ENOUGH_INFO"
                )
            else:
                # No subject parsed - default to NOT_ENOUGH_INFO for unclear claims
                decision.label = FactLabel.NOT_ENOUGH_INFO
                decision.add_note(
                    f"DYNAMIC_THRESHOLD: High contradiction but no subject parsed -> NOT_ENOUGH_INFO"
                )
            decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
            
        else:
            # Low scores - check if evidence is relevant despite low NLI score
            # Special handling for relational claims with very low scores
            
            # CRITICAL: Don't downgrade REFUTED labels that were set in the high contradiction block
            if decision.label == FactLabel.REFUTED:
                # Keep REFUTED label - don't downgrade
                decision.add_note(f"DYNAMIC_THRESHOLD: Keeping REFUTED label from high contradiction block")
                logger.info(f"DYNAMIC_THRESHOLD: Keeping REFUTED label (threshold={threshold:.2f})")
                decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                return decision
            
            claim_lower = claim.lower()
            evidence_lower = decision.best_evidence.snippet.lower() if decision.best_evidence else ""
            
            # CRITICAL: Check for numerical claims FIRST, before any other logic
            # Check if claim is a numerical claim (e.g., "X hat Y Millionen Einwohner")
            # For numerical claims, we MUST extract and compare numbers - don't rely on NLI scores alone
            is_numerical_claim_pattern = self._is_numerical_claim_pattern(claim_lower)
            if is_numerical_claim_pattern:
                # Extract numbers from claim
                claim_numbers = self._extract_numbers(claim_lower)
                if claim_numbers:
                    claim_max = max(claim_numbers)
                    # Try to extract numbers from evidence
                    evidence_numbers = self._extract_numbers(evidence_lower)
                    
                    if evidence_numbers:
                        # Both claim and evidence have numbers - compare them
                        evidence_max = max(evidence_numbers)
                        diff = abs(claim_max - evidence_max)
                        relative_diff = diff / max(claim_max, evidence_max) if max(claim_max, evidence_max) > 0 else 1.0
                        
                        # Check if numbers match (within 15% relative difference or 5 million absolute difference)
                        if relative_diff < 0.15 and diff < 5.0:
                            # Numbers match approximately - check if subject also matches
                            if parsed_claim and parsed_claim.subject:
                                subject_lower = parsed_claim.subject.lower()
                                title_lower = decision.best_evidence.title.lower() if decision.best_evidence else ""
                                snippet_lower = evidence_lower
                                
                                subject_in_title = any(token in title_lower for token in subject_lower.split() if len(token) > 3)
                                subject_in_snippet = any(token in snippet_lower for token in subject_lower.split() if len(token) > 3)
                                
                                if subject_in_title or subject_in_snippet:
                                    # Numbers match AND subject matches - SUPPORTED
                                    decision.label = FactLabel.SUPPORTED
                                    decision.add_note(f"NUMERICAL_CLAIM: Numbers match (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, diff={diff:.1f}M, relative={relative_diff:.1%}) and subject matches -> SUPPORTED")
                                    logger.info(f"NUMERICAL_CLAIM: Accepting numerical claim (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M)")
                                    decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                                    return decision
                                else:
                                    # Numbers match but subject doesn't match - NOT_ENOUGH_INFO
                                    decision.label = FactLabel.NOT_ENOUGH_INFO
                                    decision.add_note(f"NUMERICAL_CLAIM: Numbers match but subject doesn't match -> NOT_ENOUGH_INFO")
                                    logger.info(f"NUMERICAL_CLAIM: Numbers match but subject mismatch")
                                    decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                                    return decision
                            else:
                                # No subject to check - if numbers match, accept
                                decision.label = FactLabel.SUPPORTED
                                decision.add_note(f"NUMERICAL_CLAIM: Numbers match (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M) -> SUPPORTED")
                                logger.info(f"NUMERICAL_CLAIM: Accepting numerical claim (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M)")
                                decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                                return decision
                        else:
                            # Numbers differ significantly - NOT_ENOUGH_INFO or REFUTED
                            # If numbers are very different (e.g., 83M vs 10M), it's likely REFUTED
                            if relative_diff >= 0.5 or diff >= 50.0:  # Very large difference
                                decision.label = FactLabel.REFUTED
                                decision.add_note(f"NUMERICAL_CLAIM: Numbers differ significantly (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, diff={diff:.1f}M, relative={relative_diff:.1%}) -> REFUTED")
                                logger.info(f"NUMERICAL_CLAIM: Rejecting numerical claim due to large number difference (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M)")
                            else:
                                decision.label = FactLabel.NOT_ENOUGH_INFO
                                decision.add_note(f"NUMERICAL_CLAIM: Numbers differ moderately (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, diff={diff:.1f}M, relative={relative_diff:.1%}) -> NOT_ENOUGH_INFO")
                            decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                            return decision
                    else:
                        # Claim has numbers but current evidence doesn't - search ALL evidences for one with numbers about the subject
                        if parsed_claim and parsed_claim.subject:
                            subject_lower = parsed_claim.subject.lower()
                            best_evidence_with_numbers = None
                            best_evidence_numbers = None
                            best_evidence_score = -1.0
                            
                            # Search all evidences for one with numbers about the subject
                            # CRITICAL: For population claims, prioritize main article (exact title match) FIRST
                            # Then search other evidences if main article doesn't have numbers
                            logger.info(f"POPULATION_CLAIM_DEBUG: Searching {len(evidences)} evidences for one with numbers about subject '{subject_lower}'...")
                            
                            # First pass: Look for main article (exact title match)
                            main_article_ev = None
                            for ev in evidences:
                                ev_title_lower = ev.title.lower().strip()
                                if ev_title_lower == subject_lower:
                                    main_article_ev = ev
                                    logger.info(f"POPULATION_CLAIM_DEBUG: Found main article '{ev.title}' (exact match)")
                                    break
                            
                            # If main article found, try it first
                            ev_list = [main_article_ev] if main_article_ev else []
                            ev_list.extend([ev for ev in evidences if ev != main_article_ev])
                            
                            for idx, ev in enumerate(ev_list):
                                if ev is None:
                                    continue
                                    
                                # Clean HTML entities before extraction
                                ev_snippet_clean = self._clean_html_entities(ev.snippet)
                                ev_lower = ev_snippet_clean.lower()
                                ev_title_lower = ev.title.lower()
                                
                                # Check if evidence is about the subject
                                subject_tokens = [token for token in subject_lower.split() if len(token) > 3]
                                # For main article, use exact match
                                is_main_article = (ev_title_lower.strip() == subject_lower)
                                subject_in_title = is_main_article or (any(token in ev_title_lower for token in subject_tokens) if subject_tokens else False)
                                subject_in_snippet = any(token in ev_lower for token in subject_tokens) if subject_tokens else False
                                
                                # For population claims, prioritize main article (subject in title) over related articles
                                # This prevents selecting "Tourismus in Kroatien" when we need "Kroatien"
                                if subject_in_title or (subject_in_snippet and not any(kw in ev_title_lower for kw in ["tourismus", "wirtschaft", "geschichte", "kultur"])):
                                    # Try to extract numbers from this evidence (already cleaned)
                                    ev_numbers = self._extract_numbers(ev_lower, debug=True)
                                    
                                    # Filter out small numbers (< 1 million) - these are likely percentages or other metrics
                                    ev_numbers_filtered = [n for n in ev_numbers if n >= 1.0]
                                    
                                    has_demographic_kw = any(kw in ev_lower for kw in ["bevölkerung", "demografie", "demographie"])
                                    has_population_kw = any(kw in ev_lower for kw in ["einwohner", "bevölkerungszahl", "einwohnerzahl"])
                                    
                                    # Check for recent years (2020-2025) - prioritize current data
                                    recent_years = ["2024", "2023", "2022", "2021", "2020"]
                                    has_recent_year = any(year in ev.snippet for year in recent_years)
                                    
                                    # Check for current indicators
                                    current_indicators = ["aktuell", "heute", "laut", "nach", "stand", "zählung", "volkszählung"]
                                    has_current_indicator = any(ind in ev_lower for ind in current_indicators)
                                    
                                    logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] title='{ev.title}' ev_numbers={ev_numbers} filtered={ev_numbers_filtered} demographic_kw={has_demographic_kw} population_kw={has_population_kw} has_recent_year={has_recent_year} has_current_indicator={has_current_indicator}")
                                    
                                    if ev_numbers_filtered:
                                        # Score: much higher if subject is in title (main article)
                                        # CRITICAL: Main article (exact title match) gets highest priority
                                        if is_main_article:
                                            score = 50.0  # Very strong bonus for main article (exact match)
                                        elif subject_in_title:
                                            score = 10.0  # Subject in title is important
                                        else:
                                            score = 1.0
                                        
                                        # Bonus for demographic keywords (Bevölkerung, Demografie)
                                        if has_demographic_kw:
                                            score += 5.0
                                        
                                        # Bonus for population keywords
                                        if has_population_kw:
                                            score += 3.0
                                        
                                        # Strong bonus for recent year (2020-2024)
                                        if has_recent_year:
                                            score += 8.0
                                        
                                        # Bonus for current indicators
                                        if has_current_indicator:
                                            score += 4.0
                                        
                                        # Prefer larger numbers (they're more likely to be the main population figure)
                                        # But only if they're reasonable (< 2000 million for any country)
                                        max_number = max(ev_numbers_filtered)
                                        if max_number > 2000.0:
                                            # Unrealistically large - likely an error
                                            logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] has unrealistic number {max_number:.1f}M - skipping")
                                            continue
                                        
                                        logger.info(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] scored={score:.1f} (subject_in_title={subject_in_title}, demographic_kw={has_demographic_kw}, population_kw={has_population_kw}, recent_year={has_recent_year}, current_indicator={has_current_indicator}, max_number={max_number:.1f}M)")
                                        
                                        if score > best_evidence_score:
                                            best_evidence_with_numbers = ev
                                            best_evidence_numbers = ev_numbers_filtered
                                            best_evidence_score = score
                                    elif ev_numbers:
                                        logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] has numbers but all filtered out (< 1M): {ev_numbers}")
                                    else:
                                        logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] title='{ev.title}' has no numbers extracted. Snippet='{ev.snippet[:150]}...'")
                            
                            # If we found better evidence with numbers, use it
                            if best_evidence_with_numbers and best_evidence_numbers:
                                evidence_max = max(best_evidence_numbers)
                                decision.best_evidence = best_evidence_with_numbers
                                evidence_lower = best_evidence_with_numbers.snippet.lower()
                                evidence_numbers = best_evidence_numbers
                                
                                logger.info(f"POPULATION_CLAIM_DEBUG: Selected new best_evidence title='{best_evidence_with_numbers.title}' snippet='{best_evidence_with_numbers.snippet[:200]}...'")
                                
                                # Re-extract NLI scores for this evidence (if available)
                                # Try to find it in evidence_scores
                                for ev_score in decision.evidence_scores:
                                    if ev_score["evidence"].title == best_evidence_with_numbers.title and ev_score["evidence"].url == best_evidence_with_numbers.url:
                                        decision.best_entailment = ev_score.get("entailment_score", decision.best_entailment)
                                        decision.best_contradiction = ev_score.get("contradiction_score", decision.best_contradiction)
                                        decision.best_nli_score = ev_score.get("nli_score", decision.best_nli_score)
                                        break
                                
                                logger.info(f"NUMERICAL_CLAIM: Found better evidence with numbers (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M)")
                            else:
                                logger.warning(f"POPULATION_CLAIM_DEBUG: No evidence with numbers found! Searched {len(evidences)} evidences. Subject='{subject_lower}'")
                                # Log first few evidences for debugging
                                for idx, ev in enumerate(evidences[:5]):
                                    ev_lower_debug = ev.snippet.lower()
                                    has_demo = any(kw in ev_lower_debug for kw in ["bevölkerung", "demografie", "demographie"])
                                    has_pop = any(kw in ev_lower_debug for kw in ["einwohner", "bevölkerungszahl", "einwohnerzahl"])
                                    ev_nums_debug = self._extract_numbers(ev_lower_debug)
                                    logger.warning(f"POPULATION_CLAIM_DEBUG: Evidence[{idx}] title='{ev.title}' has_numbers={bool(ev_nums_debug)} demographic_kw={has_demo} population_kw={has_pop} snippet='{ev.snippet[:150]}...'")
                                
                                # No numbers found in evidence - return NOT_ENOUGH_INFO
                                decision.label = FactLabel.NOT_ENOUGH_INFO
                                decision.add_note(f"NUMERICAL_CLAIM: Claim has numbers ({claim_max:.1f}M) but no evidence contains numbers -> NOT_ENOUGH_INFO")
                                logger.info(f"NUMERICAL_CLAIM: No numbers found in evidence for claim with {claim_max:.1f}M")
                                decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                                return decision
                            
                            # No evidence with numbers found - check if current evidence is about the subject
                            title_lower = decision.best_evidence.title.lower() if decision.best_evidence else ""
                            snippet_lower = evidence_lower
                            
                            subject_in_title = any(token in title_lower for token in subject_lower.split() if len(token) > 3)
                            subject_in_snippet = any(token in snippet_lower for token in subject_lower.split() if len(token) > 3)
                            
                            # If evidence is about the subject but doesn't contain numbers, it's likely REFUTED
                            # (because the claim states a specific number but evidence doesn't mention it)
                            if subject_in_title or subject_in_snippet:
                                # Check if this is a population claim
                                is_population_claim = any(kw in claim_lower for kw in ["einwohner", "bevölkerung", "population"])
                                if is_population_claim:
                                    # For population claims, if evidence is about the entity but doesn't mention the number,
                                    # and we have high contradiction, it's likely REFUTED
                                    if decision.best_contradiction >= 0.70:
                                        decision.label = FactLabel.REFUTED
                                        decision.add_note(f"NUMERICAL_CLAIM: Population claim with number ({claim_max:.1f}M) but evidence about entity doesn't contain numbers and high contradiction -> REFUTED")
                                        logger.info(f"NUMERICAL_CLAIM: Population claim REFUTED - evidence about entity but no numbers (claim={claim_max:.1f}M)")
                                        decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                                        return decision
                                    else:
                                        decision.label = FactLabel.NOT_ENOUGH_INFO
                                        decision.add_note(f"NUMERICAL_CLAIM: Population claim but low contradiction score -> NOT_ENOUGH_INFO")
                                        decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                                        return decision
                        
                        # Default: NOT_ENOUGH_INFO if we can't verify
                        decision.label = FactLabel.NOT_ENOUGH_INFO
                        decision.add_note(f"NUMERICAL_CLAIM: Claim has numbers ({claim_max:.1f}M) but evidence doesn't contain numbers -> NOT_ENOUGH_INFO")
                        logger.warning(f"NUMERICAL_CLAIM: Evidence doesn't contain numbers for claim with {claim_max:.1f}M")
                        logger.warning(f"POPULATION_CLAIM_DEBUG: Setting NOT_ENOUGH_INFO - searched {len(evidences)} evidences, subject='{subject_lower}'")
                        logger.warning(f"POPULATION_CLAIM_DEBUG: Current best_evidence title='{decision.best_evidence.title if decision.best_evidence else 'None'}' snippet='{decision.best_evidence.snippet[:200] if decision.best_evidence else 'None'}...'")
                        decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                        return decision
                else:
                    # Claim pattern matches but no numbers extracted from claim - fall through to normal logic
                    # But don't downgrade REFUTED labels that were set earlier
                    if decision.label == FactLabel.REFUTED:
                        # Keep REFUTED label - don't downgrade
                        decision.add_note(f"NUMERICAL_CLAIM: Keeping REFUTED label (no numbers extracted from claim)")
                        logger.info(f"NUMERICAL_CLAIM: Keeping REFUTED label")
                        decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                        return decision
                    # If not REFUTED, continue to normal logic below
            else:
                # Not a numerical claim pattern - continue to normal logic below
                pass
            
            relational_markers = [
                "umkreist", "kreist um",  # Orbital relationships
                "besteht aus", "enthält", "enthalten",  # Composition
                "fließt durch", "durchfließt", "durchquert",  # Geographic flow
                "hat", "hatte", "haben", "hatten",  # Has/contains
            ]
            is_relational_claim = any(marker in claim_lower for marker in relational_markers)
            
            if parsed_claim and parsed_claim.subject and parsed_claim.year:
                subject_lower = parsed_claim.subject.lower()
                year_str = str(parsed_claim.year)
                if (subject_lower in decision.best_evidence.title.lower() or subject_lower in decision.best_evidence.snippet.lower()) and \
                   year_str in decision.best_evidence.snippet:
                    if decision.best_entailment > decision.best_contradiction and decision.best_entailment > 0.01:
                        decision.label = FactLabel.SUPPORTED
                    else:
                        decision.label = FactLabel.NOT_ENOUGH_INFO
                else:
                    decision.label = FactLabel.NOT_ENOUGH_INFO
            elif parsed_claim and parsed_claim.subject:
                subject_lower = parsed_claim.subject.lower()
                title_lower = decision.best_evidence.title.lower()
                snippet_lower = decision.best_evidence.snippet.lower()
                
                subject_in_title = any(token in title_lower for token in subject_lower.split() if len(token) > 3)
                subject_in_snippet = any(token in snippet_lower for token in subject_lower.split() if len(token) > 3)
                
                # Check if object is also mentioned (for relational claims)
                object_in_title = False
                object_in_snippet = False
                if parsed_claim.object:
                    object_lower = parsed_claim.object.lower()
                    object_in_title = any(token in title_lower for token in object_lower.split() if len(token) > 3)
                    object_in_snippet = any(token in snippet_lower for token in object_lower.split() if len(token) > 3)
                
                if subject_in_title or subject_in_snippet:
                    claim_lower = claim.lower()
                    evidence_lower = decision.best_evidence.snippet.lower()
                    
                    # CRITICAL: For numerical claims, check if evidence is about the same entity as the claim
                    # If evidence is about a different entity (e.g., claim about Portugal but evidence about Germany),
                    # reject even if numbers are similar
                    is_numerical_claim_check = self._is_numerical_claim(claim_lower, evidence_lower)
                    if is_numerical_claim_check:
                        # Extract numbers from claim and evidence
                        claim_numbers = self._extract_numbers(claim_lower)
                        evidence_numbers = self._extract_numbers(evidence_lower)
                        if claim_numbers and evidence_numbers:
                            claim_max = max(claim_numbers)
                            evidence_max = max(evidence_numbers)
                            diff = abs(claim_max - evidence_max)
                            relative_diff = diff / max(claim_max, evidence_max) if max(claim_max, evidence_max) > 0 else 1.0
                            
                            # If numbers differ significantly (more than 15% or 5 million), reject or REFUTED
                            if relative_diff >= 0.15 or diff >= 5.0:
                                # Check if this is a population claim
                                is_population_claim = any(kw in claim_lower for kw in ["einwohner", "bevölkerung", "population"])
                                if is_population_claim:
                                    # For population claims with very different numbers, mark as REFUTED
                                    decision.label = FactLabel.REFUTED
                                    decision.add_note(f"NUMERICAL_CLAIM: Population claim numbers differ significantly (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, diff={diff:.1f}M, relative={relative_diff:.1%}) -> REFUTED")
                                    logger.info(f"NUMERICAL_CLAIM: Population claim REFUTED due to large number difference (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M)")
                                else:
                                    # For other numerical claims, reject
                                    decision.label = FactLabel.NOT_ENOUGH_INFO
                                    decision.add_note(f"NUMERICAL_CLAIM: Numbers differ significantly (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, diff={diff:.1f}M, relative={relative_diff:.1%}) -> NOT_ENOUGH_INFO")
                                    logger.info(f"NUMERICAL_CLAIM: Rejecting numerical claim due to large number difference (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M)")
                                # Skip rest of relational claim logic
                                decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                                return decision
                    
                    action_words = ["gewann", "gewonnen", "siegte", "weltmeister", "titel"]
                    claim_has_action = any(word in claim_lower for word in action_words)
                    evidence_has_action = any(word in evidence_lower for word in action_words)
                    
                    if claim_has_action and evidence_has_action:
                        if decision.best_entailment > decision.best_contradiction:
                            decision.label = FactLabel.SUPPORTED
                        else:
                            decision.label = FactLabel.NOT_ENOUGH_INFO
                    elif is_relational_claim:
                        # For relational claims, if subject (and optionally object) is in evidence, accept even with very low NLI scores
                        # This handles cases where NLI model doesn't recognize paraphrased relationships
                        # For relational claims, we're more lenient: if evidence is highly relevant, accept even if contradiction > entailment
                        if subject_in_title or (subject_in_snippet and (object_in_title or object_in_snippet)):
                            # Very relevant evidence (subject in title, or subject+object in snippet)
                            # BUT: Check if this is a numerical claim with very different numbers - if so, reject
                            claim_lower_check = claim.lower()
                            evidence_lower_check = decision.best_evidence.snippet.lower()
                            is_numerical = self._is_numerical_claim(claim_lower_check, evidence_lower_check)
                            
                            if is_numerical:
                                # For numerical claims, verify that numbers are approximately equal
                                claim_numbers = self._extract_numbers(claim_lower_check)
                                evidence_numbers = self._extract_numbers(evidence_lower_check)
                                if claim_numbers and evidence_numbers:
                                    claim_max = max(claim_numbers)
                                    evidence_max = max(evidence_numbers)
                                    diff = abs(claim_max - evidence_max)
                                    relative_diff = diff / max(claim_max, evidence_max) if max(claim_max, evidence_max) > 0 else 1.0
                                    
                                    # Only accept if numbers are approximately equal (within 15% or 5 million)
                                    if relative_diff >= 0.15 and diff >= 5.0:
                                        # Numbers are too different - check if this is a population claim
                                        is_population_claim = any(kw in claim_lower_check for kw in ["einwohner", "bevölkerung", "population"])
                                        if is_population_claim:
                                            # For population claims with very different numbers, mark as REFUTED
                                            decision.label = FactLabel.REFUTED
                                            decision.add_note(f"NUMERICAL_CLAIM: Population claim numbers differ significantly (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, diff={diff:.1f}M, relative={relative_diff:.1%}) -> REFUTED")
                                            logger.info(f"NUMERICAL_CLAIM: Population claim REFUTED due to large number difference (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M)")
                                        else:
                                            # For other numerical claims, reject
                                            decision.label = FactLabel.NOT_ENOUGH_INFO
                                            decision.add_note(f"NUMERICAL_CLAIM: Numbers differ too much (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M, diff={diff:.1f}M, relative={relative_diff:.1%}) -> NOT_ENOUGH_INFO")
                                            logger.info(f"NUMERICAL_CLAIM: Rejecting numerical claim due to large number difference (claim={claim_max:.1f}M, evidence={evidence_max:.1f}M)")
                                        decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
                                        return decision
                                    else:
                                        # Numbers are approximately equal - can proceed with relational claim logic
                                        # CRITICAL: Only set SUPPORTED if entailment > contradiction AND entailment > threshold
                                        if decision.best_entailment > decision.best_contradiction and decision.best_entailment >= threshold:
                                            decision.label = FactLabel.SUPPORTED
                                            decision.add_note(f"NUMERICAL_RELATIONAL_CLAIM: accepted with low NLI score (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f}) due to relevant evidence and matching numbers")
                                        elif decision.best_contradiction > decision.best_entailment:
                                            # Contradiction is higher - mark as REFUTED or NOT_ENOUGH_INFO
                                            if decision.best_contradiction >= 0.70:
                                                decision.label = FactLabel.REFUTED
                                                decision.add_note(f"NUMERICAL_RELATIONAL_CLAIM: REFUTED due to high contradiction (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f})")
                                            else:
                                                decision.label = FactLabel.NOT_ENOUGH_INFO
                                                decision.add_note(f"NUMERICAL_RELATIONAL_CLAIM: NOT_ENOUGH_INFO (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f})")
                                else:
                                    # Could not extract numbers - fall back to standard relational claim logic
                                    # CRITICAL: Only set SUPPORTED if entailment > contradiction AND entailment > threshold
                                    if decision.best_entailment > decision.best_contradiction and decision.best_entailment >= threshold:
                                        decision.label = FactLabel.SUPPORTED
                                        decision.add_note(f"RELATIONAL_CLAIM: accepted with low NLI score (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f}) due to relevant evidence")
                                    elif decision.best_contradiction > decision.best_entailment:
                                        # Contradiction is higher - mark as REFUTED or NOT_ENOUGH_INFO
                                        if decision.best_contradiction >= 0.70:
                                            decision.label = FactLabel.REFUTED
                                            decision.add_note(f"RELATIONAL_CLAIM: REFUTED due to high contradiction (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f})")
                                        else:
                                            decision.label = FactLabel.NOT_ENOUGH_INFO
                                            decision.add_note(f"RELATIONAL_CLAIM: NOT_ENOUGH_INFO (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f})")
                            else:
                                # Non-numerical relational claim - use standard logic
                                # CRITICAL: Only set SUPPORTED if entailment > contradiction AND entailment > threshold
                                if decision.best_entailment > decision.best_contradiction and decision.best_entailment >= threshold:
                                    decision.label = FactLabel.SUPPORTED
                                    decision.add_note(f"RELATIONAL_CLAIM: accepted with low NLI score (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f}) due to relevant evidence")
                                elif decision.best_contradiction > decision.best_entailment:
                                    # Contradiction is higher - mark as REFUTED or NOT_ENOUGH_INFO
                                    if decision.best_contradiction >= 0.70:
                                        decision.label = FactLabel.REFUTED
                                        decision.add_note(f"RELATIONAL_CLAIM: REFUTED due to high contradiction (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f})")
                                    else:
                                        decision.label = FactLabel.NOT_ENOUGH_INFO
                                        decision.add_note(f"RELATIONAL_CLAIM: NOT_ENOUGH_INFO (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f})")
                                
                                # For relational claims, prefer evidence that mentions both subject AND object explicitly
                                # Extract object from parsed_claim or fallback to claim text
                                claim_object = parsed_claim.object if parsed_claim and parsed_claim.object else None
                                
                                # Fallback: try to extract object from claim text for relational claims
                                if not claim_object and is_relational_claim:
                                    claim_object = self._extract_object_from_relational_claim(claim, parsed_claim)
                                
                                # Only if current best evidence doesn't mention both subject and object
                                if claim_object and not (object_in_title or object_in_snippet):
                                    logger.info(
                                        f"RELATIONAL_CLAIM: Current best evidence doesn't mention object '{claim_object}'. "
                                        f"Searching for better evidence..."
                                    )
                                    # Try to find better evidence that mentions both subject and object
                                    # Search in ALL evidences, not just the top-3 used for NLI
                                    better_evidence = self._find_best_relational_evidence(
                                        claim, evidences, parsed_claim, decision.evidence_scores, claim_object
                                    )
                                    if better_evidence:
                                        decision.best_evidence = better_evidence["evidence"]
                                        decision.best_entailment = better_evidence["entailment"]
                                        decision.best_contradiction = better_evidence["contradiction"]
                                        decision.add_note(f"RELATIONAL_CLAIM: Updated best_evidence to one that mentions both subject and object '{claim_object}'")
                                        logger.info(
                                            f"RELATIONAL_CLAIM: Updated best_evidence to '{better_evidence['evidence'].title}' "
                                            f"(snippet: '{better_evidence['evidence'].snippet[:100]}...')"
                                        )
                                        
                                        # CRITICAL: Re-evaluate label after updating evidence
                                        # Only set SUPPORTED if entailment > contradiction AND entailment > threshold
                                        if decision.best_entailment > decision.best_contradiction and decision.best_entailment >= threshold:
                                            decision.label = FactLabel.SUPPORTED
                                            decision.add_note(f"RELATIONAL_CLAIM: SUPPORTED after evidence update (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f})")
                                        elif decision.best_contradiction > decision.best_entailment:
                                            # Contradiction is higher - mark as REFUTED or NOT_ENOUGH_INFO
                                            if decision.best_contradiction >= 0.70:
                                                decision.label = FactLabel.REFUTED
                                                decision.add_note(f"RELATIONAL_CLAIM: REFUTED after evidence update (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f})")
                                            else:
                                                decision.label = FactLabel.NOT_ENOUGH_INFO
                                                decision.add_note(f"RELATIONAL_CLAIM: NOT_ENOUGH_INFO after evidence update (entail={decision.best_entailment:.3f}, contra={decision.best_contradiction:.3f})")
                                    else:
                                        logger.info(
                                            f"RELATIONAL_CLAIM: No better evidence found that mentions both subject and object '{claim_object}'"
                                        )
                        elif subject_in_snippet:
                            # Subject in snippet but not in title - still relevant, but less confident
                            if decision.best_entailment > decision.best_contradiction and decision.best_entailment > 0:
                                decision.label = FactLabel.SUPPORTED
                                decision.add_note(f"RELATIONAL_CLAIM: accepted with low NLI score (entail={decision.best_entailment:.3f}) due to subject in snippet")
                            else:
                                decision.label = FactLabel.NOT_ENOUGH_INFO
                        else:
                            decision.label = FactLabel.NOT_ENOUGH_INFO
                    else:
                        decision.label = FactLabel.NOT_ENOUGH_INFO
                else:
                    decision.label = FactLabel.NOT_ENOUGH_INFO
            else:
                decision.label = FactLabel.NOT_ENOUGH_INFO
            decision.add_note(f"DYNAMIC_THRESHOLD: threshold={threshold:.2f}, label={decision.label.value}")
        
        return decision
    
    def _is_numerical_claim(self, claim_lower: str, evidence_lower: str) -> bool:
        """
        Check if claim is a numerical claim (e.g., "X hat Y Millionen Einwohner").
        
        Args:
            claim_lower: Lowercased claim text
            evidence_lower: Lowercased evidence text
            
        Returns:
            True if both claim and evidence contain numerical patterns
        """
        # Pattern: "hat/haben/hatte/hatten" + number + "Millionen" + "Einwohner"
        numerical_patterns = [
            r"\d+[.,]?\d*\s*(?:millionen|million|mio|milliarden|milliard)\s*(?:einwohner|bevölkerung)",
            r"(?:einwohner|bevölkerung).*?\d+[.,]?\d*\s*(?:millionen|million|mio|milliarden)",
        ]
        
        claim_has_pattern = any(re.search(pattern, claim_lower) for pattern in numerical_patterns)
        evidence_has_pattern = any(re.search(pattern, evidence_lower) for pattern in numerical_patterns)
        
        return claim_has_pattern and evidence_has_pattern
    
    def _is_numerical_claim_pattern(self, claim_lower: str) -> bool:
        """
        Check if claim text matches numerical claim pattern (e.g., "X hat Y Millionen Einwohner").
        This checks only the claim, not the evidence.
        
        Args:
            claim_lower: Lowercased claim text
            
        Returns:
            True if claim matches numerical pattern
        """
        # Pattern: "hat/haben/hatte/hatten" + number + "Millionen" + "Einwohner"
        numerical_patterns = [
            r"\d+[.,]?\d*\s*(?:millionen|million|mio|milliarden|milliard)\s*(?:einwohner|bevölkerung)",
            r"(?:einwohner|bevölkerung).*?\d+[.,]?\d*\s*(?:millionen|million|mio|milliarden)",
            r"(?:hat|hatte|haben|hatten)\s+\d+[.,]?\d*\s*(?:millionen|million|mio|milliarden)\s*(?:einwohner|bevölkerung)",
        ]
        
        return any(re.search(pattern, claim_lower) for pattern in numerical_patterns)
    
    def _clean_html_entities(self, text: str) -> str:
        """
        Clean HTML entities from text (e.g., &#160; -> space, &#91; -> [).
        
        Args:
            text: Text with HTML entities
            
        Returns:
            Cleaned text
        """
        # Decode HTML entities (e.g., &#160; -> space, &#91; -> [)
        try:
            cleaned = html.unescape(text)
        except Exception:
            cleaned = text
        # Also handle numeric entities manually (fallback)
        cleaned = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), cleaned)
        cleaned = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), cleaned)
        return cleaned
    
    def _extract_numbers(self, text: str, debug: bool = False) -> List[float]:
        """
        Extract numerical values from text (handles "83 Millionen", "83,6 Millionen", etc.).
        
        Uses unified population parser for population claims, fallback to legacy patterns for other numerical claims.
        
        Args:
            text: Text to extract numbers from (may contain HTML entities)
            debug: If True, log detailed debugging information
            
        Returns:
            List of extracted numbers (in millions if "Millionen" is present)
        """
        # Try unified population parser first (for population claims)
        text_lower = text.lower()
        is_population_context = any(kw in text_lower for kw in ["einwohner", "bevölkerung", "population", "habitantes", "inhabitants"])
        
        if is_population_context:
            pop_values = parse_population_number(text)
            if pop_values:
                # Convert absolute integers to millions (for backward compatibility)
                numbers = [v.normalized_value_int / 1_000_000.0 for v in pop_values]
                if debug:
                    logger.debug(f"EXTRACT_NUMBERS_DEBUG: Unified parser extracted {len(numbers)} numbers: {numbers}")
                return numbers
        
        # Fallback to legacy extraction for non-population numerical claims
        # Clean HTML entities first (e.g., &#160; -> space)
        text = self._clean_html_entities(text)
        numbers = []
        
        # More robust patterns that handle various formats:
        # - "83 Millionen Einwohner"
        # - "83,6 Millionen Einwohner"
        # - "83.6 Millionen Einwohner"
        # - "etwa 83 Millionen"
        # - "mit 83 Millionen"
        # - "83 Millionen" (standalone)
        # - "hat 83 Millionen Einwohner"
        # - "mit etwa 83 Millionen Einwohnern"
        # - "circa 83 Millionen Einwohner"
        # - "rund 83 Millionen Einwohner"
        # - "ca. 83 Millionen Einwohner"
        # - "83 Millionen Menschen"
        # - "83 Mio. Einwohner"
        patterns = [
            r"(\d+)[.,](\d+)\s*(?:millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?",  # "83,6 Millionen" or "83.6 Millionen Einwohner"
            r"(\d+)\s*(?:millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?",  # "83 Millionen" or "83 Millionen Einwohner"
            r"(?:einwohner|bevölkerung|menschen).*?(\d+)[.,](\d+)\s*(?:millionen|million|mio\.?)",  # "Einwohner: 83,6 Millionen"
            r"(?:einwohner|bevölkerung|menschen).*?(\d+)\s*(?:millionen|million|mio\.?)",  # "Einwohner: 83 Millionen"
            # Qualifiers before number (e.g., "mehr als 300 Millionen", "über 200 Millionen", "knapp 10 Millionen", "fast 80 Millionen", "gut 80 Millionen")
            r"(?:mehr\s+als|über|knapp|fast|gut)\s*(\d+)[.,](\d+)\s*(?:millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?",
            r"(?:mehr\s+als|über|knapp|fast|gut)\s*(\d+)\s*(?:millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?",
            r"(?:circa|etwa|ungefähr|rund|ca\.?)\s*(\d+)[.,](\d+)\s*(?:millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?",  # "circa 83,6 Millionen"
            r"(?:circa|etwa|ungefähr|rund|ca\.?)\s*(\d+)\s*(?:millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?",  # "circa 83 Millionen"
            r"(\d+)[.,](\d+)\s*(?:millionen|million|mio\.?)\b",  # Fallback: "83,6 Millionen" (without Einwohner)
            r"(\d+)\s*(?:millionen|million|mio\.?)\b",  # Fallback: "83 Millionen" (without Einwohner)
            # Additional patterns for cases where numbers appear without explicit "Einwohner" keyword
            r"(\d+)[.,](\d+)\s*(?:millionen|million|mio\.?)\s*(?:menschen|personen|bürger)?",  # "83,6 Millionen Menschen"
            r"(\d+)\s*(?:millionen|million|mio\.?)\s*(?:menschen|personen|bürger)?",  # "83 Millionen Menschen"
            r"(\d+)[.,](\d+)\s*(?:millionen|million|mio\.?)\s*(?:bewohner|einwohnerzahl|bevölkerungszahl)?",  # "83,6 Millionen Bewohner"
            r"(\d+)\s*(?:millionen|million|mio\.?)\s*(?:bewohner|einwohnerzahl|bevölkerungszahl)?",  # "83 Millionen Bewohner"
            # Patterns for numbers that appear in context (e.g., "hat 10 Millionen", "hatte 3,9 Millionen")
            # Pattern with year between verb and number (e.g., "hatte 2022 3,9 Millionen")
            r"(?:hat|hatte|haben|hatten|zählt|zählte|zählen|zählten)\s+(?:19|20)\d{2}\s+(\d+)[.,](\d+)\s*(?:millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?",  # "hatte 2022 3,9 Millionen Einwohner"
            r"(?:hat|hatte|haben|hatten|zählt|zählte|zählen|zählten)\s+(?:19|20)\d{2}\s+(\d+)\s*(?:millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?",  # "hatte 2022 10 Millionen Einwohner"
            r"(?:hat|hatte|haben|hatten|zählt|zählte|zählen|zählten)\s+(\d+)[.,](\d+)\s*(?:millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?",  # "hatte 3,9 Millionen Einwohner"
            r"(?:hat|hatte|haben|hatten|zählt|zählte|zählen|zählten)\s+(\d+)\s*(?:millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?",  # "hat 10 Millionen Einwohner"
            r"(?:hat|hatte|haben|hatten|zählt|zählte|zählen|zählten)\s+(\d+)[.,](\d+)\s*(?:millionen|million|mio\.?)",  # "hat 10,3 Millionen"
            r"(?:hat|hatte|haben|hatten|zählt|zählte|zählen|zählten)\s+(\d+)\s*(?:millionen|million|mio\.?)",  # "hat 10 Millionen"
            # Patterns for numbers in parentheses or after colons
            r"[:\(]\s*(\d+)[.,](\d+)\s*(?:millionen|million|mio\.?)",  # ": 10,3 Millionen" or "(10,3 Millionen)"
            r"[:\(]\s*(\d+)\s*(?:millionen|million|mio\.?)",  # ": 10 Millionen" or "(10 Millionen)"

            # --- NEW: Raw population integers with thousands separators (common in infoboxes) ---
            # Examples: "Einwohnerzahl: 10.467.366", "Bevölkerung 10 467 366", "10.467.366 Einwohner"
            # Also handles Portuguese "habitantes" and other languages
            r"(\d{1,3}(?:[.\s]\d{3})+)\s*(?:einwohner|einwohnerzahl|bevölkerung|bevölkerungszahl|bewohner|menschen|habitantes|inhabitants)\b",
            r"(?:einwohner|einwohnerzahl|bevölkerung|bevölkerungszahl|bewohner|menschen|habitantes|inhabitants)\D{0,30}(\d{1,3}(?:[.\s]\d{3})+)\b",
            # Pattern for "tem X habitantes" (Portuguese: "has X inhabitants")
            r"(?:tem|hat|hatte|haben|hatten|zählt|zählte|zählen|zählten)\s+(\d{1,3}(?:[.\s]\d{3})+)\s*(?:habitantes|einwohner|bewohner|menschen)\b",
            # Pattern for "X habitantes" (Portuguese: "X inhabitants")
            r"(\d{1,3}(?:[.\s]\d{3})+)\s*(?:habitantes|inhabitants)\b",
        ]
        
        if debug:
            logger.debug(f"EXTRACT_NUMBERS_DEBUG: Extracting from text='{text[:200]}...' (length={len(text)})")
        
        for pattern_idx, pattern in enumerate(patterns):
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if debug and matches:
                logger.debug(f"EXTRACT_NUMBERS_DEBUG: Pattern[{pattern_idx}] matched {len(matches)} times")
            for match in matches:
                try:
                    # Check if this match is part of a percentage (e.g., "0,03%" should be ignored)
                    match_start = match.start()
                    match_end = match.end()
                    context_before = text[max(0, match_start - 10):match_start]
                    context_after = text[match_end:min(len(text), match_end + 10)]
                    
                    # Skip if it's a percentage (look for "%" nearby)
                    if "%" in context_after or "%" in context_before:
                        if debug:
                            logger.debug(f"EXTRACT_NUMBERS_DEBUG: Pattern[{pattern_idx}] match skipped (percentage): '{match.group(0)}'")
                        continue
                    
                    # Skip if it's a very small number (< 0.1) that's not part of a larger number
                    # This filters out things like "0,03" which are likely percentages or other metrics
                    if match.lastindex == 2:
                        integer_part = match.group(1)
                        decimal_part = match.group(2)
                        number = float(f"{integer_part}.{decimal_part}")
                        if number < 0.1:
                            if debug:
                                logger.debug(f"EXTRACT_NUMBERS_DEBUG: Pattern[{pattern_idx}] match skipped (too small): {number:.3f}")
                            continue
                        numbers.append(number)
                        if debug:
                            logger.debug(f"EXTRACT_NUMBERS_DEBUG: Pattern[{pattern_idx}] extracted decimal number={number:.1f}M")
                    elif match.lastindex == 1:
                        # Integer number (e.g., "83") OR raw integer with separators (e.g., "10.467.366")
                        raw = match.group(1)
                        if raw and (("." in raw) or (" " in raw)) and re.match(r"^\d{1,3}(?:[.\s]\d{3})+$", raw.strip()):
                            # Treat as raw absolute value; convert to millions
                            cleaned = raw.replace(".", "").replace(" ", "").strip()
                            val_int = int(cleaned)
                            number = float(val_int) / 1_000_000.0
                            # Filter out very small numbers (< 0.1 million) from raw integers too
                            if number < 0.1:
                                if debug:
                                    logger.debug(f"EXTRACT_NUMBERS_DEBUG: Pattern[{pattern_idx}] match skipped (too small): {number:.3f}M")
                                continue
                            numbers.append(number)
                            if debug:
                                logger.debug(
                                    f"EXTRACT_NUMBERS_DEBUG: Pattern[{pattern_idx}] extracted raw integer='{raw}' -> {number:.3f}M"
                                )
                        else:
                            number = float(raw)
                            # Filter out very small numbers (< 0.1 million)
                            if number < 0.1:
                                if debug:
                                    logger.debug(f"EXTRACT_NUMBERS_DEBUG: Pattern[{pattern_idx}] match skipped (too small): {number:.3f}M")
                                continue
                            numbers.append(number)
                            if debug:
                                logger.debug(f"EXTRACT_NUMBERS_DEBUG: Pattern[{pattern_idx}] extracted integer number={number:.1f}M")
                except (ValueError, IndexError) as e:
                    # Skip invalid matches
                    if debug:
                        logger.debug(f"EXTRACT_NUMBERS_DEBUG: Pattern[{pattern_idx}] match failed: {e}")
                    continue
        
        if debug:
            logger.info(f"EXTRACT_NUMBERS_DEBUG: Extracted {len(numbers)} numbers: {numbers}")
        
        return numbers
    
    def _extract_object_from_relational_claim(self, claim: str, parsed_claim: Optional[ParsedClaim]) -> Optional[str]:
        """
        Extract object from relational claim text as fallback if parser didn't extract it.
        
        For claims like "Der Rhein fließt durch Deutschland", extract "Deutschland".
        """
        if not claim:
            return None
        
        import re
        
        # Common relational patterns with prepositions
        relational_patterns = [
            r"fließt\s+durch\s+(?:der|die|das|ein|eine)?\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)",  # "fließt durch Deutschland"
            r"durchfließt\s+(?:der|die|das|ein|eine)?\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)",  # "durchfließt Deutschland"
            r"durchquert\s+(?:der|die|das|ein|eine)?\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)",  # "durchquert Deutschland"
            r"umkreist\s+(?:der|die|das|ein|eine)?\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)",  # "umkreist die Sonne"
            r"kreist\s+um\s+(?:der|die|das|ein|eine)?\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)",  # "kreist um die Sonne"
            r"besteht\s+aus\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)",  # "besteht aus Wasserstoff"
        ]
        
        for pattern in relational_patterns:
            match = re.search(pattern, claim, re.IGNORECASE)
            if match:
                object_text = match.group(1).strip()
                if object_text:
                    return object_text
        
        # Fallback: look for capitalized words after relational markers
        relational_markers = ["durch", "um", "aus"]
        words = claim.split()
        for i, word in enumerate(words):
            word_lower = word.lower().rstrip('.,!?')
            if word_lower in relational_markers and i + 1 < len(words):
                # Next word might be the object
                next_word = words[i + 1]
                # Remove determiners
                if next_word.lower() in ["der", "die", "das", "ein", "eine"]:
                    if i + 2 < len(words):
                        next_word = words[i + 2]
                # Check if it's capitalized (likely a proper noun)
                if next_word and next_word[0].isupper():
                    object_text = next_word.rstrip('.,!?')
                    return object_text
        
        return None
    
    def _find_best_relational_evidence(
        self,
        claim: str,
        evidences: List[Evidence],
        parsed_claim: Optional[ParsedClaim],
        evidence_scores: List[Dict[str, Any]],
        claim_object: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find best evidence for relational claims that mentions both subject and object.
        
        Searches in ALL evidences, not just the top-3 used for NLI.
        
        Args:
            claim_object: Optional object text (if not in parsed_claim.object)
        
        Returns:
            Evidence score dict with best evidence, or None if not found
        """
        # Get subject and object
        subject = parsed_claim.subject if parsed_claim and parsed_claim.subject else None
        object_text = claim_object or (parsed_claim.object if parsed_claim and parsed_claim.object else None)
        
        if not subject or not object_text:
            return None
        
        subject_lower = subject.lower()
        object_lower = object_text.lower()
        
        # Create a map of evidence (title+url) -> scores for quick lookup
        # Use title+url as key since Evidence objects are not hashable
        evidence_to_scores = {}
        for entry in evidence_scores:
            ev = entry["evidence"]
            key = (ev.title, ev.url)
            evidence_to_scores[key] = entry
        
        # Score evidences by how well they mention both subject and object
        # Search in ALL evidences, not just the top-3
        scored_candidates = []
        for evidence in evidences:
            title_lower = evidence.title.lower()
            snippet_lower = evidence.snippet.lower()
            
            # Check if subject and object are mentioned
            subject_in_title = any(token in title_lower for token in subject_lower.split() if len(token) > 3)
            subject_in_snippet = any(token in snippet_lower for token in subject_lower.split() if len(token) > 3)
            object_in_title = any(token in title_lower for token in object_lower.split() if len(token) > 3)
            object_in_snippet = any(token in snippet_lower for token in object_lower.split() if len(token) > 3)
            
            # Score: higher if both subject and object are mentioned
            score = 0.0
            if subject_in_title and object_in_snippet:
                score = 3.0  # Best: subject in title, object in snippet
            elif subject_in_title and object_in_title:
                score = 2.5  # Good: both in title
            elif subject_in_snippet and object_in_snippet:
                score = 2.0  # Good: both in snippet
            elif subject_in_snippet and object_in_title:
                score = 1.5  # OK: one in snippet, one in title
            elif subject_in_title or object_in_title:
                score = 1.0  # Weak: only one in title
            elif subject_in_snippet or object_in_snippet:
                score = 0.5  # Very weak: only one in snippet
            
            # Only consider if both are mentioned
            if score >= 1.5:
                # Get NLI scores if available, otherwise use dummy scores
                key = (evidence.title, evidence.url)
                if key in evidence_to_scores:
                    entry = evidence_to_scores[key]
                else:
                    # Create dummy entry for evidence not in top-3
                    entry = {
                        "evidence": evidence,
                        "entailment": 0.0,
                        "contradiction": 0.0,
                        "neutral": 1.0,
                    }
                
                scored_candidates.append({
                    "entry": entry,
                    "score": score,
                    "subject_in_title": subject_in_title,
                    "object_in_snippet": object_in_snippet,
                    "both_in_snippet": subject_in_snippet and object_in_snippet,
                })
        
        if not scored_candidates:
            # No evidence found that mentions both subject and object explicitly
            logger.debug(
                f"RELATIONAL_CLAIM: No evidence found that mentions both subject '{subject}' and object '{object_text}' "
                f"(searched {len(evidences)} evidences)"
            )
            return None
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Return the best candidate
        best = scored_candidates[0]
        logger.info(
            f"RELATIONAL_CLAIM: Found better evidence with score={best['score']:.1f} "
            f"(subject_in_title={best['subject_in_title']}, object_in_snippet={best['object_in_snippet']}, "
            f"both_in_snippet={best['both_in_snippet']}, title='{best['entry']['evidence'].title}')"
        )
        return best["entry"]

