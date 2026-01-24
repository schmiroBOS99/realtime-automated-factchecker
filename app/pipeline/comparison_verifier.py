"""
Comparison claim verifier.

Handles comparison claims like "Der Nil ist länger als der Rhein" by:
1. Extracting numeric values from evidence for each entity
2. If no values found in evidence, try Wikidata as fallback
3. Comparing the values
4. Returning SUPPORTED/REFUTED/NOT_ENOUGH_INFO
"""
import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from app.pipeline.models import Evidence, FactLabel

logger = logging.getLogger(__name__)


class ComparisonVerifier:
    """Verifies comparison claims by extracting and comparing numeric values."""
    
    def __init__(self, wikidata_client: Optional[Any] = None):
        """
        Initialize ComparisonVerifier.
        
        Args:
            wikidata_client: Optional WikidataClient instance for fallback value extraction
        """
        self.wikidata_client = wikidata_client
    
    def verify(
        self,
        claim: str,
        evidences: List[Evidence],
        subclaims: List[str],
    ) -> Dict:
        """
        Verify a comparison claim by extracting numeric values and comparing them.
        
        Args:
            claim: Original claim text (e.g., "Der Nil ist länger als der Rhein")
            evidences: List of Evidence objects from subclaim retrieval
            subclaims: List of subclaims (e.g., ["nil länge", "rhein länge", "nil länger als rhein"])
            
        Returns:
            Dict with keys: label, best_evidence, scores, comparison_details
        """
        if not subclaims or len(subclaims) < 2:
            return {
                "label": FactLabel.NOT_ENOUGH_INFO,
                "best_evidence": None,
                "scores": [],
                "comparison_details": {"error": "Insufficient subclaims for comparison"},
            }
        
        # Parse comparison claim to extract entities and comparison word
        comparison_info = self._parse_comparison_claim(claim)
        if not comparison_info:
            return {
                "label": FactLabel.NOT_ENOUGH_INFO,
                "best_evidence": None,
                "scores": [],
                "comparison_details": {"error": "Failed to parse comparison claim"},
            }
        
        x_entity, y_entity, comparison_word, attribute = comparison_info
        
        # Extract numeric values for each entity
        x_value, x_evidence = self._extract_numeric_value(evidences, x_entity, attribute)
        y_value, y_evidence = self._extract_numeric_value(evidences, y_entity, attribute)
        
        # Note: Wikidata fallback will be handled asynchronously in factchecker.py
        # since verify() is synchronous and WikidataClient is async
        
        # Compare values
        if x_value is None or y_value is None:
            return {
                "label": FactLabel.NOT_ENOUGH_INFO,
                "best_evidence": x_evidence if x_evidence else (y_evidence if y_evidence else None),
                "scores": [],
                "comparison_details": {
                    "x_entity": x_entity,
                    "y_entity": y_entity,
                    "comparison_word": comparison_word,
                    "attribute": attribute,
                    "x_value": x_value,
                    "y_value": y_value,
                    "error": "Could not extract numeric values for comparison",
                },
            }
        
        # Perform comparison
        comparison_result = self._compare_values(x_value, y_value, comparison_word)
        
        # Determine label
        if comparison_result is True:
            label = FactLabel.SUPPORTED
        elif comparison_result is False:
            label = FactLabel.REFUTED
        else:
            label = FactLabel.NOT_ENOUGH_INFO
        
        # Choose best evidence (prefer evidence with both values, otherwise x_evidence)
        best_evidence = x_evidence if x_evidence else y_evidence
        
        logger.info(
            f"COMPARISON_VERIFY: {x_entity}={x_value} {comparison_word} {y_entity}={y_value} "
            f"-> {label.value}"
        )
        
        return {
            "label": label,
            "best_evidence": best_evidence,
            "scores": [],
            "comparison_details": {
                "x_entity": x_entity,
                "y_entity": y_entity,
                "x_value": x_value,
                "y_value": y_value,
                "comparison_word": comparison_word,
                "attribute": attribute,
                "result": comparison_result,
            },
        }
    
    def _parse_comparison_claim(self, claim: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Parse comparison claim to extract entities and comparison word.
        
        Args:
            claim: Comparison claim text
            
        Returns:
            Tuple of (x_entity, y_entity, comparison_word, attribute) or None
        """
        claim_lower = claim.lower().rstrip('.,!?')
        
        # Extended comparison words pattern
        comparison_words = (
            "länger|kürzer|größer|kleiner|mehr|weniger|höher|niedriger|"
            "älter|jünger|früher|später|besser|schlechter|mehr|weniger|"
            "längere|kürzere|größere|kleinere|höhere|niedrigere|"
            "ältere|jüngere|frühere|spätere"
        )
        
        # Pattern: "X ist länger als Y" or "X hat mehr Tore als Y"
        pattern1 = rf'(.+?)\s+(ist|war|sind|waren|hat|hatte|haben|hatten)\s+({comparison_words})\s+als\s+(.+)'
        match = re.search(pattern1, claim_lower)
        
        if match:
            x_entity = match.group(1).strip()
            comparison_word = match.group(3).strip()
            y_entity = match.group(4).strip().rstrip('.,')
        else:
            # Pattern: "X länger als Y"
            pattern2 = rf'(.+?)\s+({comparison_words})\s+als\s+(.+)'
            match = re.search(pattern2, claim_lower)
            if match:
                x_entity = match.group(1).strip()
                comparison_word = match.group(2).strip()
                y_entity = match.group(3).strip().rstrip('.,')
            else:
                return None
        
        # Clean entities (remove determiners)
        x_entity_cleaned = re.sub(r'^(der|die|das|ein|eine|einen|einem|einer)\s+', '', x_entity, flags=re.IGNORECASE)
        y_entity_cleaned = re.sub(r'^(der|die|das|ein|eine|einen|einem|einer)\s+', '', y_entity, flags=re.IGNORECASE)
        
        # Extended attribute mapping (use comparison_word directly, not normalized)
        attr_map = {
            # Length
            "länger": "länge",
            "längere": "länge",
            "kürzer": "länge",
            "kürzere": "länge",
            "läng": "länge",
            "kürz": "länge",
            # Size/Area
            "größer": "größe",
            "größere": "größe",
            "kleiner": "größe",
            "kleinere": "größe",
            "groß": "größe",
            "klein": "größe",
            # Height
            "höher": "höhe",
            "höhere": "höhe",
            "niedriger": "höhe",
            "niedrigere": "höhe",
            "hoch": "höhe",
            "niedrig": "höhe",
            # Age
            "älter": "alter",
            "ältere": "alter",
            "jünger": "alter",
            "jüngere": "alter",
            "alt": "alter",
            "jung": "alter",
            # Year/Date
            "früher": "jahr",
            "frühere": "jahr",
            "später": "jahr",
            "spätere": "jahr",
            "früh": "jahr",
            "spät": "jahr",
            # General quantity (needs context from claim)
            "mehr": "anzahl",  # Default, may be overridden by context
            "weniger": "anzahl",  # Default, may be overridden by context
            # Quality (rarely numeric, but included for completeness)
            "besser": "qualität",
            "schlechter": "qualität",
        }
        
        # Try to infer attribute from claim context for "mehr/weniger"
        attribute = attr_map.get(comparison_word, "anzahl")
        
        # Context-based attribute inference for "mehr/weniger"
        if comparison_word in ["mehr", "weniger"]:
            # Check claim for specific contexts
            if any(word in claim_lower for word in ["tore", "tor", "goal", "goals"]):
                attribute = "tore"
            elif any(word in claim_lower for word in ["regierungsjahre", "regierung", "amt", "amtzeit"]):
                attribute = "regierungsjahre"
            elif any(word in claim_lower for word in ["einwohner", "population", "bevölkerung"]):
                attribute = "anzahl"
            elif any(word in claim_lower for word in ["titel", "meisterschaft", "siege", "sieg"]):
                attribute = "titel"
            elif any(word in claim_lower for word in ["jahren", "jahr", "jahre alt"]):
                attribute = "alter"
            elif any(word in claim_lower for word in ["im jahr", "jahr", "seit"]):
                attribute = "jahr"
            # Default to "anzahl" if no specific context found
        
        return (x_entity_cleaned, y_entity_cleaned, comparison_word, attribute)
    
    def _extract_numeric_value(
        self, evidences: List[Evidence], entity: str, attribute: str
    ) -> Tuple[Optional[float], Optional[Evidence]]:
        """
        Extract numeric value for an entity from evidence.
        
        Args:
            evidences: List of Evidence objects
            entity: Entity name (e.g., "nil", "rhein")
            attribute: Attribute being compared (e.g., "länge", "größe")
            
        Returns:
            Tuple of (numeric_value, best_evidence) or (None, None)
        """
        entity_lower = entity.lower()
        attribute_lower = attribute.lower()
        
        # Extended keywords that might indicate the attribute
        attribute_keywords = {
            "länge": ["länge", "läng", "kilometer", "km", "meilen", "miles", "lang", "langen"],
            "größe": ["größe", "groß", "fläche", "quadratkilometer", "km²", "hektar", "flächengröße"],
            "anzahl": ["anzahl", "zahl", "einwohner", "population", "bevölkerung"],
            "höhe": ["höhe", "hoch", "meter", "m", "fuß", "feet", "metern", "meter hoch"],
            "alter": ["alter", "jahre alt", "jahr alt", "geboren", "geburtsjahr", "lebensjahre", "alt"],
            "jahr": ["jahr", "jahre", "im jahr", "seit", "von", "bis"],
            "tore": ["tore", "tor", "toren", "goal", "goals", "karrieretore", "ligatore", "torschützen"],
            "regierungsjahre": ["regierungsjahre", "regierung", "amt", "amtzeit", "regierte", "regiert", "amtszeit"],
            "titel": ["titel", "meisterschaft", "siege", "sieg", "gewann", "gewonnen", "championship"],
            "qualität": ["besser", "schlechter", "qualität", "wertung"],
        }
        keywords = attribute_keywords.get(attribute_lower, [attribute_lower])
        
        # Context filters: exclude patterns that indicate wrong attribute
        exclude_patterns = []
        if attribute_lower == "länge":
            # Exclude height indicators (ü., ü.N.N., über Normalnull, etc.)
            exclude_patterns = [
                r'\d+\s*m\s*ü\.', r'\d+\s*meter\s*ü\.', r'\d+\s*m\s*über\s*normalnull',
                r'\d+\s*m\s*ü\.?\s*n\.?\s*n\.?', r'entspringt.*\d+\s*m', r'quelle.*\d+\s*m',
                r'mündung.*\d+\s*m', r'mündet.*\d+\s*m', r'fließt.*\d+\s*m\s*ü\.',
            ]
            exclude_patterns.extend([r'\d+\s*km²', r'\d+\s*quadratkilometer', r'\d+\s*hektar', r'fläche.*\d+'])
            exclude_patterns.extend([r'\d+\s*einwohner', r'\d+\s*population', r'bevölkerung.*\d+'])
        elif attribute_lower == "höhe":
            exclude_patterns = [
                r'\d+\s*km\s*(lang|langen)', r'\d+\s*kilometer\s*(lang|langen)', r'\d+\s*meilen\s*(lang|langen)',
                r'länge.*\d+\s*km', r'\d+\s*km²', r'\d+\s*quadratkilometer', r'\d+\s*hektar',
                r'\d+\s*einwohner', r'\d+\s*population',
            ]
        elif attribute_lower == "größe":
            exclude_patterns = [
                r'\d+\s*km\s*(lang|langen)', r'\d+\s*kilometer\s*(lang|langen)', r'länge.*\d+',
                r'\d+\s*m\s*ü\.', r'\d+\s*meter\s*ü\.', r'\d+\s*einwohner', r'\d+\s*population',
            ]
        elif attribute_lower == "anzahl":
            exclude_patterns = [
                r'\d+\s*km\s*(lang|langen)', r'\d+\s*kilometer', r'\d+\s*m\s*ü\.',
                r'\d+\s*km²', r'\d+\s*quadratkilometer',
            ]
        elif attribute_lower == "alter":
            # Exclude years that are not ages (e.g., birth years, historical dates)
            exclude_patterns = [
                r'\d{4}',  # Exclude 4-digit years (likely birth years, not age)
                r'geboren.*\d{4}', r'geburt.*\d{4}',
                r'\d+\s*km', r'\d+\s*tore', r'\d+\s*einwohner',
            ]
        elif attribute_lower == "jahr":
            # Exclude ages, lengths, etc. when looking for years
            exclude_patterns = [
                r'\d{1,3}\s*jahre\s+alt', r'\d{1,3}\s+alt', r'alter.*\d+',
                r'\d+\s*km', r'\d+\s*tore',
            ]
        elif attribute_lower == "tore":
            # Exclude other numeric values when looking for goals
            exclude_patterns = [
                r'\d+\s*jahre', r'\d{4}',  # Years
                r'\d+\s*km', r'\d+\s*einwohner',
            ]
        elif attribute_lower == "regierungsjahre":
            # Exclude other numeric values when looking for years in office
            exclude_patterns = [
                r'\d+\s*tore', r'\d+\s*km',
            ]
        elif attribute_lower == "titel":
            # Exclude other numeric values when looking for titles
            exclude_patterns = [
                r'\d+\s*km', r'\d+\s*einwohner', r'\d{4}',  # Years
            ]
        
        # Score evidences by relevance
        scored_evidences = []
        for ev in evidences:
            snippet_lower = ev.snippet.lower()
            title_lower = ev.title.lower()
            
            # Check if entity is mentioned
            entity_mentioned = entity_lower in snippet_lower or entity_lower in title_lower
            
            if not entity_mentioned:
                continue
            
            # STRONG FILTER: For comparison subclaims, prefer evidences where entity is in the title
            # This prevents picking values from related entities (e.g., Zermatt's height when looking for Matterhorn)
            # Only apply this filter if we have multiple evidences (to avoid over-filtering when we have few results)
            if len(evidences) > 1:
                entity_in_title = entity_lower in title_lower
                # Skip evidences where entity is NOT in title (unless we have very few evidences)
                if not entity_in_title:
                    # Only skip if we have at least 5 evidences (enough to find one with entity in title)
                    entity_in_title_count = sum(1 for e in evidences if entity_lower in e.title.lower())
                    if entity_in_title_count >= 1:
                        # We have at least one evidence with entity in title, so skip this one
                        logger.debug(
                            f"COMPARISON_EXTRACT_FILTER: entity='{entity}' attribute='{attribute}' "
                            f"skipping evidence with title '{ev.title}' (entity not in title, "
                            f"but {entity_in_title_count} evidences with entity in title found)"
                        )
                        continue
            
            # Filter out snippets that match exclude patterns (wrong attribute context)
            if exclude_patterns:
                matches_exclude = False
                for exclude_pattern in exclude_patterns:
                    if re.search(exclude_pattern, snippet_lower):
                        logger.debug(
                            f"COMPARISON_EXTRACT_FILTER: entity='{entity}' attribute='{attribute}' "
                            f"excluded snippet due to pattern '{exclude_pattern}': '{snippet_lower[:80]}...'"
                        )
                        matches_exclude = True
                        break
                if matches_exclude:
                    continue
            
            # Look for numeric patterns with units (attribute-specific)
            keywords_pattern = '|'.join(keywords) if keywords else 'länge|größe|anzahl|höhe'
            numeric_patterns = []
            
            if attribute_lower == "länge":
                # For length: prioritize km, but also accept "lang" keyword
                numeric_patterns = [
                    r'(\d{1,5}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(km|kilometer)',
                    r'(\d{1,5}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(?:kilometer|km)\s+(?:lang|langen|länge)',
                    r'(\d{1,5}(?:[.,]\d{3})*(?:[.,]\d+)?)\s+(?:lang|langen|länge)',
                ]
            elif attribute_lower == "höhe":
                # For height: prioritize "m" but require height context
                numeric_patterns = [
                    r'(\d{1,5}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(?:meter|m)\s+(?:hoch|höhe)',
                    r'(\d{1,5}(?:[.,]\d{3})*(?:[.,]\d+)?)\s+(?:hoch|höhe)',
                    r'(\d{1,5}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(?:meter|m)(?!\s*ü\.)',
                ]
            elif attribute_lower == "größe":
                # For size/area: prioritize km², quadratkilometer
                numeric_patterns = [
                    r'(\d{1,6}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(km²|quadratkilometer|hektar)',
                    r'(\d{1,6}(?:[.,]\d{3})*(?:[.,]\d+)?)\s+(?:fläche|größe)',
                ]
            elif attribute_lower == "anzahl":
                # For population: prioritize einwohner, population
                numeric_patterns = [
                    r'(\d{1,8}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(?:einwohner|population|bevölkerung)',
                    r'(\d{1,8}(?:[.,]\d{3})*(?:[.,]\d+)?)\s+(?:hat|haben|hatte)\s+(?:einwohner|population)',
                ]
            elif attribute_lower == "alter":
                # For age: years old, born in year, age in years
                numeric_patterns = [
                    r'(\d{1,3})\s*(?:jahre|jahr)\s+alt',
                    r'(\d{1,3})\s+alt',
                    r'geboren\s+(?:im\s+)?(?:jahr\s+)?(\d{4})',  # "geboren 1950" or "geboren im Jahr 1950"
                    r'geburt\s+(?:im\s+)?(?:jahr\s+)?(\d{4})',
                    r'alter\s+von\s+(\d{1,3})',
                ]
            elif attribute_lower == "jahr":
                # For year: specific years, dates
                numeric_patterns = [
                    r'(?:im\s+)?(?:jahr\s+)?(\d{4})',  # "im Jahr 1945" or "1945"
                    r'seit\s+(\d{4})',
                    r'von\s+(\d{4})\s+bis',
                    r'(\d{4})\s+bis',
                    r'(\d{1,2})\.\s*(\d{4})',  # "1. Januar 1945" - capture year
                ]
            elif attribute_lower == "tore":
                # For goals: goals scored, career goals
                numeric_patterns = [
                    r'(\d{1,4})\s*(?:tore|tor|toren|goals|goal)',
                    r'(?:tore|tor|toren)\s+(?:in|von|bei)\s+(\d{1,4})',
                    r'karrieretore[:\s]+(\d{1,4})',
                    r'(?:hat|haben|hatte)\s+(\d{1,4})\s*(?:tore|tor|toren)',
                ]
            elif attribute_lower == "regierungsjahre":
                # For years in office: regierungsjahre, amtszeit
                numeric_patterns = [
                    r'(\d{1,2})\s*(?:jahre|jahr)\s+(?:lang|regiert|im amt|amtszeit)',
                    r'regierte\s+(?:von\s+\d{4}\s+)?bis\s+(\d{4})',  # "regierte bis 2021"
                    r'regierte\s+(\d{1,2})\s*(?:jahre|jahr)',
                    r'amtszeit[:\s]+(\d{4})\s+bis\s+(\d{4})',  # Capture both years, use end year
                    r'(\d{4})\s+bis\s+(\d{4})',  # "2005 bis 2021" - use end year
                ]
            elif attribute_lower == "titel":
                # For titles/championships: titles won, championships
                numeric_patterns = [
                    r'(\d{1,3})\s*(?:titel|meisterschaft|championship)',
                    r'(?:titel|meisterschaft)[:\s]+(\d{1,3})',
                    r'(?:hat|haben|hatte)\s+(\d{1,3})\s*(?:titel|meisterschaft)',
                ]
            else:
                # Fallback: generic patterns
                numeric_patterns = [
                    r'(\d{1,5}(?:[.,]\d{3})*(?:[.,]\d+)?)\s*(km|kilometer|meter|m)',
                    rf'(\d{{1,5}}(?:[.,]\d{{3}})*(?:[.,]\d+)?)\s*(?:{keywords_pattern})\b',
                ]
            
            value = None
            matched_pattern_idx = None
            for pattern_idx, pattern in enumerate(numeric_patterns):
                matches = re.findall(pattern, snippet_lower)
                if matches:
                    # Extract numbers from matches
                    numbers = []
                    for match in matches:
                        if isinstance(match, tuple):
                            num_str = match[0].replace('.', '').replace(',', '.')
                        else:
                            num_str = match.replace('.', '').replace(',', '.')
                        try:
                            num = float(num_str)
                            numbers.append(num)
                        except ValueError:
                            continue
                    
                    if numbers:
                        # Attribute-specific filtering and selection
                        if attribute_lower == "länge":
                            # Filter out very small numbers for length (likely errors)
                            valid_numbers = [n for n in numbers if n >= 10]  # At least 10 km
                            if valid_numbers:
                                value = max(valid_numbers)
                                matched_pattern_idx = pattern_idx
                                break
                        elif attribute_lower == "alter":
                            # Age: typically 0-120 years
                            valid_numbers = [n for n in numbers if 0 <= n <= 120]
                            if valid_numbers:
                                value = max(valid_numbers)
                                matched_pattern_idx = pattern_idx
                                break
                        elif attribute_lower == "jahr":
                            # Year: typically 1000-2100 (historical to future)
                            valid_numbers = [n for n in numbers if 1000 <= n <= 2100]
                            if valid_numbers:
                                # For year comparisons, we might want the earliest/latest depending on context
                                # Default to max (most recent)
                                value = max(valid_numbers)
                                matched_pattern_idx = pattern_idx
                                break
                        elif attribute_lower == "tore":
                            # Goals: typically 0-2000 (career goals)
                            valid_numbers = [n for n in numbers if 0 <= n <= 2000]
                            if valid_numbers:
                                value = max(valid_numbers)
                                matched_pattern_idx = pattern_idx
                                break
                        elif attribute_lower == "regierungsjahre":
                            # Years in office: for dates, use end year; for duration, use the number
                            # If we have years (4 digits), prefer those; otherwise use duration
                            year_numbers = [n for n in numbers if 1800 <= n <= 2100]
                            duration_numbers = [n for n in numbers if 0 <= n <= 50]
                            if year_numbers:
                                value = max(year_numbers)  # End year of term
                                matched_pattern_idx = pattern_idx
                                break
                            elif duration_numbers:
                                value = max(duration_numbers)
                                matched_pattern_idx = pattern_idx
                                break
                        else:
                            # Default: use largest number
                            value = max(numbers)
                            matched_pattern_idx = pattern_idx
                            break
            
            if value is not None:
                # Score: higher if entity is in title, keyword matches, etc.
                # STRONG preference for entity in title (e.g., "Matterhorn" title, not "Zermatt")
                score = 1.0
                entity_in_title = entity_lower in title_lower
                if entity_in_title:
                    score += 2.0  # Strong bonus for entity in title (increased from 1.0)
                else:
                    # Penalty if entity not in title (e.g., "Zermatt" title when looking for "Matterhorn")
                    # This helps avoid picking values from related entities (e.g., Zermatt's height when looking for Matterhorn)
                    score -= 0.5
                
                if any(kw in snippet_lower for kw in keywords):
                    score += 0.5
                
                # Bonus for attribute-specific keywords near the number
                if attribute_lower == "länge" and any(kw in snippet_lower for kw in ["lang", "länge", "kilometer", "km"]):
                    score += 0.5
                elif attribute_lower == "höhe" and any(kw in snippet_lower for kw in ["hoch", "höhe"]):
                    score += 0.5
                elif attribute_lower == "größe" and any(kw in snippet_lower for kw in ["fläche", "größe", "km²"]):
                    score += 0.5
                elif attribute_lower == "anzahl" and any(kw in snippet_lower for kw in ["einwohner", "population", "bevölkerung"]):
                    score += 0.5
                elif attribute_lower == "alter" and any(kw in snippet_lower for kw in ["jahre alt", "alter", "geboren"]):
                    score += 0.5
                elif attribute_lower == "jahr" and any(kw in snippet_lower for kw in ["jahr", "im jahr", "seit"]):
                    score += 0.5
                elif attribute_lower == "tore" and any(kw in snippet_lower for kw in ["tore", "tor", "goals"]):
                    score += 0.5
                elif attribute_lower == "regierungsjahre" and any(kw in snippet_lower for kw in ["regierung", "amt", "amtszeit"]):
                    score += 0.5
                elif attribute_lower == "titel" and any(kw in snippet_lower for kw in ["titel", "meisterschaft", "siege"]):
                    score += 0.5
                
                logger.debug(
                    f"COMPARISON_EXTRACT_DEBUG: entity='{entity}' attribute='{attribute}' "
                    f"extracted value={value} from pattern {matched_pattern_idx+1} "
                    f"snippet='{snippet_lower[:80]}...'"
                )
                scored_evidences.append((score, value, ev))
            else:
                # Debug: Log when entity is mentioned but no value extracted (only for first few snippets to avoid spam)
                if len([ev for ev in scored_evidences if ev[2] == ev]) < 3:  # Only log for first few evidences
                    logger.debug(
                        f"COMPARISON_EXTRACT_DEBUG: entity='{entity}' attribute='{attribute}' "
                        f"no numeric matches in snippet='{snippet_lower[:80]}...'"
                    )
        
        if not scored_evidences:
            # Debug: Check why no values were extracted
            entity_mentioned_evidences = [
                ev for ev in evidences
                if entity_lower in ev.snippet.lower() or entity_lower in ev.title.lower()
            ]
            entity_mentioned_count = len(entity_mentioned_evidences)
            
            # Log example snippets that mention the entity but have no values
            example_snippets = [ev.snippet[:150] for ev in entity_mentioned_evidences[:2]]
            logger.warning(
                f"COMPARISON_EXTRACT: entity='{entity}' attribute='{attribute}' "
                f"no values extracted from {len(evidences)} evidences "
                f"(entity mentioned in {entity_mentioned_count} evidences). "
                f"Example snippets: {example_snippets}"
            )
            return (None, None)
        
        # Sort by score and return best
        scored_evidences.sort(key=lambda x: x[0], reverse=True)
        best_score, best_value, best_evidence = scored_evidences[0]
        
        logger.info(
            f"COMPARISON_EXTRACT: entity='{entity}' attribute='{attribute}' "
            f"value={best_value} score={best_score:.2f}"
        )
        
        return (best_value, best_evidence)
    
    def _compare_values(
        self, x_value: float, y_value: float, comparison_word: str
    ) -> Optional[bool]:
        """
        Compare two numeric values based on comparison word.
        
        Args:
            x_value: Value for X entity
            y_value: Value for Y entity
            comparison_word: Comparison word (e.g., "länger", "größer", "mehr")
            
        Returns:
            True if comparison holds, False if not, None if uncertain
        """
        comparison_word_lower = comparison_word.lower()
        
        # Comparison logic
        if comparison_word_lower in ["länger", "größer", "mehr", "höher"]:
            # X > Y
            return x_value > y_value
        elif comparison_word_lower in ["kürzer", "kleiner", "weniger", "niedriger"]:
            # X < Y
            return x_value < y_value
        else:
            return None

