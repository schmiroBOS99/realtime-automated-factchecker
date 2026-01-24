"""
Unified population number parser.

Supports multiple languages and formats:
- German: "83 Millionen", "83,2 Mio", "3,9 Millionen", "1,3 Milliarden", "83.190.556"
- English: "83 million", "1.3 billion", "83,190,556"
- Portuguese/Spanish: "10.344.802 habitantes", "habitantes/inhabitants"

Returns absolute integers (NOT "in millions floats") with metadata.
"""
import html
import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PopulationValue:
    """Population value with metadata."""
    raw_match: str  # Original matched text
    normalized_value_int: int  # Absolute integer (e.g., 83_000_000, not 83.0)
    scale: str  # Unit: "million", "milliard", "billion", "raw"
    approx_flag: bool  # True if qualifiers like "etwa", "circa", "über" were present
    year: Optional[int] = None  # Year if mentioned in context


def parse_population_number(text: str) -> List[PopulationValue]:
    """
    Parse population numbers from text in multiple languages and formats.
    
    Args:
        text: Text to parse (may contain HTML entities)
        
    Returns:
        List of PopulationValue objects with absolute integers and metadata
    """
    # Clean HTML entities first (fixes &#91; etc.)
    text_clean = html.unescape(text or "")
    # Also handle numeric entities manually (fallback)
    text_clean = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text_clean)
    text_clean = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text_clean)
    # Normalize whitespace (including non-breaking spaces)
    text_clean = re.sub(r'[\u00A0\u2009\u202F]', ' ', text_clean)  # NBSP, thin space, narrow NBSP
    text_clean = re.sub(r'\s+', ' ', text_clean)
    # Remove Wikipedia-style reference markers: [1], [ 12 ]
    text_clean = re.sub(r"\[\s*\d+\s*\]", " ", text_clean)
    
    text_lower = text_clean.lower()
    results: List[PopulationValue] = []
    
    # Extract year from context (look for 4-digit years nearby)
    year_match = re.search(r'\b(19|20)\d{2}\b', text_clean)
    context_year = int(year_match.group(0)) if year_match else None
    
    # Pattern groups:
    # 1. Qualifiers (optional): "mehr als", "über", "knapp", "fast", "gut", "circa", "etwa", etc.
    # 2. Number formats:
    #    a) Decimal: "83,6" or "83.6" (German vs English decimal separator)
    #    b) Integer: "83"
    #    c) Raw integer with thousands separators: "10.344.802" or "10,344,802" or "10 344 802"
    # 3. Scale: "Millionen", "Million", "Mio", "Milliarden", "Milliard", "Billion", "billion"
    # 4. Population keywords: "Einwohner", "Bevölkerung", "habitantes", "inhabitants", etc.
    
    # Important: integer patterns must NOT match decimal fractions (e.g., "1,8 Mio" must not also match "8 Mio")
    patterns = [
        # German: "83,6 Millionen Einwohner", "83 Millionen", "3,9 Mio"
        (r"(?:mehr\s+als|über|knapp|fast|gut)\s*(\d+)[.,](\d+)\s*(millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?", "million", True),
        (r"(?:mehr\s+als|über|knapp|fast|gut)\s*(?<![.,\d])(\d+)\s*(millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?", "million", True),
        (r"(?:circa|etwa|ungefähr|rund|ca\.?)\s*(\d+)[.,](\d+)\s*(millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?", "million", True),
        (r"(?:circa|etwa|ungefähr|rund|ca\.?)\s*(?<![.,\d])(\d+)\s*(millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?", "million", True),
        (r"(\d+)[.,](\d+)\s*(millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?", "million", False),
        (r"(?<![.,\d])(\d+)\s*(millionen|million|mio\.?)\s*(?:einwohner|bevölkerung|menschen)?", "million", False),
        
        # German: "1,3 Milliarden", "1 Milliarde"
        (r"(?:mehr\s+als|über|knapp|fast|gut)\s*(\d+)[.,](\d+)\s*(milliarden|milliard)\s*(?:einwohner|bevölkerung|menschen)?", "milliard", True),
        (r"(?:mehr\s+als|über|knapp|fast|gut)\s*(?<![.,\d])(\d+)\s*(milliarden|milliard)\s*(?:einwohner|bevölkerung|menschen)?", "milliard", True),
        (r"(\d+)[.,](\d+)\s*(milliarden|milliard)\s*(?:einwohner|bevölkerung|menschen)?", "milliard", False),
        (r"(?<![.,\d])(\d+)\s*(milliarden|milliard)\s*(?:einwohner|bevölkerung|menschen)?", "milliard", False),
        
        # English: "83 million", "1.3 billion"
        (r"(?:more\s+than|over|about|approximately|circa)\s*(\d+)[.,](\d+)\s*(million|billion)\s*(?:inhabitants|people|population)?", "million", True),
        (r"(?:more\s+than|over|about|approximately|circa)\s*(?<![.,\d])(\d+)\s*(million|billion)\s*(?:inhabitants|people|population)?", "million", True),
        (r"(\d+)[.,](\d+)\s*(million|billion)\s*(?:inhabitants|people|population)?", "million", False),
        (r"(?<![.,\d])(\d+)\s*(million|billion)\s*(?:inhabitants|people|population)?", "million", False),
        # Handle "billion" scale
        (r"(\d+)[.,](\d+)\s*billion\s*(?:inhabitants|people|population)?", "billion", False),
        (r"(?<![.,\d])(\d+)\s*billion\s*(?:inhabitants|people|population)?", "billion", False),
        
        # Patterns with verb + year: "hatte 2022 3,9 Millionen"
        (r"(?:hat|hatte|haben|hatten|zählt|zählte|tem|tem|tem)\s+(?:19|20)\d{2}\s+(\d+)[.,](\d+)\s*(millionen|million|mio\.?|milliarden|milliard)\s*(?:einwohner|bevölkerung|habitantes|inhabitants)?", "million", False),
        (r"(?:hat|hatte|haben|hatten|zählt|zählte|tem|tem|tem)\s+(?:19|20)\d{2}\s+(\d+)\s*(millionen|million|mio\.?|milliarden|milliard)\s*(?:einwohner|bevölkerung|habitantes|inhabitants)?", "million", False),
        (r"(?:hat|hatte|haben|hatten|zählt|zählte|tem|tem|tem)\s+(\d+)[.,](\d+)\s*(millionen|million|mio\.?|milliarden|milliard)\s*(?:einwohner|bevölkerung|habitantes|inhabitants)?", "million", False),
        (r"(?:hat|hatte|haben|hatten|zählt|zählte|tem|tem|tem)\s+(\d+)\s*(millionen|million|mio\.?|milliarden|milliard)\s*(?:einwohner|bevölkerung|habitantes|inhabitants)?", "million", False),
        
        # Raw integers with thousands separators: "10.344.802 Einwohner", "10,344,802 inhabitants", "10 344 802 habitants"
        (r"(\d{1,3}(?:[.\s]\d{3})+)\s*(?:einwohner|einwohnerzahl|bevölkerung|bevölkerungszahl|bewohner|menschen|habitantes|inhabitants)\b", "raw", False),
        (r"(?:einwohner|einwohnerzahl|bevölkerung|bevölkerungszahl|bewohner|menschen|habitantes|inhabitants)\D{0,30}(\d{1,3}(?:[.\s]\d{3})+)\b", "raw", False),
        (r"(?:tem|hat|hatte|haben|hatten|zählt|zählte)\s+(\d{1,3}(?:[.\s]\d{3})+)\s*(?:habitantes|einwohner|bewohner|menschen)\b", "raw", False),
        (r"(\d{1,3}(?:[.\s]\d{3})+)\s*(?:habitantes|inhabitants)\b", "raw", False),
    ]
    
    for pattern, default_scale, is_approx in patterns:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        for match in matches:
            try:
                # Check if this match is part of a percentage
                match_start = match.start()
                match_end = match.end()
                context_before = text_clean[max(0, match_start - 10):match_start]
                context_after = text_clean[match_end:min(len(text_clean), match_end + 10)]
                
                if "%" in context_after or "%" in context_before:
                    continue
                
                raw_match = match.group(0)
                groups = match.groups()
                
                # Determine scale from match (check if "milliard" or "billion" in match)
                scale = default_scale
                match_text_lower = raw_match.lower()
                if "milliard" in match_text_lower or "milliard" in match_text_lower:
                    scale = "milliard"
                elif "billion" in match_text_lower:
                    scale = "billion"
                
                # Parse number
                if len(groups) >= 2 and groups[1] and (groups[1].isdigit() or "." in groups[1] or "," in groups[1]):
                    # Decimal number: "83,6" or "83.6"
                    integer_part = groups[0]
                    decimal_part = groups[1]
                    # Normalize decimal separator (German uses comma, English uses dot)
                    if "," in decimal_part:
                        number_float = float(f"{integer_part}.{decimal_part.replace(',', '')}")
                    else:
                        number_float = float(f"{integer_part}.{decimal_part}")
                elif len(groups) >= 1:
                    # Integer or raw integer with separators
                    raw = groups[0]
                    if raw and (("." in raw) or (" " in raw)) and re.match(r"^\d{1,3}(?:[.\s]\d{3})+$", raw.strip()):
                        # Raw integer with thousands separators
                        cleaned = raw.replace(".", "").replace(" ", "").replace(",", "").strip()
                        value_int = int(cleaned)
                        # Filter out very small values (< 100k) - likely not population
                        if value_int < 100_000:
                            continue
                        results.append(PopulationValue(
                            raw_match=raw_match,
                            normalized_value_int=value_int,
                            scale="raw",
                            approx_flag=is_approx,
                            year=context_year,
                        ))
                        continue
                    else:
                        # Regular integer
                        number_float = float(raw)
                else:
                    continue
                
                # Filter out very small numbers (< 0.1 million) - likely percentages or other metrics
                if number_float < 0.1:
                    continue
                
                # Convert to absolute integer based on scale
                if scale == "million":
                    value_int = int(number_float * 1_000_000)
                elif scale == "milliard":
                    value_int = int(number_float * 1_000_000_000)
                elif scale == "billion":
                    value_int = int(number_float * 1_000_000_000)  # English billion = 1e9
                else:
                    value_int = int(number_float)
                
                # Filter out unrealistic values (> 10 billion for any country)
                if value_int > 10_000_000_000:
                    continue
                
                results.append(PopulationValue(
                    raw_match=raw_match,
                    normalized_value_int=value_int,
                    scale=scale,
                    approx_flag=is_approx,
                    year=context_year,
                ))
            except (ValueError, IndexError) as e:
                logger.debug(f"POPULATION_PARSER: Error parsing match '{match.group(0)}': {e}")
                continue
    
    # Deduplicate by value (keep first occurrence)
    seen = set()
    deduplicated = []
    for result in results:
        if result.normalized_value_int not in seen:
            seen.add(result.normalized_value_int)
            deduplicated.append(result)
    
    return deduplicated

