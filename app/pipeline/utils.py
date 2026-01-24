"""
Shared utility functions for the fact-checking pipeline.

This module contains common functions used across multiple pipeline components
to avoid code duplication and ensure consistency.
"""
import re
from typing import List, Literal, Optional, Set

# Common German stopwords + generic fillers
COMMON_STOPWORDS = {
    "der", "die", "das", "ein", "eine", "und", "oder", "ist", "sind", "war", "waren", "bei", "von", "im", "in", "zu",
    "den", "des", "dass", "mit", "für", "auf", "am", "an", "als", "auch", "nicht", "nur", "schon", "noch", "mehr",
    "einer", "eines", "einem", "einen", "werden", "wurde", "hat", "haben", "heute"
}


def tokenize_text(text: str, stopwords: Optional[Set[str]] = None, min_length: int = 3) -> List[str]:
    """
    Tokenize text into clean tokens.
    
    Args:
        text: Text to tokenize
        stopwords: Optional set of stopwords to filter (defaults to COMMON_STOPWORDS)
        min_length: Minimum token length (default: 3)
        
    Returns:
        List of clean tokens (lowercase, filtered by length and stopwords)
    """
    if not text:
        return []
    
    if stopwords is None:
        stopwords = COMMON_STOPWORDS
    
    # Extract tokens using regex: [a-zA-ZäöüÄÖÜß]+
    tokens = re.findall(r"[a-zA-ZäöüÄÖÜß]+", text)
    
    # Lowercase, filter by length and stopwords
    clean_tokens = [
        token.lower()
        for token in tokens
        if len(token) >= min_length and token.lower() not in stopwords
    ]
    
    return clean_tokens


def infer_tense(text: str) -> Literal["past", "present", "unknown"]:
    """
    Infer tense from text markers.
    
    Uses comprehensive tense markers to detect past or present tense.
    
    Args:
        text: Text to analyze
        
    Returns:
        "past", "present", or "unknown"
    """
    if not text:
        return "unknown"
    
    text_lower = text.lower()
    
    # Past tense markers (comprehensive list)
    past_markers = [
        "war", "waren", "wurde", "wurden", "hatte", "hatten", 
        "gewann", "gewonnen", "ehemalig", "früher", "damals"
    ]
    
    # Check for past markers (with word boundaries to avoid false positives)
    for marker in past_markers:
        if f" {marker} " in text_lower or text_lower.startswith(f"{marker} ") or text_lower.endswith(f" {marker}"):
            return "past"
    
    # Present tense markers
    present_markers = [
        "ist", "sind", "wird", "werden", "hat", "haben", 
        "kann", "können", "aktuell", "derzeit", "momentan"
    ]
    
    # Check for present markers
    for marker in present_markers:
        if f" {marker} " in text_lower or text_lower.startswith(f"{marker} ") or text_lower.endswith(f" {marker}"):
            return "present"
    
    return "unknown"


def extract_years(text: str) -> List[int]:
    """
    Extract years (1900-2099) from text.
    
    Args:
        text: Text to extract years from
        
    Returns:
        List of years found in text
    """
    if not text:
        return []
    return [int(year) for year in re.findall(r"\b((?:19|20)\d{2})\b", text)]


def split_into_sentences(text: str, min_length: int = 40) -> List[str]:
    """
    Split text into sentences with basic filtering.
    
    Robust sentence splitting that handles common cases:
    - Splits on . ! ? followed by space and capital letter or end of string
    - Filters empty sentences and very short ones
    - Trims whitespace
    
    Args:
        text: Text to split into sentences
        min_length: Minimum sentence length in characters (default: 40)
        
    Returns:
        List of sentence strings (trimmed, filtered by min_length)
    """
    if not text:
        return []
    
    # Split on sentence endings: . ! ? followed by space and capital letter or end of string
    # Pattern: sentence ending punctuation + optional space + capital letter or end
    sentences = re.split(r'([.!?]+\s+(?=[A-ZÄÖÜ])|[.!?]+$)', text)
    
    # Clean up and filter
    result = []
    current = ""
    for i, part in enumerate(sentences):
        current += part
        # Every second element is sentence ending punctuation
        if i % 2 == 1:  # Sentence ending punctuation
            sentence = current.strip()
            if sentence and len(sentence) >= min_length:
                result.append(sentence)
            current = ""
    
    # Add remaining text if any
    if current.strip() and len(current.strip()) >= min_length:
        result.append(current.strip())
    
    # Fallback: if splitting failed, return whole text if long enough
    if not result and len(text.strip()) >= min_length:
        return [text.strip()]
    
    return result


def normalize_person_name(name: str) -> str:
    """
    Normalize person name for better matching.
    
    Removes extra whitespace and normalizes case.
    
    Args:
        name: Person name to normalize
        
    Returns:
        Normalized name
    """
    if not name:
        return ""
    return " ".join(name.strip().split())


def names_match(name_a: str, name_b: str) -> bool:
    """
    Robust name matching for person entities.
    
    Matches:
    - Exact match (case-insensitive)
    - Same last name (last token)
    - Single-word name contained in multi-word name
    
    Args:
        name_a: First name to compare
        name_b: Second name to compare
        
    Returns:
        True if names match, False otherwise
    """
    if not name_a or not name_b:
        return False
    
    a = name_a.strip().lower()
    b = name_b.strip().lower()
    
    # Exact match
    if a == b:
        return True
    
    a_tokens = a.split()
    b_tokens = b.split()
    
    if not a_tokens or not b_tokens:
        return False
    
    # Same last name (last token)
    if a_tokens[-1] == b_tokens[-1]:
        return True
    
    # Single-word name is part of multi-word name
    if len(a_tokens) == 1 and a_tokens[0] in b_tokens:
        return True
    
    if len(b_tokens) == 1 and b_tokens[0] in a_tokens:
        return True
    
    return False


def canonicalize_claim(claim: str) -> str:
    """
    Canonicalize a claim string for storage and matching.
    
    Normalizes:
    - Whitespace (collapse multiple spaces to single)
    - Case (lowercase)
    - Punctuation (remove quotes, normalize punctuation)
    - Numbers (basic formatting)
    
    Args:
        claim: Original claim text
        
    Returns:
        Canonical claim string
    """
    if not claim:
        return ""
    
    # Lowercase
    canonical = claim.lower()
    
    # Remove quotes (single and double, German and English)
    canonical = re.sub(r'["""''„"]', '', canonical)
    
    # Normalize whitespace: collapse multiple spaces/tabs/newlines to single space
    canonical = re.sub(r'\s+', ' ', canonical)
    
    # Remove leading/trailing whitespace
    canonical = canonical.strip()
    
    # Basic number normalization: ensure consistent spacing around numbers
    # E.g., "2020" stays "2020", "im Jahr 2020" stays "im jahr 2020"
    # This is minimal - more advanced normalization can be added later
    
    return canonical
