"""
Claim detection utility for realtime transcription processing.

MVP heuristic-based claim detector with high precision focus.
"""
import re
from typing import List, Dict, Any


# Comparison terms (German and English)
COMPARISON_TERMS = [
    "mehr als",
    "größer als",
    "weniger als",
    "höher als",
    "kleiner als",
    "länger als",  # Added: missing comparison term
    "kürzer als",  # Added: missing comparison term
    "higher than",
    "larger than",
    "more than",
    "less than",
    "smaller than",
    "longer than",  # Added: English version
    "shorter than",  # Added: English version
]

# Factual verbs (German and English)
FACTUAL_VERBS_DE = [
    "ist", "war", "hat", "liegt", "stammt", "gewann", "führt", 
    "steigt", "sinkt", "gehört", "waren", "haben", "wurde", "wurden",
    # Geographic/spatial verbs
    "fließt", "mündet", "verläuft", "grenzt", "befindet", "liegen",
    "entspringt", "durchquert", "durchfließt",
]

FACTUAL_VERBS_EN = [
    "is", "was", "has", "have", "won", "leads", "increases", 
    "decreases", "belongs", "were", "are",
    # Geographic/spatial verbs
    "flows", "runs", "borders", "located", "passes", "crosses",
]

FACTUAL_VERBS = FACTUAL_VERBS_DE + FACTUAL_VERBS_EN

# Opinion/subjective indicators (if present, reduce confidence)
OPINION_INDICATORS = [
    "meine meinung", "ich denke", "ich glaube", "ich finde", "ich meine",
    "my opinion", "i think", "i believe", "i find", "i feel",
    "meiner ansicht", "glaube ich", "denke ich"
]

# Personal pronouns (first and second person) - strong rejection signal
# These indicate personal/subjective claims that cannot be fact-checked
PERSONAL_PRONOUNS = [
    # First person (ich)
    "ich", "mir", "mich", "mein", "meine", "meinen", "meinem", "meiner",
    "i", "me", "my", "mine",
    # Second person (du/deine) - also personal/subjective
    "du", "dir", "dich", "dein", "deine", "deinen", "deinem", "deiner",
    "you", "your", "yours",
]

# Local/temporal deictic indicators (recent past, local events) - reduce confidence
LOCAL_TEMPORAL_INDICATORS = [
    "gestern", "heute", "morgen", "gerade", "vorhin", "kürzlich",
    "yesterday", "today", "tomorrow", "just now", "recently",
    # Local spatial indicators
    "gegenüber", "nebenan", "hier", "dort", "da", "hierbei",
    "opposite", "next door", "here", "there",
]


def detect_claims(sentence: str) -> List[Dict[str, Any]]:
    """
    Detect factual claims in a sentence using MVP heuristics.
    
    Returns high-precision results (confidence >= 0.7) to avoid spamming
    the factchecker with non-claim sentences.
    
    Args:
        sentence: Input sentence to analyze
        
    Returns:
        List of claim dicts, each with:
            - "claim_text": str (the sentence)
            - "confidence": float (0.7 to 0.9)
        Returns empty list if sentence doesn't meet claim criteria.
        
    Example:
        >>> detect_claims("Angela Merkel ist Bundeskanzlerin von Deutschland.")
        [{'claim_text': 'Angela Merkel ist Bundeskanzlerin von Deutschland.', 'confidence': 0.8}]
        
        >>> detect_claims("Das ist meine Meinung.")
        []
    """
    if not sentence:
        return []
    
    sentence_lower = sentence.lower()
    sentence_stripped = sentence.strip()
    
    # Check word count (must be >= 4 to filter very short fragments)
    words = sentence_stripped.split()
    if len(words) < 4:
        return []
    
    # Early rejection: Check for personal pronouns (first and second person) - strong rejection
    # Personal claims are not fact-checkable ("Ich bin 15 Jahre alt", "Du hast recht", "Deine Meinung")
    words_lower = [w.lower().rstrip('.,!?') for w in words]
    has_personal_pronoun = any(pronoun in words_lower for pronoun in PERSONAL_PRONOUNS)
    if has_personal_pronoun:
        return []  # Reject immediately - personal/subjective claims are not fact-checkable
    
    # Initialize confidence at 0.5
    confidence = 0.5
    
    # Check for digit pattern (numbers/years/percentages)
    has_digit = bool(re.search(r'\d', sentence))
    if has_digit:
        confidence += 0.1
    
    # Check for comparison terms
    # Handle encoding issues: normalize umlauts for comparison
    sentence_normalized = sentence_lower
    # Create normalized version for comparison (ö->o, ä->a, ü->u)
    sentence_normalized_ascii = sentence_lower.replace('ö', 'o').replace('ä', 'a').replace('ü', 'u')
    COMPARISON_TERMS_NORMALIZED = [t.replace('ö', 'o').replace('ä', 'a').replace('ü', 'u') for t in COMPARISON_TERMS]
    
    has_comparison = any(term in sentence_lower for term in COMPARISON_TERMS)
    # Fallback: check normalized version if original didn't match (handles encoding issues)
    if not has_comparison:
        has_comparison = any(term in sentence_normalized_ascii for term in COMPARISON_TERMS_NORMALIZED)
    
    if has_comparison:
        confidence += 0.1
    
    # Check for named entity signal
    # At least one capitalized token not at sentence start OR multiple capitalized tokens
    tokens = sentence_stripped.split()
    capitalized_tokens = []
    for i, token in enumerate(tokens):
        # Remove punctuation for checking
        token_clean = re.sub(r'[^\w]', '', token)
        if token_clean and token_clean[0].isupper():
            capitalized_tokens.append(i)
    
    has_ne_signal = False
    if len(capitalized_tokens) >= 2:
        # Multiple capitalized tokens
        has_ne_signal = True
    elif len(capitalized_tokens) == 1 and capitalized_tokens[0] > 0:
        # One capitalized token not at sentence start
        has_ne_signal = True
    
    if has_ne_signal:
        confidence += 0.1
    
    # Check for factual verb
    # Split sentence into words and check for verb matches (whole word)
    words_lower_list = sentence_lower.split()
    has_factual_verb = any(verb in words_lower_list for verb in FACTUAL_VERBS)
    if has_factual_verb:
        confidence += 0.1
    
    # Check for local/temporal indicators (reduce confidence)
    # These indicate local events or recent past, which are hard to fact-check
    # Use word boundaries to avoid false positives (e.g., "da" in "das")
    words_lower_set = set(words_lower)
    has_local_temporal = any(
        indicator in words_lower_set or 
        (len(indicator.split()) > 1 and indicator in sentence_lower)  # Multi-word indicators
        for indicator in LOCAL_TEMPORAL_INDICATORS
    )
    if has_local_temporal:
        confidence -= 0.3  # Strong penalty for local/temporal indicators
    
    # Special rule: If we have 2+ named entities (proper nouns), boost confidence
    # This handles geographic/location claims like "Der Nil fließt durch Nigeria"
    if has_ne_signal and len(capitalized_tokens) >= 2:
        # If we already have a factual verb, this is just confirmation
        # If we don't have a verb, still give a small boost for multiple entities
        if not has_factual_verb:
            confidence += 0.1  # Boost for multiple entities even without explicit verb
    
    # Check for opinion indicators (reduce confidence if present)
    has_opinion_indicator = any(indicator in sentence_lower for indicator in OPINION_INDICATORS)
    if has_opinion_indicator:
        confidence -= 0.2  # Penalty for opinion indicators
    
    # Cap confidence at 0.9
    confidence = min(confidence, 0.9)
    # Ensure confidence doesn't go below 0.0
    confidence = max(confidence, 0.0)
    
    # Require at least one signal beyond word count (confidence > 0.5)
    # AND require >= 0.7 to emit as claim
    if confidence >= 0.7:
        return [{
            "claim_text": sentence_stripped,
            "confidence": round(confidence, 2)
        }]
    
    return []

