"""
Sentence segmentation utility for realtime transcription processing.

Simple MVP implementation for splitting transcribed text into sentences.
"""
import re
from typing import List


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple MVP rules.
    
    Rules:
    - Split on sentence-ending punctuation: `.`, `!`, `?`
    - Strip whitespace from each sentence
    - Filter out very short fragments (< 2 words)
    
    Args:
        text: Input text to segment
        
    Returns:
        List of sentence strings (filtered and trimmed)
        
    Example:
        >>> split_sentences("A is B. C is D!")
        ['A is B', 'C is D']
        
        >>> split_sentences("Short. This is a longer sentence with more words.")
        ['This is a longer sentence with more words']
    """
    if not text:
        return []
    
    # Split on sentence-ending punctuation (. ! ?)
    # Use regex to split on any of these, optionally followed by whitespace
    parts = re.split(r'[.!?]+[\s]*', text)
    
    # Process each part
    sentences = []
    for part in parts:
        # Strip whitespace
        sentence = part.strip()
        
        # Skip empty strings
        if not sentence:
            continue
        
        # Filter out very short fragments (< 2 words)
        # Count words by splitting on whitespace
        words = sentence.split()
        if len(words) >= 2:
            sentences.append(sentence)
    
    return sentences

