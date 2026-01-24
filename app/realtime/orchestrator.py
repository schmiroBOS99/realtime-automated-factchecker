"""
Realtime orchestrator for processing transcribed sentences.

Handles sentence segmentation, claim detection, deduplication, caching,
and concurrent fact-checking with rate limiting.
"""
import asyncio
import re
import logging
from typing import List, Dict, Any

from app.pipeline.cache import TTLCache
from app.pipeline.service import check_claim_text
from app.realtime.segmenter import split_sentences
from app.realtime.claim_detector import detect_claims

logger = logging.getLogger(__name__)

# Cache configuration: 20 minutes TTL, max 1000 entries
CACHE_TTL_SECONDS = 20 * 60  # 20 minutes
CACHE_MAX_SIZE = 1000

# Concurrency limit: max 2 parallel fact-checks
CHECK_SEMAPHORE = asyncio.Semaphore(2)

# Global cache instance (singleton)
_factcheck_cache: TTLCache = None


def _get_cache() -> TTLCache:
    """Get or create the global factcheck cache instance."""
    global _factcheck_cache
    if _factcheck_cache is None:
        _factcheck_cache = TTLCache(
            default_ttl_seconds=CACHE_TTL_SECONDS,
            max_size=CACHE_MAX_SIZE
        )
    return _factcheck_cache


def normalize_claim(claim_text: str) -> str:
    """
    Normalize claim string for deduplication and cache key.
    
    Normalization rules:
    - Convert to lowercase
    - Remove punctuation
    - Collapse multiple whitespace to single space
    - Strip leading/trailing whitespace
    
    Args:
        claim_text: Original claim text
        
    Returns:
        Normalized claim string
        
    Example:
        >>> normalize_claim("Angela Merkel ist Bundeskanzlerin!")
        "angela merkel ist bundeskanzlerin"
    """
    if not claim_text:
        return ""
    
    # Convert to lowercase
    normalized = claim_text.lower()
    
    # Remove punctuation (keep alphanumeric, spaces, and basic characters)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # Collapse multiple whitespace to single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized


async def process_sentence(sentence: str) -> List[Dict[str, Any]]:
    """
    Process a sentence: segment, detect claims, fact-check (with dedupe/cache).
    
    This is the main orchestrator function that:
    1. Segments sentence into sub-sentences
    2. Detects claims in each segment
    3. Deduplicates and caches fact-check results
    4. Limits concurrency to avoid overwhelming the fact-checker
    
    Args:
        sentence: Input sentence to process
        
    Returns:
        List of event dicts, each with:
            - "event_type": str ("claim_detected" or "factcheck_result" or "factcheck_error")
            - "claim_text": str (original claim text)
            - "normalized_claim": str (normalized key for dedupe)
            - Additional fields based on event_type:
                * claim_detected: "confidence" (float)
                * factcheck_result: "result" (dict), "cached" (bool)
                * factcheck_error: "error" (str)
            
    Example:
        >>> events = await process_sentence("Angela Merkel ist Bundeskanzlerin.")
        >>> # Returns events for claim detection and fact-check result
    """
    events: List[Dict[str, Any]] = []
    
    if not sentence or not sentence.strip():
        return events
    
    # Step 1: Segment sentence into sub-sentences
    segments = split_sentences(sentence)
    
    if not segments:
        return events
    
    # Step 2: Detect claims in each segment
    all_claims = []
    for segment in segments:
        detected_claims = detect_claims(segment)
        all_claims.extend(detected_claims)
    
    if not all_claims:
        return events
    
    # Step 3: Process each claim (with dedupe and cache)
    cache = _get_cache()
    
    for claim in all_claims:
        claim_text = claim["claim_text"]
        confidence = claim["confidence"]
        normalized_key = normalize_claim(claim_text)
        
        # Emit claim_detected event
        events.append({
            "event_type": "claim_detected",
            "claim_text": claim_text,
            "normalized_claim": normalized_key,
            "confidence": confidence,
        })
        
        # Check cache for existing result
        cached_result = cache.get(normalized_key)
        
        if cached_result is not None:
            # Cache hit: return cached result immediately
            logger.debug(f"CACHE_HIT: normalized_key='{normalized_key[:50]}...'")
            events.append({
                "event_type": "factcheck_result",
                "claim_text": claim_text,
                "normalized_claim": normalized_key,
                "result": cached_result,
                "cached": True,
            })
        else:
            # Cache miss: fact-check with concurrency control
            logger.debug(f"CACHE_MISS: normalized_key='{normalized_key[:50]}...', checking...")
            
            async with CHECK_SEMAPHORE:
                try:
                    # Run fact-check
                    result = await check_claim_text(claim_text)
                    
                    # Store in cache
                    cache.set(normalized_key, result)
                    
                    # Emit factcheck_result event
                    events.append({
                        "event_type": "factcheck_result",
                        "claim_text": claim_text,
                        "normalized_claim": normalized_key,
                        "result": result,
                        "cached": False,
                    })
                    
                except Exception as e:
                    logger.error(f"Error fact-checking claim '{claim_text[:50]}...': {e}", exc_info=True)
                    # On error, emit error event (optional, or skip)
                    events.append({
                        "event_type": "factcheck_error",
                        "claim_text": claim_text,
                        "normalized_claim": normalized_key,
                        "error": str(e),
                    })
    
    return events

