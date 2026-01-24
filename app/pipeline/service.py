"""
Fact-checking service layer.

Provides a clean function interface for fact-checking claims that can be used
both from HTTP endpoints and WebSocket handlers.

Usage example:
    from app.pipeline.service import check_claim_text
    
    result = await check_claim_text("Emmanuel Macron is President of France.")
    print(result["label"])  # SUPPORTED, REFUTED, or NOT_ENOUGH_INFO
"""
import logging
from typing import Dict, Any, Optional
import asyncio

from app.pipeline.factchecker import FactChecker

logger = logging.getLogger(__name__)

# Singleton FactChecker instance (managed at module level)
_factchecker: Optional[FactChecker] = None
_factchecker_lock = asyncio.Lock()


async def _ensure_factchecker() -> FactChecker:
    """
    Ensure FactChecker instance is initialized (singleton pattern).
    
    Returns:
        FactChecker instance
    """
    global _factchecker
    async with _factchecker_lock:
        if _factchecker is None:
            logger.info("Initializing FactChecker service...")
            _factchecker = FactChecker()
            await _factchecker.start()
            logger.info("FactChecker service initialized successfully")
        return _factchecker


async def get_factchecker_instance() -> FactChecker:
    """
    Get the FactChecker instance (for debug endpoints or advanced usage).
    
    Returns:
        FactChecker instance
    """
    return await _ensure_factchecker()


async def check_claim_text(claim: str, max_pages: int = 3) -> Dict[str, Any]:
    """
    Check a claim and return fact-checking result.
    
    This is the core fact-checking function that can be called directly
    (without HTTP overhead) from WebSocket handlers or other async code.
    
    Args:
        claim: The claim text to verify
        max_pages: Maximum number of evidence pages to retrieve (default: 3)
        
    Returns:
        Dict with keys:
            - "claim": str (original claim)
            - "path": str (route taken: "WIKIDATA_FAST_PATH", "WIKIPEDIA_RETRIEVAL", etc.)
            - "relation_type": str
            - "parsed_claim": dict
            - "label": str ("SUPPORTED", "REFUTED", or "NOT_ENOUGH_INFO")
            - "best_evidence": dict or None
            - "scores": list of score dicts
            - "evidence_count": int
            - Additional fields based on route taken
            
    Example:
        result = await check_claim_text("Emmanuel Macron is President of France.")
        assert result["label"] in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    """
    try:
        factchecker = await _ensure_factchecker()
        
        result = await factchecker.check(
            claim=claim,
            max_pages=max_pages,
        )
        
        # Ensure result is never None
        if result is None:
            logger.error(f"check_claim_text returned None for claim '{claim[:50]}...'")
            return {
                "claim": claim,
                "path": "ERROR",
                "relation_type": "general",
                "parsed_claim": {},
                "label": "NOT_ENOUGH_INFO",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "error": "Fact-checking returned None",
            }
        
        return result
    except Exception as e:
        logger.error(f"Error in check_claim_text for claim '{claim[:50]}...': {e}", exc_info=True)
        # Return safe fallback result
        return {
            "claim": claim,
            "path": "ERROR",
            "relation_type": "general",
            "parsed_claim": {},
            "label": "NOT_ENOUGH_INFO",
            "best_evidence": None,
            "scores": [],
            "evidence_count": 0,
            "error": str(e),
        }


async def shutdown_service() -> None:
    """
    Shutdown the fact-checking service and clean up resources.
    
    Should be called when the application is shutting down.
    """
    global _factchecker
    async with _factchecker_lock:
        if _factchecker is not None:
            logger.info("Shutting down FactChecker service...")
            await _factchecker.close()
            _factchecker = None
            logger.info("FactChecker service shut down")


