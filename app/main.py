"""
FastAPI application for Pipeline 2.0 fact-checking service.

Run with:
    uvicorn app.main:app --reload --port 8000
"""
import logging
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.pipeline.service import check_claim_text, shutdown_service, get_factchecker_instance
from app.api.ws_live import router as ws_live_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI()

# Include WebSocket router
app.include_router(ws_live_router)

# Serve static files (UI)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    # If static directory doesn't exist, create a route to serve index.html
    pass

@app.get("/")
async def root():
    """Serve the UI index page."""
    try:
        return FileResponse("static/index.html")
    except FileNotFoundError:
        return {"message": "UI not found. Please ensure static/index.html exists."}


@app.on_event("startup")
async def startup_event():
    """Initialize FactChecker on application startup."""
    # Service layer manages its own singleton (lazy initialization)
    # Pre-initialize to avoid first-request delay
    await get_factchecker_instance()
    logger.info("FactChecker service initialized via service layer")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up FactChecker on application shutdown."""
    await shutdown_service()
    logger.info("FactChecker service shut down")


@app.get("/ping")
async def ping():
    """Health check endpoint."""
    return {"message": "pong"}


@app.delete("/cache/clear")
async def clear_cache():
    """
    Clear ALL caches comprehensively.
    
    This clears:
    - ClaimStore (SQLite database)
    - Wikipedia API caches (search, summary, extract)
    - Wikidata API caches (search, entity, labels)
    - All in-memory caches
    
    CRITICAL: This ensures all caches are cleared even if components are not yet initialized.
    After clearing, components will reinitialize with empty caches on next use.
    
    Returns:
        Dict with number of claims cleared and cache statistics
    """
    factchecker = await get_factchecker_instance()
    
    # CRITICAL: Ensure FactChecker is fully started before clearing caches
    # This ensures all clients are initialized and their caches exist
    if factchecker._session is None or factchecker._session.closed:
        await factchecker.start()
    
    count = factchecker.clear_cache()
    
    # Get cache statistics for response (after clearing, should all be 0)
    cache_stats = {
        "claims_cleared": count,
    }
    
    # Add Wikipedia cache stats if available
    if factchecker.wikipedia_client:
        wiki_stats = {}
        if hasattr(factchecker.wikipedia_client, '_search_cache'):
            wiki_stats["search"] = factchecker.wikipedia_client._search_cache.get_stats()
        if hasattr(factchecker.wikipedia_client, '_summary_cache'):
            wiki_stats["summary"] = factchecker.wikipedia_client._summary_cache.get_stats()
        if hasattr(factchecker.wikipedia_client, '_extract_cache'):
            wiki_stats["extract"] = factchecker.wikipedia_client._extract_cache.get_stats()
        cache_stats["wikipedia"] = wiki_stats
        logger.info(f"CACHE_CLEAR: Wikipedia cache stats after clear: {wiki_stats}")
    
    # Add Wikidata cache stats if available
    if factchecker.wikidata_client:
        wikidata_stats = {}
        if hasattr(factchecker.wikidata_client, '_search_cache'):
            wikidata_stats["search"] = factchecker.wikidata_client._search_cache.get_stats()
        if hasattr(factchecker.wikidata_client, '_entity_cache'):
            wikidata_stats["entity"] = factchecker.wikidata_client._entity_cache.get_stats()
        if hasattr(factchecker.wikidata_client, '_labels_cache'):
            wikidata_stats["labels"] = factchecker.wikidata_client._labels_cache.get_stats()
        cache_stats["wikidata"] = wikidata_stats
        logger.info(f"CACHE_CLEAR: Wikidata cache stats after clear: {wikidata_stats}")
    
    logger.info(
        f"CACHE_CLEAR: Successfully cleared {count} cached claims and all API caches. "
        f"All caches are now empty and will rebuild from scratch on next use."
    )
    
    return {
        "message": "All caches cleared successfully",
        "claims_cleared": count,
        "cache_stats": cache_stats,
        "status": "all_cleared",
    }


@app.post("/server/restart")
async def restart_server():
    """
    Restart the uvicorn server by triggering a file change.
    This only works if uvicorn is running with --reload flag.
    
    Returns:
        Dict with restart message
    """
    import os
    import time
    
    # Touch the main.py file to trigger uvicorn --reload
    try:
        # Get the path to this file
        current_file = __file__
        
        # Touch the file by updating its modification time
        os.utime(current_file, None)
        
        logger.info("SERVER_RESTART: Triggered server restart by touching main.py")
        return {"message": "Server restart triggered (if running with --reload)"}
    except Exception as e:
        logger.error(f"SERVER_RESTART: Error triggering restart: {e}")
        return {"error": f"Failed to trigger restart: {str(e)}"}, 500


class CheckRequest(BaseModel):
    """Request model for fact-checking endpoint."""
    claim: str
    max_pages: int = 3


class DetectRequest(BaseModel):
    """Request model for claim detection endpoint."""
    claim: str


@app.post("/check")
async def check_claim(request: CheckRequest):
    """
    Fact-check a claim using Pipeline 2.0.
    
    This endpoint uses the service layer function check_claim_text()
    which can also be called directly from WebSocket handlers or other async code.
    """
    result = await check_claim_text(
        claim=request.claim,
        max_pages=request.max_pages,
    )
    
    # Log one line per request: claim, relation_type, path, label, reason
    relation_type = result.get("relation_type", "unknown")
    path = result.get("path", "unknown")
    label = result.get("label", "unknown")
    reason = result.get("reason", "none")
    logger.info(
        f"CHECK: claim='{request.claim}' | relation_type={relation_type} | path={path} | label={label} | reason={reason}"
    )
    
    return result


@app.post("/detect")
async def detect_claim(request: DetectRequest):
    """
    Detect and parse a claim without performing fact-checking.
    
    Returns parsed claim information including:
    - parsed_claim (subject, predicate, object, category, relation_type, etc.)
    - is_hard_claim (whether claim requires decomposition)
    - subclaims (list of decomposed subclaims if applicable)
    - category (claim category)
    
    This endpoint is useful for evaluating claim detection/decomposition
    separately from the full fact-checking pipeline.
    """
    try:
        factchecker = await get_factchecker_instance()
        
        # Check if claim is personal/subjective (should not be fact-checked)
        is_personal = factchecker._claim_parser.is_personal_or_subjective_claim(request.claim)
        
        # Parse claim
        parsed_claim = factchecker._claim_parser.parse_claim(request.claim)
        
        # Build parsed_claim dict
        parsed_claim_dict: Dict[str, Any] = {}
        if parsed_claim.subject:
            parsed_claim_dict["subject"] = parsed_claim.subject
        if parsed_claim.predicate:
            parsed_claim_dict["predicate"] = parsed_claim.predicate
        if parsed_claim.object:
            parsed_claim_dict["object"] = parsed_claim.object
        if parsed_claim.year:
            parsed_claim_dict["year"] = parsed_claim.year
        if parsed_claim.party:
            parsed_claim_dict["party"] = parsed_claim.party
        if parsed_claim.role:
            parsed_claim_dict["role"] = parsed_claim.role
        if parsed_claim.event:
            parsed_claim_dict["event"] = parsed_claim.event
        if parsed_claim.sport:
            parsed_claim_dict["sport"] = parsed_claim.sport
        if parsed_claim.canonical_claim:
            parsed_claim_dict["canonical_claim"] = parsed_claim.canonical_claim
        if parsed_claim.subclaims:
            parsed_claim_dict["subclaims"] = parsed_claim.subclaims
        parsed_claim_dict["is_hard_claim"] = parsed_claim.is_hard_claim
        
        return {
            "claim": request.claim,
            "is_personal_or_subjective": is_personal,
            "parsed_claim": parsed_claim_dict,
            "category": parsed_claim.category,
            "relation_type": parsed_claim.relation_type,
            "is_hard_claim": parsed_claim.is_hard_claim,
            "subclaims": parsed_claim.subclaims if parsed_claim.subclaims else [],
            "canonical_claim": parsed_claim.canonical_claim,
        }
    except Exception as e:
        logger.error(f"Error in detect_claim for claim '{request.claim[:50]}...': {e}", exc_info=True)
        return {
            "claim": request.claim,
            "error": str(e),
            "is_personal_or_subjective": False,
            "parsed_claim": {},
            "category": "unknown",
            "relation_type": "unknown",
            "is_hard_claim": False,
            "subclaims": [],
            "canonical_claim": "",
        }


@app.get("/debug/wiki")
async def debug_wiki(q: str):
    """
    Temporary debug endpoint to inspect Wikipedia search results.
    
    Args:
        q: Search query string
        
    Returns:
        JSON with query, raw search results, and count
    """
    factchecker = await get_factchecker_instance()
    
    # Get WikipediaClient from FactChecker
    wiki_client = factchecker.wikipedia_client
    if wiki_client is None:
        return {"q": q, "results": [], "count": 0, "error": "WikipediaClient not initialized"}
    
    # Call search
    results = await wiki_client.search(query=q, limit=5)
    
    return {
        "q": q,
        "results": results,
        "count": len(results),
    }


@app.get("/debug/wikidata/person")
async def debug_wikidata_person(q: str):
    """
    Debug endpoint to inspect Wikidata person search results.
    
    Args:
        q: Person name to search for
        
    Returns:
        JSON with search results, chosen QID, and whether entity is human (Q5)
    """
    factchecker = await get_factchecker_instance()
    
    wikidata_client = factchecker.wikidata_client
    if wikidata_client is None:
        return {"q": q, "error": "WikidataClient not initialized"}
    
    # Search for entity
    search_results = await wikidata_client.search_entity(name=q, language="de", limit=5)
    
    # Get labels
    candidate_qids = [hit.get("id") for hit in search_results if hit.get("id")]
    labels = {}
    if candidate_qids:
        labels = await wikidata_client.get_labels(candidate_qids, lang="de")
    
    # Try to resolve conservatively and check if human
    resolved = await factchecker._resolve_entity_conservative(q, is_person=True)
    chosen_qid = None
    chosen_label = None
    is_human = False
    
    if resolved:
        chosen_qid, chosen_label = resolved
        # Check if entity is human (Q5)
        entity = await wikidata_client.get_entity(chosen_qid, language="de")
        if entity:
            claims = entity.get("claims", {})
            instance_of = claims.get("P31", [])
            is_human = any(
                stmt.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == "Q5"
                for stmt in instance_of
            )
    
    # Format results
    formatted_results = []
    for hit in search_results:
        qid = hit.get("id", "")
        label = labels.get(qid, hit.get("label", ""))
        formatted_results.append({
            "qid": qid,
            "label": label,
            "description": hit.get("description", ""),
            "score": hit.get("score", 0),
        })
    
    return {
        "q": q,
        "search_results": formatted_results,
        "chosen_qid": chosen_qid,
        "chosen_label": chosen_label,
        "is_human": is_human,
    }


@app.get("/debug/wikidata/entity")
async def debug_wikidata_entity(qid: str):
    """
    Debug endpoint to inspect Wikidata entity structure.
    
    Args:
        qid: Wikidata QID (e.g., "Q567")
        
    Returns:
        JSON with compact view of claims keys, especially P39 and P102 counts
    """
    factchecker = await get_factchecker_instance()
    
    wikidata_client = factchecker.wikidata_client
    if wikidata_client is None:
        return {"qid": qid, "error": "WikidataClient not initialized"}
    
    # Get entity
    entity = await wikidata_client.get_entity(qid, language="de")
    if not entity:
        return {"qid": qid, "error": "Entity not found"}
    
    # Extract claims summary
    claims = entity.get("claims", {})
    claims_summary = {}
    for prop_id, statements in claims.items():
        claims_summary[prop_id] = len(statements)
    
    # Get detailed info for P39 and P102
    p39_details = []
    p39_statements = claims.get("P39", [])
    for stmt in p39_statements[:5]:  # Limit to first 5
        mainsnak = stmt.get("mainsnak", {})
        if mainsnak.get("snaktype") == "value":
            position_id = mainsnak.get("datavalue", {}).get("value", {}).get("id")
            qualifiers = stmt.get("qualifiers", {})
            p582 = qualifiers.get("P582", [])  # End time
            has_end_time = len(p582) > 0
            p39_details.append({
                "position_id": position_id,
                "has_end_time": has_end_time,
            })
    
    p102_details = []
    p102_statements = claims.get("P102", [])
    for stmt in p102_statements[:5]:  # Limit to first 5
        mainsnak = stmt.get("mainsnak", {})
        if mainsnak.get("snaktype") == "value":
            party_id = mainsnak.get("datavalue", {}).get("value", {}).get("id")
            p102_details.append({"party_id": party_id})
    
    # Get labels for positions and parties
    position_ids = [d["position_id"] for d in p39_details if d.get("position_id")]
    party_ids = [d["party_id"] for d in p102_details if d.get("party_id")]
    
    position_labels = {}
    if position_ids:
        position_labels = await wikidata_client.get_labels(position_ids, lang="de")
    
    party_labels = {}
    if party_ids:
        party_labels = await wikidata_client.get_labels(party_ids, lang="de")
    
    # Add labels to details
    for detail in p39_details:
        if detail.get("position_id"):
            detail["position_label"] = position_labels.get(detail["position_id"], "")
    
    for detail in p102_details:
        if detail.get("party_id"):
            detail["party_label"] = party_labels.get(detail["party_id"], "")
    
    # Get entity label
    entity_labels = await wikidata_client.get_labels([qid], lang="de")
    entity_label = entity_labels.get(qid, "")
    
    return {
        "qid": qid,
        "label": entity_label,
        "claims_summary": claims_summary,
        "p39_count": len(p39_statements),
        "p39_details": p39_details,
        "p102_count": len(p102_statements),
        "p102_details": p102_details,
    }

