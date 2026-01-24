"""
Async Wikidata API client with TTL caching.

Usage example:
    import aiohttp
    from app.pipeline.wikidata_client import WikidataClient
    
    async with aiohttp.ClientSession() as session:
        client = WikidataClient(session)
        results = await client.search_entity("Angela Merkel", limit=5)
        entity = await client.get_entity("Q567")
        labels = await client.get_labels(["Q567", "Q183"], lang="en")
"""
import aiohttp
import asyncio
import logging
import os
from typing import Dict, List, Optional, Any

from app.pipeline.cache import TTLCache

logger = logging.getLogger(__name__)

# Timeout configuration (realtime defaults, configurable via env)
SEARCH_TIMEOUT = int(os.getenv("FACTCHECK_WIKIDATA_SEARCH_TIMEOUT", "3"))
ENTITY_TIMEOUT = int(os.getenv("FACTCHECK_WIKIDATA_ENTITY_TIMEOUT", "5"))
LABELS_TIMEOUT = int(os.getenv("FACTCHECK_WIKIDATA_LABELS_TIMEOUT", "5"))


class WikidataClient:
    """
    Async Wikidata API client with TTL caching.
    Uses a shared aiohttp.ClientSession (do NOT create a session per request).
    """
    
    WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
    WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    
    # Cache TTL: 6 hours (21600 seconds)
    CACHE_TTL_SECONDS = 21600
    
    # Default headers for Wikimedia API requests
    DEFAULT_HEADERS = {
        "User-Agent": "FactCheckerService/0.1 (Maro Schmitt Rodrigues; contact: maro.schmitt_rodrigues@student.reutlingen-university.de)",
        "Accept": "application/json",
    }
    
    def __init__(self, session: aiohttp.ClientSession):
        """
        Initialize Wikidata client with a shared aiohttp session.
        
        Args:
            session: Shared aiohttp.ClientSession instance
        """
        self.session = session
        self._search_cache = TTLCache(default_ttl_seconds=self.CACHE_TTL_SECONDS, max_size=5000)
        self._entity_cache = TTLCache(default_ttl_seconds=self.CACHE_TTL_SECONDS, max_size=5000)
        self._labels_cache = TTLCache(default_ttl_seconds=self.CACHE_TTL_SECONDS, max_size=5000)
    
    async def search_entity(self, name: str, language: str = "de", limit: int = 5) -> List[Dict]:
        """
        Search Wikidata entities by name.
        
        Args:
            name: Entity name to search for
            language: Language code (default: "de")
            limit: Maximum number of results (default: 5)
            
        Returns:
            List of search result dicts with keys: id, label, description, etc.
            Returns empty list on error or timeout.
        """
        if not name:
            return []
        
        # Check cache
        cache_key = f"search:{name}:{language}:{limit}"
        cached_result = self._search_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            async with self.session.get(
                self.WIKIDATA_API_URL,
                params={
                    "action": "wbsearchentities",
                    "search": name,
                    "language": language,
                    "format": "json",
                    "limit": limit,
                },
                headers=self.DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=SEARCH_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Wikidata search API returned status {resp.status} for name '{name}'")
                    return []
                
                data = await resp.json()
                search_results = data.get("search", [])
                
                # Cache successful results
                self._search_cache.set(cache_key, search_results)
                return search_results
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Return empty list on error/timeout
            return []
    
    async def get_entity(self, qid: str, language: str = "de") -> Dict:
        """
        Get Wikidata entity data by QID.
        
        Args:
            qid: Wikidata QID (e.g., "Q567")
            language: Language code (default: "de")
            
        Returns:
            Entity dict with claims, labels, etc.
            Returns empty dict on error or timeout.
        """
        if not qid:
            return {}
        
        # Check cache
        cache_key = f"entity:{qid}:{language}"
        cached_result = self._entity_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            url = self.WIKIDATA_ENTITY_URL.format(qid=qid)
            async with self.session.get(
                url,
                params={"language": language},
                headers=self.DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=ENTITY_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Wikidata entity API returned status {resp.status} for QID '{qid}'")
                    return {}
                
                data = await resp.json()
                entity = data.get("entities", {}).get(qid)
                
                if entity:
                    # Cache successful results
                    self._entity_cache.set(cache_key, entity)
                    return entity
                else:
                    return {}
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Return empty dict on error/timeout
            return {}
    
    async def get_labels(self, ids: List[str], lang: str = "en") -> Dict[str, str]:
        """
        Get labels for multiple Wikidata QIDs.
        
        Args:
            ids: List of QIDs (e.g., ["Q567", "Q183"])
            lang: Language code (default: "en")
            
        Returns:
            Dict mapping QID -> label string
            Returns dict with empty strings for missing labels on error/timeout.
        """
        if not ids:
            return {}
        
        # Check cache for each ID and collect missing ones
        result: Dict[str, str] = {}
        missing_ids: List[str] = []
        
        for qid in ids:
            cache_key = f"label:{qid}:{lang}"
            cached_label = self._labels_cache.get(cache_key)
            if cached_label is not None:
                result[qid] = cached_label
            else:
                missing_ids.append(qid)
        
        # Fetch missing labels
        if missing_ids:
            try:
                async with self.session.get(
                    self.WIKIDATA_API_URL,
                    params={
                        "action": "wbgetentities",
                        "ids": "|".join(missing_ids),
                        "props": "labels",
                        "languages": lang,
                        "format": "json",
                    },
                    headers=self.DEFAULT_HEADERS,
                    timeout=aiohttp.ClientTimeout(total=LABELS_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        entities = data.get("entities", {})
                        
                        for qid in missing_ids:
                            cache_key = f"label:{qid}:{lang}"
                            entity = entities.get(qid, {})
                            label = (
                                entity.get("labels", {})
                                .get(lang, {})
                                .get("value")
                            )
                            if label:
                                result[qid] = label
                                # Cache successful labels
                                self._labels_cache.set(cache_key, label)
                            else:
                                result[qid] = ""
                    else:
                        # On non-200, set empty strings for missing IDs
                        logger.warning(f"Wikidata labels API returned status {resp.status} for QIDs {missing_ids}")
                        for qid in missing_ids:
                            result[qid] = ""
                            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # On error/timeout, set empty strings for missing IDs
                for qid in missing_ids:
                    result[qid] = ""
        
        return result
    
    async def get_population(self, entity_qid: str, claim_year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get population data (P1082) for an entity.
        
        Args:
            entity_qid: Wikidata QID (e.g., "Q183")
            claim_year: Optional year from claim (for matching point_in_time)
            
        Returns:
            List of population entries with keys:
            - value_int: Absolute integer population
            - point_in_time: Year if available (int or None)
            - reference: Reference info if available
            Sorted by point_in_time descending (most recent first)
        """
        if not entity_qid:
            return []
        
        # Check cache
        cache_key = f"population:{entity_qid}:{claim_year}"
        cached_result = self._entity_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            # Get entity data
            entity = await self.get_entity(entity_qid, language="de")
            if not entity:
                return []
            
            claims = entity.get("claims", {})
            p1082_statements = claims.get("P1082", [])  # Population property
            
            if not p1082_statements:
                return []
            
            results = []
            for stmt in p1082_statements:
                mainsnak = stmt.get("mainsnak", {})
                if mainsnak.get("snaktype") != "value":
                    continue
                
                datavalue = mainsnak.get("datavalue", {})
                value_obj = datavalue.get("value", {})
                
                # Extract population value
                if "amount" in value_obj:
                    amount_str = value_obj["amount"].lstrip("+")
                    try:
                        value_int = int(float(amount_str))
                    except (ValueError, TypeError):
                        continue
                else:
                    continue
                
                # Extract point_in_time (P585) from qualifiers
                qualifiers = stmt.get("qualifiers", {})
                p585 = qualifiers.get("P585", [])  # Point in time
                point_in_time = None
                if p585:
                    for time_stmt in p585:
                        if time_stmt.get("snaktype") == "value":
                            time_value = time_stmt.get("datavalue", {}).get("value", {}).get("time")
                            if time_value:
                                # Wikidata time format: +2022-01-01T00:00:00Z
                                try:
                                    point_in_time = int(time_value[1:5])  # Extract year
                                except (ValueError, IndexError):
                                    pass
                                break
                
                # Extract reference info (optional)
                references = stmt.get("references", [])
                reference = None
                if references:
                    # Get first reference
                    ref = references[0]
                    ref_snaks = ref.get("snaks", {})
                    # Try to get source (P248: stated in)
                    p248 = ref_snaks.get("P248", [])
                    if p248:
                        source_qid = p248[0].get("datavalue", {}).get("value", {}).get("id")
                        if source_qid:
                            reference = {"source_qid": source_qid}
                
                results.append({
                    "value_int": value_int,
                    "point_in_time": point_in_time,
                    "reference": reference,
                })
            
            # Sort by point_in_time descending (most recent first), then by value_int descending
            results.sort(key=lambda x: (x["point_in_time"] if x["point_in_time"] else 0, x["value_int"]), reverse=True)
            
            # Cache successful results
            self._entity_cache.set(cache_key, results)
            return results
            
        except Exception as e:
            logger.warning(f"WIKIDATA_POPULATION: Error fetching population for {entity_qid}: {e}")
            return []

