"""
Async Wikipedia API client with TTL caching.

Usage example:
    import aiohttp
    from app.pipeline.wikipedia_client import WikipediaClient
    
    async with aiohttp.ClientSession() as session:
        client = WikipediaClient(session)
        results = await client.search("Angela Merkel", limit=5)
        summary = await client.get_summary("Angela Merkel")
"""
import aiohttp
import asyncio
import logging
import os
from typing import Dict, List, Optional
from urllib.parse import quote

from app.pipeline.cache import TTLCache

logger = logging.getLogger(__name__)

# Timeout configuration (realtime defaults, configurable via env)
SEARCH_TIMEOUT = int(os.getenv("FACTCHECK_WIKI_SEARCH_TIMEOUT", "3"))
SUMMARY_TIMEOUT = int(os.getenv("FACTCHECK_WIKI_SUMMARY_TIMEOUT", "5"))
EXTRACT_TIMEOUT = int(os.getenv("FACTCHECK_WIKI_EXTRACT_TIMEOUT", "5"))


class WikipediaClient:
    """
    Async Wikipedia API client with TTL caching.
    Uses a shared aiohttp.ClientSession (do NOT create a session per request).
    """
    
    WIKIPEDIA_API_URL = "https://de.wikipedia.org/w/api.php"
    WIKIPEDIA_SUMMARY_URL = "https://de.wikipedia.org/api/rest_v1/page/summary/"
    
    # Cache TTL: 6 hours (21600 seconds)
    CACHE_TTL_SECONDS = 21600
    
    # Default headers for Wikimedia API requests
    DEFAULT_HEADERS = {
        "User-Agent": "FactCheckerService/0.1 (Maro Schmitt Rodrigues; contact: maro.schmitt_rodrigues@student.reutlingen-university.de)",
        "Accept": "application/json",
    }
    
    def __init__(self, session: aiohttp.ClientSession):
        """
        Initialize Wikipedia client with a shared aiohttp session.
        
        Args:
            session: Shared aiohttp.ClientSession instance
        """
        self.session = session
        self._search_cache = TTLCache(default_ttl_seconds=self.CACHE_TTL_SECONDS, max_size=5000)
        self._summary_cache = TTLCache(default_ttl_seconds=self.CACHE_TTL_SECONDS, max_size=5000)
        self._extract_cache = TTLCache(default_ttl_seconds=self.CACHE_TTL_SECONDS, max_size=5000)
    
    async def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search Wikipedia articles by query.
        
        Args:
            query: Search query string
            limit: Maximum number of results (default: 5)
            
        Returns:
            List of search result dicts with keys: title, snippet, wordcount, etc.
            Returns empty list on error or timeout.
        """
        if not query:
            return []
        
        # Check cache
        cache_key = f"search:{query}:{limit}"
        cached_result = self._search_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            logger.debug(f"WIKIPEDIA_SEARCH: Searching for query='{query}' limit={limit}")
            async with self.session.get(
                self.WIKIPEDIA_API_URL,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": limit,
                },
                headers=self.DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=SEARCH_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Wikipedia search API returned status {resp.status} for query '{query}'")
                    return []
                
                data = await resp.json()
                
                # Debug: Log API response structure
                logger.debug(f"WIKIPEDIA_SEARCH_DEBUG: API response keys: {list(data.keys())}")
                if "query" in data:
                    logger.debug(f"WIKIPEDIA_SEARCH_DEBUG: query keys: {list(data['query'].keys())}")
                    if "search" in data["query"]:
                        logger.info(f"WIKIPEDIA_SEARCH_DEBUG: Found {len(data['query']['search'])} results for query='{query}'")
                        if data["query"]["search"]:
                            logger.info(f"WIKIPEDIA_SEARCH_DEBUG: First result title='{data['query']['search'][0].get('title', 'N/A')}'")
                    else:
                        logger.warning(f"WIKIPEDIA_SEARCH_DEBUG: No 'search' key in query response. Available keys: {list(data['query'].keys())}")
                else:
                    logger.warning(f"WIKIPEDIA_SEARCH_DEBUG: No 'query' key in API response. Available keys: {list(data.keys())}")
                    if "error" in data:
                        logger.error(f"WIKIPEDIA_SEARCH_DEBUG: API error: {data.get('error', {})}")
                
                hits = data.get("query", {}).get("search", [])
                
                # Cache successful results
                self._search_cache.set(cache_key, hits)
                return hits
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Return empty list on error/timeout
            logger.error(f"WIKIPEDIA_SEARCH_ERROR: Error searching for '{query}': {e}")
            return []
        except Exception as e:
            logger.error(f"WIKIPEDIA_SEARCH_ERROR: Unexpected error searching for '{query}': {e}", exc_info=True)
            return []
    
    async def get_summary(self, title: str) -> Dict:
        """
        Get Wikipedia article summary (title, URL, extract).
        
        Args:
            title: Article title
            
        Returns:
            Dict with keys: title, url, snippet
            Returns empty dict on error or timeout.
        """
        if not title:
            return {}
        
        # Check cache
        cache_key = f"summary:{title}"
        cached_result = self._summary_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            encoded_title = quote(title, safe="")
            async with self.session.get(
                self.WIKIPEDIA_SUMMARY_URL + encoded_title,
                headers=self.DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=SUMMARY_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Wikipedia summary API returned status {resp.status} for title '{title}'")
                    return {}
                
                data = await resp.json()
                
                url = (
                    data.get("content_urls", {})
                    .get("desktop", {})
                    .get("page", "")
                )
                extract = data.get("extract", "")
                
                summary = {
                    "title": data.get("title", title),
                    "url": url,
                    "snippet": extract[:500],  # Limit to ~500 chars
                }
                
                # Cache successful results
                self._summary_cache.set(cache_key, summary)
                return summary
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Return empty dict on error/timeout
            return {}
    
    async def get_extract(self, title: str, sentences: int = 5) -> Dict:
        """
        Get Wikipedia article extract (full text, not just summary).
        
        Args:
            title: Article title
            sentences: Number of sentences to extract (default: 5)
                      Note: Wikipedia API has a limit, typically returns max ~20-30 sentences
            
        Returns:
            Dict with keys: title, extract
            Returns empty dict on error or timeout.
        """
        if not title:
            return {}
        
        # Check cache
        cache_key = f"extract:{title}:{sentences}"
        cached_result = self._extract_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            # For large sentence counts (>50), use exlimit=max to get more content
            # But note: Wikipedia API still has limits, typically max ~20-30 sentences
            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": "1",
                "exintro": "0",
                "redirects": "1",
                "titles": title,
            }
            
            # If requesting many sentences, try to get more content
            if sentences > 50:
                # Use exlimit=max to get maximum content (but API still limits to ~20-30 sentences)
                params["exlimit"] = "max"
                # Don't set exsentences for max content
            else:
                params["exsentences"] = str(sentences)
            
            async with self.session.get(
                self.WIKIPEDIA_API_URL,
                params=params,
                headers=self.DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=EXTRACT_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Wikipedia extract API returned status {resp.status} for title '{title}'")
                    return {}
                
                data = await resp.json()
                pages = data.get("query", {}).get("pages", {})
                
                # Get first page (there should be only one)
                page_data = None
                for page_id, page_info in pages.items():
                    page_data = page_info
                    break
                
                if not page_data:
                    return {}
                
                extract = page_data.get("extract", "")
                if not extract:
                    return {}
                
                result = {
                    "title": page_data.get("title", title),
                    "extract": extract,
                }
                
                # Cache successful results
                self._extract_cache.set(cache_key, result)
                return result
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Return empty dict on error/timeout
            return {}

    async def get_full_extract(self, title: str, max_chars: int = 25000) -> Dict:
        """
        Get a larger plaintext extract by character count (helps reach non-intro sections).

        This is useful for pages like countries where important facts (e.g., Bevölkerung)
        are not in the lead.
        """
        if not title:
            return {}

        max_chars = max(1000, min(int(max_chars), 50000))

        cache_key = f"full_extract:{title}:{max_chars}"
        cached_result = self._extract_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        try:
            async with self.session.get(
                self.WIKIPEDIA_API_URL,
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "explaintext": "1",
                    "exsectionformat": "plain",
                    "exintro": "0",
                    "redirects": "1",
                    "exchars": str(max_chars),
                    "titles": title,
                },
                headers=self.DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=EXTRACT_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Wikipedia full extract API returned status {resp.status} for title '{title}'")
                    return {}

                data = await resp.json()
                pages = data.get("query", {}).get("pages", {})

                page_data = None
                for _, page_info in pages.items():
                    page_data = page_info
                    break

                if not page_data:
                    return {}

                extract = page_data.get("extract", "")
                if not extract:
                    return {}

                result = {
                    "title": page_data.get("title", title),
                    "extract": extract,
                }
                self._extract_cache.set(cache_key, result)
                return result

        except (aiohttp.ClientError, asyncio.TimeoutError):
            return {}
        except Exception as e:
            logger.debug(f"WIKIPEDIA_FULL_EXTRACT: Unexpected error for '{title}': {e}", exc_info=True)
            return {}
    
    async def get_section(self, title: str, section_name: str = "Bevölkerung") -> Dict:
        """
        Get a specific section from a Wikipedia article (by section title).

        Uses MediaWiki API `action=parse`:
        - First call: `prop=sections` to find the section index for `section_name`
        - Second call: `prop=text&section=<index>` to get HTML, then strip tags
        
        Args:
            title: Article title
            section_name: Section name to extract (e.g., "Bevölkerung", "Demografie")
            
        Returns:
            Dict with keys: title, extract (section content)
            Returns empty dict on error or timeout.
        """
        if not title:
            return {}
        
        # Check cache
        cache_key = f"section:{title}:{section_name}"
        cached_result = self._extract_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            section_index = await self._find_section_index(title=title, section_name=section_name)
            if not section_index:
                return {}

            html = await self._get_section_html(title=title, section_index=section_index)
            if not html:
                return {}

            # Strip HTML tags (keep it lightweight; we only need plaintext for sentence splitting)
            import re

            text_content = re.sub(r"<[^>]+>", " ", html)
            text_content = re.sub(r"\s+", " ", text_content).strip()
            if not text_content or len(text_content) < 50:
                return {}

            result = {"title": title, "extract": text_content}
            self._extract_cache.set(cache_key, result)
            logger.info(
                f"WIKIPEDIA_SECTION: Retrieved section '{section_name}' (index={section_index}) "
                f"from article '{title}' ({len(text_content)} chars)"
            )
            return result
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"WIKIPEDIA_SECTION: Error fetching section '{section_name}' from '{title}': {e}")
            return {}
        except Exception as e:
            logger.debug(f"WIKIPEDIA_SECTION: Unexpected error fetching section '{section_name}' from '{title}': {e}")
            return {}

    async def _find_section_index(self, title: str, section_name: str) -> Optional[str]:
        """Find MediaWiki section index for a section title."""
        if not title or not section_name:
            return None

        try:
            async with self.session.get(
                self.WIKIPEDIA_API_URL,
                params={
                    "action": "parse",
                    "format": "json",
                    "page": title,
                    "prop": "sections",
                    "redirects": "1",
                },
                headers=self.DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=EXTRACT_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                sections = data.get("parse", {}).get("sections", []) or []
                if not sections:
                    return None

                target = section_name.strip().lower()

                # Prefer exact match; fallback to containment
                exact = None
                contains = None
                for sec in sections:
                    line = (sec.get("line") or "").strip()
                    idx = sec.get("index")
                    if not line or not idx:
                        continue
                    line_l = line.lower()
                    if line_l == target:
                        exact = idx
                        break
                    if target in line_l:
                        contains = contains or idx

                return exact or contains
        except Exception:
            return None

    async def _get_section_html(self, title: str, section_index: str) -> str:
        """Fetch section HTML via MediaWiki API action=parse."""
        if not title or not section_index:
            return ""

        async with self.session.get(
            self.WIKIPEDIA_API_URL,
            params={
                "action": "parse",
                "format": "json",
                "page": title,
                "prop": "text",
                "section": str(section_index),
                "redirects": "1",
            },
            headers=self.DEFAULT_HEADERS,
            timeout=aiohttp.ClientTimeout(total=EXTRACT_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json()
            text_obj = data.get("parse", {}).get("text", {}) or {}
            # MediaWiki returns {"*": "<div>...</div>"}
            html = text_obj.get("*") if isinstance(text_obj, dict) else ""
            return html or ""

