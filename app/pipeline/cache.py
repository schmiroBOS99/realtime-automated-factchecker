import time
from typing import Dict, Optional, Tuple, Any


class TTLCache:
    """
    Simple TTL cache with max size eviction.
    Stores entries as {key: (expires_at, value)}.
    """
    
    def __init__(self, default_ttl_seconds: int = 3600, max_size: int = 5000):
        self.default_ttl_seconds = default_ttl_seconds
        self.max_size = max_size
        self._data: Dict[str, Tuple[float, Any]] = {}
        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key. Returns None if missing or expired.
        """
        if key not in self._data:
            self._misses += 1
            return None
        
        expires_at, value = self._data[key]
        
        # Check if expired
        if time.time() > expires_at:
            del self._data[key]
            self._misses += 1
            return None
        
        self._hits += 1
        return value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Set value for key with TTL. Uses default_ttl_seconds if ttl_seconds not provided.
        Evicts oldest or expired entries if max_size exceeded.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        expires_at = time.time() + ttl
        
        # Evict if needed
        if len(self._data) >= self.max_size and key not in self._data:
            self._evict_one()
        
        self._data[key] = (expires_at, value)
    
    def _evict_one(self) -> None:
        """
        Evict one entry: prefer expired entries, otherwise entry with smallest expires_at (oldest).
        """
        now = time.time()
        
        # First try to evict expired entries
        for key, (expires_at, _) in list(self._data.items()):
            if now > expires_at:
                del self._data[key]
                self._evictions += 1
                return
        
        # If no expired entries, evict entry with smallest expires_at (oldest)
        if self._data:
            oldest_key = min(self._data.keys(), key=lambda k: self._data[k][0])
            del self._data[oldest_key]
            self._evictions += 1
    
    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        self._data.clear()
    
    def __contains__(self, key: str) -> bool:
        """
        Check if key exists and is not expired.
        """
        return self.get(key) is not None
    
    @property
    def hits(self) -> int:
        """Number of cache hits."""
        return self._hits
    
    @property
    def misses(self) -> int:
        """Number of cache misses."""
        return self._misses
    
    @property
    def evictions(self) -> int:
        """Number of cache evictions."""
        return self._evictions
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict with keys: hits, misses, evictions, size
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "size": len(self._data),
        }

