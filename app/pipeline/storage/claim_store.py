"""
SQLite-based claim store for caching fact-check results.

Stores canonical claims with embeddings, labels, and evidence for fast retrieval.
"""
import sqlite3
import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ClaimStore:
    """
    SQLite-based store for cached fact-check results.
    
    Stores:
    - canonical_claim: Normalized claim text
    - embedding: Dense vector (stored as JSON array)
    - label: FactLabel value
    - evidence_json: JSON-serialized best_evidence dict
    - created_at: Timestamp
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize claim store.
        
        Args:
            db_path: Path to SQLite database file.
                    If None, uses default: ./data/claim_cache.db
        """
        if db_path is None:
            # Default: create data directory if it doesn't exist
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "claim_cache.db")
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_claim TEXT UNIQUE NOT NULL,
                embedding TEXT,  -- JSON array of floats
                label TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on canonical_claim for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_canonical_claim 
            ON claims(canonical_claim)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"CLAIM_STORE: Initialized database at {self.db_path}")
    
    def store(
        self,
        canonical_claim: str,
        label: str,
        evidence: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> None:
        """
        Store a claim result in the cache.
        
        Args:
            canonical_claim: Canonical claim text
            label: FactLabel value (SUPPORTED, REFUTED, NOT_ENOUGH_INFO)
            evidence: Best evidence dict (will be JSON-serialized)
            embedding: Optional embedding vector
        """
        if not canonical_claim:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_json = json.dumps(embedding) if embedding else None
        evidence_json = json.dumps(evidence)
        
        # Upsert: update if exists, insert if not
        cursor.execute("""
            INSERT INTO claims (canonical_claim, embedding, label, evidence_json, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(canonical_claim) DO UPDATE SET
                embedding = excluded.embedding,
                label = excluded.label,
                evidence_json = excluded.evidence_json,
                updated_at = CURRENT_TIMESTAMP
        """, (canonical_claim, embedding_json, label, evidence_json))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"CLAIM_STORE: Stored claim '{canonical_claim[:50]}...' with label {label}")
    
    def get_exact(self, canonical_claim: str) -> Optional[Dict[str, Any]]:
        """
        Get exact match for canonical claim.
        
        Args:
            canonical_claim: Canonical claim text
            
        Returns:
            Dict with keys: canonical_claim, label, evidence_json, embedding, created_at
            or None if not found
        """
        if not canonical_claim:
            return None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT canonical_claim, label, evidence_json, embedding, created_at
            FROM claims
            WHERE canonical_claim = ?
        """, (canonical_claim,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        canonical, label, evidence_json, embedding_json, created_at = row
        
        embedding = json.loads(embedding_json) if embedding_json else None
        evidence = json.loads(evidence_json)
        
        return {
            "canonical_claim": canonical,
            "label": label,
            "evidence": evidence,
            "embedding": embedding,
            "created_at": created_at,
        }
    
    def get_all_with_embeddings(self) -> List[Dict[str, Any]]:
        """
        Get all claims with embeddings for similarity search.
        
        Returns:
            List of dicts with keys: canonical_claim, embedding, label, evidence_json
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT canonical_claim, embedding, label, evidence_json
            FROM claims
            WHERE embedding IS NOT NULL
        """)
        
        results = []
        for row in cursor.fetchall():
            canonical, embedding_json, label, evidence_json = row
            if embedding_json:
                embedding = json.loads(embedding_json)
                evidence = json.loads(evidence_json)
                results.append({
                    "canonical_claim": canonical,
                    "embedding": embedding,
                    "label": label,
                    "evidence": evidence,
                })
        
        conn.close()
        return results
    
    def count(self) -> int:
        """Get total number of cached claims."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM claims")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def clear(self) -> int:
        """
        Clear all cached claims from the database.
        
        Returns:
            Number of claims deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get count before deletion
        cursor.execute("SELECT COUNT(*) FROM claims")
        count = cursor.fetchone()[0]
        
        # Delete all claims
        cursor.execute("DELETE FROM claims")
        
        conn.commit()
        conn.close()
        
        logger.info(f"CLAIM_STORE: Cleared {count} cached claims from database")
        return count

