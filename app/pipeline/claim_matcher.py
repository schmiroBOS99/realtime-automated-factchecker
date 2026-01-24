"""
Claim matching system for cache lookup.

Uses dense embeddings + cross-encoder rerank to find semantically similar claims.
"""
import logging
from typing import Optional, Dict, Any, List
from enum import Enum

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    import torch  # type: ignore[import]
except ImportError:
    SentenceTransformer = None
    torch = None
    logging.warning("sentence-transformers not installed. Claim matching will use lexical only.")

from app.pipeline.storage.claim_store import ClaimStore
from app.pipeline.utils import canonicalize_claim
from app.pipeline.models import FactLabel
from app.pipeline.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class MatchType(str, Enum):
    """Match type for claim matching."""
    EXACT = "EXACT"
    RELATED = "RELATED"
    NO_MATCH = "NO_MATCH"


class ClaimMatcher:
    """
    Claim matcher for cache lookup.
    
    Uses 2-stage matching:
    1. Dense embedding similarity (fast, top_k candidates)
    2. Cross-encoder rerank (precise, on top_k)
    """
    
    # Thresholds
    THRESH_STRONG = 0.99  # Auto-return cached result (conservative: only very similar claims)
    THRESH_RELATED = 0.75  # Show as related but don't auto-return
    
    def __init__(
        self,
        claim_store: ClaimStore,
        embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        verifier_scorer: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
    ):
        """
        Initialize claim matcher.
        
        Args:
            claim_store: ClaimStore instance for persistence
            embedding_model_name: SentenceTransformer model for dense embeddings
            verifier_scorer: Optional callable for cross-encoder scoring (claim, candidate_claim) -> score
                           If None, uses embedding similarity only
            embedding_model: Optional pre-loaded SentenceTransformer model (shared instance)
                           If None, will use EmbeddingService singleton
        """
        self.store = claim_store
        self.verifier_scorer = verifier_scorer
        self.embedding_model_name = embedding_model_name
        
        # Use provided model, or get from EmbeddingService singleton
        if embedding_model is not None:
            self.embedding_model = embedding_model
            logger.info(f"CLAIM_MATCHER: Using provided embedding model")
        else:
            embedding_service = EmbeddingService()
            self.embedding_model = embedding_service.get_model(embedding_model_name)
            
            if self.embedding_model is None and SentenceTransformer:
                # Fallback: load locally if service failed
                try:
                    logger.info(f"CLAIM_MATCHER: Fallback: Loading model '{embedding_model_name}' locally")
                    self.embedding_model = SentenceTransformer(embedding_model_name)
                    logger.info(f"CLAIM_MATCHER: Model loaded successfully (fallback)")
                except Exception as e:
                    logger.warning(f"CLAIM_MATCHER: Failed to load embedding model: {e}")
                    self.embedding_model = None
            elif self.embedding_model is None:
                logger.warning("CLAIM_MATCHER: SentenceTransformer not available, using lexical matching only")
    
    def match(self, claim: str) -> Dict[str, Any]:
        """
        Match a claim against cached claims.
        
        Args:
            claim: Original claim text
            
        Returns:
            Dict with keys:
                - match_type: MatchType (EXACT, RELATED, NO_MATCH)
                - match_score: float (0.0 to 1.0)
                - cached_result: Optional dict with label, evidence if match found
                - canonical_claim: Canonical claim text
        """
        if not claim:
            return {
                "match_type": MatchType.NO_MATCH,
                "match_score": 0.0,
                "cached_result": None,
                "canonical_claim": "",
            }
        
        canonical = canonicalize_claim(claim)
        
        # Stage 0: Exact match (fastest)
        exact_match = self.store.get_exact(canonical)
        if exact_match:
            logger.info(f"CLAIM_MATCH: EXACT match for '{canonical[:50]}...'")
            return {
                "match_type": MatchType.EXACT,
                "match_score": 1.0,
                "cached_result": {
                    "label": exact_match["label"],
                    "evidence": exact_match["evidence"],
                },
                "canonical_claim": canonical,
            }
        
        # Stage 1: Dense embedding similarity (if embeddings available)
        if self.embedding_model:
            candidates = self._find_similar_embeddings(canonical, top_k=20)
            
            if candidates:
                # Stage 2: Cross-encoder rerank (if scorer available)
                if self.verifier_scorer:
                    best_match = self._rerank_with_crossencoder(claim, candidates)
                else:
                    # Use embedding similarity only
                    best_match = candidates[0] if candidates else None
                
                if best_match and best_match["score"] >= self.THRESH_RELATED:
                    match_type = MatchType.RELATED if best_match["score"] < self.THRESH_STRONG else MatchType.EXACT
                    logger.info(
                        f"CLAIM_MATCH: {match_type.value} match (score={best_match['score']:.3f}) "
                        f"for '{canonical[:50]}...'"
                    )
                    return {
                        "match_type": match_type,
                        "match_score": best_match["score"],
                        "cached_result": {
                            "label": best_match["label"],
                            "evidence": best_match["evidence"],
                        },
                        "canonical_claim": canonical,
                    }
        
        # No match found
        logger.debug(f"CLAIM_MATCH: NO_MATCH for '{canonical[:50]}...'")
        return {
            "match_type": MatchType.NO_MATCH,
            "match_score": 0.0,
            "cached_result": None,
            "canonical_claim": canonical,
        }
    
    def _find_similar_embeddings(self, canonical_claim: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Find similar claims using dense embeddings.
        
        Args:
            canonical_claim: Canonical claim text
            top_k: Number of candidates to return
            
        Returns:
            List of candidate dicts with keys: canonical_claim, label, evidence, score
        """
        if not self.embedding_model:
            return []
        
        # Get all cached claims with embeddings
        cached_claims = self.store.get_all_with_embeddings()
        
        if not cached_claims:
            return []
        
        # Embed query claim
        try:
            query_embedding = self.embedding_model.encode(canonical_claim, convert_to_tensor=True)
        except Exception as e:
            logger.warning(f"CLAIM_MATCHER: Error embedding query: {e}")
            return []
        
        # Compute cosine similarities
        candidates = []
        for cached in cached_claims:
            if not cached.get("embedding"):
                continue
            
            cached_embedding = torch.tensor(cached["embedding"], dtype=torch.float32)
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                cached_embedding.unsqueeze(0)
            ).item()
            
            candidates.append({
                "canonical_claim": cached["canonical_claim"],
                "label": cached["label"],
                "evidence": cached["evidence"],
                "score": similarity,
            })
        
        # Sort by similarity descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return candidates[:top_k]
    
    def _rerank_with_crossencoder(
        self,
        claim: str,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            claim: Original claim text
            candidates: List of candidate dicts from embedding search
            
        Returns:
            Best candidate dict with updated score, or None
        """
        if not self.verifier_scorer or not candidates:
            return candidates[0] if candidates else None
        
        # Extract candidate claim texts
        candidate_texts = [c["canonical_claim"] for c in candidates]
        
        try:
            # Score relevance: claim vs each candidate claim
            scores = self.verifier_scorer(claim, candidate_texts)
            
            # Update candidate scores with cross-encoder scores
            for i, candidate in enumerate(candidates):
                if i < len(scores):
                    # Combine embedding similarity (0.5 weight) with cross-encoder (0.5 weight)
                    # Cross-encoder score is relevance (0-1), so we can combine directly
                    embedding_score = candidate["score"]
                    crossencoder_score = scores[i]
                    combined_score = (embedding_score * 0.5) + (crossencoder_score * 0.5)
                    candidate["score"] = combined_score
            
            # Sort by combined score
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            return candidates[0] if candidates else None
            
        except Exception as e:
            logger.warning(f"CLAIM_MATCHER: Error in cross-encoder rerank: {e}")
            return candidates[0] if candidates else None
    
    def store_result(
        self,
        claim: str,
        label: FactLabel,
        evidence: Dict[str, Any],
    ) -> None:
        """
        Store a verification result in the cache.
        
        Args:
            claim: Original claim text
            label: FactLabel result
            evidence: Best evidence dict
        """
        canonical = canonicalize_claim(claim)
        
        # Generate embedding if model available
        embedding = None
        if self.embedding_model:
            try:
                embedding_list = self.embedding_model.encode(canonical, convert_to_tensor=False).tolist()
                embedding = embedding_list
            except Exception as e:
                logger.warning(f"CLAIM_MATCHER: Error generating embedding: {e}")
        
        # Store in database
        self.store.store(
            canonical_claim=canonical,
            label=label.value,
            evidence=evidence,
            embedding=embedding,
        )
        
        logger.debug(f"CLAIM_MATCHER: Stored result for '{canonical[:50]}...' with label {label.value}")

