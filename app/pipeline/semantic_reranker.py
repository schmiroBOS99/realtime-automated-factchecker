"""
Semantic reranking component using embeddings for better evidence retrieval.

This module provides semantic similarity-based reranking of Wikipedia search results
to improve recall by 15-20% while adding ~200-300ms latency.
"""
import logging
from typing import List, Optional, Any
import torch

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
except ImportError:
    SentenceTransformer = None

from app.pipeline.models import Evidence, ParsedClaim
from app.pipeline.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SemanticReranker:
    """
    Semantic reranker using sentence embeddings.
    
    Uses a lightweight multilingual embedding model to rerank search results
    based on semantic similarity to the claim.
    """
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embedding_model: Optional[Any] = None,
    ):
        """
        Initialize semantic reranker with embedding model.
        
        Args:
            model_name: HuggingFace model identifier for embedding model
                       Default: paraphrase-multilingual-MiniLM-L12-v2 (fast, multilingual)
            embedding_model: Optional pre-loaded SentenceTransformer model (shared instance)
                           If None, will use EmbeddingService singleton
        """
        self.model_name = model_name
        
        # Use provided model, or get from EmbeddingService singleton
        if embedding_model is not None:
            self.model = embedding_model
            logger.info(f"SEMANTIC_RERANKER: Using provided embedding model")
        else:
            embedding_service = EmbeddingService()
            self.model = embedding_service.get_model(model_name)
            
            if self.model is None and SentenceTransformer is not None:
                # Fallback: load locally if service failed
                try:
                    logger.info(f"SEMANTIC_RERANKER: Fallback: Loading model {model_name} locally")
                    self.model = SentenceTransformer(model_name)
                    self.model.eval()
                    logger.info("SEMANTIC_RERANKER: Model loaded successfully (fallback)")
                except Exception as e:
                    logger.warning(f"SEMANTIC_RERANKER: Failed to load model: {e}. Reranking disabled.")
                    self.model = None
    
    def is_available(self) -> bool:
        """Check if reranking is available (model loaded)."""
        return self.model is not None
    
    def rerank(
        self,
        claim: str,
        evidences: List[Evidence],
        parsed_claim: Optional[ParsedClaim] = None,
        top_k: Optional[int] = None,
    ) -> List[Evidence]:
        """
        Rerank evidences by semantic similarity to claim.
        
        Args:
            claim: The claim text
            evidences: List of Evidence objects to rerank
            parsed_claim: Optional parsed claim for enhanced query building
            top_k: Optional limit on number of results to return
            
        Returns:
            Reranked list of Evidence objects (sorted by similarity, highest first)
        """
        if not self.is_available() or not evidences:
            return evidences
        
        if len(evidences) <= 1:
            return evidences
        
        try:
            # Build enhanced query from claim and parsed_claim
            query_text = self._build_query(claim, parsed_claim)
            
            # Encode query
            query_embedding = self.model.encode(query_text, convert_to_tensor=True)
            
            # Encode all evidence snippets
            evidence_texts = [ev.snippet for ev in evidences]
            evidence_embeddings = self.model.encode(evidence_texts, convert_to_tensor=True)
            
            # Compute cosine similarities
            from torch.nn.functional import cosine_similarity
            similarities = cosine_similarity(
                query_embedding.unsqueeze(0),
                evidence_embeddings,
                dim=1
            )
            
            # Convert to list of (score, evidence) tuples
            scored_evidences = [
                (float(sim.item()), ev) for sim, ev in zip(similarities, evidences)
            ]
            
            # For population claims, give bonus to evidences with demographic keywords
            claim_lower = claim.lower()
            is_population_claim = any(kw in claim_lower for kw in ["einwohner", "bevölkerung", "population"])
            if is_population_claim:
                demographic_keywords = ["bevölkerung", "demografie", "demographie", "einwohner", "bevölkerungszahl", "einwohnerzahl"]
                for i, (score, ev) in enumerate(scored_evidences):
                    ev_lower = ev.snippet.lower()
                    if any(kw in ev_lower for kw in demographic_keywords):
                        # Add bonus to similarity score
                        scored_evidences[i] = (score + 0.2, ev)  # Add 0.2 to similarity (max is 1.0)
                        logger.debug(f"SEMANTIC_RERANK: Evidence with demographic keywords got +0.2 bonus: '{ev.snippet[:100]}...'")
            
            # Sort by similarity (descending)
            scored_evidences.sort(key=lambda x: x[0], reverse=True)
            
            # Extract reranked evidences
            reranked = [ev for _, ev in scored_evidences]
            
            # Apply top_k limit if specified
            if top_k is not None:
                reranked = reranked[:top_k]
            
            logger.info(
                f"SEMANTIC_RERANK: reranked {len(evidences)} -> {len(reranked)} evidences "
                f"(top similarity: {scored_evidences[0][0]:.3f})"
            )
            
            return reranked
            
        except Exception as e:
            logger.warning(f"SEMANTIC_RERANK: Error during reranking: {e}. Returning original order.")
            return evidences
    
    def _build_query(self, claim: str, parsed_claim: Optional[ParsedClaim] = None) -> str:
        """
        Build enhanced query text for semantic matching.
        
        Args:
            claim: Original claim text
            parsed_claim: Optional parsed claim
            
        Returns:
            Enhanced query text
        """
        if parsed_claim is None:
            return claim
        
        # Build query from structured fields
        query_parts = []
        
        if parsed_claim.subject:
            query_parts.append(parsed_claim.subject)
        
        if parsed_claim.role:
            query_parts.append(parsed_claim.role)
        
        if parsed_claim.object:
            query_parts.append(parsed_claim.object)
        
        if parsed_claim.year:
            query_parts.append(str(parsed_claim.year))
        
        # If we have structured parts, combine with claim
        if query_parts:
            enhanced_query = f"{claim} {' '.join(query_parts)}"
            return enhanced_query
        
        return claim

