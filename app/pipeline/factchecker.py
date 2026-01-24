"""
End-to-end fact-checking pipeline orchestrator.

Usage example:
    from app.pipeline.factchecker import FactChecker
    
    async def main():
        factchecker = FactChecker()
        await factchecker.start()
        
        try:
            result = await factchecker.check(
                claim="Angela Merkel ist Bundeskanzlerin",
                max_pages=3
            )
            # Returns:
            # {
            #     "claim": "Angela Merkel ist Bundeskanzlerin",
            #     "label": "SUPPORTED",  # or "REFUTED" or "NOT_ENOUGH_INFO"
            #     "best_evidence": {
            #         "title": "Angela Merkel",
            #         "url": "https://...",
            #         "snippet": "...",
            #         "source": "wikipedia"
            #     },
            #     "scores": [
            #         {"entailment": 0.85, "contradiction": 0.10, "neutral": 0.05}
            #     ],
            #     "evidence_count": 3
            # }
            print(result)
        finally:
            await factchecker.close()
    
    # Run with: asyncio.run(main())
"""
import aiohttp
import asyncio
import re
import logging
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

from app.pipeline.wikipedia_client import WikipediaClient
from app.pipeline.wikidata_client import WikidataClient
from app.pipeline.retriever import Retriever
from app.pipeline.verifier import Verifier
from app.pipeline.claim_parser import ClaimParser
from app.pipeline.models import ParsedClaim, Evidence, FactLabel, RunResult, RouteType
from app.pipeline.claim_matcher import ClaimMatcher, MatchType
from app.pipeline.storage.claim_store import ClaimStore
from app.pipeline.embedding_service import EmbeddingService
from app.pipeline.comparison_verifier import ComparisonVerifier
from app.pipeline.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)

# Top-K evidences to use for NLI verification (reduces computation)
NLI_TOP_K = 3

# Party QIDs mapping for fast-path Wikidata lookup
# Used in _resolve_party_entity_conservative() to avoid ambiguous searches
# Format: "party_shortname" or "Full Party Name": "Wikidata QID"
PARTY_QIDS = {
    # Germany
    "CDU": "Q49722",
    "CSU": "Q102313",
    "SPD": "Q49768",
    "AfD": "Q450140",
    "Bündnis 90/Die Grünen": "Q1009322",
    "FDP": "Q49766",
    "Die Linke": "Q182436",
    
    # USA
    "Democratic Party": "Q29552",
    "Republican Party": "Q29468",
    "Democrats": "Q29552",
    "Republicans": "Q29468",
    
    # UK
    "Conservative Party": "Q9626",
    "Labour Party": "Q9630",
    "Liberal Democrats": "Q9624",
    "Scottish National Party": "Q10658",
    "SNP": "Q10658",
    "Tories": "Q9626",
    
    # France
    "La République En Marche": "Q23731823",
    "Renaissance": "Q112567854",  # Renamed from LREM in 2022
    "Les Républicains": "Q20012759",
    "Rassemblement National": "Q205150",
    "Front National": "Q205150",  # Former name of RN
    "La France Insoumise": "Q23731823",
    "Parti Socialiste": "Q170972",
    
    # Spain
    "Partido Popular": "Q208242",
    "PSOE": "Q202012",
    "Partido Socialista Obrero Español": "Q202012",
    "Vox": "Q17101759",
    "Podemos": "Q17301071",
    "Ciudadanos": "Q1139031",
    
    # Italy
    "Fratelli d'Italia": "Q1757843",
    "Lega": "Q47750",
    "Lega Nord": "Q47750",
    "Partito Democratico": "Q47729",
    "Forza Italia": "Q14924303",
    "Movimento 5 Stelle": "Q14957018",
    "M5S": "Q14957018",
    
    # Japan
    "Liberal Democratic Party": "Q208121",  # Japan's LDP
    "Constitutional Democratic Party": "Q30634601",
    "Komeito": "Q205330",
    "Japanese Communist Party": "Q329055",
    "Democratic Party for the People": "Q97436247",
}

# Party name normalization: map shortcuts to full names
PARTY_NORMALIZATION = {
    # Germany
    "CDU": "Christlich Demokratische Union Deutschlands",
    "CSU": "Christlich-Soziale Union in Bayern",
    "SPD": "Sozialdemokratische Partei Deutschlands",
    "FDP": "Freie Demokratische Partei",
    "GRUENE": "Bündnis 90/Die Grünen",
    "GRÜNE": "Bündnis 90/Die Grünen",
    "GRUEN": "Bündnis 90/Die Grünen",
    "AfD": "Alternative für Deutschland",
    "LINKE": "Die Linke",
    
    # USA
    "Democrats": "Democratic Party",
    "Republicans": "Republican Party",
    "GOP": "Republican Party",
    
    # UK
    "Tories": "Conservative Party",
    "Labour": "Labour Party",
    "Lib Dems": "Liberal Democrats",
    "SNP": "Scottish National Party",
    
    # France
    "LREM": "Renaissance",
    "RN": "Rassemblement National",
    "FN": "Rassemblement National",
    "LFI": "La France Insoumise",
    "PS": "Parti Socialiste",
    
    # Spain
    "PP": "Partido Popular",
    
    # Italy
    "FdI": "Fratelli d'Italia",
    "Lega Nord": "Lega",
    "PD": "Partito Democratico",
    "FI": "Forza Italia",
    "M5S": "Movimento 5 Stelle",
}


class FactChecker:
    """
    End-to-end fact-checking pipeline orchestrator.
    Manages Wikipedia retrieval and NLI-based verification.
    """
    
    def __init__(self, enable_claim_cache: bool = True, cache_db_path: Optional[str] = None):
        """
        Initialize FactChecker.
        Components are created lazily when start() is called.
        
        Args:
            enable_claim_cache: Enable claim matching cache (default: True)
            cache_db_path: Optional path to cache database (default: ./data/claim_cache.db)
        """
        self._session: Optional[aiohttp.ClientSession] = None
        self._wikipedia_client: Optional[WikipediaClient] = None
        self._wikidata_client: Optional[WikidataClient] = None
        self._retriever: Optional[Retriever] = None
        self._verifier: Optional[Verifier] = None
        self._comparison_verifier: Optional[ComparisonVerifier] = None
        self._response_formatter: Optional[ResponseFormatter] = None
        self._claim_parser: Optional[ClaimParser] = None
        self._claim_matcher: Optional[ClaimMatcher] = None
        self._claim_store: Optional[ClaimStore] = None
        self.enable_claim_cache = enable_claim_cache
        self.cache_db_path = cache_db_path
    
    async def start(self) -> None:
        """
        Start the fact-checker pipeline.
        Creates aiohttp session and initializes components.
        """
        if self._session is None or self._session.closed:
            # Create session with default timeout
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        
        # Initialize components
        if self._wikipedia_client is None:
            self._wikipedia_client = WikipediaClient(self._session)
        
        if self._wikidata_client is None:
            self._wikidata_client = WikidataClient(self._session)
        
        if self._verifier is None:
            self._verifier = Verifier()
        
        if self._comparison_verifier is None:
            self._comparison_verifier = ComparisonVerifier()
        
        if self._response_formatter is None:
            self._response_formatter = ResponseFormatter()
        
        if self._retriever is None:
            # Initialize retriever with semantic reranking and query expansion enabled
            # Cross-encoder reranking (Stage 2) configurable via env: FACTCHECK_ENABLE_CROSSENCODER
            enable_crossencoder = os.getenv("FACTCHECK_ENABLE_CROSSENCODER", "").lower() in ("1", "true", "yes")
            relevance_scorer = None
            if enable_crossencoder and self._verifier:
                # Pass verifier's batch scoring method as callable to avoid circular imports
                relevance_scorer = self._verifier.score_relevance_batch
                logger.info("RETRIEVER: Cross-encoder reranking enabled (FACTCHECK_ENABLE_CROSSENCODER=1)")
            
            # Get shared embedding model from EmbeddingService
            embedding_service = EmbeddingService()
            shared_embedding_model = embedding_service.get_model()
            
            self._retriever = Retriever(
                self._wikipedia_client,
                enable_semantic_reranking=True,
                enable_query_expansion=False,  # Default OFF, only as fallback
                enable_crossencoder_rerank=enable_crossencoder,
                crossencoder_prune_top_n=50,  # Prune to top 50 before Stage 2
                relevance_scorer=relevance_scorer,
                embedding_model=shared_embedding_model,
            )
        
        if self._claim_parser is None:
            self._claim_parser = ClaimParser()
        
        # Initialize claim cache if enabled
        if self.enable_claim_cache:
            if self._claim_store is None:
                self._claim_store = ClaimStore(db_path=self.cache_db_path)
            
            if self._claim_matcher is None:
                # Get verifier scorer for cross-encoder rerank
                verifier_scorer = None
                if self._verifier:
                    verifier_scorer = self._verifier.score_relevance_batch
                
                # Get shared embedding model from EmbeddingService
                embedding_service = EmbeddingService()
                shared_embedding_model = embedding_service.get_model()
                
                self._claim_matcher = ClaimMatcher(
                    claim_store=self._claim_store,
                    verifier_scorer=verifier_scorer,
                    embedding_model=shared_embedding_model,
                )
                logger.info(f"CLAIM_CACHE: Initialized (store at {self._claim_store.db_path}, {self._claim_store.count()} cached claims)")
    
    @property
    def wikipedia_client(self) -> Optional[WikipediaClient]:
        """Get WikipediaClient instance (for debugging)."""
        return self._wikipedia_client
    
    @property
    def wikidata_client(self) -> Optional[WikidataClient]:
        """Get WikidataClient instance (for debugging)."""
        return self._wikidata_client
    
    def clear_cache(self) -> int:
        """
        Clear ALL caches comprehensively.
        
        This clears:
        - SQLite database (ClaimStore)
        - Wikipedia API caches (search, summary, extract)
        - Wikidata API caches (search, entity, labels)
        - In-memory embeddings cache in ClaimMatcher (if available)
        
        Returns:
            Number of claims cleared from ClaimStore
        """
        count = 0
        wikipedia_caches_cleared = 0
        wikidata_caches_cleared = 0
        
        # 1. Clear ClaimStore (SQLite database)
        if self._claim_store:
            count = self._claim_store.clear()
            logger.info(f"CACHE_CLEAR: Cleared {count} claims from SQLite database")
        
        # 2. Clear Wikipedia API caches (TTLCache instances)
        if self._wikipedia_client:
            if hasattr(self._wikipedia_client, '_search_cache'):
                self._wikipedia_client._search_cache.clear()
                wikipedia_caches_cleared += 1
            if hasattr(self._wikipedia_client, '_summary_cache'):
                self._wikipedia_client._summary_cache.clear()
                wikipedia_caches_cleared += 1
            if hasattr(self._wikipedia_client, '_extract_cache'):
                self._wikipedia_client._extract_cache.clear()
                wikipedia_caches_cleared += 1
            logger.info(f"CACHE_CLEAR: Cleared {wikipedia_caches_cleared} Wikipedia API caches")
        
        # 3. Clear Wikidata API caches (TTLCache instances)
        if self._wikidata_client:
            if hasattr(self._wikidata_client, '_search_cache'):
                self._wikidata_client._search_cache.clear()
                wikidata_caches_cleared += 1
            if hasattr(self._wikidata_client, '_entity_cache'):
                self._wikidata_client._entity_cache.clear()
                wikidata_caches_cleared += 1
            if hasattr(self._wikidata_client, '_labels_cache'):
                self._wikidata_client._labels_cache.clear()
                wikidata_caches_cleared += 1
            logger.info(f"CACHE_CLEAR: Cleared {wikidata_caches_cleared} Wikidata API caches")
        
        # 4. Clear ClaimMatcher in-memory cache (if any)
        if self._claim_matcher:
            # ClaimMatcher loads embeddings from ClaimStore on-demand, so clearing
            # the store should be sufficient, but we log it for clarity
            logger.info(f"CACHE_CLEAR: ClaimMatcher will reload embeddings from empty store on next match")
        
        logger.info(
            f"CACHE_CLEAR: Total caches cleared - ClaimStore: {count} claims, "
            f"Wikipedia: {wikipedia_caches_cleared} caches, Wikidata: {wikidata_caches_cleared} caches"
        )
        
        return count
    
    async def close(self) -> None:
        """
        Close the fact-checker pipeline.
        Closes aiohttp session and cleans up resources.
        """
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        
        # Clear component references
        self._wikipedia_client = None
        self._wikidata_client = None
        self._retriever = None
        self._verifier = None
        self._claim_parser = None
    
    async def check(self, claim: str, max_pages: int = 3) -> Dict[str, Any]:
        """
        Check a claim end-to-end: parse, route, retrieve evidence and verify.
        
        Args:
            claim: The claim text to verify
            max_pages: Maximum number of evidence pages to retrieve
            
        Returns:
            JSON-serializable dict with:
                - "claim": str
                - "path": str ("WIKIDATA_FAST_PATH" or "WIKIPEDIA_RETRIEVAL")
                - "relation_type": str
                - "parsed_claim": dict with subject/predicate/object/year/party/role if present
                - "label": str ("SUPPORTED", "REFUTED", or "NOT_ENOUGH_INFO")
                - "best_evidence": dict with title, url, snippet, source, or None
                - "scores": list of score dicts (entailment, contradiction, neutral)
                - "evidence_count": int
                - Additional fields for Wikidata fast path diagnostics
        """
        # Ensure session is started
        if self._session is None or self._session.closed:
            await self.start()
        
        # ===== STEP 0: Check if claim is personal/subjective/colloquial =====
        # These claims should not be fact-checked and should return NOT_ENOUGH_INFO immediately
        if self._claim_parser.is_personal_or_subjective_claim(claim):
            logger.info(
                f"CLAIM_FILTER: Skipping personal/subjective/colloquial claim '{claim[:60]}...' "
                f"(not treating as fact-checkable claim)"
            )
            return {
                "claim": claim,
                "path": "CLAIM_FILTERED",
                "relation_type": "personal_subjective",
                "parsed_claim": {},
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "answer": "Diese Aussage ist persönlich oder subjektiv und kann nicht als faktische Behauptung überprüft werden.",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
            }
        
        # ===== STEP 1: Claim Cache Lookup =====
        related_info = None  # Initialize for later use
        if self.enable_claim_cache and self._claim_matcher:
            match_result = self._claim_matcher.match(claim)
            match_type = match_result["match_type"]
            match_score = match_result["match_score"]
            cached_result = match_result["cached_result"]
            
            logger.info(
                f"CLAIM_CACHE: match_type={match_type.value} match_score={match_score:.3f} "
                f"for claim '{claim[:50]}...'"
            )
            
            # Auto-return if strong match (score >= THRESH_STRONG)
            # BUT: Skip cache for comparison claims (they need full verification)
            # CRITICAL: Only use cache for EXACT matches (score >= 0.99) to be conservative
            # Also check for temporal differences ("ist" vs "war") and numerical differences
            if match_type == MatchType.EXACT and match_score >= ClaimMatcher.THRESH_STRONG:
                # Parse claim to check if it's a comparison claim
                parsed_claim_check = self._claim_parser.parse_claim(claim)
                is_comparison = (
                    parsed_claim_check.category == "comparison" or
                    parsed_claim_check.relation_type == "comparison"
                )
                
                # Skip cache for comparison claims (they need comparison verification)
                if is_comparison:
                    logger.info(
                        f"CLAIM_CACHE: Skipping cache for comparison claim '{claim[:50]}...' "
                        f"(comparison claims require full verification)"
                    )
                elif cached_result:
                    canonical_claim = match_result.get("canonical_claim", "")
                    
                    # CRITICAL: Check for temporal differences ("ist" vs "war")
                    # "Angela Merkel ist Bundeskanzlerin" should NOT match "Angela Merkel war Bundeskanzlerin"
                    claim_lower = claim.lower()
                    canonical_lower = canonical_claim.lower()
                    claim_has_ist = " ist " in claim_lower or claim_lower.startswith("ist ")
                    claim_has_war = " war " in claim_lower or claim_lower.startswith("war ")
                    canonical_has_ist = " ist " in canonical_lower or canonical_lower.startswith("ist ")
                    canonical_has_war = " war " in canonical_lower or canonical_lower.startswith("war ")
                    
                    # If temporal tense differs, skip cache
                    if (claim_has_ist and canonical_has_war) or (claim_has_war and canonical_has_ist):
                        logger.info(
                            f"CLAIM_CACHE: Skipping cache for temporal difference "
                            f"(claim='{claim[:50]}...' vs cached='{canonical_claim[:50]}...', ist/war mismatch)"
                        )
                        cached_result = None  # Prevent cache return
                    
                    # Additional check: For numerical claims, verify that numbers match
                    # This prevents cache hits for claims with different numbers (e.g., "27 Mitgliedstaaten" vs "30 Mitgliedstaaten")
                    # IMPORTANT: Also check EXACT matches, because cached entries might be incorrect
                    if cached_result and self._is_numerical_claim_cache(claim, canonical_claim):
                        if not self._numerical_claims_match(claim, canonical_claim):
                            logger.info(
                                f"CLAIM_CACHE: Skipping cache for numerical claim with different numbers "
                                f"(match_type={match_type.value}, claim='{claim[:50]}...', cached='{canonical_claim[:50]}...')"
                            )
                            cached_result = None  # Prevent cache return
                    
                    if cached_result:
                        logger.info(f"CLAIM_CACHE: Returning cached result (score={match_score:.3f})")
                        # Parse claim for answer generation
                        parsed_claim_for_answer = self._claim_parser.parse_claim(claim)
                        parsed_claim_dict_for_answer = {}
                        if parsed_claim_for_answer.subject:
                            parsed_claim_dict_for_answer["subject"] = parsed_claim_for_answer.subject
                        if parsed_claim_for_answer.object:
                            parsed_claim_dict_for_answer["object"] = parsed_claim_for_answer.object
                        if parsed_claim_for_answer.role:
                            parsed_claim_dict_for_answer["role"] = parsed_claim_for_answer.role
                        
                        # Generate answer
                        answer = self._response_formatter.format_answer(
                            claim=claim,
                            label=cached_result["label"],
                            parsed_claim=parsed_claim_dict_for_answer,
                            best_evidence=cached_result.get("evidence"),
                            population_correction=None,
                            geo_correction=None,
                        )
                        
                        return {
                            "claim": claim,
                            "path": "CLAIM_CACHE",
                            "relation_type": "cached",
                            "parsed_claim": {},
                            "label": cached_result["label"],
                            "answer": answer,
                            "best_evidence": cached_result["evidence"],
                            "scores": [],
                            "evidence_count": 1,
                            "cache_match": {
                                "match_type": match_type.value,
                                "match_score": match_score,
                                "canonical_claim": match_result["canonical_claim"],
                            },
                        }
            
            # For related matches below strong threshold, continue with full pipeline
            # but include related info in response
            if match_type == MatchType.RELATED and match_score >= ClaimMatcher.THRESH_RELATED:
                related_info = {
                    "match_type": match_type.value,
                    "match_score": match_score,
                    "cached_label": cached_result["label"] if cached_result else None,
                }
        
        # Parse claim using ClaimParser
        parsed_claim = self._claim_parser.parse_claim(claim)
        
        # Build parsed_claim dict for response (only include non-empty fields)
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
        
        # A) Definition/Containment Claim Guard: Skip decomposition for simple definition-like claims
        # These claims often produce broken subclaims and benefit from full-claim retrieval
        # BUT: Never skip decomposition for comparison claims - they need it to work properly
        is_comparison = (
            parsed_claim.category == "comparison" or
            parsed_claim.relation_type == "comparison"
        )
        
        if parsed_claim.subclaims and not is_comparison and self._is_definition_like_claim(parsed_claim.canonical_claim, parsed_claim.subject):
            logger.info(
                f"CLAIM_DECOMP: Skipping decomposition for definition-like claim "
                f"(canonical='{parsed_claim.canonical_claim[:60]}...', subject='{parsed_claim.subject}')"
            )
            parsed_claim.subclaims = []
            parsed_claim.is_hard_claim = False  # Treat as simple claim for retrieval
        elif is_comparison:
            logger.debug(
                f"CLAIM_DECOMP: Keeping decomposition for comparison claim "
                f"(category='{parsed_claim.category}', subclaims={len(parsed_claim.subclaims)})"
            )
        
        # Log normalization and decomposition
        logger.info(
            f"CLAIM_NORM: original='{claim[:60]}...' "
            f"canonical='{parsed_claim.canonical_claim[:60]}...' "
            f"hard_claim={parsed_claim.is_hard_claim} "
            f"subclaims={len(parsed_claim.subclaims)}"
        )
        if parsed_claim.subclaims:
            logger.info(f"CLAIM_DECOMP: subclaims={parsed_claim.subclaims}")
        
        # Route based on relation_type
        # Try Wikidata first for political_office/party_membership/nationality, but fallback to Wikipedia if it fails
        wikidata_result = None
        if parsed_claim.relation_type in {"political_office", "party_membership", "nationality"}:
            # Try Wikidata fast path (non-blocking, degrade gracefully)
            try:
                wikidata_result = await self._evaluate_wikidata_fast_path(claim, parsed_claim, parsed_claim_dict)
                label_str = wikidata_result.get("label", "")
                
                # Only use Wikidata result if confident (SUPPORTED or REFUTED)
                # If NOT_ENOUGH_INFO or error, fallback to Wikipedia retrieval
                if label_str in {FactLabel.SUPPORTED.value, FactLabel.REFUTED.value}:
                    logger.info(
                        f"WIKIDATA_SUCCESS: label={label_str} for '{claim[:50]}...', "
                        f"using Wikidata result"
                    )
                    
                    # Generate answer for Wikidata result
                    answer = self._response_formatter.format_answer(
                        claim=claim,
                        label=label_str,
                        parsed_claim=parsed_claim_dict,
                        best_evidence=wikidata_result.get("best_evidence"),
                        population_correction=None,
                        geo_correction=None,
                    )
                    wikidata_result["answer"] = answer
                    
                    # Auto-populate cache for successful Wikidata results
                    if self.enable_claim_cache and self._claim_matcher:
                        self._claim_matcher.store_result(
                            claim=claim,
                            label=FactLabel(label_str),
                            evidence=wikidata_result.get("best_evidence") or {},
                        )
                    return wikidata_result
                else:
                    logger.info(
                        f"WIKIDATA_FALLBACK: label={label_str} for '{claim[:50]}...', "
                        f"falling back to Wikipedia retrieval"
                    )
            except Exception as e:
                logger.warning(f"WIKIDATA_ERROR: Error in Wikidata evaluation: {e}, falling back to Wikipedia")
                wikidata_result = None
        
        # Wikipedia retrieval path (always used as backbone, or as fallback from Wikidata)
        path = "WIKIPEDIA_RETRIEVAL"
        if wikidata_result:
            path = "WIKIPEDIA_RETRIEVAL_WIKIDATA_FALLBACK"
        
        # STEP 5: Handle hard claims with subclaims
        if parsed_claim.is_hard_claim and parsed_claim.subclaims:
                # Retrieve evidence for each subclaim
                all_evidences = []
                evidence_counts_per_subclaim = []
                
                for subclaim in parsed_claim.subclaims:
                    # Create a minimal parsed_claim for subclaim (reuse main parsed_claim structure)
                    subclaim_parsed = ParsedClaim(raw=subclaim)
                    # For comparison claims, don't reuse subject (each subclaim has its own entity)
                    # Only reuse subject for non-comparison claims
                    if parsed_claim.category != "comparison":
                        subclaim_parsed.subject = parsed_claim.subject
                    # Otherwise, leave subject empty (will be extracted from subclaim text if needed)
                    subclaim_parsed.year = parsed_claim.year
                    subclaim_parsed.category = parsed_claim.category
                    
                    # Retrieve evidence for subclaim
                    subclaim_evidences = await self._retriever.retrieve(
                        claim=subclaim,
                        parsed_claim=subclaim_parsed,
                        max_pages=max_pages,
                    )
                    
                    evidence_counts_per_subclaim.append(len(subclaim_evidences))
                    all_evidences.extend(subclaim_evidences)
                    
                    logger.info(
                        f"SUBCLAIM_RETRIEVAL: subclaim='{subclaim[:50]}...' "
                        f"retrieved={len(subclaim_evidences)} evidences"
                    )
                
                # Aggregate and deduplicate evidences
                evidences = self._aggregate_and_deduplicate_evidences(all_evidences, max_total=80)
                
                logger.info(
                    f"CLAIM_DECOMP_AGGREGATE: subclaims={len(parsed_claim.subclaims)} "
                    f"evidence_counts={evidence_counts_per_subclaim} "
                    f"total_before_dedup={len(all_evidences)} "
                    f"total_after_dedup={len(evidences)}"
                )
        else:
            # Normal retrieval for non-hard claims
            evidences = await self._retriever.retrieve(
                claim=claim,
                parsed_claim=parsed_claim,
                max_pages=max_pages,
            )
        
        # Subject consistency filter: ensure evidences mention the same person/subject
        # BUT: Only apply if subject is a specific entity (not generic nouns like "Jahr", "Tag")
        # Also skip for definition-like claims (they need all evidences from main article)
        is_definition_like = self._is_definition_like_claim(
            parsed_claim.canonical_claim, parsed_claim.subject
        )
        
        if parsed_claim.subject and not is_definition_like:
            # Check if subject is generic (skip filtering for generic subjects)
            from app.pipeline.postprocessors.subject_voting import SubjectVotingPostProcessor
            subject_voting = SubjectVotingPostProcessor()
            is_generic = subject_voting._is_generic_subject(parsed_claim.subject)
            
            if not is_generic:
                filtered_evidences = []
                subject_lower = parsed_claim.subject.lower()
                subject_tokens = set(subject_lower.split())
                surname = subject_tokens.pop() if subject_tokens else ""
                
                for ev in evidences:
                    title_lower = ev.title.lower()
                    snippet_lower = ev.snippet.lower()
                    
                    # Check if subject matches (exact match, surname match, or token overlap)
                    if (subject_lower in title_lower or surname in title_lower or
                        subject_lower in snippet_lower or surname in snippet_lower):
                        filtered_evidences.append(ev)
                    else:
                        logger.debug(f"SUBJECT_FILTER: Dropping '{ev.title}' for subject '{parsed_claim.subject}'")
                
                if filtered_evidences:
                    evidences = filtered_evidences
                    logger.info(f"SUBJECT_FILTER: {len(evidences)} evidences after filtering")
                else:
                    # CRITICAL: If filter removes everything, keep original evidences (don't delete all)
                    logger.warning(
                        f"SUBJECT_FILTER: No evidences match subject '{parsed_claim.subject}', "
                        f"keeping all {len(evidences)} original evidences to avoid empty result"
                    )
            else:
                logger.debug(f"SUBJECT_FILTER: Skipping filter for generic subject '{parsed_claim.subject}'")
        elif is_definition_like:
            logger.debug(f"SUBJECT_FILTER: Skipping filter for definition-like claim")
        
        # Pre-filter to top-k evidences before NLI (cheap heuristic)
        evidences_topk = self._select_topk_evidences(
            claim=claim,
            evidences=evidences,
            parsed_claim=parsed_claim,
            k=NLI_TOP_K,
        )
        
        # Log evidence counts
        logger.info(
            f"NLI_INPUT: retrieved={len(evidences)} used={len(evidences_topk)}"
        )
        
        # Special handling for comparison claims
        # Check both category and relation_type to be safe
        is_comparison = (
            parsed_claim.category == "comparison" or 
            parsed_claim.relation_type == "comparison"
        )
        if is_comparison and parsed_claim.subclaims:
            # Use comparison verifier for comparison claims
            logger.info(
                f"COMPARISON_VERIFY: Using ComparisonVerifier for comparison claim "
                f"(category={parsed_claim.category}, relation_type={parsed_claim.relation_type})"
            )
            # First, try synchronous verification
            result = self._comparison_verifier.verify(
                claim=claim,
                evidences=evidences,  # Use all evidences, not just topk
                subclaims=parsed_claim.subclaims,
            )
            
            # If values are missing, try Wikidata fallback
            comparison_details = result.get("comparison_details", {})
            x_value = comparison_details.get("x_value")
            y_value = comparison_details.get("y_value")
            
            if (x_value is None or y_value is None) and self._wikidata_client:
                logger.info(
                    f"COMPARISON_WIKIDATA_FALLBACK: Missing values (x={x_value}, y={y_value}), "
                    f"trying Wikidata fallback"
                )
                try:
                    # Try to get missing values from Wikidata
                    result = await self._comparison_verify_with_wikidata_fallback(
                        claim=claim,
                        evidences=evidences,
                        subclaims=parsed_claim.subclaims,
                        current_result=result,
                    )
                except Exception as e:
                    logger.error(
                        f"COMPARISON_WIKIDATA_FALLBACK: Error in Wikidata fallback: {e}",
                        exc_info=True
                    )
                    # Continue with original result if Wikidata fails
        else:
            # Standard NLI verification for non-comparison claims
            # Verify claim against evidence (run in executor to avoid blocking event loop)
            # For numerical claims, pass ALL evidences to verifier so post-processors can search through all of them
            is_numerical_claim = parsed_claim.category == "numerical" or any(
                kw in claim.lower() for kw in ["einwohner", "bevölkerung", "population"]
            )
            is_population_claim = any(kw in claim.lower() for kw in ["einwohner", "bevölkerung", "population"])
            
            # For population claims: try Wikidata fast-path first
            metadata = {}
            if is_population_claim and parsed_claim.subject and self._wikidata_client:
                wikidata_population = await self._try_wikidata_population_fast_path(
                    parsed_claim.subject, parsed_claim.year
                )
                if wikidata_population:
                    metadata["wikidata_population"] = wikidata_population
                    logger.info(
                        f"POPULATION_WIKIDATA: Found population data for '{parsed_claim.subject}': "
                        f"{wikidata_population['value_int']:,} (year={wikidata_population.get('point_in_time')})"
                    )
            
            # For geo location claims: try Wikidata P17 (country) fast-path
            # Also include "Hauptstadt" claims (e.g., "Berlin ist die Hauptstadt von Frankreich")
            claim_lower = claim.lower()
            is_geo_claim = any(kw in claim_lower for kw in ["liegt in", "befindet sich in", "gehört zu", "ist in"])
            is_hauptstadt_claim = "hauptstadt" in claim_lower and "ist" in claim_lower
            if (is_geo_claim or is_hauptstadt_claim) and parsed_claim.subject and self._wikidata_client:
                wikidata_geo = await self._try_wikidata_geo_fast_path(parsed_claim.subject)
                if wikidata_geo:
                    metadata["wikidata_geo"] = wikidata_geo
                    logger.info(
                        f"GEO_WIKIDATA: Found geo data for '{parsed_claim.subject}': "
                        f"country={wikidata_geo.get('country_label')} ({wikidata_geo.get('country_qid')})"
                    )
            
            # For population claims with subclaims: only use population-specific evidences
            # Don't aggregate subclaim[0] ("Portugal") into population evaluation
            if is_population_claim and parsed_claim.is_hard_claim and parsed_claim.subclaims:
                # Filter evidences to only those from population subclaim or main article
                population_evidences = self._filter_population_evidences(
                    evidences, parsed_claim.subject, claim
                )
                evidences_for_verifier = population_evidences if population_evidences else evidences
                logger.info(
                    f"POPULATION_EVIDENCE_FILTER: Filtered to {len(evidences_for_verifier)} population-specific evidences "
                    f"(from {len(evidences)} total)"
                )
            else:
                evidences_for_verifier = evidences if is_numerical_claim else evidences_topk
            
            if is_numerical_claim:
                logger.info(
                    f"NUMERICAL_CLAIM_VERIFY: Passing all {len(evidences_for_verifier)} evidences to verifier "
                    f"(instead of top-{len(evidences_topk)}) for numerical claim processing"
                )
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._verifier.verify,
                claim,
                evidences_for_verifier,  # Use all evidences for numerical claims, topk for others
                parsed_claim,  # Pass parsed_claim for temporal reasoning
                metadata if metadata else None,  # Pass Wikidata data through metadata
            )
        
        # Convert to JSON-serializable format
        best_evidence_dict = None
        if result.get("best_evidence") is not None:
            ev = result["best_evidence"]
            # Handle both Evidence objects and dicts
            if isinstance(ev, dict):
                best_evidence_dict = ev
            else:
                best_evidence_dict = {
                    "title": ev.title,
                    "url": ev.url,
                    "snippet": ev.snippet,
                    "source": ev.source,
                }
        
        # Ensure scores are floats (JSON-serializable)
        scores = []
        if "scores" in result and result["scores"]:
            for score_dict in result["scores"]:
                scores.append({
                    "entailment": float(score_dict.get("entailment", 0.0)),
                    "contradiction": float(score_dict.get("contradiction", 0.0)),
                    "neutral": float(score_dict.get("neutral", 0.0)),
                })
        
        # Get label value (handle both FactLabel enum and string)
        label_value = result["label"]
        if isinstance(label_value, FactLabel):
            label_value = label_value.value
        
        # Extract comparison details if available
        comparison_details = result.get("comparison_details")
        
        # Generate natural language answer (population/geo claims may provide structured correction)
        population_correction = result.get("population_correction")
        geo_correction = result.get("geo_correction")
        answer = self._response_formatter.format_answer(
            claim=claim,
            label=label_value,
            parsed_claim=parsed_claim_dict,
            best_evidence=best_evidence_dict,
            comparison_details=comparison_details,
            population_correction=population_correction,
            geo_correction=geo_correction,
        )
        
        response = {
            "claim": claim,
            "path": path,
            "relation_type": parsed_claim.relation_type,
            "parsed_claim": parsed_claim_dict,
            "label": label_value,
            "answer": answer,  # Natural language answer
            "best_evidence": best_evidence_dict,
            "scores": scores,
            "evidence_count": len(evidences),
        }

        # Attach structured population correction (for UI)
        if population_correction:
            response["correct_value"] = population_correction.get("correct_value")
            response["correct_value_year"] = population_correction.get("correct_value_year")
            response["correct_value_source"] = population_correction.get("correct_value_source")
            response["correct_value_str"] = population_correction.get("correct_value_str")
        
        # Add comparison details if available
        if comparison_details:
            response["comparison_details"] = comparison_details
        
        # Add related cache info if available
        if related_info:
            response["cache_related"] = related_info
        
        # Auto-populate cache if result is confident (SUPPORTED or REFUTED)
        if self.enable_claim_cache and self._claim_matcher:
            label_enum_or_str = result["label"]
            # Handle both FactLabel enum and string
            if isinstance(label_enum_or_str, str):
                try:
                    label_enum = FactLabel(label_enum_or_str)
                except ValueError:
                    label_enum = None
            else:
                label_enum = label_enum_or_str
            
            if label_enum and label_enum in {FactLabel.SUPPORTED, FactLabel.REFUTED}:
                # Don't cache comparison claims (they need full verification each time)
                is_comparison_result = (
                    parsed_claim.category == "comparison" or
                    parsed_claim.relation_type == "comparison"
                )
                if not is_comparison_result:
                    # Store in cache for future lookups
                    self._claim_matcher.store_result(
                        claim=claim,
                        label=label_enum,
                        evidence=best_evidence_dict or {},
                    )
                    logger.debug(f"CLAIM_CACHE: Stored result for claim '{claim[:50]}...' with label {label_enum.value}")
                else:
                    logger.debug(f"CLAIM_CACHE: Skipped caching comparison claim '{claim[:50]}...'")
        
        return response
    
    async def _evaluate_wikidata_fast_path(
        self, claim: str, parsed_claim: ParsedClaim, parsed_claim_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate claim using Wikidata fast path for political_office and party_membership.
        
        Returns:
            Response dict with label, reason, wikidata diagnostics, and best_evidence.
        """
        path = "WIKIDATA_FAST_PATH"
        
        # Extract person name
        person_name = parsed_claim.subject
        if not person_name:
            # Fallback: try to extract from claim
            person_name = self._claim_parser._extract_person_name(claim) or claim.split()[0] if claim.split() else ""
        
        if not person_name:
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_NO_PERSON_NAME",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {},
            }
        
        # Resolve person entity with conservative matching
        person_result = await self._resolve_entity_conservative(person_name, is_person=True)
        if not person_result:
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_ENTITY_NO_CONFIDENT_MATCH",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {"person_qid": None, "person_label": None},
            }
        
        person_qid, person_label = person_result
        
        # Get person entity data
        person_entity = await self._wikidata_client.get_entity(person_qid, language="de")
        if not person_entity:
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_ENTITY_FETCH_FAILED",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {"person_qid": person_qid, "person_label": person_label},
            }
        
        # Route to specific evaluation based on relation_type
        if parsed_claim.relation_type == "political_office":
            return await self._evaluate_political_office(
                claim, parsed_claim, parsed_claim_dict, person_qid, person_label, person_entity
            )
        elif parsed_claim.relation_type == "party_membership":
            return await self._evaluate_party_membership(
                claim, parsed_claim, parsed_claim_dict, person_qid, person_label, person_entity
            )
        elif parsed_claim.relation_type == "nationality":
            return await self._evaluate_nationality(
                claim, parsed_claim, parsed_claim_dict, person_qid, person_label, person_entity
            )
        else:
            # Should not happen, but handle gracefully
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_UNSUPPORTED_RELATION_TYPE",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {"person_qid": person_qid, "person_label": person_label},
            }
    
    async def _comparison_verify_with_wikidata_fallback(
        self,
        claim: str,
        evidences: List[Evidence],
        subclaims: List[str],
        current_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fallback to Wikidata for missing numeric values in comparison claims.
        
        Args:
            claim: Original claim text
            evidences: List of Evidence objects
            subclaims: List of subclaims
            current_result: Current result from ComparisonVerifier.verify()
            
        Returns:
            Updated result dict with Wikidata values if found
        """
        comparison_details = current_result.get("comparison_details", {})
        x_entity = comparison_details.get("x_entity")
        y_entity = comparison_details.get("y_entity")
        attribute = comparison_details.get("attribute")
        x_value = comparison_details.get("x_value")
        y_value = comparison_details.get("y_value")
        
        logger.debug(
            f"COMPARISON_WIKIDATA_FALLBACK_DETAILS: x_entity={x_entity}, y_entity={y_entity}, "
            f"attribute={attribute}, x_value={x_value}, y_value={y_value}"
        )
        
        if not x_entity or not y_entity or not attribute:
            logger.warning(
                f"COMPARISON_WIKIDATA_FALLBACK: Missing required fields in comparison_details: "
                f"x_entity={x_entity}, y_entity={y_entity}, attribute={attribute}"
            )
            return current_result
        
        # Try to get missing values from Wikidata
        if x_value is None:
            x_value_wikidata = await self._get_wikidata_numeric_value(x_entity, attribute, evidences)
            if x_value_wikidata is not None:
                x_value = x_value_wikidata
                logger.info(
                    f"COMPARISON_WIKIDATA_FALLBACK: Found x_value={x_value} for entity='{x_entity}' "
                    f"attribute='{attribute}' from Wikidata"
                )
        
        if y_value is None:
            y_value_wikidata = await self._get_wikidata_numeric_value(y_entity, attribute, evidences)
            if y_value_wikidata is not None:
                y_value = y_value_wikidata
                logger.info(
                    f"COMPARISON_WIKIDATA_FALLBACK: Found y_value={y_value} for entity='{y_entity}' "
                    f"attribute='{attribute}' from Wikidata"
                )
        
        # If we now have both values, re-run comparison
        if x_value is not None and y_value is not None:
            comparison_word = comparison_details.get("comparison_word", "")
            
            # Compare values
            comparison_result = self._compare_comparison_values(x_value, y_value, comparison_word)
            
            # Update result
            if comparison_result is True:
                label = FactLabel.SUPPORTED
            elif comparison_result is False:
                label = FactLabel.REFUTED
            else:
                label = FactLabel.NOT_ENOUGH_INFO
            
            # Update best_evidence if we have one from evidences
            best_evidence = current_result.get("best_evidence")
            if not best_evidence:
                # Try to find evidence with entity mentioned
                for ev in evidences:
                    if x_entity.lower() in ev.title.lower() or y_entity.lower() in ev.title.lower():
                        best_evidence = ev
                        break
            
            comparison_details.update({
                "x_value": x_value,
                "y_value": y_value,
                "result": comparison_result,
            })
            
            logger.info(
                f"COMPARISON_WIKIDATA_VERIFY: x_value={x_value} y_value={y_value} "
                f"comparison_word='{comparison_word}' -> {label.value}"
            )
            
            return {
                "label": label,
                "best_evidence": best_evidence,
                "scores": [],
                "comparison_details": comparison_details,
            }
        
        return current_result
    
    def _compare_comparison_values(
        self, x_value: float, y_value: float, comparison_word: str
    ) -> Optional[bool]:
        """
        Compare two numeric values based on comparison word (helper for Wikidata fallback).
        
        Args:
            x_value: Value for X entity
            y_value: Value for Y entity
            comparison_word: Comparison word (e.g., "länger", "größer")
            
        Returns:
            True if comparison holds, False if not, None if uncertain
        """
        comparison_word_lower = comparison_word.lower()
        
        if comparison_word_lower in ["länger", "größer", "mehr", "höher"]:
            return x_value > y_value
        elif comparison_word_lower in ["kürzer", "kleiner", "weniger", "niedriger"]:
            return x_value < y_value
        else:
            return None
    
    async def _get_wikidata_numeric_value(
        self, entity_name: str, attribute: str, evidences: List[Evidence]
    ) -> Optional[float]:
        """
        Get numeric value for an entity from Wikidata.
        
        Args:
            entity_name: Entity name (e.g., "weser", "wolga")
            attribute: Attribute being compared (e.g., "länge", "größe")
            evidences: List of Evidence objects (to find Wikipedia article title)
            
        Returns:
            Numeric value if found, None otherwise
        """
        # Map attribute to Wikidata property
        attribute_to_property = {
            "länge": "P2043",  # Length
            "größe": "P2046",  # Area
            "anzahl": "P1082",  # Population
            "höhe": "P2044",  # Elevation above sea level
            "alter": "P569",  # Date of birth (for age calculation)
            "jahr": None,  # Year depends on context
            "tore": None,  # Goals - depends on context
            "regierungsjahre": None,  # Years in office - depends on context
            "titel": None,  # Titles - depends on context
        }
        
        property_id = attribute_to_property.get(attribute.lower())
        if not property_id:
            return None
        
        # Try to find Wikipedia article title from evidences
        entity_lower = entity_name.lower()
        wikipedia_title = None
        exact_matches = []
        partial_matches = []
        
        for ev in evidences:
            title_lower = ev.title.lower()
            
            # CRITICAL: Filter out disambiguation/list pages
            if any(kw in title_lower for kw in ["begriffsklärung", "liste", "geschichte", "krieg", "chronik"]):
                continue
            
            # CRITICAL: Filter out compound titles that contain entity name but aren't the main article
            # E.g., "Main-Donau-Kanal" should not match "Donau"
            # Only accept if entity name is the first word or exact match
            title_words = title_lower.split()
            if len(title_words) > 0:
                # Remove determiners and compare
                title_clean = " ".join([w for w in title_words if w not in ["der", "die", "das"]])
                title_clean_lower = title_clean.lower()
                
                # Extract first word (before any hyphens or parentheses)
                first_word = title_clean_lower.split()[0].split("-")[0].split("(")[0].strip()
                
                # CRITICAL: Filter out compound titles with hyphens (e.g., "Afrika-Cup" should not match "Afrika")
                # Only accept if title is exactly the entity name, or starts with entity name + space/parenthesis
                has_hyphen = "-" in title_clean_lower
                
                # Exact match (title equals entity name exactly)
                if title_clean_lower == entity_lower:
                    exact_matches.append(ev.title)
                elif title_clean_lower.startswith(entity_lower + " "):
                    # Title starts with entity name followed by space (e.g., "Afrika (Kontinent)")
                    exact_matches.append(ev.title)
                elif title_clean_lower.startswith(entity_lower + "("):
                    # Title starts with entity name followed by parenthesis (e.g., "Afrika(Kontinent)")
                    exact_matches.append(ev.title)
                elif title_clean_lower == entity_lower + "s":
                    # Genitive form (e.g., "Afrikas" for "Afrika")
                    exact_matches.append(ev.title)
                elif first_word == entity_lower and not has_hyphen:
                    # First word matches entity name exactly, but only if no hyphen (avoids "Afrika-Cup")
                    exact_matches.append(ev.title)
                # Partial match (entity name is contained in title, but not at start)
                # CRITICAL: Exclude compound titles with hyphens (e.g., "Afrika-Cup", "Main-Donau-Kanal")
                elif entity_lower in title_clean_lower and not has_hyphen:
                    # Avoid false positives: "afrika" in "südafrika", "donau" in "main-donau-kanal"
                    # Only accept if entity name is a word boundary match AND it's the first word
                    if re.search(r'\b' + re.escape(entity_lower) + r'\b', title_clean_lower):
                        # Additional check: entity should be at the start of the title
                        if title_clean_lower.startswith(entity_lower):
                            partial_matches.append(ev.title)
        
        # Prefer exact matches over partial matches
        if exact_matches:
            wikipedia_title = exact_matches[0]
        elif partial_matches:
            wikipedia_title = partial_matches[0]
        
        if not wikipedia_title:
            # Try direct search (try both lowercase and capitalized)
            logger.debug(
                f"COMPARISON_WIKIDATA_SEARCH: No Wikipedia title found, searching directly for '{entity_name}'"
            )
            # Try capitalized version first (more likely to match)
            search_name = entity_name.capitalize() if entity_name else entity_name
            search_results = await self._wikidata_client.search_entity(
                name=search_name, language="de", limit=10  # Get more results for disambiguation
            )
            # If no results, try original (lowercase)
            if not search_results and entity_name != search_name:
                logger.debug(
                    f"COMPARISON_WIKIDATA_SEARCH: No results for '{search_name}', trying '{entity_name}'"
                )
                search_results = await self._wikidata_client.search_entity(
                    name=entity_name, language="de", limit=10
                )
            
            if not search_results:
                logger.warning(
                    f"COMPARISON_WIKIDATA_SEARCH: No search results for entity='{entity_name}' "
                    f"(tried '{search_name}' and '{entity_name}')"
                )
                return None
            
            # Disambiguate: For geographical comparisons, prefer continents over countries, rivers over other entities
            # Check if attribute is "größe" (area) or "länge" (length) - likely geographic comparison
            is_geographic_comparison = attribute.lower() in ["größe", "länge", "höhe"]
            is_length_comparison = attribute.lower() == "länge"
            
            if is_geographic_comparison and len(search_results) > 1:
                # Score candidates based on description and instance type
                scored_candidates = []
                for result in search_results:
                    qid_candidate = result.get("id")
                    label = result.get("label", "")
                    description = result.get("description", "").lower()
                    
                    score = 0.0
                    
                    # Check if it has the required property (if not, penalize heavily)
                    try:
                        entity_data = await self._wikidata_client.get_entity(qid_candidate, language="de")
                        if entity_data:
                            claims = entity_data.get("claims", {})
                            has_property = property_id in claims
                            if not has_property:
                                score -= 50.0  # Heavy penalty if property is missing
                                logger.debug(
                                    f"COMPARISON_WIKIDATA_DISAMBIG: QID={qid_candidate} ({label}) missing property {property_id}"
                                )
                                scored_candidates.append({
                                    "result": result,
                                    "qid": qid_candidate,
                                    "score": score,
                                })
                                continue
                            
                            instance_of = claims.get("P31", [])
                            instance_ids = [
                                stmt.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                                for stmt in instance_of
                                if stmt.get("mainsnak", {}).get("snaktype") == "value"
                            ]
                            
                            # For length comparisons, prefer rivers (Q4022)
                            if is_length_comparison:
                                if "Q4022" in instance_ids:  # River
                                    score += 30.0  # Strong preference for rivers
                                    logger.debug(
                                        f"COMPARISON_WIKIDATA_DISAMBIG: QID={qid_candidate} ({label}) is a river (Q4022)"
                                    )
                                else:
                                    score -= 10.0  # Penalize non-rivers for length comparisons
                            
                            # Q5107 = continent (for area comparisons)
                            if "Q5107" in instance_ids:
                                score += 20.0  # Strong preference for continents
                                logger.debug(
                                    f"COMPARISON_WIKIDATA_DISAMBIG: QID={qid_candidate} ({label}) is a continent (Q5107)"
                                )
                            # Q6256 = country (penalize for geographic comparisons)
                            elif "Q6256" in instance_ids:
                                score -= 5.0
                                logger.debug(
                                    f"COMPARISON_WIKIDATA_DISAMBIG: QID={qid_candidate} ({label}) is a country (Q6256)"
                                )
                            
                            # Prefer continents for area comparisons
                            if attribute.lower() == "größe":
                                continent_keywords = ["kontinent", "continent", "erdteil"]
                                if any(kw in description for kw in continent_keywords):
                                    score += 10.0
                    except Exception as e:
                        logger.debug(f"COMPARISON_WIKIDATA_DISAMBIG: Error checking entity type for {qid_candidate}: {e}")
                    
                    scored_candidates.append({
                        "result": result,
                        "qid": qid_candidate,
                        "score": score,
                    })
                
                # Sort by score descending and pick best
                scored_candidates.sort(key=lambda x: x["score"], reverse=True)
                if scored_candidates and scored_candidates[0]["score"] > -50:  # Only use if score is reasonable
                    best_candidate = scored_candidates[0]
                    qid = best_candidate["qid"]
                    best_result = best_candidate["result"]
                    
                    logger.info(
                        f"COMPARISON_WIKIDATA_DISAMBIG: Selected QID={qid} for entity='{entity_name}' "
                        f"(label={best_result.get('label', 'N/A')}, score={best_candidate['score']:.1f}, "
                        f"total_candidates={len(scored_candidates)})"
                    )
                else:
                    # No good candidate found, use first result anyway
                    best_result = search_results[0]
                    qid = best_result.get("id")
                    logger.warning(
                        f"COMPARISON_WIKIDATA_DISAMBIG: No good candidate found for '{entity_name}', "
                        f"using first result QID={qid} ({best_result.get('label', 'N/A')})"
                    )
            else:
                # No disambiguation needed, use first result
                best_result = search_results[0]
                qid = best_result.get("id")
                logger.debug(
                    f"COMPARISON_WIKIDATA_SEARCH: Found QID={qid} for entity='{entity_name}' "
                    f"(label={best_result.get('label', 'N/A')}, searched as '{search_name}')"
                )
        else:
            # Search for entity by Wikipedia article title
            logger.debug(
                f"COMPARISON_WIKIDATA_SEARCH: Searching for Wikipedia title '{wikipedia_title}' "
                f"(entity='{entity_name}')"
            )
            search_results = await self._wikidata_client.search_entity(
                name=wikipedia_title, language="de", limit=10  # Get more results for disambiguation
            )
            if not search_results:
                logger.warning(
                    f"COMPARISON_WIKIDATA_SEARCH: No search results for Wikipedia title='{wikipedia_title}'"
                )
                return None
            
            # Apply disambiguation logic here too (for geographic comparisons)
            is_geographic_comparison = attribute.lower() in ["größe", "länge", "höhe"]
            is_length_comparison = attribute.lower() == "länge"
            
            if is_geographic_comparison and len(search_results) > 1:
                # Score candidates based on description and instance type
                scored_candidates = []
                for result in search_results:
                    qid_candidate = result.get("id")
                    label = result.get("label", "")
                    description = result.get("description", "").lower()
                    
                    score = 0.0
                    
                    # Check if it has the required property (if not, penalize heavily)
                    try:
                        entity_data = await self._wikidata_client.get_entity(qid_candidate, language="de")
                        if entity_data:
                            claims = entity_data.get("claims", {})
                            has_property = property_id in claims
                            if not has_property:
                                score -= 50.0  # Heavy penalty if property is missing
                                logger.debug(
                                    f"COMPARISON_WIKIDATA_DISAMBIG: QID={qid_candidate} ({label}) missing property {property_id}"
                                )
                                scored_candidates.append({
                                    "result": result,
                                    "qid": qid_candidate,
                                    "score": score,
                                })
                                continue
                            
                            instance_of = claims.get("P31", [])
                            instance_ids = [
                                stmt.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                                for stmt in instance_of
                                if stmt.get("mainsnak", {}).get("snaktype") == "value"
                            ]
                            
                            # For length comparisons, prefer rivers (Q4022)
                            if is_length_comparison:
                                if "Q4022" in instance_ids:  # River
                                    score += 30.0  # Strong preference for rivers
                                    logger.debug(
                                        f"COMPARISON_WIKIDATA_DISAMBIG: QID={qid_candidate} ({label}) is a river (Q4022)"
                                    )
                                else:
                                    score -= 10.0  # Penalize non-rivers for length comparisons
                            
                            # Q5107 = continent (for area comparisons)
                            if "Q5107" in instance_ids:
                                score += 20.0  # Strong preference for continents
                                logger.debug(
                                    f"COMPARISON_WIKIDATA_DISAMBIG: QID={qid_candidate} ({label}) is a continent (Q5107)"
                                )
                            # Q6256 = country (penalize for geographic comparisons)
                            elif "Q6256" in instance_ids:
                                score -= 5.0
                                logger.debug(
                                    f"COMPARISON_WIKIDATA_DISAMBIG: QID={qid_candidate} ({label}) is a country (Q6256)"
                                )
                            
                            # Prefer continents for area comparisons
                            if attribute.lower() == "größe":
                                continent_keywords = ["kontinent", "continent", "erdteil"]
                                if any(kw in description for kw in continent_keywords):
                                    score += 10.0
                    except Exception as e:
                        logger.debug(f"COMPARISON_WIKIDATA_DISAMBIG: Error checking entity type for {qid_candidate}: {e}")
                    
                    scored_candidates.append({
                        "result": result,
                        "qid": qid_candidate,
                        "score": score,
                    })
                
                # Sort by score descending and pick best
                scored_candidates.sort(key=lambda x: x["score"], reverse=True)
                if scored_candidates and scored_candidates[0]["score"] > -50:  # Only use if score is reasonable
                    best_candidate = scored_candidates[0]
                    qid = best_candidate["qid"]
                    result = best_candidate["result"]
                    
                    logger.info(
                        f"COMPARISON_WIKIDATA_DISAMBIG: Selected QID={qid} for Wikipedia title='{wikipedia_title}' "
                        f"(label={result.get('label', 'N/A')}, score={best_candidate['score']:.1f}, "
                        f"total_candidates={len(scored_candidates)})"
                    )
                else:
                    # No good candidate found, use first result anyway
                    result = search_results[0]
                    qid = result.get("id")
                    logger.warning(
                        f"COMPARISON_WIKIDATA_DISAMBIG: No good candidate found for Wikipedia title='{wikipedia_title}', "
                        f"using first result QID={qid} ({result.get('label', 'N/A')})"
                    )
            else:
                # No disambiguation needed, use first result
                qid = None
                for result in search_results:
                    qid = result.get("id")
                    logger.debug(
                        f"COMPARISON_WIKIDATA_SEARCH: Found QID={qid} for Wikipedia title='{wikipedia_title}' "
                        f"(label={result.get('label', 'N/A')})"
                    )
                    break
        
        if not qid:
            return None
        
        # Get entity data
        logger.debug(
            f"COMPARISON_WIKIDATA_ENTITY: Fetching entity data for QID={qid} "
            f"(entity='{entity_name}', property={property_id})"
        )
        entity_data = await self._wikidata_client.get_entity(qid, language="de")
        if not entity_data:
            logger.warning(
                f"COMPARISON_WIKIDATA_ENTITY: Failed to fetch entity data for QID={qid}"
            )
            return None
        
        # Extract property value
        claims = entity_data.get("claims", {})
        property_claims = claims.get(property_id, [])
        
        if not property_claims:
            logger.warning(
                f"COMPARISON_WIKIDATA_PROPERTY: Property {property_id} not found for QID={qid} "
                f"(entity='{entity_name}', attribute='{attribute}'). "
                f"Available properties: {list(claims.keys())[:10]}"
            )
            return None
        
        logger.debug(
            f"COMPARISON_WIKIDATA_PROPERTY: Found {len(property_claims)} statements for "
            f"property={property_id} (entity='{entity_name}')"
        )
        
        # Get first value (prefer preferred rank if available)
        for stmt_idx, stmt in enumerate(property_claims):
            mainsnak = stmt.get("mainsnak", {})
            snaktype = mainsnak.get("snaktype")
            
            if snaktype != "value":
                logger.debug(
                    f"COMPARISON_WIKIDATA_STMT: Statement {stmt_idx} has snaktype='{snaktype}', skipping"
                )
                continue
            
            datavalue = mainsnak.get("datavalue", {})
            value_obj = datavalue.get("value", {})
            
            logger.debug(
                f"COMPARISON_WIKIDATA_STMT: Processing statement {stmt_idx}, "
                f"value_obj keys: {list(value_obj.keys())}"
            )
            
            # Handle quantity values (for length, area, etc.)
            if "amount" in value_obj:
                amount_str = value_obj["amount"]
                # Remove + sign if present
                amount_str = amount_str.lstrip("+")
                try:
                    value = float(amount_str)
                    
                    # Get unit (Wikidata uses URIs like "http://www.wikidata.org/entity/Q11573" for meters)
                    unit = value_obj.get("unit", "")
                    
                    # Handle unit conversion if needed
                    # For length: P2043 can be in meters (Q11573) or kilometers (Q828224)
                    if attribute.lower() == "länge":
                        # Q11573 = meter, Q828224 = kilometer
                        if "Q11573" in unit:
                            # Unit is meter, convert to kilometers
                            original_value = value
                            value = value / 1000.0
                            logger.debug(
                                f"COMPARISON_WIKIDATA: Converted {original_value} m to {value} km "
                                f"for entity='{entity_name}' (unit=Q11573/meter)"
                            )
                        elif "Q828224" in unit:
                            # Unit is already kilometer, use as-is
                            logger.debug(
                                f"COMPARISON_WIKIDATA: Value already in km for entity='{entity_name}' "
                                f"(value={value}, unit=Q828224/kilometer)"
                            )
                        elif unit == "1" or not unit:
                            # No unit specified - heuristic: if value > 1000, assume km; else assume m
                            if value < 1000:
                                # Likely meters, convert to km
                                original_value = value
                                value = value / 1000.0
                                logger.debug(
                                    f"COMPARISON_WIKIDATA: No unit specified, converted {original_value} m to {value} km "
                                    f"(heuristic: value < 1000) for entity='{entity_name}'"
                                )
                            else:
                                # Likely already in km
                                logger.debug(
                                    f"COMPARISON_WIKIDATA: No unit specified, assuming km "
                                    f"(heuristic: value >= 1000) for entity='{entity_name}'"
                                )
                    
                    logger.info(
                        f"COMPARISON_WIKIDATA: entity='{entity_name}' attribute='{attribute}' "
                        f"property={property_id} value={value} (unit={unit[:50] if unit else 'none'})"
                    )
                    return value
                except (ValueError, KeyError) as e:
                    logger.warning(
                        f"COMPARISON_WIKIDATA: Error parsing amount '{amount_str}' for entity='{entity_name}': {e}"
                    )
                    continue
            
            # Handle time values (for birth dates, etc.)
            elif "time" in value_obj:
                time_str = value_obj["time"]
                logger.debug(
                    f"COMPARISON_WIKIDATA_TIME: Found time value '{time_str}' for entity='{entity_name}'"
                )
                # Wikidata time format: +1950-01-01T00:00:00Z
                try:
                    year = int(time_str[1:5])  # Skip + and get year
                    if attribute.lower() == "alter":
                        # Calculate age (current year - birth year)
                        from datetime import datetime
                        current_year = datetime.now().year
                        age = current_year - year
                        logger.info(
                            f"COMPARISON_WIKIDATA: entity='{entity_name}' attribute='{attribute}' "
                            f"calculated age={age} from birth year={year}"
                        )
                        return float(age)
                    elif attribute.lower() == "jahr":
                        logger.info(
                            f"COMPARISON_WIKIDATA: entity='{entity_name}' attribute='{attribute}' "
                            f"year={year}"
                        )
                        return float(year)
                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"COMPARISON_WIKIDATA: Error parsing time '{time_str}': {e}"
                    )
                    continue
            else:
                logger.debug(
                    f"COMPARISON_WIKIDATA_STMT: Statement {stmt_idx} has value_obj keys "
                    f"{list(value_obj.keys())}, but no 'amount' or 'time' field"
                )
        
        logger.warning(
            f"COMPARISON_WIKIDATA: No valid value found for entity='{entity_name}' "
            f"attribute='{attribute}' property={property_id} after processing {len(property_claims)} statements"
        )
        return None
    
    async def _resolve_entity_conservative(
        self, name: str, is_person: bool = False
    ) -> Optional[Tuple[str, str]]:
        """
        Resolve entity with conservative matching.
        
        Args:
            name: Entity name to search for
            is_person: If True, prefer entities that are humans (instance of Q5)
            
        Returns:
            Tuple of (qid, label) if confident match found, None otherwise
        """
        if not name:
            return None
        
        search_results = await self._wikidata_client.search_entity(name=name, language="de", limit=5)
        if not search_results:
            return None
        
        # Get labels for all candidates
        candidate_qids = [hit.get("id") for hit in search_results if hit.get("id")]
        if not candidate_qids:
            return None
        
        labels = await self._wikidata_client.get_labels(candidate_qids, lang="de")
        
        # Normalize search name for comparison
        name_normalized = name.strip().lower()
        name_parts = name_normalized.split()
        
        # Score candidates
        best_candidate = None
        best_score = -1
        
        for hit in search_results:
            qid = hit.get("id")
            if not qid:
                continue
            
            label = labels.get(qid, hit.get("label", "")).strip()
            label_normalized = label.lower()
            
            # Check for exact match (case-insensitive)
            if label_normalized == name_normalized:
                score = 100
            # Check if label contains all name parts
            elif all(part in label_normalized for part in name_parts if len(part) > 2):
                score = 80
            # Check if first token matches
            elif name_parts and name_parts[0] in label_normalized:
                score = 50
            else:
                score = 20
            
            # For persons, check if entity is a human (instance of Q5)
            if is_person:
                entity = await self._wikidata_client.get_entity(qid, language="de")
                if entity:
                    claims = entity.get("claims", {})
                    instance_of = claims.get("P31", [])
                    is_human = any(
                        stmt.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == "Q5"
                        for stmt in instance_of
                    )
                    if not is_human:
                        # Not a human, reduce score significantly
                        score = max(0, score - 50)
            
            if score > best_score:
                best_score = score
                best_candidate = (qid, label)
        
        # Only return if we have a confident match (score >= 50)
        if best_score >= 50 and best_candidate:
            return best_candidate
        
        return None
    
    async def _get_person_country_hint(self, person_qid: str) -> Optional[Dict[str, str]]:
        """
        Get person's country of citizenship (P27) as a hint for party disambiguation.
        
        Args:
            person_qid: Wikidata QID of the person
            
        Returns:
            Dict with keys: country_qid, country_label if found, None otherwise
        """
        if not person_qid:
            return None
        
        try:
            # Fetch person entity
            person_entity = await self._wikidata_client.get_entity(person_qid, language="de")
            if not person_entity:
                return None
            
            # Extract P27 (country of citizenship) claims
            claims = person_entity.get("claims", {})
            p27_statements = claims.get("P27", [])
            
            if not p27_statements:
                return None
            
            # Get first P27 value QID (if multiple, pick the first)
            country_qid = None
            for stmt in p27_statements:
                mainsnak = stmt.get("mainsnak", {})
                if mainsnak.get("snaktype") == "value":
                    country_qid = mainsnak.get("datavalue", {}).get("value", {}).get("id")
                    if country_qid:
                        break
            
            if not country_qid:
                return None
            
            # Resolve country label
            labels = await self._wikidata_client.get_labels([country_qid], lang="de")
            country_label = labels.get(country_qid, "")
            
            if not country_label:
                return None
            
            logger.info(
                f"PERSON_COUNTRY: person={person_qid} country={country_label} ({country_qid})"
            )
            
            return {
                "country_qid": country_qid,
                "country_label": country_label,
            }
        except Exception as e:
            logger.warning(f"PERSON_COUNTRY: Error fetching country for {person_qid}: {e}")
            return None
    
    def _is_likely_party_entity(self, entity: Dict[str, Any]) -> bool:
        """
        Check if a search result entity is likely a political party.
        
        Args:
            entity: Search result dict from wbsearchentities with fields like:
                   id, label, description, match, concepturi
                   
        Returns:
            True if entity appears to be a party, False if it's clearly not (e.g., parliamentary group)
        """
        label = (entity.get("label", "") or "").lower()
        description = (entity.get("description", "") or "").lower()
        text = f"{label} {description}"
        
        # Reject if it's clearly a parliamentary group/faction
        non_party_indicators = [
            "bundestagsfraktion",
            "fraktion",
            "faction",
            "parliamentary group",
        ]
        if any(indicator in text for indicator in non_party_indicators):
            return False
        
        # Accept if it contains party indicators
        party_indicators = [
            "partei",
            "political party",
            "party",
        ]
        if any(indicator in text for indicator in party_indicators):
            return True
        
        # If no description, default to True (will be filtered by later checks)
        if not description:
            return True
        
        # Default: not clearly a party, but don't reject yet
        return True
    
    async def _resolve_party_conservative(
        self, party_name: str, parsed_claim: ParsedClaim, country_hint: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve party entity with conservative confidence gating.
        Avoids wrong entities like "CDU (DDR)" and supports non-German contexts.
        
        Args:
            party_name: Party name to search for
            parsed_claim: Parsed claim (for context like subject, object)
            country_hint: Optional country label hint (from person's citizenship)
            
        Returns:
            Dict with keys: id, label, description if confident match found, None otherwise
        """
        if not party_name:
            return None
        
        # Fetch candidates using primary query
        candidates = await self._wikidata_client.search_entity(
            name=party_name, language="de", limit=10
        )
        
        # Additionally, if parsed_claim.subject exists, try context query
        if parsed_claim.subject:
            context_query = f"{party_name} Partei"
            context_candidates = await self._wikidata_client.search_entity(
                name=context_query, language="de", limit=10
            )
            
            # Merge candidates (dedupe by id)
            seen_ids = {c.get("id") for c in candidates if c.get("id")}
            for candidate in context_candidates:
                candidate_id = candidate.get("id")
                if candidate_id and candidate_id not in seen_ids:
                    candidates.append(candidate)
                    seen_ids.add(candidate_id)
        
        # Optional: if parsed_claim.object exists (country hint from claim), try additional query
        if parsed_claim.object:
            country_hint_from_claim = parsed_claim.object
            country_query = f"{party_name} {country_hint_from_claim}"
            country_candidates = await self._wikidata_client.search_entity(
                name=country_query, language="de", limit=10
            )
            
            # Merge candidates (dedupe by id)
            seen_ids = {c.get("id") for c in candidates if c.get("id")}
            for candidate in country_candidates:
                candidate_id = candidate.get("id")
                if candidate_id and candidate_id not in seen_ids:
                    candidates.append(candidate)
                    seen_ids.add(candidate_id)
        
        # If country_hint is provided (from person's citizenship), try additional query
        if country_hint:
            country_query = f"{party_name} {country_hint}"
            country_candidates = await self._wikidata_client.search_entity(
                name=country_query, language="de", limit=10
            )
            
            # Merge candidates (dedupe by id)
            seen_ids = {c.get("id") for c in candidates if c.get("id")}
            for candidate in country_candidates:
                candidate_id = candidate.get("id")
                if candidate_id and candidate_id not in seen_ids:
                    candidates.append(candidate)
                    seen_ids.add(candidate_id)
        
        if not candidates:
            logger.info(f"PARTY_RESOLVE: ambiguous for {party_name} -> NEI (no candidates)")
            return None
        
        # Filter candidates using _is_likely_party_entity
        filtered_candidates = [
            c for c in candidates if self._is_likely_party_entity(c)
        ]
        
        if not filtered_candidates:
            logger.info(f"PARTY_RESOLVE: ambiguous for {party_name} -> NEI (all filtered out)")
            return None
        
        # Score candidates
        scored_candidates = []
        party_name_lower = party_name.lower()
        
        for candidate in filtered_candidates:
            label = (candidate.get("label", "") or "").strip()
            description = (candidate.get("description", "") or "").strip()
            label_lower = label.lower()
            description_lower = description.lower()
            match_text = (candidate.get("match", {}).get("text", "") or "").lower()
            
            # Base score from match text (if available)
            score = 0.0
            if match_text and match_text == party_name_lower:
                score += 5.0  # Exact match bonus
            
            # Strong negative penalty for DDR/East Germany references
            if "(ddr)" in label_lower or "(ddr)" in description_lower:
                score -= 5.0
            if "ost-deutschland" in description_lower or "east germany" in description_lower:
                score -= 3.0
            
            # Label equals party_name (case-insensitive)
            if label_lower == party_name_lower:
                score += 3.0
            
            # "Partei" in description
            if "partei" in description_lower:
                score += 2.0
            
            # Small bonus for label containing party name
            if party_name_lower in label_lower:
                score += 1.0
            
            scored_candidates.append({
                "candidate": candidate,
                "score": score,
            })
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Second-stage refinement: if country_hint is provided and we have >=2 candidates with close scores
        if country_hint and len(scored_candidates) >= 2:
            # Check if top candidates have close scores (within 2.0 points)
            top_score = scored_candidates[0]["score"]
            second_score = scored_candidates[1]["score"] if len(scored_candidates) > 1 else -999
            score_diff = top_score - second_score
            
            # If scores are close, apply country-based refinement to top 3 candidates
            if score_diff < 2.0:
                country_hint_lower = country_hint.lower()
                
                # Refine top 3 candidates
                for i, scored_entry in enumerate(scored_candidates[:3]):
                    candidate = scored_entry["candidate"]
                    candidate_qid = candidate.get("id")
                    
                    if not candidate_qid:
                        continue
                    
                    try:
                        # Fetch party entity
                        party_entity = await self._wikidata_client.get_entity(candidate_qid, language="de")
                        if not party_entity:
                            continue
                        
                        # Extract country properties in order: P17, P495, P1001
                        claims = party_entity.get("claims", {})
                        party_country_qid = None
                        
                        # Try P17 (country) first
                        p17_statements = claims.get("P17", [])
                        for stmt in p17_statements:
                            mainsnak = stmt.get("mainsnak", {})
                            if mainsnak.get("snaktype") == "value":
                                party_country_qid = mainsnak.get("datavalue", {}).get("value", {}).get("id")
                                if party_country_qid:
                                    break
                        
                        # Try P495 (country of origin) if P17 not found
                        if not party_country_qid:
                            p495_statements = claims.get("P495", [])
                            for stmt in p495_statements:
                                mainsnak = stmt.get("mainsnak", {})
                                if mainsnak.get("snaktype") == "value":
                                    party_country_qid = mainsnak.get("datavalue", {}).get("value", {}).get("id")
                                    if party_country_qid:
                                        break
                        
                        # Try P1001 (applies to jurisdiction) if still not found
                        if not party_country_qid:
                            p1001_statements = claims.get("P1001", [])
                            for stmt in p1001_statements:
                                mainsnak = stmt.get("mainsnak", {})
                                if mainsnak.get("snaktype") == "value":
                                    party_country_qid = mainsnak.get("datavalue", {}).get("value", {}).get("id")
                                    if party_country_qid:
                                        break
                        
                        # If we found a country QID, resolve its label and check match
                        if party_country_qid:
                            country_labels = await self._wikidata_client.get_labels([party_country_qid], lang="de")
                            party_country_label = country_labels.get(party_country_qid, "").lower()
                            
                            # Check if country matches hint (exact or contains)
                            if party_country_label == country_hint_lower or country_hint_lower in party_country_label:
                                # Add bonus
                                scored_entry["score"] += 5.0
                                logger.info(
                                    f"PARTY_COUNTRY: party_qid={candidate_qid} country={party_country_label} "
                                    f"({party_country_qid}) hint={country_hint} bonus=+5.0"
                                )
                            else:
                                logger.info(
                                    f"PARTY_COUNTRY: party_qid={candidate_qid} country={party_country_label} "
                                    f"({party_country_qid}) hint={country_hint} bonus=0.0 (no match)"
                                )
                    except Exception as e:
                        logger.warning(f"PARTY_COUNTRY: Error checking country for {candidate_qid}: {e}")
                        continue
                
                # Re-sort after applying country bonuses
                scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Confidence gate (re-run after country bonus)
        if not scored_candidates:
            logger.info(f"PARTY_RESOLVE: ambiguous for {party_name} -> NEI (no scored candidates)")
            return None
        
        top_score = scored_candidates[0]["score"]
        if top_score < 2.0:
            logger.info(
                f"PARTY_RESOLVE: ambiguous for {party_name} -> NEI (top_score={top_score:.2f} < 2.0)"
            )
            return None
        
        # Check if ambiguous (top_score - second_score < 1)
        if len(scored_candidates) > 1:
            second_score = scored_candidates[1]["score"]
            if top_score - second_score < 1.0:
                logger.info(
                    f"PARTY_RESOLVE: ambiguous for {party_name} -> NEI "
                    f"(top_score={top_score:.2f}, second_score={second_score:.2f}, diff={top_score - second_score:.2f} < 1.0)"
                )
                return None
        
        # Return best candidate
        best = scored_candidates[0]["candidate"]
        best_label = best.get("label", "").strip()
        best_id = best.get("id", "")
        
        logger.info(
            f"PARTY_RESOLVE: input={party_name} candidates={len(candidates)} "
            f"best={best_id} ({best_label}) score={top_score:.2f}"
        )
        
        return {
            "id": best_id,
            "label": best_label,
            "description": best.get("description", "").strip(),
        }
    
    async def _resolve_party_entity_conservative(
        self, search_name: str, original_party_name: str
    ) -> Optional[Tuple[str, str]]:
        """
        Resolve party entity with conservative matching, avoiding parliamentary groups/factions.
        
        Args:
            search_name: Normalized party name to search for
            original_party_name: Original party name (for QID lookup)
            
        Returns:
            Tuple of (qid, label) if confident match found, None otherwise
        """
        if not search_name:
            return None
        
        # First, try direct QID lookup if we have the party name in our mapping
        if original_party_name in PARTY_QIDS:
            party_qid = PARTY_QIDS[original_party_name]
            # Verify it's still a valid entity
            entity = await self._wikidata_client.get_entity(party_qid, language="de")
            if entity:
                labels = await self._wikidata_client.get_labels([party_qid], lang="de")
                party_label = labels.get(party_qid, "")
                # Verify it's a political party (instance of Q7278)
                claims = entity.get("claims", {})
                instance_of = claims.get("P31", [])
                is_party = any(
                    stmt.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == "Q7278"
                    for stmt in instance_of
                )
                if is_party:
                    logger.info(f"WIKIDATA_PARTY: Found via QID lookup: {party_qid} ({party_label})")
                    return (party_qid, party_label)
        
        # Fallback to search
        search_results = await self._wikidata_client.search_entity(name=search_name, language="de", limit=5)
        if not search_results:
            return None
        
        # Get labels for all candidates
        candidate_qids = [hit.get("id") for hit in search_results if hit.get("id")]
        if not candidate_qids:
            return None
        
        labels = await self._wikidata_client.get_labels(candidate_qids, lang="de")
        
        # Normalize search name for comparison
        name_normalized = search_name.strip().lower()
        name_parts = name_normalized.split()
        
        # Score candidates, preferring political parties over parliamentary groups
        best_candidate = None
        best_score = -1
        
        for hit in search_results:
            qid = hit.get("id")
            if not qid:
                continue
            
            label = labels.get(qid, hit.get("label", "")).strip()
            label_normalized = label.lower()
            
            # Fetch entity to check instance type
            entity = await self._wikidata_client.get_entity(qid, language="de")
            if not entity:
                continue
            
            claims = entity.get("claims", {})
            instance_of = claims.get("P31", [])
            instance_ids = [
                stmt.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                for stmt in instance_of
                if stmt.get("mainsnak", {}).get("snaktype") == "value"
            ]
            
            # Q7278 = political party
            # Q1126647 = parliamentary group (avoid these)
            # Q330557 = parliamentary faction (avoid these)
            is_party = "Q7278" in instance_ids
            is_parliamentary_group = "Q1126647" in instance_ids or "Q330557" in instance_ids
            
            if is_parliamentary_group and not is_party:
                # Skip parliamentary groups/factions that are not also parties
                logger.info(f"WIKIDATA_PARTY: Rejecting parliamentary group: {qid} ({label})")
                continue
            
            # Score based on label match
            if label_normalized == name_normalized:
                score = 100
            elif all(part in label_normalized for part in name_parts if len(part) > 2):
                score = 80
            elif name_parts and name_parts[0] in label_normalized:
                score = 50
            else:
                score = 20
            
            # Boost score if it's a political party
            if is_party:
                score += 30
            # Reduce score if it's only a parliamentary group
            elif is_parliamentary_group:
                score -= 40
            
            if score > best_score:
                best_score = score
                best_candidate = (qid, label)
        
        # Only return if we have a confident match (score >= 50) and it's a party
        if best_score >= 50 and best_candidate:
            # Double-check the best candidate is actually a party
            qid, label = best_candidate
            entity = await self._wikidata_client.get_entity(qid, language="de")
            if entity:
                claims = entity.get("claims", {})
                instance_of = claims.get("P31", [])
                instance_ids = [
                    stmt.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                    for stmt in instance_of
                    if stmt.get("mainsnak", {}).get("snaktype") == "value"
                ]
                is_party = "Q7278" in instance_ids
                is_parliamentary_group = "Q1126647" in instance_ids or "Q330557" in instance_ids
                
                if is_parliamentary_group and not is_party:
                    # Only parliamentary group, not a party - reject
                    logger.info(f"WIKIDATA_PARTY: Rejecting best match (only parliamentary group): {qid} ({label})")
                    return None
                
                logger.info(f"WIKIDATA_PARTY: Accepted match: {qid} ({label}) | is_party={is_party}")
                return best_candidate
        
        logger.info(f"WIKIDATA_PARTY: No confident match found for '{search_name}'")
        return None
    
    async def _evaluate_political_office(
        self,
        claim: str,
        parsed_claim: ParsedClaim,
        parsed_claim_dict: Dict[str, Any],
        person_qid: str,
        person_label: str,
        person_entity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate political office claim using P39 (position held).
        
        Currently supports: Präsident (France), Kanzler (Germany)
        """
        path = "WIKIDATA_FAST_PATH"
        
        # Extract country from parsed_claim.object or claim
        # Normalize genitive forms like "Frankreichs" -> "Frankreich"
        country_name = None
        if parsed_claim.object:
            # Normalize genitive: remove trailing 's' if present
            country_name = parsed_claim.object.rstrip("s")
        else:
            # Try to extract from claim (handle genitive forms like "Frankreichs")
            claim_lower = claim.lower()
            # Check for country mentions (handle genitive: "frankreichs" -> "frankreich")
            if "frankreich" in claim_lower or "france" in claim_lower:
                country_name = "Frankreich"
            elif "deutschland" in claim_lower or "germany" in claim_lower:
                country_name = "Deutschland"
            # Also check for patterns like "Präsident von Frankreich" or "Präsident Frankreichs"
            if not country_name:
                # Pattern: "von Frankreich" or "Frankreichs" (genitive)
                match = re.search(r"(?:von|der|des)\s+([A-ZÄÖÜ][a-zäöüß]+(?:s)?)", claim, re.IGNORECASE)
                if match:
                    potential_country = match.group(1).rstrip("s")  # Remove genitive 's'
                    potential_lower = potential_country.lower()
                    if "frankreich" in potential_lower:
                        country_name = "Frankreich"
                    elif "deutschland" in potential_lower:
                        country_name = "Deutschland"
        
        # Final normalization: ensure we have the base form
        if country_name:
            country_name = country_name.rstrip("s")  # Remove any remaining genitive 's'
        
        if not country_name:
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_NO_COUNTRY",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {"person_qid": person_qid, "person_label": person_label},
            }
        
        # Resolve country entity
        country_result = await self._resolve_entity_conservative(country_name, is_person=False)
        if not country_result:
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_COUNTRY_NO_MATCH",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {"person_qid": person_qid, "person_label": person_label},
            }
        
        country_qid, country_label = country_result
        
        # Determine role keywords from parsed_claim.role or claim text
        role_lower = (parsed_claim.role or "").lower()
        claim_lower = claim.lower()
        
        president_keywords = ["präsident", "president", "bundespräsident", "bundespräsidentin"]
        chancellor_keywords = ["kanzler", "chancellor", "bundeskanzler", "bundeskanzlerin"]
        
        # Determine which role type we're looking for
        is_president = any(kw in role_lower for kw in president_keywords) or any(kw in claim_lower for kw in president_keywords)
        is_chancellor = any(kw in role_lower for kw in chancellor_keywords) or any(kw in claim_lower for kw in chancellor_keywords)
        
        if not is_president and not is_chancellor:
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_OFFICE_NOT_SUPPORTED",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {
                    "person_qid": person_qid,
                    "person_label": person_label,
                    "country_qid": country_qid,
                    "country_label": country_label,
                },
            }
        
        # Determine desired role keywords for matching
        desired_role_keywords = president_keywords if is_president else chancellor_keywords
        
        # Determine country keywords (extendable structure)
        country_keywords = []
        if country_label:
            country_label_lower = country_label.lower()
            # France synonyms
            if country_label_lower in ["frankreich", "france"]:
                country_keywords = ["frankreich", "französ", "france"]
            # Germany synonyms
            elif country_label_lower in ["deutschland", "germany"]:
                country_keywords = ["deutschland", "bundesrepublik", "german", "germany"]
            # For other countries, use the country label itself
            else:
                country_keywords = [country_label_lower]
        
        # Check P39 (position held) statements
        claims = person_entity.get("claims", {})
        p39_statements = claims.get("P39", [])
        
        if not p39_statements:
            logger.info(f"WIKIDATA_OFFICE: person_qid={person_qid} | reason=NO_P39_STATEMENTS")
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_NO_P39_STATEMENTS",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {
                    "person_qid": person_qid,
                    "person_label": person_label,
                    "country_qid": country_qid,
                    "country_label": country_label,
                },
            }
        
        # Collect all P39 position QIDs and check if current
        position_data = []
        for stmt in p39_statements:
            mainsnak = stmt.get("mainsnak", {})
            if mainsnak.get("snaktype") != "value":
                continue
            
            position_id = mainsnak.get("datavalue", {}).get("value", {}).get("id")
            if not position_id:
                continue
            
            # Check qualifiers for end time (P582)
            qualifiers = stmt.get("qualifiers", {})
            p582 = qualifiers.get("P582", [])  # End time
            
            # Check if position is current (no end time or end time in future)
            is_current = True
            has_end_time = len(p582) > 0
            if p582:
                for end_time_stmt in p582:
                    if end_time_stmt.get("snaktype") == "value":
                        end_time = end_time_stmt.get("datavalue", {}).get("value", {}).get("time")
                        if end_time:
                            # Parse Wikidata time format: +2024-01-01T00:00:00Z
                            try:
                                year = int(end_time[1:5])  # Skip + and get year
                                if year < datetime.now().year:
                                    is_current = False
                            except (ValueError, IndexError):
                                pass
            
            position_data.append({
                "position_id": position_id,
                "is_current": is_current,
                "qualifiers": qualifiers,
            })
        
        if not position_data:
            logger.info(f"WIKIDATA_OFFICE: person_qid={person_qid} | reason=NO_VALID_P39")
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_NO_P39_STATEMENTS",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {
                    "person_qid": person_qid,
                    "person_label": person_label,
                    "country_qid": country_qid,
                    "country_label": country_label,
                },
            }
        
        # Resolve labels for all position QIDs
        position_ids = [p["position_id"] for p in position_data]
        position_labels = await self._wikidata_client.get_labels(position_ids, lang="de")
        
        # Log position labels for debugging
        position_label_list = [position_labels.get(pid, "") for pid in position_ids[:5]]  # Limit to first 5
        logger.info(
            "WIKIDATA_OFFICE_LABELS: person=%s labels=%s", person_qid, position_label_list
        )
        
        # Match positions by label keywords
        matching_statements = []
        
        for pos_data in position_data:
            position_id = pos_data["position_id"]
            position_label = position_labels.get(position_id, "")
            
            if not position_label:
                continue
            
            position_label_lower = position_label.lower()
            
            # Check role match: position label must contain one of desired_role_keywords
            role_match = any(kw in position_label_lower for kw in desired_role_keywords)
            
            if not role_match:
                continue
            
            # Check country match: if country_label is known, require country keyword match
            country_match = True  # Default to True
            if country_label and country_keywords:
                country_label_lower = country_label.lower()
                # Special case: "Bundespräsident" or "Bundeskanzler" implies Germany
                if "bundespräsident" in position_label_lower or "bundeskanzler" in position_label_lower:
                    # These positions are always German, so match if country is Germany
                    if country_label_lower in ["deutschland", "germany"]:
                        country_match = True
                    else:
                        country_match = False
                else:
                    # Position label must contain country_label or one of its synonyms
                    country_match = (
                        country_label_lower in position_label_lower or
                        any(kw in position_label_lower for kw in country_keywords)
                    )
            
            if country_match:
                matching_statements.append(pos_data)
                logger.info(
                    "WIKIDATA_OFFICE_MATCH: person=%s match=%s current=%s",
                    person_qid,
                    position_label,
                    pos_data["is_current"]
                )
        
        # Evaluate result
        if matching_statements:
            # Check if any matching statement is current
            current_matches = [m for m in matching_statements if m["is_current"]]
            if current_matches:
                # SUPPORTED: person currently holds the position
                position_id = current_matches[0]["position_id"]
                position_label = position_labels.get(position_id, "Amt")
                
                evidence_text = f"Wikidata: {person_label} ist aktuell {position_label} von {country_label}."
                
                logger.info(
                    "WIKIDATA_OFFICE_MATCH: person=%s match=%s current=True -> SUPPORTED",
                    person_qid,
                    position_label
                )
                
                return {
                    "claim": claim,
                    "path": path,
                    "relation_type": parsed_claim.relation_type,
                    "parsed_claim": parsed_claim_dict,
                    "label": FactLabel.SUPPORTED.value,
                    "reason": "WIKIDATA_OFFICE_MATCH",
                    "best_evidence": {
                        "title": person_label,
                        "url": f"https://www.wikidata.org/wiki/{person_qid}",
                        "snippet": evidence_text,
                        "source": "wikidata",
                    },
                    "scores": [],
                    "evidence_count": 1,
                    "wikidata": {
                        "person_qid": person_qid,
                        "person_label": person_label,
                        "country_qid": country_qid,
                        "country_label": country_label,
                        "position_id": position_id,
                        "position_label": position_label,
                    },
                }
            else:
                # Past position: Check if claim is also in past tense
                past_matches = [m for m in matching_statements if not m["is_current"]]
                if past_matches:
                    # If claim is in past tense and we found past position -> SUPPORTED
                    if parsed_claim.tense == "past":
                        position_id = past_matches[0]["position_id"]
                        position_label = position_labels.get(position_id, "Amt")
                        
                        evidence_text = f"Wikidata: {person_label} war {position_label} von {country_label}."
                        
                        logger.info(
                            "WIKIDATA_OFFICE_MATCH: person=%s match=%s past=True claim_tense=past -> SUPPORTED",
                            person_qid,
                            position_label
                        )
                        
                        return {
                            "claim": claim,
                            "path": path,
                            "relation_type": parsed_claim.relation_type,
                            "parsed_claim": parsed_claim_dict,
                            "label": FactLabel.SUPPORTED.value,
                            "reason": "WIKIDATA_OFFICE_PAST_MATCH",
                            "best_evidence": {
                                "title": person_label,
                                "url": f"https://www.wikidata.org/wiki/{person_qid}",
                                "snippet": evidence_text,
                                "source": "wikidata",
                            },
                            "scores": [],
                            "evidence_count": 1,
                            "wikidata": {
                                "person_qid": person_qid,
                                "person_label": person_label,
                                "country_qid": country_qid,
                                "country_label": country_label,
                                "position_id": position_id,
                                "position_label": position_label,
                            },
                        }
                    else:
                        # Present tense claim but only past position -> REFUTED
                        position_id = past_matches[0]["position_id"]
                        position_label = position_labels.get(position_id, "Amt")
                        
                        evidence_text = f"Wikidata: {person_label} war {position_label} von {country_label}, ist aber nicht mehr im Amt."
                        
                        logger.info(
                            "WIKIDATA_OFFICE_MATCH: person=%s match=%s past=True claim_tense=present -> REFUTED",
                            person_qid,
                            position_label
                        )
                        
                        return {
                            "claim": claim,
                            "path": path,
                            "relation_type": parsed_claim.relation_type,
                            "parsed_claim": parsed_claim_dict,
                            "label": FactLabel.REFUTED.value,
                            "reason": "WIKIDATA_OFFICE_PAST_ONLY_PRESENT_CLAIM",
                            "best_evidence": {
                                "title": person_label,
                                "url": f"https://www.wikidata.org/wiki/{person_qid}",
                                "snippet": evidence_text,
                                "source": "wikidata",
                            },
                            "scores": [],
                            "evidence_count": 1,
                            "wikidata": {
                                "person_qid": person_qid,
                                "person_label": person_label,
                                "country_qid": country_qid,
                                "country_label": country_label,
                                "position_id": position_id,
                                "position_label": position_label,
                            },
                        }
                else:
                    # No matches at all
                    logger.info(
                        "WIKIDATA_OFFICE_MATCH: person=%s reason=NO_MATCH found_positions=%d",
                        person_qid,
                        len(position_data)
                    )
                    return {
                        "claim": claim,
                        "path": path,
                        "relation_type": parsed_claim.relation_type,
                        "parsed_claim": parsed_claim_dict,
                        "label": FactLabel.NOT_ENOUGH_INFO.value,
                        "reason": "WIKIDATA_OFFICE_NO_MATCH",
                        "best_evidence": None,
                        "scores": [],
                        "evidence_count": 0,
                        "wikidata": {
                            "person_qid": person_qid,
                            "person_label": person_label,
                            "country_qid": country_qid,
                            "country_label": country_label,
                        },
                    }
        else:
            # No matching position found: NOT_ENOUGH_INFO (don't automatically REFUTE)
            logger.info(
                "WIKIDATA_OFFICE_MATCH: person=%s reason=NO_MATCH found_positions=%d",
                person_qid,
                len(position_data)
            )
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_OFFICE_NO_MATCH",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {
                    "person_qid": person_qid,
                    "person_label": person_label,
                    "country_qid": country_qid,
                    "country_label": country_label,
                },
            }
    
    async def _evaluate_party_membership(
        self,
        claim: str,
        parsed_claim: ParsedClaim,
        parsed_claim_dict: Dict[str, Any],
        person_qid: str,
        person_label: str,
        person_entity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate party membership claim using P102 (member of political party).
        """
        path = "WIKIDATA_FAST_PATH"
        
        # Get party name from parsed_claim.party or extract from claim
        party_name = parsed_claim.party
        if not party_name:
            # Try to extract from claim
            claim_lower = claim.lower()
            from app.pipeline.claim_parser import PARTY_KEYWORDS
            for key, value in PARTY_KEYWORDS.items():
                if key in claim_lower:
                    party_name = value
                    break
        
        if not party_name:
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_NO_PARTY_NAME",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {"person_qid": person_qid, "person_label": person_label},
            }
        
        # Get person's country hint for better party disambiguation
        country_hint_dict = await self._get_person_country_hint(person_qid)
        country_hint_label = None
        person_country_qid = None
        person_country_label = None
        
        if country_hint_dict:
            country_hint_label = country_hint_dict["country_label"]
            person_country_qid = country_hint_dict["country_qid"]
            person_country_label = country_hint_dict["country_label"]
        
        # Resolve party entity with conservative matching (context-aware, avoids wrong entities)
        party_result = await self._resolve_party_conservative(
            party_name, parsed_claim, country_hint=country_hint_label
        )
        if not party_result:
            # Build wikidata dict with country info if available
            wikidata_dict = {"person_qid": person_qid, "person_label": person_label}
            if person_country_qid:
                wikidata_dict["person_country_qid"] = person_country_qid
                wikidata_dict["person_country_label"] = person_country_label
            
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_PARTY_AMBIGUOUS",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": wikidata_dict,
            }
        
        party_qid = party_result["id"]
        party_label = party_result["label"]
        
        # Check P102 (member of political party) statements
        claims = person_entity.get("claims", {})
        p102_statements = claims.get("P102", [])
        
        if not p102_statements:
            # Build wikidata dict with country info if available
            wikidata_dict = {
                "person_qid": person_qid,
                "person_label": person_label,
                "party_qid": party_qid,
                "party_label": party_label,
            }
            if person_country_qid:
                wikidata_dict["person_country_qid"] = person_country_qid
                wikidata_dict["person_country_label"] = person_country_label
            
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_NO_P102_STATEMENTS",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": wikidata_dict,
            }
        
        # Check if any P102 statement matches the party QID
        has_match = False
        for stmt in p102_statements:
            mainsnak = stmt.get("mainsnak", {})
            if mainsnak.get("snaktype") != "value":
                continue
            
            member_party_id = mainsnak.get("datavalue", {}).get("value", {}).get("id")
            if member_party_id == party_qid:
                has_match = True
                break
        
        if has_match:
            # SUPPORTED: person is a member of the claimed party
            evidence_text = f"Wikidata: {person_label} ist Mitglied der {party_label}."
            
            # Build wikidata dict with country info if available
            wikidata_dict = {
                "person_qid": person_qid,
                "person_label": person_label,
                "party_qid": party_qid,
                "party_label": party_label,
            }
            if person_country_qid:
                wikidata_dict["person_country_qid"] = person_country_qid
                wikidata_dict["person_country_label"] = person_country_label
            
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.SUPPORTED.value,
                "reason": "WIKIDATA_PARTY_MATCH",
                "best_evidence": {
                    "title": person_label,
                    "url": f"https://www.wikidata.org/wiki/{person_qid}",
                    "snippet": evidence_text,
                    "source": "wikidata",
                },
                "scores": [],
                "evidence_count": 1,
                "wikidata": wikidata_dict,
            }
        else:
            # REFUTED: person has party memberships but not the claimed one
            # Get all party QIDs for diagnostics
            member_party_ids = []
            for stmt in p102_statements:
                mainsnak = stmt.get("mainsnak", {})
                if mainsnak.get("snaktype") == "value":
                    member_party_id = mainsnak.get("datavalue", {}).get("value", {}).get("id")
                    if member_party_id:
                        member_party_ids.append(member_party_id)
            
            member_party_labels = {}
            if member_party_ids:
                member_party_labels = await self._wikidata_client.get_labels(member_party_ids, lang="de")
            
            evidence_text = f"Wikidata: {person_label} ist nicht Mitglied der {party_label}."
            
            # Build wikidata dict with country info if available
            wikidata_dict = {
                "person_qid": person_qid,
                "person_label": person_label,
                "party_qid": party_qid,
                "party_label": party_label,
                "member_party_ids": member_party_ids,
                "member_party_labels": member_party_labels,
            }
            if person_country_qid:
                wikidata_dict["person_country_qid"] = person_country_qid
                wikidata_dict["person_country_label"] = person_country_label
            
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.REFUTED.value,
                "reason": "WIKIDATA_PARTY_NO_MATCH",
                "best_evidence": {
                    "title": person_label,
                    "url": f"https://www.wikidata.org/wiki/{person_qid}",
                    "snippet": evidence_text,
                    "source": "wikidata",
                },
                "scores": [],
                "evidence_count": 1,
                "wikidata": wikidata_dict,
            }
    
    async def _evaluate_nationality(
        self,
        claim: str,
        parsed_claim: ParsedClaim,
        parsed_claim_dict: Dict[str, Any],
        person_qid: str,
        person_label: str,
        person_entity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate nationality claim using P27 (country of citizenship).
        
        For claims like "Messi ist Argentinier", checks if person has Argentina (Q414) as P27.
        """
        path = "WIKIDATA_FAST_PATH"
        
        # Extract claimed nationality/demonym from claim
        # Pattern: "ist/war" + demonym (e.g., "Argentinier", "Deutscher")
        claim_lower = claim.lower()
        object_claim = parsed_claim.object or ""
        
        # Map demonyms to country QIDs (common cases)
        demonym_to_country = {
            "argentinier": "Q414",  # Argentina
            "argentinierin": "Q414",
            "argentinisch": "Q414",
            "deutscher": "Q183",  # Germany
            "deutsche": "Q183",
            "deutsch": "Q183",
            "franzose": "Q142",  # France
            "französin": "Q142",
            "französisch": "Q142",
            "spanier": "Q29",  # Spain
            "spanierin": "Q29",
            "spanisch": "Q29",
            "brasilianer": "Q155",  # Brazil
            "brasilianerin": "Q155",
            "brasilianisch": "Q155",
            "portugiese": "Q45",  # Portugal
            "portugiesin": "Q45",
            "portugiesisch": "Q45",
            "italiener": "Q38",  # Italy
            "italienerin": "Q38",
            "italienisch": "Q38",
            "engländer": "Q21",  # England (part of UK)
            "engländerin": "Q21",
            "englisch": "Q21",
            "amerikaner": "Q30",  # United States
            "amerikanerin": "Q30",
            "amerikanisch": "Q30",
        }
        
        # Try to extract demonym from object or claim
        claimed_demonym = None
        if object_claim:
            claimed_demonym = object_claim.lower().strip()
        else:
            # Try to extract from claim pattern
            nationality_match = re.search(
                r'\bist\s+(?:ein|eine|der|die|das)?\s*([a-zäöüß]+(?:er|in|isch))\b',
                claim_lower
            )
            if nationality_match:
                claimed_demonym = nationality_match.group(1).lower()
        
        if not claimed_demonym:
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_NO_DEMONYM",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {"person_qid": person_qid, "person_label": person_label},
            }
        
        # Resolve country QID from demonym
        claimed_country_qid = demonym_to_country.get(claimed_demonym)
        if not claimed_country_qid:
            # Try to resolve country by searching for demonym
            # For now, return NOT_ENOUGH_INFO if demonym not in mapping
            # (Could be extended to search Wikidata for demonym -> country mapping)
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_DEMONYM_NOT_MAPPED",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {
                    "person_qid": person_qid,
                    "person_label": person_label,
                    "claimed_demonym": claimed_demonym,
                },
            }
        
        # Get country label for evidence
        country_labels = await self._wikidata_client.get_labels([claimed_country_qid], lang="de")
        claimed_country_label = country_labels.get(claimed_country_qid, "")
        
        # Check P27 (country of citizenship) statements
        claims = person_entity.get("claims", {})
        p27_statements = claims.get("P27", [])
        
        if not p27_statements:
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.NOT_ENOUGH_INFO.value,
                "reason": "WIKIDATA_NO_P27_STATEMENTS",
                "best_evidence": None,
                "scores": [],
                "evidence_count": 0,
                "wikidata": {
                    "person_qid": person_qid,
                    "person_label": person_label,
                    "claimed_country_qid": claimed_country_qid,
                    "claimed_country_label": claimed_country_label,
                },
            }
        
        # Check if any P27 statement matches the claimed country QID
        has_match = False
        all_country_qids = []
        for stmt in p27_statements:
            mainsnak = stmt.get("mainsnak", {})
            if mainsnak.get("snaktype") != "value":
                continue
            
            country_qid = mainsnak.get("datavalue", {}).get("value", {}).get("id")
            if country_qid:
                all_country_qids.append(country_qid)
                if country_qid == claimed_country_qid:
                    has_match = True
        
        # Get labels for all countries (for correction if REFUTED)
        all_country_labels = {}
        if all_country_qids:
            all_country_labels = await self._wikidata_client.get_labels(all_country_qids, lang="de")
        
        if has_match:
            # SUPPORTED: person has the claimed country as citizenship
            evidence_text = f"Wikidata: {person_label} ist {claimed_demonym.capitalize()} (Staatsbürgerschaft: {claimed_country_label})."
            
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.SUPPORTED.value,
                "reason": "WIKIDATA_NATIONALITY_MATCH",
                "best_evidence": {
                    "title": person_label,
                    "url": f"https://www.wikidata.org/wiki/{person_qid}",
                    "snippet": evidence_text,
                    "source": "wikidata",
                },
                "scores": [],
                "evidence_count": 1,
                "wikidata": {
                    "person_qid": person_qid,
                    "person_label": person_label,
                    "claimed_country_qid": claimed_country_qid,
                    "claimed_country_label": claimed_country_label,
                    "all_country_qids": all_country_qids,
                    "all_country_labels": all_country_labels,
                },
            }
        else:
            # REFUTED: person has citizenship but not the claimed one
            # Use first country as correction
            correct_country_label = ""
            if all_country_qids and all_country_labels:
                correct_country_qid = all_country_qids[0]
                correct_country_label = all_country_labels.get(correct_country_qid, "")
            
            if correct_country_label:
                evidence_text = f"Wikidata: {person_label} ist nicht {claimed_demonym.capitalize()}, sondern hat die Staatsbürgerschaft {correct_country_label}."
            else:
                evidence_text = f"Wikidata: {person_label} ist nicht {claimed_demonym.capitalize()}."
            
            return {
                "claim": claim,
                "path": path,
                "relation_type": parsed_claim.relation_type,
                "parsed_claim": parsed_claim_dict,
                "label": FactLabel.REFUTED.value,
                "reason": "WIKIDATA_NATIONALITY_NO_MATCH",
                "best_evidence": {
                    "title": person_label,
                    "url": f"https://www.wikidata.org/wiki/{person_qid}",
                    "snippet": evidence_text,
                    "source": "wikidata",
                },
                "scores": [],
                "evidence_count": 1,
                "wikidata": {
                    "person_qid": person_qid,
                    "person_label": person_label,
                    "claimed_country_qid": claimed_country_qid,
                    "claimed_country_label": claimed_country_label,
                    "all_country_qids": all_country_qids,
                    "all_country_labels": all_country_labels,
                    "correct_country_label": correct_country_label,
                },
            }
    
    def _select_topk_evidences(
        self,
        claim: str,
        evidences: List[Evidence],
        parsed_claim: ParsedClaim,
        k: int = 3,
    ) -> List[Evidence]:
        """
        Pre-filter evidences using cheap token overlap heuristic before NLI.
        
        Args:
            claim: The claim text
            evidences: List of Evidence objects to filter
            parsed_claim: Parsed claim (for subject extraction)
            k: Number of top evidences to keep
            
        Returns:
            List of top-k Evidence objects sorted by relevance score
        """
        if len(evidences) <= k:
            return evidences
        
        # Tokenize claim: split on non-letters, lowercase, filter short tokens
        claim_tokens = set(
            token.lower()
            for token in re.findall(r"[a-zA-ZäöüÄÖÜß]+", claim)
            if len(token) >= 3
        )
        
        # Extract subject tokens if available
        subject_tokens = set()
        if parsed_claim.subject:
            subject_tokens = set(
                token.lower()
                for token in re.findall(r"[a-zA-ZäöüÄÖÜß]+", parsed_claim.subject)
                if len(token) >= 3
            )
        
        # Score each evidence
        scored_evidences = []
        for evidence in evidences:
            # Tokenize evidence snippet
            snippet_tokens = set(
                token.lower()
                for token in re.findall(r"[a-zA-ZäöüÄÖÜß]+", evidence.snippet)
                if len(token) >= 3
            )
            
            # Tokenize evidence title
            title_tokens = set(
                token.lower()
                for token in re.findall(r"[a-zA-ZäöüÄÖÜß]+", evidence.title)
                if len(token) >= 3
            )
            
            # Calculate token overlap score
            snippet_overlap = len(claim_tokens & snippet_tokens)
            title_overlap = len(claim_tokens & title_tokens)
            
            # Base score: token overlap with snippet
            score = snippet_overlap
            
            # Bonus: if title contains subject tokens
            if subject_tokens and title_tokens:
                subject_in_title = len(subject_tokens & title_tokens)
                if subject_in_title > 0:
                    score += 2.0  # Small bonus for subject match in title
            
            # Small bonus for title overlap
            if title_overlap > 0:
                score += 1.0
            
            scored_evidences.append((score, evidence))
        
        # Sort by score descending and take top k
        scored_evidences.sort(key=lambda x: x[0], reverse=True)
        topk_evidences = [evidence for _, evidence in scored_evidences[:k]]
        
        return topk_evidences
    
    def _aggregate_and_deduplicate_evidences(
        self,
        evidences: List[Evidence],
        max_total: int = 80,
    ) -> List[Evidence]:
        """
        Aggregate and deduplicate evidences from multiple subclaims.
        
        Deduplication by:
        - Same URL + sentence_index (exact duplicate)
        - Same snippet text (content duplicate)
        
        Args:
            evidences: List of Evidence objects from multiple subclaims
            max_total: Maximum total evidences to return (default: 80)
            
        Returns:
            Deduplicated list of Evidence objects (max max_total)
        """
        if not evidences:
            return []
        
        seen = set()
        deduplicated = []
        
        for ev in evidences:
            # Create deduplication key: (url, sentence_index) or snippet text
            if ev.sentence_index is not None:
                key = (ev.url, ev.sentence_index)
            else:
                # Fallback: use snippet text (first 100 chars)
                key = ("snippet", ev.snippet[:100])
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(ev)
                
                # Early exit if we've reached max_total
                if len(deduplicated) >= max_total:
                    break
        
        logger.info(
            f"EVIDENCE_DEDUP: input={len(evidences)} "
            f"output={len(deduplicated)} "
            f"removed={len(evidences) - len(deduplicated)}"
        )
        
        return deduplicated
    
    async def check_with_timing(
        self,
        claim: str,
        max_pages: int = 3,
        enable_quality_proxies: bool = True,
    ) -> RunResult:
        """
        Check a claim with full timing instrumentation and return RunResult.
        
        This method is designed for evaluation runs and provides detailed timing
        and quality proxy information.
        
        Args:
            claim: The claim text to verify
            max_pages: Maximum number of evidence pages to retrieve
            enable_quality_proxies: Whether to compute quality proxies (default: True)
            
        Returns:
            RunResult object with timing, evidence, and quality proxy information
        """
        total_start = time.perf_counter()
        timings_ms = {
            "parse_ms": 0.0,
            "claim_match_ms": 0.0,
            "retrieve_ms": 0.0,
            "stage1_rerank_ms": 0.0,
            "stage2_crossencoder_ms": 0.0,
            "verify_ms": 0.0,
            "total_ms": 0.0,
        }
        
        debug_info: Dict[str, Any] = {}
        route_taken = RouteType.FULL_PIPELINE
        
        try:
            # Ensure session is started
            if self._session is None or self._session.closed:
                await self.start()
            
            # ===== STEP 1: Claim Cache Lookup =====
            claim_match_start = time.perf_counter()
            related_info = None
            cache_match_info = None
            
            if self.enable_claim_cache and self._claim_matcher:
                match_result = self._claim_matcher.match(claim)
                match_type = match_result["match_type"]
                match_score = match_result["match_score"]
                cached_result = match_result["cached_result"]
                
                timings_ms["claim_match_ms"] = (time.perf_counter() - claim_match_start) * 1000
                
                cache_match_info = {
                    "match_type": match_type.value,
                    "match_score": float(match_score),
                    "canonical_claim": match_result.get("canonical_claim", ""),
                }
                debug_info["cache_match"] = cache_match_info
                
                # Auto-return if strong match
                if match_type in {MatchType.EXACT, MatchType.RELATED} and match_score >= ClaimMatcher.THRESH_STRONG:
                    if cached_result:
                        route_taken = RouteType.CACHE_HIT
                        total_ms = (time.perf_counter() - total_start) * 1000
                        timings_ms["total_ms"] = total_ms
                        
                        return RunResult(
                            claim_raw=claim,
                            claim_canonical=match_result.get("canonical_claim", ""),
                            route_taken=route_taken,
                            label_final=cached_result["label"],
                            confidence_final=float(match_score),
                            best_evidence=cached_result.get("evidence"),
                            evidence_top_k=[],
                            debug=debug_info,
                            timings_ms=timings_ms,
                        )
                
                # For related matches below strong threshold
                if match_type == MatchType.RELATED and match_score >= ClaimMatcher.THRESH_RELATED:
                    related_info = {
                        "match_type": match_type.value,
                        "match_score": float(match_score),
                        "cached_label": cached_result["label"] if cached_result else None,
                    }
                    route_taken = RouteType.RELATED_ONLY
            else:
                timings_ms["claim_match_ms"] = (time.perf_counter() - claim_match_start) * 1000
            
            # ===== STEP 2: Parse claim =====
            parse_start = time.perf_counter()
            parsed_claim = self._claim_parser.parse_claim(claim)
            timings_ms["parse_ms"] = (time.perf_counter() - parse_start) * 1000
            
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
            
            # Check if decomposition was used
            if parsed_claim.is_hard_claim and parsed_claim.subclaims:
                route_taken = RouteType.FULL_PIPELINE_WITH_DECOMP
            
            # Route based on relation_type
            # Try Wikidata first for political_office/party_membership, but fallback to Wikipedia if it fails
            wikidata_result = None
            if parsed_claim.relation_type in {"political_office", "party_membership"}:
                try:
                    wikidata_result = await self._evaluate_wikidata_fast_path(claim, parsed_claim, parsed_claim_dict)
                    label_str = wikidata_result.get("label", "")
                    
                    # Only use Wikidata result if confident (SUPPORTED or REFUTED)
                    if label_str in {FactLabel.SUPPORTED.value, FactLabel.REFUTED.value}:
                        route_taken = RouteType.WIKIDATA_FAST_PATH
                        
                        # Auto-populate cache
                        if self.enable_claim_cache and self._claim_matcher:
                            self._claim_matcher.store_result(
                                claim=claim,
                                label=FactLabel(label_str),
                                evidence=wikidata_result.get("best_evidence") or {},
                            )
                        
                        # Convert to RunResult
                        total_ms = (time.perf_counter() - total_start) * 1000
                        timings_ms["total_ms"] = total_ms
                        
                        return RunResult(
                            claim_raw=claim,
                            claim_canonical=parsed_claim.canonical_claim,
                            route_taken=route_taken,
                            label_final=label_str,
                            confidence_final=1.0,  # Wikidata fast path is deterministic
                            best_evidence=wikidata_result.get("best_evidence"),
                            evidence_top_k=[],
                            debug={
                                **debug_info,
                                "wikidata": wikidata_result.get("wikidata", {}),
                                "reason": wikidata_result.get("reason", ""),
                            },
                            timings_ms=timings_ms,
                        )
                    else:
                        # Wikidata returned NOT_ENOUGH_INFO, fallback to Wikipedia
                        logger.info(
                            f"WIKIDATA_FALLBACK: label={label_str} for '{claim[:50]}...', "
                            f"falling back to Wikipedia retrieval"
                        )
                except Exception as e:
                    logger.warning(f"WIKIDATA_ERROR: Error in Wikidata evaluation: {e}, falling back to Wikipedia")
                    wikidata_result = None
            
            # Wikipedia retrieval path (backbone or fallback from Wikidata)
            
            # ===== STEP 3: Wikipedia retrieval =====
            retrieve_start = time.perf_counter()
            path = "WIKIPEDIA_RETRIEVAL"
            
            # Handle hard claims with subclaims
            if parsed_claim.is_hard_claim and parsed_claim.subclaims:
                all_evidences = []
                evidence_counts_per_subclaim = []
                
                for subclaim in parsed_claim.subclaims:
                    subclaim_parsed = ParsedClaim(raw=subclaim)
                    # For comparison claims, don't reuse subject (each subclaim has its own entity)
                    if parsed_claim.category != "comparison":
                        subclaim_parsed.subject = parsed_claim.subject
                    subclaim_parsed.year = parsed_claim.year
                    subclaim_parsed.category = parsed_claim.category
                    
                    subclaim_evidences = await self._retriever.retrieve(
                        claim=subclaim,
                        parsed_claim=subclaim_parsed,
                        max_pages=max_pages,
                    )
                    
                    evidence_counts_per_subclaim.append(len(subclaim_evidences))
                    all_evidences.extend(subclaim_evidences)
                
                evidences = self._aggregate_and_deduplicate_evidences(all_evidences, max_total=80)
                debug_info["subclaim_evidence_counts"] = evidence_counts_per_subclaim
            else:
                evidences = await self._retriever.retrieve(
                    claim=claim,
                    parsed_claim=parsed_claim,
                    max_pages=max_pages,
                )
            
            timings_ms["retrieve_ms"] = (time.perf_counter() - retrieve_start) * 1000
            
            # Subject consistency filter
            if parsed_claim.subject:
                filtered_evidences = []
                subject_lower = parsed_claim.subject.lower()
                subject_tokens = set(subject_lower.split())
                surname = subject_tokens.pop() if subject_tokens else ""
                
                for ev in evidences:
                    title_lower = ev.title.lower()
                    snippet_lower = ev.snippet.lower()
                    
                    if (subject_lower in title_lower or surname in title_lower or
                        subject_lower in snippet_lower or surname in snippet_lower):
                        filtered_evidences.append(ev)
                
                if filtered_evidences:
                    evidences = filtered_evidences
            
            # Pre-filter to top-k evidences
            evidences_topk = self._select_topk_evidences(
                claim=claim,
                evidences=evidences,
                parsed_claim=parsed_claim,
                k=NLI_TOP_K,
            )
            
            # ===== STEP 4: Verify =====
            verify_start = time.perf_counter()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._verifier.verify,
                claim,
                evidences_topk,
                parsed_claim,
            )
            timings_ms["verify_ms"] = (time.perf_counter() - verify_start) * 1000
            
            # Convert to JSON-serializable format
            best_evidence_dict = None
            if result["best_evidence"] is not None:
                ev = result["best_evidence"]
                best_evidence_dict = {
                    "title": ev.title,
                    "url": ev.url,
                    "snippet": ev.snippet,
                    "source": ev.source,
                }
            
            scores = []
            for score_dict in result["scores"]:
                scores.append({
                    "entailment": float(score_dict["entailment"]),
                    "contradiction": float(score_dict["contradiction"]),
                    "neutral": float(score_dict["neutral"]),
                })
            
            # Build evidence_top_k list (truncated to top 10 for evaluation)
            evidence_top_k_list = []
            for ev in evidences[:10]:
                evidence_top_k_list.append({
                    "title": ev.title,
                    "url": ev.url,
                    "snippet": ev.snippet[:200],  # Truncate for evaluation
                    "source": ev.source,
                })
            
            # Add debug info
            debug_info["notes"] = result.get("notes", [])
            if related_info:
                debug_info["cache_related"] = related_info
            
            # Compute quality proxies if enabled
            quality_proxies = {}
            if enable_quality_proxies:
                from app.pipeline.eval.metrics import compute_quality_proxies
                quality_proxies = compute_quality_proxies(
                    claim=claim,
                    parsed_claim=parsed_claim,
                    evidences=evidences,
                    scores=scores,
                )
            
            # Auto-populate cache
            if self.enable_claim_cache and self._claim_matcher:
                label_enum = result["label"]
                if label_enum in {FactLabel.SUPPORTED, FactLabel.REFUTED}:
                    self._claim_matcher.store_result(
                        claim=claim,
                        label=label_enum,
                        evidence=best_evidence_dict or {},
                    )
            
            total_ms = (time.perf_counter() - total_start) * 1000
            timings_ms["total_ms"] = total_ms
            
            return RunResult(
                claim_raw=claim,
                claim_canonical=parsed_claim.canonical_claim,
                route_taken=route_taken,
                label_final=result["label"].value,
                confidence_final=result.get("confidence", 0.0),
                best_evidence=best_evidence_dict,
                evidence_top_k=evidence_top_k_list,
                debug=debug_info,
                timings_ms=timings_ms,
                quality_proxies=quality_proxies,
            )
            
        except Exception as e:
            # Conservative fallback: return RunResult with error note
            logger.error(f"Error in check_with_timing for claim '{claim[:50]}...': {e}", exc_info=True)
            total_ms = (time.perf_counter() - total_start) * 1000
            timings_ms["total_ms"] = total_ms
            
            debug_info["error"] = str(e)
            debug_info["notes"] = debug_info.get("notes", []) + [f"Pipeline error: {str(e)}"]
            
            return RunResult(
                claim_raw=claim,
                claim_canonical="",
                route_taken=RouteType.FULL_PIPELINE,
                label_final=FactLabel.NOT_ENOUGH_INFO.value,
                confidence_final=0.0,
                best_evidence=None,
                evidence_top_k=[],
                debug=debug_info,
                timings_ms=timings_ms,
            )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    @staticmethod
    def _is_numerical_claim_cache(claim: str, cached_claim: str) -> bool:
        """
        Check if both claims are numerical claims (e.g., "X hat Y Millionen Einwohner" or "X hat Y Mitgliedstaaten").
        
        Args:
            claim: Current claim text
            cached_claim: Cached claim text
            
        Returns:
            True if both claims contain numerical patterns
        """
        claim_lower = claim.lower()
        cached_lower = cached_claim.lower()
        
        # Pattern: number + "Millionen"/"Million"/"Mio" + "Einwohner" or "Einwohner" + number + "Millionen"
        # Also include: number + "Mitgliedstaaten" (for EU, etc.)
        numerical_patterns = [
            r"\d+[.,]?\d*\s*(?:millionen|million|mio|milliarden|milliard)\s*(?:einwohner|bevölkerung)",
            r"(?:einwohner|bevölkerung).*?\d+[.,]?\d*\s*(?:millionen|million|mio|milliarden)",
            r"\d+\s*mitgliedstaaten",  # "27 Mitgliedstaaten", "30 Mitgliedstaaten"
            r"mitgliedstaaten.*?\d+",  # "Mitgliedstaaten: 27"
        ]
        
        claim_has_pattern = any(re.search(pattern, claim_lower) for pattern in numerical_patterns)
        cached_has_pattern = any(re.search(pattern, cached_lower) for pattern in numerical_patterns)
        
        return claim_has_pattern and cached_has_pattern
    
    @staticmethod
    def _is_definition_like_claim(canonical: str, subject: Optional[str] = None) -> bool:
        """
        Detect definition/containment-like claims that should skip decomposition.
        
        Examples:
        - "ein jahr hat zwölf monate"
        - "das jahr umfasst 12 monate"
        - "eine woche besteht aus sieben tagen"
        
        Args:
            canonical: Canonical claim text (normalized, lowercase)
            subject: Optional subject extracted from claim
            
        Returns:
            True if claim appears to be a simple definition/containment claim
        """
        if not canonical:
            return False
        
        canonical_lower = canonical.lower()
        
        # CRITICAL: Exclude comparison claims - they need decomposition to work properly
        # Comparison claims have "als" which is a clear indicator
        comparison_indicators = [" als ", "länger", "kürzer", "größer", "kleiner", "mehr", "weniger", "höher", "niedriger"]
        if any(indicator in canonical_lower for indicator in comparison_indicators):
            # This is a comparison claim, NOT a definition claim - don't skip decomposition
            return False
        
        # Pattern 1: Check for definition/containment verbs
        definition_patterns = [" hat ", " umfasst ", " besteht aus ", " ist "]
        has_definition_pattern = any(pattern in canonical_lower for pattern in definition_patterns)
        
        if not has_definition_pattern:
            return False
        
        # Pattern 2: Check for time-unit definition patterns (e.g., "jahr hat 10 monate", "jahr hat zwölf monate")
        # These are definition-like even if they contain numbers
        time_unit_patterns = [
            r"(jahr|monat|woche|tag|stunde|minute|sekunde)\s+hat\s+(\d+|zwölf|elf|zehn|neun|acht|sieben|sechs|fünf|vier|drei|zwei|eins)\s+(monat|monate|woche|wochen|tag|tage|stunde|stunden|minute|minuten|sekunde|sekunden)",
            r"(jahr|monat|woche|tag)\s+umfasst\s+(\d+|zwölf|elf|zehn)\s+(monat|monate|woche|wochen|tag|tage)",
        ]
        is_time_unit_definition = any(re.search(pattern, canonical_lower) for pattern in time_unit_patterns)
        if is_time_unit_definition:
            logger.debug(f"DEFINITION_LIKE: Detected time-unit definition pattern in '{canonical_lower}'")
            return True
        
        # Pattern 3: Claim should be relatively short (definition claims are concise)
        tokens = canonical_lower.split()
        if len(tokens) > 12:
            return False
        
        # Pattern 3: Subject should be generic (optional but helps avoid false positives)
        is_generic = False
        if subject:
            subject_lower = subject.lower().strip()
            # Remove articles
            articles = ["der", "die", "das", "ein", "eine", "einen", "einem", "einer"]
            subject_words = subject_lower.split()
            if subject_words and subject_words[0] in articles:
                subject_clean = " ".join(subject_words[1:]) if len(subject_words) > 1 else subject_words[0] if subject_words else ""
            else:
                subject_clean = subject_lower
            
            # Generic time/calendar units
            generic_subjects = {
                "jahr", "jahre", "jahrs",
                "monat", "monate",
                "woche", "wochen",
                "tag", "tage",
                "stunde", "stunden",
                "minute", "minuten",
                "sekunde", "sekunden",
                "kalenderjahr",
            }
            
            # Check if subject is generic (1-2 tokens, matches generic set)
            subject_tokens = subject_clean.split()
            is_generic = (
                len(subject_tokens) <= 2 and
                any(token in generic_subjects for token in subject_tokens)
            )
            
            # Also accept if subject is 1-2 tokens (heuristic for simple entities)
            if not is_generic and len(subject_tokens) <= 2:
                # Additional check: if claim is very short and has definition pattern, likely definition-like
                if len(tokens) <= 8:
                    return True
        else:
            # No subject extracted, but if claim is short with definition pattern, likely definition-like
            if len(tokens) <= 8:
                return True
        
        # If we have a generic subject, definitely skip decomposition
        if is_generic:
            return True
        
        # Otherwise, only skip if claim is very short (likely definition)
        return len(tokens) <= 8
    
    @staticmethod
    def _extract_numbers_cache(text: str) -> List[float]:
        """
        Extract numerical values from text using unified parser.
        
        Args:
            text: Text to extract numbers from
            
        Returns:
            List of extracted numbers (in millions for backward compatibility)
        """
        from app.pipeline.population_parser import parse_population_number
        
        pop_values = parse_population_number(text)
        if not pop_values:
            return []
        
        # Convert absolute integers to millions (for backward compatibility)
        return [v.normalized_value_int / 1_000_000.0 for v in pop_values]
    
    @staticmethod
    def _numerical_claims_match(claim: str, cached_claim: str) -> bool:
        """
        Check if numerical values in two claims match (within 15% or 5 million difference).
        
        For small numbers (< 1000), use absolute difference of 2.
        For large numbers (millions), use relative difference of 15% or absolute difference of 5 million.
        
        Args:
            claim: Current claim text
            cached_claim: Cached claim text
            
        Returns:
            True if numbers are approximately equal, False otherwise
        """
        from app.pipeline.population_parser import parse_population_number
        
        # Extract raw numbers (not converted to millions)
        claim_values = parse_population_number(claim)
        cached_values = parse_population_number(cached_claim)
        
        if not claim_values or not cached_values:
            # Can't compare numbers, allow cache (fallback to semantic similarity)
            return True
        
        # Get largest absolute integer values
        claim_max = max(v.normalized_value_int for v in claim_values)
        cached_max = max(v.normalized_value_int for v in cached_values)
        
        diff = abs(claim_max - cached_max)
        
        # CRITICAL: For small numbers (< 1000, e.g., "27 Mitgliedstaaten", "30 Mitgliedstaaten"),
        # use strict absolute difference (max 2) instead of relative difference
        # This prevents "27 Mitgliedstaaten" from matching "30 Mitgliedstaaten" via cache
        if claim_max < 1000 and cached_max < 1000:
            # Small numbers: absolute difference must be <= 2
            match = diff <= 2
            logger.debug(
                f"NUMERICAL_CACHE_CHECK (small): claim_max={claim_max} cached_max={cached_max} "
                f"diff={diff} -> match={match}"
            )
            return match
        
        # For large numbers (millions): use relative difference or absolute difference
        # Convert to millions for comparison
        claim_max_millions = claim_max / 1_000_000.0
        cached_max_millions = cached_max / 1_000_000.0
        diff_millions = abs(claim_max_millions - cached_max_millions)
        relative_diff = diff_millions / max(claim_max_millions, cached_max_millions) if max(claim_max_millions, cached_max_millions) > 0 else 1.0
        
        # Numbers match if within 15% relative difference or 5 million absolute difference
        match = relative_diff < 0.15 or diff_millions < 5.0
        
        logger.debug(
            f"NUMERICAL_CACHE_CHECK (large): claim_max={claim_max_millions:.1f}M cached_max={cached_max_millions:.1f}M "
            f"diff={diff_millions:.1f}M relative_diff={relative_diff:.1%} -> match={match}"
        )
        
        return match
    
    async def _try_wikidata_population_fast_path(
        self, subject: str, claim_year: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """
        Try to get population from Wikidata P1082 (fast-path for population claims).
        
        Args:
            subject: Entity name (e.g., "Portugal", "Vereinigte Staaten")
            claim_year: Optional year from claim
            
        Returns:
            Dict with value_int, point_in_time, or None if not found
        """
        if not self._wikidata_client or not subject:
            return None
        
        try:
            # Search for entity
            search_results = await self._wikidata_client.search_entity(
                name=subject, language="de", limit=5
            )
            if not search_results:
                return None
            
            # Get first result (most relevant)
            entity_qid = search_results[0].get("id")
            if not entity_qid:
                return None
            
            # Get population data
            population_data = await self._wikidata_client.get_population(
                entity_qid, claim_year=claim_year
            )
            if not population_data:
                return None
            
            # Select best entry:
            # - If claim_year provided, prefer exact match, then closest
            # - Otherwise, prefer most recent
            if claim_year:
                # Find exact match first
                exact_match = next(
                    (entry for entry in population_data if entry.get("point_in_time") == claim_year),
                    None
                )
                if exact_match:
                    exact_match = dict(exact_match)
                    exact_match["entity_qid"] = entity_qid
                    return exact_match
                
                # Find closest match
                best_entry = min(
                    population_data,
                    key=lambda e: abs((e.get("point_in_time") or 0) - claim_year)
                )
                best_entry = dict(best_entry)
                best_entry["entity_qid"] = entity_qid
                return best_entry
            else:
                # Return most recent (first in sorted list)
                best_entry = dict(population_data[0])
                best_entry["entity_qid"] = entity_qid
                return best_entry
                
        except Exception as e:
            logger.warning(f"FACTCHECKER_WIKIDATA_POPULATION: Error in Wikidata lookup for '{subject}': {e}")
            return None
    
    async def _try_wikidata_geo_fast_path(
        self, subject: str
    ) -> Optional[Dict[str, Any]]:
        """
        Try to get country/location from Wikidata P17 (country) or P131 (located in administrative territorial entity).
        
        Args:
            subject: Entity name (e.g., "Berlin", "Rom")
            
        Returns:
            Dict with country_qid, country_label, or None if not found
        """
        if not self._wikidata_client or not subject:
            return None
        
        try:
            # Search for entity
            search_results = await self._wikidata_client.search_entity(
                name=subject, language="de", limit=5
            )
            if not search_results:
                return None
            
            # Get first result (most relevant)
            entity_qid = search_results[0].get("id")
            if not entity_qid:
                return None
            
            # Get entity data
            entity = await self._wikidata_client.get_entity(entity_qid, language="de")
            if not entity:
                return None
            
            # Extract P17 (country) or P131 (located in administrative territorial entity)
            claims = entity.get("claims", {})
            
            # Try P17 (country) first
            country_qid = None
            country_label = None
            location_qid = None  # For P131 recursive lookup
            
            # Historical entities to filter out (should not be used as countries)
            historical_entities = {
                "Q2277",  # Roman Empire
                "Q12548",  # Holy Roman Empire
                "Q10763",  # Historical region
                "Q1520223",  # Historical country
            }
            
            if "P17" in claims:
                # P17 is country property
                p17_claims = claims["P17"]
                if p17_claims and len(p17_claims) > 0:
                    # Get the main value (first claim)
                    # Handle both list and single claim formats
                    claim_item = p17_claims[0] if isinstance(p17_claims, list) else p17_claims
                    main_snak = claim_item.get("mainsnak", {})
                    if main_snak.get("datatype") == "wikibase-item":
                        datavalue = main_snak.get("datavalue", {})
                        if datavalue:
                            value = datavalue.get("value", {})
                            if isinstance(value, dict):
                                potential_country_qid = value.get("id")
                            elif isinstance(value, str):
                                potential_country_qid = value
                            else:
                                potential_country_qid = None
                            
                            # CRITICAL: Filter out historical entities (e.g., Roman Empire for London)
                            if potential_country_qid and potential_country_qid not in historical_entities:
                                country_qid = potential_country_qid
                            elif potential_country_qid:
                                logger.info(
                                    f"FACTCHECKER_WIKIDATA_GEO: Filtered out historical entity P17={potential_country_qid} "
                                    f"for '{subject}', will try P131 instead"
                                )
            
            # If no P17, try P131 (located in administrative territorial entity) and recursively find country
            if not country_qid and "P131" in claims:
                p131_claims = claims["P131"]
                if p131_claims and len(p131_claims) > 0:
                    # Get the main value (first claim)
                    # Handle both list and single claim formats
                    claim_item = p131_claims[0] if isinstance(p131_claims, list) else p131_claims
                    main_snak = claim_item.get("mainsnak", {})
                    if main_snak.get("datatype") == "wikibase-item":
                        datavalue = main_snak.get("datavalue", {})
                        if datavalue:
                            value = datavalue.get("value", {})
                            if isinstance(value, dict):
                                location_qid = value.get("id")
                            elif isinstance(value, str):
                                location_qid = value
                            
                            # Recursively find country from location (P131 chain)
                            if location_qid:
                                country_qid, country_label = await self._get_country_from_location_recursive(
                                    location_qid, max_depth=5
                                )
            
            if not country_qid:
                return None
            
            # Get country label if not already retrieved
            if not country_label:
                country_labels = await self._wikidata_client.get_labels([country_qid], lang="de")
                country_label = country_labels.get(country_qid, "")
            
            if not country_label:
                return None
            
            return {
                "country_qid": country_qid,
                "country_label": country_label,
                "entity_qid": entity_qid,
            }
            
        except Exception as e:
            logger.warning(f"FACTCHECKER_WIKIDATA_GEO: Error in Wikidata lookup for '{subject}': {e}")
            return None
    
    async def _get_country_from_location_recursive(
        self, location_qid: str, max_depth: int = 5, current_depth: int = 0
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Recursively find country (P17) from a location entity (P131 chain).
        
        Filters out historical entities (Roman Empire, etc.) and handles special cases like England->UK.
        
        Args:
            location_qid: Wikidata QID of the location entity
            max_depth: Maximum recursion depth (default: 5)
            current_depth: Current recursion depth
            
        Returns:
            Tuple of (country_qid, country_label) or (None, None) if not found
        """
        if current_depth >= max_depth or not location_qid:
            return None, None
        
        try:
            # Get location entity
            location_entity = await self._wikidata_client.get_entity(location_qid, language="de")
            if not location_entity:
                return None, None
            
            claims = location_entity.get("claims", {})
            
            # CRITICAL: Filter out historical entities (P31 = instance of)
            # Historical entities to ignore: Roman Empire (Q2277), Holy Roman Empire (Q12548), etc.
            historical_entities = {
                "Q2277",  # Roman Empire
                "Q12548",  # Holy Roman Empire
                "Q10763",  # Historical region
                "Q1520223",  # Historical country
            }
            
            instance_of = claims.get("P31", [])
            instance_ids = []
            for stmt in instance_of:
                mainsnak = stmt.get("mainsnak", {})
                if mainsnak.get("snaktype") == "value":
                    instance_id = mainsnak.get("datavalue", {}).get("value", {})
                    if isinstance(instance_id, dict):
                        instance_id = instance_id.get("id")
                    if isinstance(instance_id, str):
                        instance_ids.append(instance_id)
            
            # Skip if this entity is a historical entity
            if any(inst_id in historical_entities for inst_id in instance_ids):
                logger.info(
                    f"FACTCHECKER_WIKIDATA_GEO_RECURSIVE: Skipping historical entity '{location_qid}' "
                    f"(instances={instance_ids}) at depth {current_depth}"
                )
                return None, None
            
            # Check P17 (country) first
            if "P17" in claims:
                p17_claims = claims["P17"]
                if p17_claims and len(p17_claims) > 0:
                    claim_item = p17_claims[0] if isinstance(p17_claims, list) else p17_claims
                    main_snak = claim_item.get("mainsnak", {})
                    if main_snak.get("datatype") == "wikibase-item":
                        datavalue = main_snak.get("datavalue", {})
                        if datavalue:
                            value = datavalue.get("value", {})
                            if isinstance(value, dict):
                                country_qid = value.get("id")
                            elif isinstance(value, str):
                                country_qid = value
                            
                            if country_qid:
                                # CRITICAL: Handle special case - England (Q21) should map to UK (Q145)
                                if country_qid == "Q21":  # England
                                    country_qid = "Q145"  # United Kingdom
                                    logger.info(
                                        f"FACTCHECKER_WIKIDATA_GEO_RECURSIVE: Mapped England (Q21) to UK (Q145) "
                                        f"at depth {current_depth}"
                                    )
                                
                                # Get country label
                                country_labels = await self._wikidata_client.get_labels([country_qid], lang="de")
                                country_label = country_labels.get(country_qid, "")
                                if country_label:
                                    logger.info(
                                        f"FACTCHECKER_WIKIDATA_GEO_RECURSIVE: Found country '{country_label}' "
                                        f"({country_qid}) at depth {current_depth}"
                                    )
                                    return country_qid, country_label
            
            # If no P17, try P131 (located in administrative territorial entity) and recurse
            if "P131" in claims:
                p131_claims = claims["P131"]
                if p131_claims and len(p131_claims) > 0:
                    claim_item = p131_claims[0] if isinstance(p131_claims, list) else p131_claims
                    main_snak = claim_item.get("mainsnak", {})
                    if main_snak.get("datatype") == "wikibase-item":
                        datavalue = main_snak.get("datavalue", {})
                        if datavalue:
                            value = datavalue.get("value", {})
                            if isinstance(value, dict):
                                parent_location_qid = value.get("id")
                            elif isinstance(value, str):
                                parent_location_qid = value
                            
                            if parent_location_qid:
                                # Recursively find country from parent location
                                return await self._get_country_from_location_recursive(
                                    parent_location_qid, max_depth=max_depth, current_depth=current_depth + 1
                                )
            
            return None, None
            
        except Exception as e:
            logger.warning(f"FACTCHECKER_WIKIDATA_GEO_RECURSIVE: Error in recursive lookup for '{location_qid}': {e}")
            return None, None
    
    def _filter_population_evidences(
        self,
        evidences: List[Evidence],
        subject: str,
        claim: str,
    ) -> List[Evidence]:
        """
        Filter evidences for population claims.
        
        Only keeps evidences that:
        - Are from main article (title == subject) OR
        - Contain population keywords (einwohner, bevölkerung, etc.) AND
        - Contain parseable population numbers
        
        This prevents using wrong evidences like "Lissabon" for "Portugal" population claims.
        
        Args:
            evidences: List of all evidences
            subject: Subject entity (e.g., "Portugal")
            claim: Original claim text
            
        Returns:
            Filtered list of population-specific evidences
        """
        from app.pipeline.population_parser import parse_population_number
        
        subject_lower = subject.lower()
        claim_lower = claim.lower()
        is_population_claim = any(kw in claim_lower for kw in ["einwohner", "bevölkerung", "population"])
        
        if not is_population_claim:
            return evidences
        
        filtered = []
        pop_keywords = ["einwohner", "bevölkerung", "population", "habitantes", "inhabitants"]
        
        for ev in evidences:
            title_lower = ev.title.lower()
            snippet_lower = ev.snippet.lower()
            
            # Always include main article (exact title match)
            is_main_article = title_lower.strip() == subject_lower
            if is_main_article:
                filtered.append(ev)
                continue
            
            # Exclude disambiguation/list pages
            if any(kw in title_lower for kw in ["begriffsklärung", "liste", "geschichte", "krieg", "chronik"]):
                continue
            
            # Must contain population keywords
            has_pop_keyword = any(kw in snippet_lower for kw in pop_keywords)
            if not has_pop_keyword:
                continue
            
            # Must contain parseable population number
            pop_values = parse_population_number(ev.snippet)
            if not pop_values:
                continue
            
            # Filter out unrealistic values
            valid_values = [v for v in pop_values if 100_000 <= v.normalized_value_int <= 10_000_000_000]
            if not valid_values:
                continue
            
            # Subject should be mentioned (in title or snippet)
            subject_tokens = [t for t in subject_lower.split() if len(t) > 3]
            subject_in_title = any(token in title_lower for token in subject_tokens) if subject_tokens else False
            subject_in_snippet = any(token in snippet_lower for token in subject_tokens) if subject_tokens else False
            
            if subject_in_title or subject_in_snippet:
                filtered.append(ev)
        
        return filtered

