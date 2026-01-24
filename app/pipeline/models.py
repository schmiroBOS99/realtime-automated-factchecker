from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict, Any

# Label-Typ
from enum import Enum

class FactLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"



# -----------------------------------------------------------
# Datenstruktur für Evidenz
# -----------------------------------------------------------

@dataclass
class Evidence:
    source: str
    title: str
    snippet: str  # For sentence-level: contains a single sentence
    url: str
    parsed: Optional["ParsedEvidence"] = None
    # Optional metadata for sentence-level evidence
    sentence_index: Optional[int] = None  # Position of sentence in source article
    source_title: Optional[str] = None  # Article title (same as title, kept for clarity)
    score: Optional[float] = None  # Optional relevance score from cross-encoder reranking


@dataclass
class EvidenceCandidate:
    title: str
    snippet: str
    url: str
    source: str = "wikipedia"


@dataclass
class SentenceEvidence:
    sentence: str
    title: str
    url: str
    source: str = "wikipedia"
    full_snippet: str = ""  # Keep full snippet for context
    score: float = 0.0  # Semantic similarity score


@dataclass
class ParsedClaim:
    raw: str
    subject: str = ""
    predicate: str = ""
    object: Optional[str] = None
    modality: Optional[str] = None
    tense: Literal["present", "past", "unknown"] = "unknown"
    numbers: List[int] = field(default_factory=list)
    category: str = "general"
    relation_type: str = "general"  # Maps from category: "political_role", "party_membership", "sports_title", "biological_capability", etc.
    year: Optional[int] = None  # First year from numbers if available
    party: Optional[str] = None
    role: Optional[str] = None
    event: Optional[str] = None
    sport: Optional[str] = None
    # Claim normalization & decomposition fields
    canonical_claim: str = ""  # Normalized claim text
    subclaims: List[str] = field(default_factory=list)  # Decomposed subclaims (max 3-4)
    is_hard_claim: bool = False  # Flag indicating if claim requires decomposition


@dataclass
class ParsedEvidence:
    raw: str
    subject: str = ""
    predicate: str = ""
    object: Optional[str] = None
    modality: Optional[str] = None
    tense: Literal["present", "past", "unknown"] = "unknown"
    numbers: List[int] = field(default_factory=list)
    party: Optional[str] = None
    role: Optional[str] = None
    mentions_negation: bool = False


# -----------------------------------------------------------
# Evaluation Harness Models
# -----------------------------------------------------------

class RouteType(str, Enum):
    """Route taken through the pipeline."""
    CACHE_HIT = "CACHE_HIT"
    FULL_PIPELINE = "FULL_PIPELINE"
    FULL_PIPELINE_WITH_DECOMP = "FULL_PIPELINE_WITH_DECOMP"
    RELATED_ONLY = "RELATED_ONLY"
    WIKIDATA_FAST_PATH = "WIKIDATA_FAST_PATH"


@dataclass
class RunResult:
    """
    Standardized result structure for evaluation runs.
    
    Contains all information needed for evaluation, debugging, and analysis.
    """
    # Claim information
    claim_raw: str
    claim_canonical: str = ""
    
    # Route and decision
    route_taken: RouteType = RouteType.FULL_PIPELINE
    label_final: str = ""  # FactLabel value as string
    confidence_final: float = 0.0
    
    # Evidence
    best_evidence: Optional[Dict[str, Any]] = None  # title, url, snippet/sentence, source
    evidence_top_k: List[Dict[str, Any]] = field(default_factory=list)  # Optional, may be truncated
    
    # Debug information
    debug: Dict[str, Any] = field(default_factory=dict)  # thresholds, notes, postprocessor_changes, match_score, match_type
    
    # Timing information (milliseconds)
    timings_ms: Dict[str, float] = field(default_factory=lambda: {
        "parse_ms": 0.0,
        "claim_match_ms": 0.0,
        "retrieve_ms": 0.0,
        "stage1_rerank_ms": 0.0,
        "stage2_crossencoder_ms": 0.0,
        "verify_ms": 0.0,
        "total_ms": 0.0,
    })
    
    # Quality proxies (optional, computed post-hoc)
    quality_proxies: Dict[str, float] = field(default_factory=dict)

