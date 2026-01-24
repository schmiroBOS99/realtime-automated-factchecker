"""
Quality proxy metrics for evaluation.

Simple, robust proxies for evidence and decision quality without requiring ground truth.
"""
import re
from typing import List, Dict, Any, Optional

from app.pipeline.models import Evidence, FactLabel, ParsedClaim
from app.pipeline.utils import tokenize_text


def compute_quality_proxies(
    claim: str,
    parsed_claim: ParsedClaim,
    evidences: List[Evidence],
    scores: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Compute quality proxies for a claim check result.
    
    Args:
        claim: Original claim text
        parsed_claim: Parsed claim structure
        evidences: List of evidence objects
        scores: List of NLI score dictionaries with 'entailment', 'contradiction', 'neutral' keys
        
    Returns:
        Dictionary of quality proxy metrics
    """
    proxies: Dict[str, float] = {
        "evidence_count_returned": len(evidences),
        "avg_sentence_length_chars": 0.0,
        "avg_sentence_length_tokens": 0.0,
        "claim_token_overlap_mean": 0.0,
        "subject_coverage_rate": 0.0,
        "confidence_margin": 0.0,
        "max_entailment": 0.0,
        "max_contradiction": 0.0,
    }
    
    if not evidences:
        return proxies
    
    # Evidence-level metrics
    total_chars = 0
    total_tokens = 0
    total_overlap = 0
    subject_covered_count = 0
    
    claim_tokens = set(tokenize_text(claim))
    subject_tokens = set(tokenize_text(parsed_claim.subject)) if parsed_claim.subject else set()
    
    for ev in evidences:
        snippet = ev.snippet
        total_chars += len(snippet)
        
        ev_tokens = tokenize_text(snippet)
        total_tokens += len(ev_tokens)
        
        # Token overlap
        overlap = len(claim_tokens.intersection(ev_tokens))
        total_overlap += overlap
        
        # Subject coverage
        if subject_tokens and subject_tokens.intersection(ev_tokens):
            subject_covered_count += 1
    
    proxies["avg_sentence_length_chars"] = total_chars / len(evidences)
    proxies["avg_sentence_length_tokens"] = total_tokens / len(evidences)
    proxies["claim_token_overlap_mean"] = total_overlap / len(evidences)
    
    if subject_tokens:
        proxies["subject_coverage_rate"] = subject_covered_count / len(evidences)
    
    # Decision-level metrics
    if scores:
        max_entailment = 0.0
        max_contradiction = 0.0
        
        for score_dict in scores:
            entailment = score_dict.get("entailment", 0.0)
            contradiction = score_dict.get("contradiction", 0.0)
            
            max_entailment = max(max_entailment, entailment)
            max_contradiction = max(max_contradiction, contradiction)
        
        proxies["max_entailment"] = max_entailment
        proxies["max_contradiction"] = max_contradiction
        proxies["confidence_margin"] = abs(max_entailment - max_contradiction)
    
    return proxies


def calculate_evidence_metrics(
    claim: str,
    evidences: List[Evidence],
    parsed_claim_subject: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculates simple quality proxies for a list of evidences.
    
    DEPRECATED: Use compute_quality_proxies instead.
    """
    metrics = {
        "evidence_count_returned": len(evidences),
        "avg_sentence_length_chars": 0.0,
        "avg_sentence_length_tokens": 0.0,
        "claim_token_overlap_mean": 0.0,
        "subject_coverage_rate": 0.0,
    }
    
    if not evidences:
        return metrics
    
    total_chars = 0
    total_tokens = 0
    total_overlap = 0
    subject_covered_count = 0
    
    claim_tokens = set(tokenize_text(claim))
    subject_tokens = set(tokenize_text(parsed_claim_subject)) if parsed_claim_subject else set()
    
    for ev in evidences:
        snippet = ev.snippet
        total_chars += len(snippet)
        
        ev_tokens = tokenize_text(snippet)
        total_tokens += len(ev_tokens)
        
        total_overlap += len(claim_tokens.intersection(ev_tokens))
        
        if subject_tokens and subject_tokens.intersection(ev_tokens):
            subject_covered_count += 1
    
    metrics["avg_sentence_length_chars"] = total_chars / len(evidences)
    metrics["avg_sentence_length_tokens"] = total_tokens / len(evidences)
    metrics["claim_token_overlap_mean"] = total_overlap / len(evidences)
    
    if subject_tokens:
        metrics["subject_coverage_rate"] = subject_covered_count / len(evidences)
    
    return metrics


def calculate_decision_metrics(
    final_label: FactLabel,
    final_confidence: float,
    raw_nli_scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculates simple quality proxies for the final decision.
    
    DEPRECATED: Use compute_quality_proxies instead.
    """
    metrics = {
        "confidence_margin": 0.0,
        "max_entailment": 0.0,
        "max_contradiction": 0.0,
    }
    
    if not raw_nli_scores:
        return metrics
    
    max_entailment = 0.0
    max_contradiction = 0.0
    
    for score_entry in raw_nli_scores:
        entailment = score_entry.get("entailment", 0.0)
        contradiction = score_entry.get("contradiction", 0.0)
        
        max_entailment = max(max_entailment, entailment)
        max_contradiction = max(max_contradiction, contradiction)
    
    metrics["max_entailment"] = max_entailment
    metrics["max_contradiction"] = max_contradiction
    metrics["confidence_margin"] = abs(max_entailment - max_contradiction)
    
    return metrics
