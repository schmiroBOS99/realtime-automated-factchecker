"""
NLI-based fact verification component.

Usage example:
    from app.pipeline.verifier import Verifier
    from app.pipeline.models import Evidence
    
    verifier = Verifier()
    
    # Score a single claim-evidence pair
    scores = verifier.score_pair(
        claim="Angela Merkel ist Bundeskanzlerin",
        evidence_text="Angela Merkel war von 2005 bis 2021 Bundeskanzlerin von Deutschland."
    )
    # Returns: {"entailment": 0.85, "contradiction": 0.10, "neutral": 0.05}
    
    # Verify a claim against multiple evidences
    evidences = [
        Evidence(
            source="wikipedia",
            title="Angela Merkel",
            snippet="Angela Merkel war von 2005 bis 2021 Bundeskanzlerin.",
            url="https://..."
        )
    ]
    result = verifier.verify(
        claim="Angela Merkel ist Bundeskanzlerin",
        evidences=evidences
    )
    # Returns: {
    #     "label": FactLabel.SUPPORTED,
    #     "best_evidence": Evidence(...),
    #     "scores": [{"entailment": 0.85, "contradiction": 0.10, "neutral": 0.05}]
    # }
"""
import re
import logging
import torch
from typing import Dict, List, Optional, Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.pipeline.models import Evidence, FactLabel, ParsedClaim
from app.pipeline.postprocessors.base import Decision
from app.pipeline.postprocessors import get_all_processors

logger = logging.getLogger(__name__)


class Verifier:
    """
    NLI-based fact verifier with modular post-processors.
    
    Uses a minimal NLI kernel (score → aggregate → label) and optional
    post-processors for domain-specific heuristics.
    """
    
    # Fixed kernel thresholds
    KERNEL_SUPPORT_THRESHOLD = 0.75
    KERNEL_REFUTE_THRESHOLD = 0.75
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        postprocessors: Optional[List] = None,
    ):
        """
        Initialize verifier with NLI model and optional post-processors.
        
        Args:
            model_name: HuggingFace model identifier for NLI model
            postprocessors: Optional list of PostProcessor instances.
                          If None, uses all default processors.
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Map label indices to names
        self.label_map = {
            int(idx): label.lower()
            for idx, label in self.model.config.id2label.items()
        }
        
        # Find label indices (case-insensitive search)
        self.entail_idx = self._get_label_index("entail", default=2)
        self.contra_idx = self._get_label_index("contrad", default=0)
        self.neutral_idx = self._get_label_index("neutral", default=1)
        
        # Initialize post-processors
        if postprocessors is None:
            # Default: use all processors
            self.postprocessors = get_all_processors()
        else:
            self.postprocessors = postprocessors
        
        logger.info(f"VERIFIER: Initialized with {len(self.postprocessors)} post-processors")
    
    def _get_label_index(self, keyword: str, default: int) -> int:
        """
        Find label index by searching for keyword in label names.
        
        Args:
            keyword: Keyword to search for (e.g., "entail", "contrad")
            default: Default index if not found
            
        Returns:
            Label index
        """
        for idx, label in self.label_map.items():
            if keyword in label:
                return idx
        return default
    
    
    @torch.no_grad()
    def score_pair(self, claim: str, evidence_text: str, parsed_claim: Optional[ParsedClaim] = None) -> Dict[str, float]:
        """
        Score a claim-evidence pair using NLI model (raw scores, no post-processing).
        
        This method returns raw NLI scores. For full verification with post-processors,
        use verify() instead.
        
        Args:
            claim: The claim to verify
            evidence_text: The evidence text to check against
            parsed_claim: Optional parsed claim (currently unused, kept for compatibility)
            
        Returns:
            Dict with keys: "entailment", "contradiction", "neutral"
            Values are probabilities (0.0 to 1.0)
        """
        if not claim or not evidence_text:
            return {
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 1.0,
            }
        
        # Tokenize: evidence first, then claim (standard NLI format)
        inputs = self.tokenizer(
            evidence_text,
            claim,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        
        # Get model predictions
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        
        # Extract probabilities
        entailment = probs[self.entail_idx].item()
        contradiction = probs[self.contra_idx].item()
        
        # Calculate neutral probability
        if self.neutral_idx is not None:
            neutral = probs[self.neutral_idx].item()
        else:
            # Fallback: ensure probabilities sum to 1.0
            neutral = max(0.0, 1.0 - entailment - contradiction)
        
        return {
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral,
        }
    
    def score_pairs_batch(
        self,
        claim: str,
        evidence_texts: List[str],
    ) -> List[Dict[str, float]]:
        """
        Score multiple claim-evidence pairs in batch (kernel method).
        
        Args:
            claim: The claim to verify
            evidence_texts: List of evidence texts to score
            
        Returns:
            List of score dicts, each with "entailment", "contradiction", "neutral"
        """
        if not claim or not evidence_texts:
            return []
        
        scores_list = []
        for evidence_text in evidence_texts:
            scores = self.score_pair(claim, evidence_text)
            scores_list.append(scores)
        
        return scores_list
    
    def aggregate_scores(
        self,
        evidence_scores: List[Dict[str, Any]],
        strategy: str = "max",
    ) -> Dict[str, float]:
        """
        Aggregate multiple evidence scores into a single decision (kernel method).
        
        Args:
            evidence_scores: List of score dicts with "entailment", "contradiction", "neutral"
            strategy: Aggregation strategy ("max" or "top2_mean")
            
        Returns:
            Aggregated scores dict with "entailment", "contradiction", "neutral", "confidence"
        """
        if not evidence_scores:
            return {
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 1.0,
                "confidence": 0.0,
            }
        
        if strategy == "max":
            # Use best (highest entailment) evidence
            best_entry = max(evidence_scores, key=lambda x: x["entailment"])
            entailment = best_entry["entailment"]
            contradiction = best_entry["contradiction"]
            neutral = best_entry.get("neutral", 1.0 - entailment - contradiction)
            
        elif strategy == "top2_mean":
            # Average top 2 evidences
            sorted_scores = sorted(evidence_scores, key=lambda x: x["entailment"], reverse=True)
            top2 = sorted_scores[:2]
            entailment = sum(e["entailment"] for e in top2) / len(top2)
            contradiction = sum(e["contradiction"] for e in top2) / len(top2)
            neutral = sum(e.get("neutral", 1.0 - e["entailment"] - e["contradiction"]) for e in top2) / len(top2)
        else:
            # Default to max
            best_entry = max(evidence_scores, key=lambda x: x["entailment"])
            entailment = best_entry["entailment"]
            contradiction = best_entry["contradiction"]
            neutral = best_entry.get("neutral", 1.0 - entailment - contradiction)
        
        # Multi-evidence aggregation: boost confidence if multiple evidences agree
        if len(evidence_scores) >= 2:
            sorted_scores = sorted(evidence_scores, key=lambda x: x["entailment"], reverse=True)
            others = sorted_scores[1:]
            
            # Check for agreeing evidences (entailment > 0.6)
            agreeing = [e for e in others if e["entailment"] > 0.6]
            disagreeing = [e for e in others if e["contradiction"] > 0.6]
            
            if len(agreeing) >= 1:
                # Boost: weighted average with agreeing evidences
                avg_entailment = sum(e["entailment"] for e in agreeing) / len(agreeing)
                entailment = (entailment * 0.6) + (avg_entailment * 0.4)
                entailment = min(1.0, entailment * 1.1)  # Small boost
                logger.debug(
                    f"KERNEL_AGGREGATE: {len(agreeing)} agreeing evidences, boosted entailment to {entailment:.3f}"
                )
            
            if len(disagreeing) >= 1:
                # Reduce confidence if strong disagreement
                avg_contradiction = sum(e["contradiction"] for e in disagreeing) / len(disagreeing)
                if avg_contradiction > 0.7:
                    entailment *= 0.85  # Reduce by 15%
                    logger.debug(
                        f"KERNEL_AGGREGATE: {len(disagreeing)} disagreeing evidences, reduced entailment to {entailment:.3f}"
                    )
        
        confidence = max(entailment, contradiction)
        
        return {
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral,
            "confidence": confidence,
        }
    
    def decide_label(
        self,
        aggregate_scores: Dict[str, float],
        support_threshold: Optional[float] = None,
        refute_threshold: Optional[float] = None,
    ) -> FactLabel:
        """
        Decide label from aggregated scores using fixed thresholds (kernel method).
        
        Args:
            aggregate_scores: Aggregated scores dict with "entailment", "contradiction"
            support_threshold: Threshold for SUPPORTED (default: KERNEL_SUPPORT_THRESHOLD)
            refute_threshold: Threshold for REFUTED (default: KERNEL_REFUTE_THRESHOLD)
            
        Returns:
            FactLabel decision
        """
        if support_threshold is None:
            support_threshold = self.KERNEL_SUPPORT_THRESHOLD
        if refute_threshold is None:
            refute_threshold = self.KERNEL_REFUTE_THRESHOLD
        
        entailment = aggregate_scores["entailment"]
        contradiction = aggregate_scores["contradiction"]
        
        if entailment >= support_threshold and entailment > contradiction:
            return FactLabel.SUPPORTED
        elif contradiction >= refute_threshold and contradiction > entailment:
            return FactLabel.REFUTED
        else:
            return FactLabel.NOT_ENOUGH_INFO
    
    def score_relevance_batch(
        self,
        claim: str,
        sentences: List[str],
        max_length: int = 256,
    ) -> List[float]:
        """
        Score relevance of multiple sentences against a claim using NLI model.
        
        This is a fast, batched relevance scoring method for retrieval reranking.
        Relevance is defined as: max(entailment_prob, contradiction_prob)
        Rationale: both entail and contradict indicate high claim-relatedness;
        neutral usually indicates not about the claim.
        
        Args:
            claim: The claim text
            sentences: List of sentence texts to score
            max_length: Maximum sequence length for tokenization (default: 256)
            
        Returns:
            List of relevance scores (float, 0.0 to 1.0) for each sentence
        """
        if not claim or not sentences:
            return [0.0] * len(sentences) if sentences else []
        
        if len(sentences) == 0:
            return []
        
        try:
            # Batch tokenize: evidence first, then claim (standard NLI format)
            # Truncate aggressively to keep latency down
            inputs = self.tokenizer(
                sentences,
                [claim] * len(sentences),  # Repeat claim for each sentence
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            
            # Get model predictions (batched)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
            
            # Extract relevance scores: max(entailment, contradiction) for each sentence
            relevance_scores = []
            for i in range(len(sentences)):
                entailment = probs[i][self.entail_idx].item()
                contradiction = probs[i][self.contra_idx].item()
                # Relevance = max(entail, contradict) - both indicate claim-relatedness
                relevance = max(entailment, contradiction)
                relevance_scores.append(relevance)
            
            return relevance_scores
            
        except Exception as e:
            logger.warning(f"RELEVANCE_SCORE_BATCH: Error during batch scoring: {e}. Returning zeros.")
            return [0.0] * len(sentences)
    
    def verify(
        self,
        claim: str,
        evidences: List[Evidence],
        parsed_claim: Optional[ParsedClaim] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Verify a claim against multiple evidences using NLI kernel + post-processors.
        
        Flow:
        1. Kernel: score → aggregate → decide_label (fixed thresholds)
        2. Post-processors: apply heuristics sequentially
        
        Args:
            claim: The claim to verify
            evidences: List of Evidence objects to check against
            parsed_claim: Optional parsed claim for context
            
        Returns:
            Dict with keys:
                - "label": FactLabel (SUPPORTED, REFUTED, or NOT_ENOUGH_INFO)
                - "best_evidence": Evidence object with highest entailment score, or None
                - "scores": List of score dicts (one per evidence)
        """
        if not evidences:
            return {
                "label": FactLabel.NOT_ENOUGH_INFO,
                "best_evidence": None,
                "scores": [],
            }
        
        if not claim:
            return {
                "label": FactLabel.NOT_ENOUGH_INFO,
                "best_evidence": None,
                "scores": [],
            }
        
        # ===== KERNEL: Step 1 - Score all evidence pairs =====
        evidence_scores: List[Dict] = []
        evidence_texts = []
        
        for evidence in evidences:
            # Use evidence.parsed.raw if available, else evidence.snippet
            evidence_text = (
                evidence.parsed.raw
                if evidence.parsed and evidence.parsed.raw
                else evidence.snippet
            )
            evidence_texts.append(evidence_text)
            
            # Get raw NLI scores (no post-processing in score_pair)
            scores = self.score_pair(claim, evidence_text)
            evidence_scores.append({
                "evidence": evidence,
                "entailment": scores["entailment"],
                "contradiction": scores["contradiction"],
                "neutral": scores["neutral"],
            })
        
        # ===== KERNEL: Step 2 - Aggregate scores =====
        aggregate = self.aggregate_scores(evidence_scores, strategy="max")
        
        # Find best evidence
        best_entry = max(evidence_scores, key=lambda x: x["entailment"])
        best_evidence = best_entry["evidence"]
        
        # ===== KERNEL: Step 3 - Decide label with fixed thresholds =====
        label = self.decide_label(aggregate)
        
        logger.info(
            f"KERNEL: aggregate_entail={aggregate['entailment']:.3f} "
            f"aggregate_contra={aggregate['contradiction']:.3f} -> {label.value}"
        )
        
        # ===== Create initial Decision object =====
        decision = Decision(
            label=label,
            confidence=aggregate["confidence"],
            best_evidence=best_evidence,
            best_entailment=aggregate["entailment"],
            best_contradiction=aggregate["contradiction"],
            threshold=self.KERNEL_SUPPORT_THRESHOLD,  # Start with fixed threshold
            evidence_scores=evidence_scores,
            metadata=metadata,  # Pass metadata through to postprocessors
        )
        
        # ===== POST-PROCESSORS: Apply sequentially =====
        for processor in self.postprocessors:
            old_label = decision.label
            old_confidence = decision.confidence
            
            decision = processor.process(claim, evidences, decision, parsed_claim)
            
            # Log if processor changed the decision
            if decision.label != old_label or abs(decision.confidence - old_confidence) > 0.01:
                logger.info(
                    f"POSTPROCESSOR[{processor.name}]: "
                    f"label={old_label.value}->{decision.label.value} "
                    f"confidence={old_confidence:.3f}->{decision.confidence:.3f}"
                )
                if decision.notes:
                    logger.debug(f"POSTPROCESSOR[{processor.name}] notes: {decision.notes[-1]}")
        
        # ===== Build final response =====
        # Find best evidence index for guard flag
        best_entry_idx = evidence_scores.index(best_entry)
        scores_list = []
        for i, entry in enumerate(evidence_scores):
            score_dict = {
                "entailment": entry["entailment"],
                "contradiction": entry["contradiction"],
                "neutral": entry["neutral"],
            }
            # Add guard flag if this was the best evidence and it was downgraded
            if i == best_entry_idx and decision.label == FactLabel.NOT_ENOUGH_INFO and decision.best_entailment >= 0.70:
                score_dict["guard"] = "PREDICATE_WEAK"
            scores_list.append(score_dict)
        
        population_correction = None
        if decision.metadata and "population_correction" in decision.metadata:
            population_correction = decision.metadata.get("population_correction")
        
        geo_correction = None
        if decision.metadata and "geo_correction" in decision.metadata:
            geo_correction = decision.metadata.get("geo_correction")

        return {
            "label": decision.label,
            "best_evidence": decision.best_evidence,
            "scores": scores_list,
            "population_correction": population_correction,
            "geo_correction": geo_correction,
            "notes": decision.notes or [],
        }

