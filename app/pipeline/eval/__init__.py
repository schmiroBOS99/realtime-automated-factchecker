"""
Evaluation harness for Pipeline 2.0.

Provides tools for measuring latency, quality proxies, and running A/B tests.
"""

from app.pipeline.eval.harness import EvaluationHarness
from app.pipeline.eval.metrics import compute_quality_proxies
from app.pipeline.eval.datasets import load_claims_from_jsonl

__all__ = [
    "EvaluationHarness",
    "compute_quality_proxies",
    "load_claims_from_jsonl",
]

