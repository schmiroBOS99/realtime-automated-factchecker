"""
Evaluation harness for Pipeline 2.0.

Provides tools for running evaluation runs, collecting metrics, and generating reports.
"""
import json
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.pipeline.factchecker import FactChecker
from app.pipeline.models import RunResult, RouteType, FactLabel

logger = logging.getLogger(__name__)


class EvaluationHarness:
    """
    Evaluation harness for running reproducible evaluation runs.
    
    Supports:
    - Running claims through pipeline with timing
    - JSONL output for individual results
    - Aggregation and summary reports
    - A/B testing via feature flags
    """
    
    def __init__(
        self,
        factchecker: FactChecker,
        output_dir: str = "eval_outputs",
    ):
        """
        Initialize evaluation harness.
        
        Args:
            factchecker: FactChecker instance (configured with desired feature flags)
            output_dir: Directory for output files
        """
        self.factchecker = factchecker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_evaluation(
        self,
        claims: List[Dict[str, Any]],
        preset_name: str = "default",
    ) -> tuple[str, str]:
        """
        Run evaluation on a list of claims.
        
        Args:
            claims: List of claim dictionaries (must have "claim" field)
            preset_name: Name of the preset/configuration (for output file naming)
            
        Returns:
            Tuple of (jsonl_output_path, summary_output_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = self.output_dir / f"{preset_name}_run_{timestamp}.jsonl"
        summary_path = self.output_dir / f"{preset_name}_summary_{timestamp}.json"
        
        results: List[RunResult] = []
        
        logger.info(f"EVAL: Starting evaluation run '{preset_name}' with {len(claims)} claims")
        
        # Process each claim
        for i, claim_dict in enumerate(claims, 1):
            claim_text = claim_dict.get("claim", "")
            if not claim_text:
                logger.warning(f"EVAL: Claim {i} missing 'claim' field, skipping")
                continue
            
            try:
                result = await self.factchecker.check_with_timing(claim_text)
                results.append(result)
                
                # Write JSONL line immediately
                result_dict = self._runresult_to_dict(result)
                with open(jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                
                if i % 10 == 0:
                    logger.info(f"EVAL: Processed {i}/{len(claims)} claims")
                    
            except Exception as e:
                logger.error(f"EVAL: Error processing claim {i} '{claim_text[:50]}...': {e}", exc_info=True)
                # Create error result
                error_result = RunResult(
                    claim_raw=claim_text,
                    claim_canonical="",
                    route_taken=RouteType.FULL_PIPELINE,
                    label_final=FactLabel.NOT_ENOUGH_INFO.value,
                    confidence_final=0.0,
                    best_evidence=None,
                    evidence_top_k=[],
                    debug={"error": str(e)},
                    timings_ms={"total_ms": 0.0},
                )
                results.append(error_result)
                
                result_dict = self._runresult_to_dict(error_result)
                with open(jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        
        # Generate summary
        summary = self._aggregate_results(results, preset_name)
        
        # Write summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"EVAL: Evaluation complete. Results: {jsonl_path}, Summary: {summary_path}")
        
        return str(jsonl_path), str(summary_path)
    
    def _runresult_to_dict(self, result: RunResult) -> Dict[str, Any]:
        """Convert RunResult to dictionary for JSON serialization."""
        return {
            "claim_raw": result.claim_raw,
            "claim_canonical": result.claim_canonical,
            "route_taken": result.route_taken.value,
            "label_final": result.label_final,
            "confidence_final": result.confidence_final,
            "best_evidence": result.best_evidence,
            "evidence_top_k": result.evidence_top_k,
            "debug": result.debug,
            "timings_ms": result.timings_ms,
            "quality_proxies": result.quality_proxies,
        }
    
    def _aggregate_results(
        self,
        results: List[RunResult],
        preset_name: str,
    ) -> Dict[str, Any]:
        """
        Aggregate results into summary statistics.
        
        Computes:
        - Latency p50/p95 total_ms + breakdown means
        - Label distribution
        - Cache hit rate
        - Cross-encoder usage rate
        - Decomposition usage rate
        """
        if not results:
            return {
                "preset": preset_name,
                "total_claims": 0,
                "error": "No results to aggregate",
            }
        
        # Latency statistics
        total_times = [r.timings_ms.get("total_ms", 0.0) for r in results]
        parse_times = [r.timings_ms.get("parse_ms", 0.0) for r in results]
        retrieve_times = [r.timings_ms.get("retrieve_ms", 0.0) for r in results]
        verify_times = [r.timings_ms.get("verify_ms", 0.0) for r in results]
        
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            index = int(len(sorted_data) * p / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]
        
        latency_stats = {
            "total_ms": {
                "mean": statistics.mean(total_times) if total_times else 0.0,
                "median": statistics.median(total_times) if total_times else 0.0,
                "p50": percentile(total_times, 50),
                "p95": percentile(total_times, 95),
                "p99": percentile(total_times, 99),
            },
            "parse_ms": {
                "mean": statistics.mean(parse_times) if parse_times else 0.0,
            },
            "retrieve_ms": {
                "mean": statistics.mean(retrieve_times) if retrieve_times else 0.0,
            },
            "verify_ms": {
                "mean": statistics.mean(verify_times) if verify_times else 0.0,
            },
        }
        
        # Label distribution
        label_counts = {}
        for result in results:
            label = result.label_final
            label_counts[label] = label_counts.get(label, 0) + 1
        
        label_distribution = {
            label: {
                "count": count,
                "percentage": (count / len(results)) * 100,
            }
            for label, count in label_counts.items()
        }
        
        # Route distribution
        route_counts = {}
        for result in results:
            route = result.route_taken.value
            route_counts[route] = route_counts.get(route, 0) + 1
        
        route_distribution = {
            route: {
                "count": count,
                "percentage": (count / len(results)) * 100,
            }
            for route, count in route_counts.items()
        }
        
        # Usage rates
        cache_hits = sum(1 for r in results if r.route_taken == RouteType.CACHE_HIT)
        cache_hit_rate = (cache_hits / len(results)) * 100 if results else 0.0
        
        decomposition_used = sum(
            1 for r in results 
            if r.route_taken == RouteType.FULL_PIPELINE_WITH_DECOMP
        )
        decomposition_rate = (decomposition_used / len(results)) * 100 if results else 0.0
        
        # Quality proxies (averages)
        quality_proxy_means = {}
        if results and results[0].quality_proxies:
            proxy_keys = results[0].quality_proxies.keys()
            for key in proxy_keys:
                values = [r.quality_proxies.get(key, 0.0) for r in results if r.quality_proxies]
                if values:
                    quality_proxy_means[key] = statistics.mean(values)
        
        summary = {
            "preset": preset_name,
            "total_claims": len(results),
            "timestamp": datetime.now().isoformat(),
            "latency_stats": latency_stats,
            "label_distribution": label_distribution,
            "route_distribution": route_distribution,
            "usage_rates": {
                "cache_hit_rate": cache_hit_rate,
                "decomposition_rate": decomposition_rate,
            },
            "quality_proxies_mean": quality_proxy_means,
        }
        
        return summary
    
    def compare_presets(
        self,
        preset_results: Dict[str, List[RunResult]],
    ) -> Dict[str, Any]:
        """
        Compare multiple preset runs and compute label flips.
        
        Args:
            preset_results: Dict mapping preset_name -> List[RunResult]
            
        Returns:
            Comparison summary with label flip counts
        """
        if len(preset_results) < 2:
            return {"error": "Need at least 2 presets to compare"}
        
        preset_names = list(preset_results.keys())
        baseline_name = preset_names[0]
        baseline_results = preset_results[baseline_name]
        
        # Create claim -> label mapping for baseline
        baseline_labels = {
            r.claim_raw: r.label_final
            for r in baseline_results
        }
        
        comparisons = {}
        for preset_name in preset_names[1:]:
            preset_results_list = preset_results[preset_name]
            
            label_flips = {}
            total_flips = 0
            
            for result in preset_results_list:
                claim = result.claim_raw
                baseline_label = baseline_labels.get(claim)
                current_label = result.label_final
                
                if baseline_label and baseline_label != current_label:
                    flip_key = f"{baseline_label} -> {current_label}"
                    label_flips[flip_key] = label_flips.get(flip_key, 0) + 1
                    total_flips += 1
            
            comparisons[preset_name] = {
                "total_flips": total_flips,
                "flip_rate": (total_flips / len(preset_results_list)) * 100 if preset_results_list else 0.0,
                "label_flips": label_flips,
            }
        
        return {
            "baseline": baseline_name,
            "comparisons": comparisons,
        }

