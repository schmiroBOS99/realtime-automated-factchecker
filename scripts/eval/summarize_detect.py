"""
Summarize claim detection evaluation results.

Computes filter metrics, detection coverage, and relation-type accuracy.

Usage:
    python scripts/eval/summarize_detect.py results/detect_stress_200.jsonl
    python scripts/eval/summarize_detect.py results/detect_stress_200.jsonl --json results/summary.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter


def load_results(infile: Path) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    
    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping invalid JSONL line: {e}", file=sys.stderr)
    
    return results


def compute_filter_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute filter metrics (should_filter)."""
    tp = 0  # predicted filter=true AND expected filter=true
    fp = 0  # predicted filter=true AND expected filter=false
    fn = 0  # predicted filter=false AND expected filter=true
    tn = 0  # predicted filter=false AND expected filter=false
    
    for r in results:
        if 'error' in r:
            continue
        
        expected_filter = r.get('expected_should_filter')
        if expected_filter is None:
            continue
        
        predicted_filter = r.get('is_personal_or_subjective', False)
        
        if predicted_filter and expected_filter:
            tp += 1
        elif predicted_filter and not expected_filter:
            fp += 1
        elif not predicted_filter and expected_filter:
            fn += 1
        else:
            tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def compute_detection_coverage(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute detection coverage (should_detect)."""
    should_detect = [r for r in results if r.get('expected_should_detect') is True and 'error' not in r]
    
    detected = 0
    for r in should_detect:
        parsed_claim = r.get('parsed_claim', {})
        # Consider detected if subject exists OR category is not None
        if parsed_claim.get('subject') or r.get('category'):
            detected += 1
    
    recall = detected / len(should_detect) if should_detect else 0.0
    
    return {
        'total_should_detect': len(should_detect),
        'detected': detected,
        'recall': recall,
    }


def compute_relation_type_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute relation-type accuracy."""
    with_expected = [r for r in results if r.get('expected_relation_type') and 'error' not in r]
    
    exact_match = 0
    coarse_match = 0  # e.g., "geo_location" matches "geo_city"
    
    for r in with_expected:
        expected = r.get('expected_relation_type', '').lower()
        predicted = r.get('relation_type', '').lower()
        
        if expected == predicted:
            exact_match += 1
        elif expected in predicted or predicted in expected:
            coarse_match += 1
    
    exact_accuracy = exact_match / len(with_expected) if with_expected else 0.0
    coarse_accuracy = (exact_match + coarse_match) / len(with_expected) if with_expected else 0.0
    
    return {
        'total_with_expected': len(with_expected),
        'exact_match': exact_match,
        'coarse_match': coarse_match,
        'exact_accuracy': exact_accuracy,
        'coarse_accuracy': coarse_accuracy,
    }


def compute_decomposition_sanity(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute decomposition sanity checks."""
    comparison_keywords = ['länger', 'größer', 'kleiner', 'mehr', 'weniger', 'höher', 'niedriger', 'als']
    definition_keywords = ['hat', 'ist', 'besteht']
    
    comparison_cases = []
    definition_cases = []
    
    for r in results:
        if 'error' in r:
            continue
        
        text_lower = r.get('text', '').lower()
        
        if any(kw in text_lower for kw in comparison_keywords):
            comparison_cases.append(r)
        
        if any(kw in text_lower for kw in definition_keywords):
            definition_cases.append(r)
    
    # Comparison: should have subclaims
    comparison_with_subclaims = sum(1 for r in comparison_cases if len(r.get('subclaims', [])) > 0)
    comparison_subclaim_rate = comparison_with_subclaims / len(comparison_cases) if comparison_cases else 0.0
    
    # Definition: should NOT have subclaims (guard should prevent)
    definition_without_subclaims = sum(1 for r in definition_cases if len(r.get('subclaims', [])) == 0 and not r.get('is_hard_claim', False))
    definition_guard_rate = definition_without_subclaims / len(definition_cases) if definition_cases else 0.0
    
    return {
        'comparison_cases': len(comparison_cases),
        'comparison_with_subclaims': comparison_with_subclaims,
        'comparison_subclaim_rate': comparison_subclaim_rate,
        'definition_cases': len(definition_cases),
        'definition_without_subclaims': definition_without_subclaims,
        'definition_guard_rate': definition_guard_rate,
    }


def print_summary(stats: Dict[str, Any]) -> None:
    """Print formatted summary."""
    print("=" * 80)
    print("CLAIM DETECTOR EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal Cases: {stats['total']}")
    
    # Filter metrics
    filter_metrics = stats['filter_metrics']
    print(f"\n{'=' * 80}")
    print("FILTER METRICS (should_filter)")
    print("=" * 80)
    print(f"True Positives (TP):  {filter_metrics['tp']:4d} (predicted filter=true, expected filter=true)")
    print(f"False Positives (FP): {filter_metrics['fp']:4d} (predicted filter=true, expected filter=false)")
    print(f"False Negatives (FN): {filter_metrics['fn']:4d} (predicted filter=false, expected filter=true)")
    print(f"True Negatives (TN):  {filter_metrics['tn']:4d} (predicted filter=false, expected filter=false)")
    print(f"\nPrecision: {filter_metrics['precision']:.2%}")
    print(f"Recall:    {filter_metrics['recall']:.2%}")
    print(f"F1-Score:  {filter_metrics['f1']:.2%}")
    
    # Detection coverage
    detection = stats['detection_coverage']
    print(f"\n{'=' * 80}")
    print("DETECTION COVERAGE (should_detect)")
    print("=" * 80)
    print(f"Total should_detect cases: {detection['total_should_detect']}")
    print(f"Detected (subject OR category exists): {detection['detected']}")
    print(f"Recall: {detection['recall']:.2%}")
    
    # Relation type
    relation = stats['relation_type_accuracy']
    print(f"\n{'=' * 80}")
    print("RELATION TYPE ACCURACY")
    print("=" * 80)
    print(f"Total with expected_relation_type: {relation['total_with_expected']}")
    print(f"Exact match: {relation['exact_match']} ({relation['exact_accuracy']:.2%})")
    print(f"Coarse match: {relation['coarse_match']} (additional)")
    print(f"Coarse accuracy: {relation['coarse_accuracy']:.2%}")
    
    # Decomposition sanity
    decomp = stats['decomposition_sanity']
    print(f"\n{'=' * 80}")
    print("DECOMPOSITION SANITY")
    print("=" * 80)
    print(f"Comparison cases (with keywords): {decomp['comparison_cases']}")
    print(f"  With subclaims: {decomp['comparison_with_subclaims']} ({decomp['comparison_subclaim_rate']:.2%})")
    print(f"Definition cases (with keywords): {decomp['definition_cases']}")
    print(f"  Without subclaims (guard working): {decomp['definition_without_subclaims']} ({decomp['definition_guard_rate']:.2%})")
    
    # Latency
    latencies = stats['latencies']
    print(f"\n{'=' * 80}")
    print("LATENCY STATISTICS (ms)")
    print("=" * 80)
    print(f"Mean:  {latencies['mean']:8.1f} ms")
    print(f"p50:   {latencies['p50']:8.1f} ms")
    print(f"p90:   {latencies['p90']:8.1f} ms")
    print(f"p95:   {latencies['p95']:8.1f} ms")
    
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Summarize claim detection evaluation results")
    parser.add_argument("infile", type=Path, help="Input JSONL file with detection results")
    parser.add_argument("--json", type=Path, help="Optional: save summary as JSON")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.infile)
    
    # Compute statistics
    filter_metrics = compute_filter_metrics(results)
    detection_coverage = compute_detection_coverage(results)
    relation_type_accuracy = compute_relation_type_accuracy(results)
    decomposition_sanity = compute_decomposition_sanity(results)
    
    # Latency statistics
    latencies = [r.get('latency_ms', 0) for r in results if r.get('latency_ms') is not None]
    latencies_sorted = sorted(latencies) if latencies else []
    
    def percentile(data, p):
        if not data:
            return 0.0
        index = int(len(data) * p / 100)
        return data[min(index, len(data) - 1)]
    
    latency_stats = {
        'mean': sum(latencies) / len(latencies) if latencies else 0.0,
        'p50': percentile(latencies_sorted, 50),
        'p90': percentile(latencies_sorted, 90),
        'p95': percentile(latencies_sorted, 95),
    }
    
    stats = {
        'total': len(results),
        'filter_metrics': filter_metrics,
        'detection_coverage': detection_coverage,
        'relation_type_accuracy': relation_type_accuracy,
        'decomposition_sanity': decomposition_sanity,
        'latencies': latency_stats,
    }
    
    # Print summary
    print_summary(stats)
    
    # Save JSON if requested
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
