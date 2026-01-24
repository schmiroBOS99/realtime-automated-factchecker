"""
Summarize batch fact-checking evaluation results.

Computes accuracy, label distribution, and latency statistics.

Usage:
    python scripts/eval/summarize.py results/results.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List


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


def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy metrics."""
    total_with_expected = 0
    correct = 0
    by_label = {}
    
    for result in results:
        if 'error' in result:
            continue
        
        expected = result.get('expected')
        if not expected:
            continue
        
        predicted = result.get('label', '')
        if not predicted:
            continue
        
        total_with_expected += 1
        
        # Normalize labels (uppercase)
        expected_upper = expected.upper()
        predicted_upper = predicted.upper()
        
        is_correct = expected_upper == predicted_upper
        
        if expected_upper not in by_label:
            by_label[expected_upper] = {'total': 0, 'correct': 0}
        
        by_label[expected_upper]['total'] += 1
        if is_correct:
            correct += 1
            by_label[expected_upper]['correct'] += 1
    
    accuracy = correct / total_with_expected if total_with_expected > 0 else 0.0
    
    # Compute per-label accuracy
    per_label_accuracy = {}
    for label, counts in by_label.items():
        per_label_accuracy[label] = counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
    
    return {
        'total_with_expected': total_with_expected,
        'correct': correct,
        'accuracy': accuracy,
        'per_label_accuracy': per_label_accuracy,
        'per_label_counts': {k: v['total'] for k, v in by_label.items()},
    }


def compute_label_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute label distribution."""
    labels = []
    
    for result in results:
        if 'error' in result:
            labels.append('ERROR')
        else:
            label = result.get('label', 'UNKNOWN')
            labels.append(label.upper())
    
    return dict(Counter(labels))


def compute_latency_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute latency statistics (p50, p90, p95, mean, min, max)."""
    latencies = []
    
    for result in results:
        latency_ms = result.get('latency_ms')
        if latency_ms is not None and isinstance(latency_ms, (int, float)):
            latencies.append(latency_ms)
    
    if not latencies:
        return {}
    
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    
    def percentile(p: float) -> float:
        """Compute p-th percentile."""
        idx = int((p / 100.0) * n)
        if idx >= n:
            idx = n - 1
        return latencies_sorted[idx]
    
    return {
        'count': n,
        'mean': sum(latencies) / n,
        'min': latencies_sorted[0],
        'max': latencies_sorted[-1],
        'p50': percentile(50),
        'p90': percentile(90),
        'p95': percentile(95),
        'p99': percentile(99),
    }


def compute_cache_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute cache hit statistics."""
    total = 0
    cache_hits = 0
    cache_misses = 0
    
    for result in results:
        if 'error' in result:
            continue
        
        if 'cache_hit' in result:
            total += 1
            if result['cache_hit']:
                cache_hits += 1
            else:
                cache_misses += 1
    
    cache_hit_rate = cache_hits / total if total > 0 else 0.0
    
    return {
        'total': total,
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'cache_hit_rate': cache_hit_rate,
    }


def compute_path_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute path distribution (e.g., WIKIPEDIA_RETRIEVAL, WIKIDATA_FAST_PATH)."""
    paths = []
    
    for result in results:
        if 'error' in result:
            paths.append('ERROR')
        else:
            path = result.get('path', 'unknown')
            paths.append(path)
    
    return dict(Counter(paths))


def main():
    parser = argparse.ArgumentParser(
        description="Summarize batch fact-checking evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'infile',
        type=Path,
        help='Input JSONL results file',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output summary as JSON',
    )
    
    args = parser.parse_args()
    
    if not args.infile.exists():
        print(f"ERROR: Input file not found: {args.infile}", file=sys.stderr)
        sys.exit(1)
    
    # Load results
    results = load_results(args.infile)
    total_results = len(results)
    
    if total_results == 0:
        print("ERROR: No results to summarize", file=sys.stderr)
        sys.exit(1)
    
    # Compute metrics
    accuracy_metrics = compute_accuracy(results)
    label_dist = compute_label_distribution(results)
    latency_stats = compute_latency_stats(results)
    cache_stats = compute_cache_stats(results)
    path_dist = compute_path_distribution(results)
    
    # Count errors
    error_count = sum(1 for r in results if 'error' in r)
    
    # Build summary
    summary = {
        'total_results': total_results,
        'error_count': error_count,
        'accuracy': accuracy_metrics,
        'label_distribution': label_dist,
        'latency_stats_ms': latency_stats,
        'cache_stats': cache_stats,
        'path_distribution': path_dist,
    }
    
    # Output
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        # Human-readable output
        print("=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\nTotal Results: {total_results}")
        print(f"Errors: {error_count}")
        
        print("\n" + "-" * 70)
        print("ACCURACY")
        print("-" * 70)
        if accuracy_metrics['total_with_expected'] > 0:
            print(f"Overall Accuracy: {accuracy_metrics['accuracy']:.2%}")
            print(f"  ({accuracy_metrics['correct']}/{accuracy_metrics['total_with_expected']} correct)")
            print("\nPer-Label Accuracy:")
            for label, acc in accuracy_metrics['per_label_accuracy'].items():
                count = accuracy_metrics['per_label_counts'][label]
                print(f"  {label}: {acc:.2%} ({int(acc * count)}/{count})")
        else:
            print("No expected labels found in results")
        
        print("\n" + "-" * 70)
        print("LABEL DISTRIBUTION")
        print("-" * 70)
        for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_results * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        print("\n" + "-" * 70)
        print("LATENCY STATISTICS (ms)")
        print("-" * 70)
        if latency_stats:
            print(f"  Count: {latency_stats['count']}")
            print(f"  Mean: {latency_stats['mean']:.1f}")
            print(f"  Min: {latency_stats['min']:.1f}")
            print(f"  Max: {latency_stats['max']:.1f}")
            print(f"  p50: {latency_stats['p50']:.1f}")
            print(f"  p90: {latency_stats['p90']:.1f}")
            print(f"  p95: {latency_stats['p95']:.1f}")
            print(f"  p99: {latency_stats['p99']:.1f}")
        else:
            print("  No latency data available")
        
        print("\n" + "-" * 70)
        print("CACHE STATISTICS")
        print("-" * 70)
        if cache_stats['total'] > 0:
            print(f"  Total: {cache_stats['total']}")
            print(f"  Hits: {cache_stats['cache_hits']} ({cache_stats['cache_hit_rate']:.2%})")
            print(f"  Misses: {cache_stats['cache_misses']} ({1 - cache_stats['cache_hit_rate']:.2%})")
        else:
            print("  No cache data available")
        
        print("\n" + "-" * 70)
        print("PATH DISTRIBUTION")
        print("-" * 70)
        for path, count in sorted(path_dist.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_results * 100
            print(f"  {path}: {count} ({pct:.1f}%)")
        
        print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
