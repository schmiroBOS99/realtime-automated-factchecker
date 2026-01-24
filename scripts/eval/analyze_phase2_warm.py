"""
Phase 2.2 Warm Run Analysis

Analyzes seed1000_warm.jsonl and compares with cold run.
"""
import json
import statistics
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

def percentile(data: List[float], p: float) -> float:
    """Calculate percentile."""
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]

def analyze_run(results_file: Path) -> Dict[str, Any]:
    """Analyze run results."""
    results = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    
    total = len(results)
    
    # Accuracy metrics
    correct = sum(1 for r in results if r.get('expected') and r.get('label') == r.get('expected'))
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Accuracy excluding NOT_ENOUGH_INFO (traditional: all non-NEI cases)
    non_nei = [r for r in results if r.get('expected') != 'NOT_ENOUGH_INFO']
    correct_non_nei = sum(1 for r in non_nei if r.get('label') == r.get('expected'))
    accuracy_excl_nei = (correct_non_nei / len(non_nei) * 100) if non_nei else 0
    
    # Accuracy excluding NOT_ENOUGH_INFO predictions (only count SUPPORTED/REFUTED predictions)
    verifiable = [
        r for r in results
        if r.get('expected') in ['SUPPORTED', 'REFUTED']
        and r.get('label') in ['SUPPORTED', 'REFUTED']
    ]
    correct_verifiable = sum(1 for r in verifiable if r.get('label') == r.get('expected'))
    accuracy_verifiable = (correct_verifiable / len(verifiable) * 100) if verifiable else 0
    
    # Per-label accuracy (excluding NOT_ENOUGH_INFO predictions)
    supported_verifiable = [r for r in results if r.get('expected') == 'SUPPORTED' and r.get('label') in ['SUPPORTED', 'REFUTED']]
    supported_correct_verifiable = sum(1 for r in supported_verifiable if r.get('label') == 'SUPPORTED')
    supported_accuracy_verifiable = (supported_correct_verifiable / len(supported_verifiable) * 100) if supported_verifiable else 0
    
    refuted_verifiable = [r for r in results if r.get('expected') == 'REFUTED' and r.get('label') in ['SUPPORTED', 'REFUTED']]
    refuted_correct_verifiable = sum(1 for r in refuted_verifiable if r.get('label') == 'REFUTED')
    refuted_accuracy_verifiable = (refuted_correct_verifiable / len(refuted_verifiable) * 100) if refuted_verifiable else 0
    
    # Count excluded cases
    supported_to_nei = sum(1 for r in results if r.get('expected') == 'SUPPORTED' and r.get('label') == 'NOT_ENOUGH_INFO')
    refuted_to_nei = sum(1 for r in results if r.get('expected') == 'REFUTED' and r.get('label') == 'NOT_ENOUGH_INFO')
    
    # Label distribution
    predicted_labels = Counter(r.get('label', 'UNKNOWN') for r in results)
    expected_labels = Counter(r.get('expected', 'UNKNOWN') for r in results)
    
    # Latency statistics
    latencies = [r.get('latency_ms', 0) for r in results if r.get('latency_ms') is not None]
    mean_latency = statistics.mean(latencies) if latencies else 0
    p50_latency = percentile(latencies, 50) if latencies else 0
    p90_latency = percentile(latencies, 90) if latencies else 0
    p95_latency = percentile(latencies, 95) if latencies else 0
    
    # Path distribution
    paths = Counter(r.get('path', 'UNKNOWN') for r in results)
    
    # Cache hit rate
    cache_hits = sum(1 for r in results if r.get('cache_hit') is True or r.get('path') == 'CLAIM_CACHE')
    cache_hit_rate = (cache_hits / total * 100) if total > 0 else 0
    
    # Confusion matrix
    confusion = {}
    for r in results:
        expected = r.get('expected', 'UNKNOWN')
        predicted = r.get('label', 'UNKNOWN')
        key = f"{expected} -> {predicted}"
        confusion[key] = confusion.get(key, 0) + 1
    
    return {
        'total': total,
        'accuracy': accuracy,
        'accuracy_excl_nei': accuracy_excl_nei,
        'correct': correct,
        'correct_non_nei': correct_non_nei,
        'accuracy_verifiable': accuracy_verifiable,
        'correct_verifiable': correct_verifiable,
        'supported_accuracy_verifiable': supported_accuracy_verifiable,
        'supported_correct_verifiable': supported_correct_verifiable,
        'refuted_accuracy_verifiable': refuted_accuracy_verifiable,
        'refuted_correct_verifiable': refuted_correct_verifiable,
        'verifiable_total': len(verifiable),
        'supported_verifiable_total': len(supported_verifiable),
        'refuted_verifiable_total': len(refuted_verifiable),
        'supported_to_nei': supported_to_nei,
        'refuted_to_nei': refuted_to_nei,
        'predicted_labels': dict(predicted_labels),
        'expected_labels': dict(expected_labels),
        'latency': {
            'mean': mean_latency,
            'p50': p50_latency,
            'p90': p90_latency,
            'p95': p95_latency,
        },
        'paths': dict(paths),
        'cache_hit_rate': cache_hit_rate,
        'cache_hits': cache_hits,
        'confusion': confusion,
    }

def compare_runs(cold_stats: Dict[str, Any], warm_stats: Dict[str, Any]) -> None:
    """Compare cold and warm run statistics."""
    print("=" * 80)
    print("PHASE 2.2: WARM RUN vs COLD RUN COMPARISON")
    print("=" * 80)
    
    print(f"\n{'=' * 80}")
    print("ACCURACY COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<50} {'Cold':<15} {'Warm':<15} {'Delta':<15}")
    print("-" * 80)
    print(f"{'Overall Accuracy':<50} {cold_stats['accuracy']:>6.2f}%        {warm_stats['accuracy']:>6.2f}%        {warm_stats['accuracy'] - cold_stats['accuracy']:>+6.2f}%")
    print(f"{'Accuracy (excl NEI expected)':<50} {cold_stats['accuracy_excl_nei']:>6.2f}%        {warm_stats['accuracy_excl_nei']:>6.2f}%        {warm_stats['accuracy_excl_nei'] - cold_stats['accuracy_excl_nei']:>+6.2f}%")
    print(f"{'Accuracy (excl NEI predictions)':<50} {cold_stats['accuracy_verifiable']:>6.2f}%        {warm_stats['accuracy_verifiable']:>6.2f}%        {warm_stats['accuracy_verifiable'] - cold_stats['accuracy_verifiable']:>+6.2f}%")
    print(f"{'SUPPORTED Accuracy (verifiable)':<50} {cold_stats['supported_accuracy_verifiable']:>6.2f}%        {warm_stats['supported_accuracy_verifiable']:>6.2f}%        {warm_stats['supported_accuracy_verifiable'] - cold_stats['supported_accuracy_verifiable']:>+6.2f}%")
    print(f"{'REFUTED Accuracy (verifiable)':<50} {cold_stats['refuted_accuracy_verifiable']:>6.2f}%        {warm_stats['refuted_accuracy_verifiable']:>6.2f}%        {warm_stats['refuted_accuracy_verifiable'] - cold_stats['refuted_accuracy_verifiable']:>+6.2f}%")
    
    print(f"\n{'=' * 80}")
    print("LATENCY COMPARISON (ms)")
    print("=" * 80)
    print(f"{'Metric':<20} {'Cold':<15} {'Warm':<15} {'Speedup':<15}")
    print("-" * 80)
    print(f"{'Mean':<20} {cold_stats['latency']['mean']:>8.1f}        {warm_stats['latency']['mean']:>8.1f}        {cold_stats['latency']['mean'] / warm_stats['latency']['mean']:>6.2f}x" if warm_stats['latency']['mean'] > 0 else f"{'Mean':<20} {cold_stats['latency']['mean']:>8.1f}        {warm_stats['latency']['mean']:>8.1f}        N/A")
    print(f"{'p50':<20} {cold_stats['latency']['p50']:>8.1f}        {warm_stats['latency']['p50']:>8.1f}        {cold_stats['latency']['p50'] / warm_stats['latency']['p50']:>6.2f}x" if warm_stats['latency']['p50'] > 0 else f"{'p50':<20} {cold_stats['latency']['p50']:>8.1f}        {warm_stats['latency']['p50']:>8.1f}        N/A")
    print(f"{'p90':<20} {cold_stats['latency']['p90']:>8.1f}        {warm_stats['latency']['p90']:>8.1f}        {cold_stats['latency']['p90'] / warm_stats['latency']['p90']:>6.2f}x" if warm_stats['latency']['p90'] > 0 else f"{'p90':<20} {cold_stats['latency']['p90']:>8.1f}        {warm_stats['latency']['p90']:>8.1f}        N/A")
    print(f"{'p95':<20} {cold_stats['latency']['p95']:>8.1f}        {warm_stats['latency']['p95']:>8.1f}        {cold_stats['latency']['p95'] / warm_stats['latency']['p95']:>6.2f}x" if warm_stats['latency']['p95'] > 0 else f"{'p95':<20} {cold_stats['latency']['p95']:>8.1f}        {warm_stats['latency']['p95']:>8.1f}        N/A")
    
    print(f"\n{'=' * 80}")
    print("CACHE BEHAVIOR")
    print("=" * 80)
    print(f"Cold Run Cache Hits: {cold_stats['cache_hits']}/{cold_stats['total']} ({cold_stats['cache_hit_rate']:.2f}%)")
    print(f"Warm Run Cache Hits: {warm_stats['cache_hits']}/{warm_stats['total']} ({warm_stats['cache_hit_rate']:.2f}%)")
    print(f"Cache Hit Rate Increase: {warm_stats['cache_hit_rate'] - cold_stats['cache_hit_rate']:+.2f}%")
    
    print(f"\n{'=' * 80}")
    print("PATH DISTRIBUTION COMPARISON")
    print("=" * 80)
    all_paths = set(cold_stats['paths'].keys()) | set(warm_stats['paths'].keys())
    print(f"{'Path':<50} {'Cold':<15} {'Warm':<15} {'Delta':<15}")
    print("-" * 80)
    for path in sorted(all_paths):
        cold_count = cold_stats['paths'].get(path, 0)
        warm_count = warm_stats['paths'].get(path, 0)
        print(f"{path:<50} {cold_count:>6d}        {warm_count:>6d}        {warm_count - cold_count:>+6d}")
    
    print(f"\n{'=' * 80}")
    print("LABEL DISTRIBUTION COMPARISON")
    print("=" * 80)
    print("\nPredicted Labels:")
    all_labels = set(cold_stats['predicted_labels'].keys()) | set(warm_stats['predicted_labels'].keys())
    print(f"{'Label':<20} {'Cold':<15} {'Warm':<15} {'Delta':<15}")
    print("-" * 80)
    for label in sorted(all_labels):
        cold_count = cold_stats['predicted_labels'].get(label, 0)
        warm_count = warm_stats['predicted_labels'].get(label, 0)
        print(f"{label:<20} {cold_count:>6d}        {warm_count:>6d}        {warm_count - cold_count:>+6d}")
    
    print(f"\n{'=' * 80}")
    print("STEP 3 COMPLETE")
    print("=" * 80)
    print()

if __name__ == "__main__":
    cold_file = Path("results/seed1000_cold.jsonl")
    warm_file = Path("results/seed1000_warm.jsonl")
    
    print("Loading cold run results...")
    cold_stats = analyze_run(cold_file)
    
    print("Loading warm run results...")
    warm_stats = analyze_run(warm_file)
    
    compare_runs(cold_stats, warm_stats)
