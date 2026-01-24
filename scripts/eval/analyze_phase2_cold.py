"""
Phase 2.1 Cold Run Analysis

Analyzes seed1000_cold.jsonl and provides detailed metrics.
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

def analyze_cold_run(results_file: Path) -> Dict[str, Any]:
    """Analyze cold run results."""
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
    # This means: exclude cases where predicted = NOT_ENOUGH_INFO
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

def print_report(stats: Dict[str, Any]) -> None:
    """Print formatted report."""
    print("=" * 80)
    print("PHASE 2.1: COLD RUN SUMMARY")
    print("=" * 80)
    print(f"\nTotal Claims Processed: {stats['total']}")
    
    # Calculate non-NEI count
    non_nei_total = sum(1 for exp in stats['expected_labels'].values() if exp != 'NOT_ENOUGH_INFO')
    non_nei_total = stats['total'] - stats['expected_labels'].get('NOT_ENOUGH_INFO', 0)
    
    print(f"\n{'=' * 80}")
    print("ACCURACY METRICS")
    print("=" * 80)
    print(f"Overall Accuracy: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    print(f"Accuracy (excluding NOT_ENOUGH_INFO expected): {stats['accuracy_excl_nei']:.2f}% ({stats['correct_non_nei']}/{non_nei_total})")
    
    print(f"\n{'=' * 80}")
    print("ACCURACY (EXCLUDING NOT_ENOUGH_INFO PREDICTIONS)")
    print("=" * 80)
    print("Only counting cases where system made a definitive decision (SUPPORTED or REFUTED):")
    print(f"\nOverall Accuracy: {stats['accuracy_verifiable']:.2f}% ({stats['correct_verifiable']}/{stats['verifiable_total']})")
    print(f"  (Excludes {stats['total'] - stats['verifiable_total']} cases where predicted = NOT_ENOUGH_INFO)")
    
    print(f"\nSUPPORTED Accuracy: {stats['supported_accuracy_verifiable']:.2f}% ({stats['supported_correct_verifiable']}/{stats['supported_verifiable_total']})")
    print(f"  (Excludes {stats['supported_to_nei']} cases where SUPPORTED -> NOT_ENOUGH_INFO)")
    
    print(f"\nREFUTED Accuracy: {stats['refuted_accuracy_verifiable']:.2f}% ({stats['refuted_correct_verifiable']}/{stats['refuted_verifiable_total']})")
    print(f"  (Excludes {stats['refuted_to_nei']} cases where REFUTED -> NOT_ENOUGH_INFO)")
    
    print(f"\nInterpretation:")
    print(f"  - When system makes a definitive decision, it is correct in {stats['accuracy_verifiable']:.1f}% of cases")
    print(f"  - {stats['supported_to_nei'] + stats['refuted_to_nei']} claims ({((stats['supported_to_nei'] + stats['refuted_to_nei']) / stats['total'] * 100):.1f}%) were classified as NOT_ENOUGH_INFO")
    print(f"    despite being expected as SUPPORTED/REFUTED (conservative behavior)")
    
    print(f"\n{'=' * 80}")
    print("LABEL DISTRIBUTION")
    print("=" * 80)
    print("\nPredicted Labels:")
    for label, count in sorted(stats['predicted_labels'].items()):
        pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {label:20s}: {count:4d} ({pct:5.1f}%)")
    
    print("\nExpected Labels:")
    for label, count in sorted(stats['expected_labels'].items()):
        pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {label:20s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n{'=' * 80}")
    print("LATENCY STATISTICS (ms)")
    print("=" * 80)
    print(f"  Mean:  {stats['latency']['mean']:8.1f} ms")
    print(f"  p50:   {stats['latency']['p50']:8.1f} ms")
    print(f"  p90:   {stats['latency']['p90']:8.1f} ms")
    print(f"  p95:   {stats['latency']['p95']:8.1f} ms")
    
    print(f"\n{'=' * 80}")
    print("PATH DISTRIBUTION")
    print("=" * 80)
    for path, count in sorted(stats['paths'].items()):
        pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {path:40s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n{'=' * 80}")
    print("CACHE BEHAVIOR")
    print("=" * 80)
    print(f"Cache Hits: {stats['cache_hits']}/{stats['total']}")
    print(f"Cache Hit Rate: {stats['cache_hit_rate']:.2f}%")
    print(f"Expected: ~0% (cold run)")
    
    print(f"\n{'=' * 80}")
    print("CONFUSION MATRIX (Top 20)")
    print("=" * 80)
    sorted_confusion = sorted(stats['confusion'].items(), key=lambda x: x[1], reverse=True)
    for key, count in sorted_confusion[:20]:
        print(f"  {key:40s}: {count:4d}")
    
    print(f"\n{'=' * 80}")
    print("STEP 2 COMPLETE")
    print("=" * 80)
    print()

if __name__ == "__main__":
    results_file = Path("results/seed1000_cold.jsonl")
    stats = analyze_cold_run(results_file)
    print_report(stats)
