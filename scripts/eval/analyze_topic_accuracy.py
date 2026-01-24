"""
Analyze accuracy per topic for seed1000 evaluation.

Computes:
1. Standard accuracy (all labels)
2. Accuracy with NOT_ENOUGH_INFO as neutral (only SUPPORTED/REFUTED count)
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter


def load_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def compute_accuracy_per_topic(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute accuracy per topic (standard and with NEI as neutral)."""
    topic_stats = defaultdict(lambda: {
        'total': 0,
        'correct_standard': 0,
        'correct_nei_neutral': 0,
        'total_nei_neutral': 0,  # Only SUPPORTED/REFUTED cases
        'expected_distribution': Counter(),
        'predicted_distribution': Counter(),
        'confusion_matrix': defaultdict(lambda: defaultdict(int)),
    })
    
    for r in results:
        topic = r.get('topic', 'unknown')
        expected = r.get('expected', '')
        predicted = r.get('label', '')
        
        topic_stats[topic]['total'] += 1
        topic_stats[topic]['expected_distribution'][expected] += 1
        topic_stats[topic]['predicted_distribution'][predicted] += 1
        topic_stats[topic]['confusion_matrix'][expected][predicted] += 1
        
        # Standard accuracy: exact match
        if expected == predicted:
            topic_stats[topic]['correct_standard'] += 1
        
        # NEI as neutral: only count SUPPORTED/REFUTED expected cases
        # NEI predicted is IGNORED (not counted as error, not counted in total)
        # Only count cases where predicted is SUPPORTED or REFUTED (system made a decision)
        if expected in {'SUPPORTED', 'REFUTED'} and predicted in {'SUPPORTED', 'REFUTED'}:
            topic_stats[topic]['total_nei_neutral'] += 1
            # Correct if predicted matches expected
            if predicted == expected:
                topic_stats[topic]['correct_nei_neutral'] += 1
    
    # Calculate accuracies
    topic_accuracies = {}
    for topic, stats in topic_stats.items():
        standard_accuracy = (stats['correct_standard'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
        nei_neutral_accuracy = (stats['correct_nei_neutral'] / stats['total_nei_neutral'] * 100) if stats['total_nei_neutral'] > 0 else 0.0
        
        topic_accuracies[topic] = {
            'total': stats['total'],
            'standard_accuracy': standard_accuracy,
            'nei_neutral_accuracy': nei_neutral_accuracy,
            'correct_standard': stats['correct_standard'],
            'correct_nei_neutral': stats['correct_nei_neutral'],
            'total_nei_neutral': stats['total_nei_neutral'],
            'expected_distribution': dict(stats['expected_distribution']),
            'predicted_distribution': dict(stats['predicted_distribution']),
            'confusion_matrix': {k: dict(v) for k, v in stats['confusion_matrix'].items()},
        }
    
    return topic_accuracies


def print_accuracy_table(topic_accuracies: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted accuracy table."""
    print("=" * 100)
    print("ACCURACY PER TOPIC")
    print("=" * 100)
    print(f"\n{'Topic':<30} {'Total':<8} {'Standard':<12} {'NEI=Neutral':<15} {'Correct (Std)':<15} {'Correct (NEI)':<15}")
    print("-" * 100)
    
    # Sort by topic name
    sorted_topics = sorted(topic_accuracies.items())
    
    for topic, stats in sorted_topics:
        print(f"{topic:<30} {stats['total']:<8} {stats['standard_accuracy']:>6.2f}%     "
              f"{stats['nei_neutral_accuracy']:>6.2f}%        "
              f"{stats['correct_standard']:<15} {stats['correct_nei_neutral']:<15}")
    
    # Overall statistics
    total_all = sum(s['total'] for s in topic_accuracies.values())
    correct_standard_all = sum(s['correct_standard'] for s in topic_accuracies.values())
    correct_nei_all = sum(s['correct_nei_neutral'] for s in topic_accuracies.values())
    total_nei_all = sum(s['total_nei_neutral'] for s in topic_accuracies.values())
    
    overall_standard = (correct_standard_all / total_all * 100) if total_all > 0 else 0.0
    overall_nei = (correct_nei_all / total_nei_all * 100) if total_nei_all > 0 else 0.0
    
    print("-" * 100)
    print(f"{'OVERALL':<30} {total_all:<8} {overall_standard:>6.2f}%     "
          f"{overall_nei:>6.2f}%        "
          f"{correct_standard_all:<15} {correct_nei_all:<15}")
    print("=" * 100)


def print_detailed_stats(topic_accuracies: Dict[str, Dict[str, Any]]) -> None:
    """Print detailed statistics per topic."""
    print("\n" + "=" * 100)
    print("DETAILED STATISTICS PER TOPIC")
    print("=" * 100)
    
    sorted_topics = sorted(topic_accuracies.items())
    
    for topic, stats in sorted_topics:
        print(f"\n{'=' * 100}")
        print(f"TOPIC: {topic}")
        print(f"{'=' * 100}")
        print(f"Total cases: {stats['total']}")
        print(f"Standard Accuracy: {stats['standard_accuracy']:.2f}% ({stats['correct_standard']}/{stats['total']})")
        print(f"NEI=Neutral Accuracy: {stats['nei_neutral_accuracy']:.2f}% ({stats['correct_nei_neutral']}/{stats['total_nei_neutral']})")
        
        print(f"\nExpected Distribution:")
        for label, count in sorted(stats['expected_distribution'].items()):
            pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0.0
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        print(f"\nPredicted Distribution:")
        for label, count in sorted(stats['predicted_distribution'].items()):
            pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0.0
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        # Confusion matrix
        print(f"\nConfusion Matrix (Expected -> Predicted):")
        expected_labels = sorted(set(stats['expected_distribution'].keys()) | set(stats['predicted_distribution'].keys()))
        print(f"  {'Expected\\Predicted':<20}", end="")
        for pred_label in expected_labels:
            print(f"{pred_label:<15}", end="")
        print()
        
        for exp_label in expected_labels:
            print(f"  {exp_label:<20}", end="")
            for pred_label in expected_labels:
                count = stats['confusion_matrix'].get(exp_label, {}).get(pred_label, 0)
                print(f"{count:<15}", end="")
            print()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze accuracy per topic")
    parser.add_argument("--infile", type=Path, default=Path("results/seed1000_cold.jsonl"), help="Input JSONL file")
    parser.add_argument("--json", type=Path, help="Optional: save results as JSON")
    parser.add_argument("--detailed", action="store_true", help="Print detailed statistics")
    
    args = parser.parse_args()
    
    results = load_results(args.infile)
    topic_accuracies = compute_accuracy_per_topic(results)
    
    print_accuracy_table(topic_accuracies)
    
    if args.detailed:
        print_detailed_stats(topic_accuracies)
    
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(topic_accuracies, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.json}")


if __name__ == "__main__":
    main()
