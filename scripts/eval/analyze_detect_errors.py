"""
Analyze errors from claim detector evaluation.

Identifies false negatives (should be filtered but weren't) and false positives.
"""
import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter


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


def analyze_errors(results: List[Dict[str, Any]]) -> None:
    """Analyze filter errors."""
    false_negatives = []  # should_filter=true but predicted=false
    false_positives = []  # should_filter=false but predicted=true
    
    for r in results:
        if 'error' in r:
            continue
        
        expected_filter = r.get('expected_should_filter')
        predicted_filter = r.get('is_personal_or_subjective', False)
        
        if expected_filter is True and not predicted_filter:
            false_negatives.append(r)
        elif expected_filter is False and predicted_filter:
            false_positives.append(r)
    
    print("=" * 80)
    print("CLAIM DETECTOR ERROR ANALYSIS")
    print("=" * 80)
    
    print(f"\nFalse Negatives (should_filter=true but predicted=false): {len(false_negatives)}")
    print("=" * 80)
    if false_negatives:
        print("\nThese cases should have been filtered but weren't:")
        for i, fn in enumerate(false_negatives[:20], 1):  # Show first 20
            print(f"\n{i}. [{fn['id']}] {fn['text']}")
            print(f"   Notes: {fn.get('notes', 'N/A')}")
            print(f"   Category: {fn.get('category', 'N/A')}")
            print(f"   Relation Type: {fn.get('relation_type', 'N/A')}")
        
        if len(false_negatives) > 20:
            print(f"\n... and {len(false_negatives) - 20} more")
        
        # Group by pattern
        print(f"\n{'=' * 80}")
        print("False Negatives by Pattern")
        print("=" * 80)
        patterns = Counter()
        for fn in false_negatives:
            text_lower = fn.get('text', '').lower()
            if '?' in fn.get('text', ''):
                patterns['Question'] += 1
            elif any(p in text_lower for p in ['ich', 'du', 'deine', 'dein']):
                patterns['Personal pronoun'] += 1
            elif any(p in text_lower for p in ['beste', 'schönste', 'wunderschön', 'toll']):
                patterns['Subjective adjective'] += 1
            elif any(p in text_lower for p in ['er war', 'sie ist', 'es ist']):
                patterns['Missing context (pronoun)'] += 1
            elif any(p in text_lower for p in ['die hölle los', 'halt']):
                patterns['Colloquial'] += 1
            else:
                patterns['Other'] += 1
        
        for pattern, count in patterns.most_common():
            print(f"  {pattern}: {count}")
    else:
        print("\nNo false negatives!")
    
    print(f"\n{'=' * 80}")
    print(f"False Positives (should_filter=false but predicted=true): {len(false_positives)}")
    print("=" * 80)
    if false_positives:
        print("\nThese cases were incorrectly filtered:")
        for i, fp in enumerate(false_positives[:10], 1):  # Show first 10
            print(f"\n{i}. [{fp['id']}] {fp['text']}")
            print(f"   Notes: {fp.get('notes', 'N/A')}")
            print(f"   Category: {fp.get('category', 'N/A')}")
    else:
        print("\nNo false positives! Perfect precision.")
    
    # Detection coverage errors
    print(f"\n{'=' * 80}")
    print("DETECTION COVERAGE ERRORS")
    print("=" * 80)
    
    should_detect = [r for r in results if r.get('expected_should_detect') is True and 'error' not in r]
    not_detected = []
    for r in should_detect:
        parsed_claim = r.get('parsed_claim', {})
        if not (parsed_claim.get('subject') or r.get('category')):
            not_detected.append(r)
    
    print(f"\nShould detect but not detected: {len(not_detected)}")
    if not_detected:
        for i, nd in enumerate(not_detected[:10], 1):
            print(f"\n{i}. [{nd['id']}] {nd['text']}")
            print(f"   Notes: {nd.get('notes', 'N/A')}")
            print(f"   Category: {nd.get('category', 'N/A')}")
            print(f"   Subject: {nd.get('parsed_claim', {}).get('subject', 'N/A')}")
    else:
        print("\nAll should_detect cases were detected! Perfect recall.")
    
    # Relation type errors
    print(f"\n{'=' * 80}")
    print("RELATION TYPE ERRORS")
    print("=" * 80)
    
    with_expected = [r for r in results if r.get('expected_relation_type') and 'error' not in r]
    relation_errors = []
    for r in with_expected:
        expected = r.get('expected_relation_type', '').lower()
        predicted = r.get('relation_type', '').lower()
        if expected != predicted and expected not in predicted and predicted not in expected:
            relation_errors.append(r)
    
    print(f"\nRelation type mismatches: {len(relation_errors)}")
    if relation_errors:
        # Group by expected type
        by_expected = Counter()
        for re in relation_errors:
            by_expected[re.get('expected_relation_type', 'N/A')] += 1
        
        print("\nBy expected relation type:")
        for rel_type, count in by_expected.most_common():
            print(f"  {rel_type}: {count}")
        
        print("\nSample errors:")
        for i, re in enumerate(relation_errors[:10], 1):
            print(f"\n{i}. [{re['id']}] {re['text']}")
            print(f"   Expected: {re.get('expected_relation_type', 'N/A')}")
            print(f"   Predicted: {re.get('relation_type', 'N/A')}")
    
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    results_file = Path("results/detect_stress_200.jsonl")
    results = load_results(results_file)
    analyze_errors(results)
