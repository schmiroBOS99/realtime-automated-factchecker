"""
Calculate accuracy excluding NOT_ENOUGH_INFO cases.

NOT_ENOUGH_INFO predictions are treated as neutral (neither correct nor incorrect).
Only SUPPORTED and REFUTED cases are counted for accuracy.
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python accuracy_excluding_nei.py <results.jsonl>")
        sys.exit(1)
    
    infile = Path(sys.argv[1])
    if not infile.exists():
        print(f"ERROR: File not found: {infile}")
        sys.exit(1)
    
    # Load results
    results = []
    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Filter: Only include cases where both expected and predicted are SUPPORTED or REFUTED
    # This excludes:
    # - Cases where expected is NOT_ENOUGH_INFO
    # - Cases where predicted is NOT_ENOUGH_INFO
    verifiable = [
        r for r in results
        if r.get('expected') in ['SUPPORTED', 'REFUTED']
        and r.get('label') in ['SUPPORTED', 'REFUTED']
        and 'error' not in r
    ]
    
    # Count correct predictions
    correct = sum(1 for r in verifiable if r.get('expected') == r.get('label'))
    total = len(verifiable)
    
    # Count excluded cases
    nei_excluded = len(results) - total
    
    # Breakdown by label
    supported_total = sum(1 for r in verifiable if r.get('expected') == 'SUPPORTED')
    supported_correct = sum(1 for r in verifiable if r.get('expected') == 'SUPPORTED' and r.get('label') == 'SUPPORTED')
    
    refuted_total = sum(1 for r in verifiable if r.get('expected') == 'REFUTED')
    refuted_correct = sum(1 for r in verifiable if r.get('expected') == 'REFUTED' and r.get('label') == 'REFUTED')
    
    # Print results
    print("=" * 70)
    print("ACCURACY (NOT_ENOUGH_INFO EXCLUDED)")
    print("=" * 70)
    print(f"\nTotal verifiable cases: {total}/{len(results)}")
    print(f"NOT_ENOUGH_INFO cases excluded: {nei_excluded}")
    print(f"\nOverall Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    
    print("\n" + "-" * 70)
    print("BREAKDOWN BY EXPECTED LABEL")
    print("-" * 70)
    if supported_total > 0:
        print(f"SUPPORTED: {supported_correct}/{supported_total} = {supported_correct/supported_total*100:.2f}%")
    else:
        print("SUPPORTED: 0/0")
    
    if refuted_total > 0:
        print(f"REFUTED: {refuted_correct}/{refuted_total} = {refuted_correct/refuted_total*100:.2f}%")
    else:
        print("REFUTED: 0/0")
    
    # Show excluded cases breakdown
    print("\n" + "-" * 70)
    print("EXCLUDED CASES (NOT_ENOUGH_INFO)")
    print("-" * 70)
    excluded_by_expected = {}
    excluded_by_predicted = {}
    
    for r in results:
        if 'error' in r:
            continue
        expected = r.get('expected', '')
        predicted = r.get('label', '')
        
        if expected == 'NOT_ENOUGH_INFO':
            excluded_by_expected[expected] = excluded_by_expected.get(expected, 0) + 1
        
        if predicted == 'NOT_ENOUGH_INFO' and expected != 'NOT_ENOUGH_INFO':
            excluded_by_predicted[predicted] = excluded_by_predicted.get(predicted, 0) + 1
    
    if excluded_by_expected:
        print(f"Expected=NOT_ENOUGH_INFO: {excluded_by_expected.get('NOT_ENOUGH_INFO', 0)}")
    if excluded_by_predicted:
        print(f"Predicted=NOT_ENOUGH_INFO (but expected was SUPPORTED/REFUTED): {excluded_by_predicted.get('NOT_ENOUGH_INFO', 0)}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
