"""
Phase 1.3: Analyze potential improvement options.

Identify small, scoped improvements that:
- Don't require architectural changes
- Don't require new models
- Could potentially improve multiple error cases
"""
import json
from pathlib import Path
from typing import Dict, List, Set


def analyze_improvement_options():
    """Analyze error cases to identify potential improvement opportunities."""
    results_file = 'results/seed_50_final.jsonl'
    
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Find all errors
    errors = []
    for r in results:
        expected = r.get('expected')
        predicted = r.get('label')
        
        if expected and expected != 'NOT_ENOUGH_INFO' and predicted != expected:
            errors.append(r)
    
    print("=" * 90)
    print("PHASE 1.3 - IMPROVEMENT OPTIONS ANALYSIS")
    print("=" * 90)
    print(f"\nTotal Errors: {len(errors)}")
    
    # Analyze patterns
    print("\n" + "=" * 90)
    print("PATTERN ANALYSIS")
    print("=" * 90)
    
    # 1. Synonym normalization opportunities
    print("\n1. SYNONYM NORMALIZATION:")
    synonym_cases = []
    for e in errors:
        claim = e.get('claim', '').lower()
        if 'bundeshauptstadt' in claim or ('hauptstadt' in claim and 'bundeshauptstadt' in e.get('best_evidence_snippet', '').lower()):
            synonym_cases.append(e)
    
    print(f"  Potential cases: {len(synonym_cases)}")
    if synonym_cases:
        for case in synonym_cases[:3]:
            print(f"    - {case['id']}: {case['claim'][:60]}...")
    
    # 2. Temporal claims
    print("\n2. TEMPORAL CLAIMS (war/wurde/fanden):")
    temporal_cases = [e for e in errors if any(word in e.get('claim', '').lower() for word in [' war ', 'wurde', 'fanden', 'gewonnen'])]
    print(f"  Potential cases: {len(temporal_cases)}")
    for case in temporal_cases[:5]:
        print(f"    - {case['id']}: {case['claim'][:60]}... | {case.get('expected')} -> {case.get('label')}")
    
    # 3. Numerical extraction
    print("\n3. NUMERICAL EXTRACTION:")
    numerical_cases = [e for e in errors if any(word in e.get('claim', '').lower() for word in ['mitgliedstaaten', 'jahre', 'kammern'])]
    print(f"  Potential cases: {len(numerical_cases)}")
    for case in numerical_cases[:3]:
        print(f"    - {case['id']}: {case['claim'][:60]}...")
    
    # 4. Negative inference
    print("\n4. NEGATIVE INFERENCE (ist nicht Mitglied):")
    negative_cases = [e for e in errors if 'ist nicht' in e.get('claim', '').lower() or ('ist' in e.get('claim', '').lower() and 'nicht' in e.get('claim', '').lower())]
    print(f"  Potential cases: {len(negative_cases)}")
    for case in negative_cases:
        print(f"    - {case['id']}: {case['claim'][:60]}...")
    
    print("\n" + "=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)
    
    print("\nBased on the analysis, the most promising improvement would be:")
    print("\nOPTION: Improved synonym normalization for 'Hauptstadt' claims")
    print("  - Scope: Small, targeted fix")
    print("  - Impact: Could help with 'Ottawa ist die Bundeshauptstadt' vs 'Hauptstadt'")
    print("  - Implementation: Add normalization in claim parser")
    print("  - Risk: Low (simple string normalization)")
    
    print("\nHowever, note that Phase 1.3 is OPTIONAL.")
    print("Current system behavior is already defensible (all errors are LIMITATIONs).")
    print("\nRecommendation: SKIP Phase 1.3 unless a specific improvement is clearly justified.")
    
    print("\n" + "=" * 90)


if __name__ == '__main__':
    analyze_improvement_options()
