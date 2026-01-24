"""
Phase 1.1: Error & Decision Taxonomy Analysis

Analyze all error cases and classify them by root cause and type (BUG/LIMITATION/DESIGN).
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


def classify_root_cause(claim: str, expected: str, predicted: str, result: Dict[str, Any]) -> Tuple[str, str]:
    """
    Classify root cause and type for an error case.
    
    Returns: (root_cause, classification_type)
    """
    claim_lower = claim.lower()
    path = result.get('path', '')
    relation_type = result.get('relation_type', '')
    best_evidence_title = result.get('best_evidence_title', '')
    best_evidence_snippet = result.get('best_evidence_snippet', '')
    
    # Get scores if available
    entailment = result.get('entailment_score', 0.0)
    contradiction = result.get('contradiction_score', 0.0)
    
    # FALSE NOT_ENOUGH_INFO cases
    if predicted == 'NOT_ENOUGH_INFO' and expected in ['SUPPORTED', 'REFUTED']:
        # Very low scores (< 0.01) - NLI uncertainty / paraphrase sensitivity
        if entailment < 0.01 and contradiction < 0.01:
            # Check if it's a temporal claim (past tense)
            if any(word in claim_lower for word in ['war', 'wurde', 'fanden', 'gewonnen']):
                return ('NLI uncertainty / paraphrase sensitivity', 'LIMITATION')
            # Check if it's a sports/event claim
            if any(word in claim_lower for word in ['fifa', 'weltmeisterschaft', 'olympischen', 'sommerspiele']):
                return ('Retrieval limitation', 'LIMITATION')
            # Check if it's a complex claim (multiple clauses)
            if any(word in claim_lower for word in ['wirken gegen', 'nicht gegen']):
                return ('NLI uncertainty / paraphrase sensitivity', 'LIMITATION')
            # Check if it's a definition/biological claim
            if any(word in claim_lower for word in ['hat vier kammern', 'hat zwölf monate']):
                return ('Retrieval limitation', 'LIMITATION')
            return ('NLI uncertainty / paraphrase sensitivity', 'LIMITATION')
        
        # High scores but got NEI - should be rare after fixes
        if entailment >= 0.95:
            return ('Post-processor logic (residual)', 'BUG')
        
        # Medium scores (0.01 - 0.70)
        if entailment >= 0.01 or contradiction >= 0.01:
            return ('Intentional conservative decision', 'DESIGN DECISION')
        
        return ('NLI uncertainty / paraphrase sensitivity', 'LIMITATION')
    
    # FALSE SUPPORTED (predicted SUPPORTED, expected REFUTED)
    if predicted == 'SUPPORTED' and expected == 'REFUTED':
        # Check if it's a comparison claim
        if any(word in claim_lower for word in ['größer als', 'länger als', 'höher als']):
            return ('Comparison handling', 'LIMITATION')
        
        # Check if it's a numerical claim
        if any(word in claim_lower for word in ['mitgliedstaaten', 'einwohner', 'bevölkerung']):
            if relation_type == 'cached':
                return ('Cache behavior (numerical similarity)', 'BUG')
            return ('Relation-type mismatch', 'LIMITATION')
        
        # Check if evidence mentions subject but wrong fact
        if best_evidence_title:
            return ('Retrieval limitation', 'LIMITATION')
        
        return ('NLI uncertainty / paraphrase sensitivity', 'LIMITATION')
    
    # FALSE REFUTED (predicted REFUTED, expected SUPPORTED)
    if predicted == 'REFUTED' and expected == 'SUPPORTED':
        # Check if it's a comparison claim
        if any(word in claim_lower for word in ['größer als', 'länger als']):
            return ('Comparison handling', 'LIMITATION')
        
        # Check if evidence contradicts due to parsing issue
        if best_evidence_snippet:
            return ('Relation-type mismatch', 'LIMITATION')
        
        return ('NLI uncertainty / paraphrase sensitivity', 'LIMITATION')
    
    return ('Unknown', 'LIMITATION')


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_errors.py <results.jsonl>")
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
    
    # Find all error cases
    errors = []
    for r in results:
        expected = r.get('expected')
        predicted = r.get('label')
        
        # Skip if no expected label
        if not expected:
            continue
        
        # Skip if expected is NOT_ENOUGH_INFO (that's acceptable)
        if expected == 'NOT_ENOUGH_INFO':
            continue
        
        # Error if predicted != expected
        if predicted != expected:
            errors.append(r)
    
    # Group errors
    false_nei = [r for r in errors if r.get('label') == 'NOT_ENOUGH_INFO']
    false_supported = [r for r in errors if r.get('label') == 'SUPPORTED' and r.get('expected') == 'REFUTED']
    false_refuted = [r for r in errors if r.get('label') == 'REFUTED' and r.get('expected') == 'SUPPORTED']
    
    print("=" * 80)
    print("ERROR TAXONOMY ANALYSIS")
    print("=" * 80)
    print(f"\nTotal Errors: {len(errors)}")
    print(f"  - False NOT_ENOUGH_INFO: {len(false_nei)}")
    print(f"  - False SUPPORTED: {len(false_supported)}")
    print(f"  - False REFUTED: {len(false_refuted)}")
    
    # Classify each error
    classified = {
        'BUG': [],
        'LIMITATION': [],
        'DESIGN DECISION': [],
    }
    
    root_causes = {}
    
    for error in errors:
        root_cause, classification = classify_root_cause(
            error.get('claim', ''),
            error.get('expected', ''),
            error.get('label', ''),
            error
        )
        
        classified[classification].append({
            'id': error.get('id', ''),
            'claim': error.get('claim', ''),
            'expected': error.get('expected', ''),
            'predicted': error.get('label', ''),
            'root_cause': root_cause,
            'classification': classification,
            'path': error.get('path', ''),
            'entailment': error.get('entailment_score', 0.0),
            'contradiction': error.get('contradiction_score', 0.0),
        })
        
        if root_cause not in root_causes:
            root_causes[root_cause] = {'BUG': 0, 'LIMITATION': 0, 'DESIGN DECISION': 0}
        root_causes[root_cause][classification] += 1
    
    # Print taxonomy table
    print("\n" + "=" * 80)
    print("ERROR TAXONOMY TABLE")
    print("=" * 80)
    print(f"\n{'Root Cause':<50} {'BUG':<8} {'LIMIT':<8} {'DESIGN':<8} {'Total':<8}")
    print("-" * 80)
    
    for root_cause in sorted(root_causes.keys()):
        counts = root_causes[root_cause]
        total = sum(counts.values())
        print(f"{root_cause:<50} {counts['BUG']:<8} {counts['LIMITATION']:<8} {counts['DESIGN DECISION']:<8} {total:<8}")
    
    print("\n" + "=" * 80)
    print("DETAILED ERROR LIST")
    print("=" * 80)
    
    for classification in ['BUG', 'LIMITATION', 'DESIGN DECISION']:
        items = classified[classification]
        if not items:
            continue
        
        print(f"\n--- {classification} ({len(items)} cases) ---")
        for item in items:
            print(f"\n{item['id']}: {item['claim'][:70]}...")
            print(f"  Expected: {item['expected']} | Predicted: {item['predicted']}")
            print(f"  Root Cause: {item['root_cause']}")
            print(f"  Scores: E={item['entailment']:.3f}, C={item['contradiction']:.3f}")
            print(f"  Path: {item['path']}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
