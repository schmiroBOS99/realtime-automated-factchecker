"""
Phase 1.1: Manual Error Taxonomy Analysis

Manually classify each error case by examining the evidence and scores.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple


def main():
    # Load results
    infile = Path('results/seed_50_final.jsonl')
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
        
        if not expected or expected == 'NOT_ENOUGH_INFO':
            continue
        
        if predicted != expected:
            errors.append(r)
    
    # Manual classification based on detailed analysis
    # Format: (id, expected, predicted, root_cause, classification, notes)
    classifications = [
        # FALSE NOT_ENOUGH_INFO (should be SUPPORTED or REFUTED)
        ('seed50_009', 'SUPPORTED', 'NOT_ENOUGH_INFO', 
         'NLI uncertainty / paraphrase sensitivity', 'LIMITATION',
         'Temporal claim (war) - evidence lacks explicit date range'),
        
        ('seed50_012', 'SUPPORTED', 'NOT_ENOUGH_INFO',
         'NLI uncertainty / paraphrase sensitivity', 'LIMITATION',
         'Complex claim structure - evidence mentions "Bundesregierung" not exact phrasing'),
        
        ('seed50_013', 'REFUTED', 'NOT_ENOUGH_INFO',
         'NLI uncertainty / paraphrase sensitivity', 'LIMITATION',
         'Evidence doesn\'t explicitly state "nicht Mitglied" - requires negative inference'),
        
        ('seed50_018', 'REFUTED', 'NOT_ENOUGH_INFO',
         'Comparison handling', 'LIMITATION',
         'Comparison claim - no evidence retrieved or comparison values missing'),
        
        ('seed50_024', 'SUPPORTED', 'NOT_ENOUGH_INFO',
         'NLI uncertainty / paraphrase sensitivity', 'LIMITATION',
         'Complex claim with negation ("nicht gegen Viren") - evidence about different topic'),
        
        ('seed50_026', 'SUPPORTED', 'NOT_ENOUGH_INFO',
         'NLI uncertainty / paraphrase sensitivity', 'LIMITATION',
         'Evidence mentions team but not explicit "spielt in Bundesliga"'),
        
        ('seed50_028', 'SUPPORTED', 'NOT_ENOUGH_INFO',
         'NLI uncertainty / paraphrase sensitivity', 'LIMITATION',
         'Temporal event claim (fanden) - evidence snippet too short'),
        
        ('seed50_029', 'REFUTED', 'NOT_ENOUGH_INFO',
         'Retrieval limitation', 'LIMITATION',
         'No evidence retrieved for this specific claim'),
        
        ('seed50_037', 'SUPPORTED', 'NOT_ENOUGH_INFO',
         'Retrieval limitation', 'LIMITATION',
         'Evidence about wrong person (mother, not Beethoven)'),
        
        ('seed50_040', 'REFUTED', 'NOT_ENOUGH_INFO',
         'Retrieval limitation', 'LIMITATION',
         'Evidence about "Roman" (genre) not specific book "1984"'),
        
        ('seed50_044', 'SUPPORTED', 'NOT_ENOUGH_INFO',
         'Retrieval limitation', 'LIMITATION',
         'Evidence about "Herz" but doesn\'t mention "vier Kammern" explicitly'),
        
        ('seed50_048', 'REFUTED', 'NOT_ENOUGH_INFO',
         'NLI uncertainty / paraphrase sensitivity', 'LIMITATION',
         'Evidence describes concept but doesn\'t state explicit age'),
        
        # FALSE SUPPORTED (should be REFUTED)
        ('seed50_004', 'REFUTED', 'SUPPORTED',
         'Retrieval limitation', 'LIMITATION',
         'Evidence says "Hauptstadt des Bundesstaates" - partial match misleads NLI'),
        
        ('seed50_023', 'REFUTED', 'SUPPORTED',
         'Retrieval limitation', 'LIMITATION',
         'Evidence mentions "drei Lungenlappen" (anatomy detail) - NLI matches on "drei"'),
        
        # FALSE REFUTED (should be SUPPORTED)
        ('seed50_010', 'SUPPORTED', 'REFUTED',
         'Relation-type mismatch', 'LIMITATION',
         'Evidence about different time period - temporal mismatch'),
        
        ('seed50_014', 'SUPPORTED', 'REFUTED',
         'Relation-type mismatch', 'LIMITATION',
         'Evidence discusses nuanced terminology ("Verfassung" vs "Grundgesetz")'),
        
        ('seed50_022', 'SUPPORTED', 'REFUTED',
         'Relation-type mismatch', 'LIMITATION',
         'Evidence discusses electrons but focuses on different property'),
        
        ('seed50_033', 'SUPPORTED', 'REFUTED',
         'Relation-type mismatch', 'LIMITATION',
         'Evidence definition slightly different phrasing - NLI sees contradiction'),
        
        ('seed50_039', 'SUPPORTED', 'REFUTED',
         'Retrieval limitation', 'LIMITATION',
         'Evidence about English language, not German - wrong article retrieved'),
    ]
    
    # Create lookup
    classification_lookup = {item[0]: item for item in classifications}
    
    # Group by classification type
    by_type = {'BUG': [], 'LIMITATION': [], 'DESIGN DECISION': []}
    by_root_cause = {}
    
    for error in errors:
        item_id = error.get('id', '')
        if item_id in classification_lookup:
            _, expected, predicted, root_cause, classification, notes = classification_lookup[item_id]
        else:
            # Fallback classification
            root_cause = 'Unknown'
            classification = 'LIMITATION'
            notes = 'Not manually classified'
        
        entry = {
            'id': item_id,
            'claim': error.get('claim', ''),
            'expected': expected if item_id in classification_lookup else error.get('expected'),
            'predicted': predicted if item_id in classification_lookup else error.get('label'),
            'root_cause': root_cause,
            'classification': classification,
            'notes': notes,
            'path': error.get('path', ''),
            'entailment': error.get('entailment_score', 0.0),
            'contradiction': error.get('contradiction_score', 0.0),
        }
        
        by_type[classification].append(entry)
        
        if root_cause not in by_root_cause:
            by_root_cause[root_cause] = {'BUG': 0, 'LIMITATION': 0, 'DESIGN DECISION': 0}
        by_root_cause[root_cause][classification] += 1
    
    # Print taxonomy
    print("=" * 90)
    print("ERROR TAXONOMY - PHASE 1.1")
    print("=" * 90)
    print(f"\nTotal Errors: {len(errors)}")
    print(f"  - False NOT_ENOUGH_INFO: {len([e for e in errors if e.get('label') == 'NOT_ENOUGH_INFO'])}")
    print(f"  - False SUPPORTED: {len([e for e in errors if e.get('label') == 'SUPPORTED' and e.get('expected') == 'REFUTED'])}")
    print(f"  - False REFUTED: {len([e for e in errors if e.get('label') == 'REFUTED' and e.get('expected') == 'SUPPORTED'])}")
    
    print("\n" + "=" * 90)
    print("ROOT CAUSE TAXONOMY TABLE")
    print("=" * 90)
    print(f"\n{'Root Cause':<50} {'BUG':<6} {'LIMIT':<6} {'DESIGN':<6} {'Total':<6}")
    print("-" * 90)
    
    for root_cause in sorted(by_root_cause.keys()):
        counts = by_root_cause[root_cause]
        total = sum(counts.values())
        print(f"{root_cause:<50} {counts['BUG']:<6} {counts['LIMITATION']:<6} {counts['DESIGN DECISION']:<6} {total:<6}")
    
    print("\n" + "=" * 90)
    print("CLASSIFICATION SUMMARY")
    print("=" * 90)
    for class_type in ['BUG', 'LIMITATION', 'DESIGN DECISION']:
        count = len(by_type[class_type])
        print(f"{class_type}: {count} cases ({count/len(errors)*100:.1f}%)")
    
    print("\n" + "=" * 90)
    print("DETAILED ERROR CLASSIFICATIONS")
    print("=" * 90)
    
    for class_type in ['BUG', 'LIMITATION', 'DESIGN DECISION']:
        items = by_type[class_type]
        if not items:
            continue
        
        print(f"\n--- {class_type} ({len(items)} cases) ---\n")
        
        # Group by root cause
        by_root = {}
        for item in items:
            rc = item['root_cause']
            if rc not in by_root:
                by_root[rc] = []
            by_root[rc].append(item)
        
        for root_cause in sorted(by_root.keys()):
            print(f"\n{root_cause}:")
            for item in by_root[root_cause]:
                print(f"  [{item['id']}] {item['claim'][:65]}...")
                print(f"      Expected: {item['expected']} | Predicted: {item['predicted']}")
                print(f"      {item['notes']}")
                if item['entailment'] > 0 or item['contradiction'] > 0:
                    print(f"      Scores: E={item['entailment']:.3f}, C={item['contradiction']:.3f}")
    
    print("\n" + "=" * 90)


if __name__ == '__main__':
    main()
