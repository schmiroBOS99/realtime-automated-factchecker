"""
Phase 2.3: Error Pattern Confirmation

Analyzes errors from seed1000 results and compares with Phase 1.1 error taxonomy.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Set

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

def identify_errors(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify misclassified cases (excluding NOT_ENOUGH_INFO expected)."""
    errors = []
    
    for r in results:
        expected = r.get('expected')
        predicted = r.get('label')
        
        # Skip if no expected label
        if not expected:
            continue
        
        # Skip if expected is NOT_ENOUGH_INFO (we focus on SUPPORTED/REFUTED errors)
        if expected == 'NOT_ENOUGH_INFO':
            continue
        
        # Error if predicted != expected
        if predicted != expected:
            errors.append({
                'id': r.get('id', ''),
                'claim': r.get('claim', ''),
                'expected': expected,
                'predicted': predicted,
                'topic': r.get('topic', ''),
                'path': r.get('path', ''),
                'confidence': r.get('confidence', 0.0),
                'best_evidence_title': r.get('best_evidence', {}).get('title', '') if isinstance(r.get('best_evidence'), dict) else '',
            })
    
    return errors

def categorize_error(error: Dict[str, Any]) -> str:
    """
    Categorize error based on Phase 1.1 taxonomy.
    
    Categories from Phase 1.1:
    - Retrieval limitation
    - NLI uncertainty / paraphrase sensitivity
    - Relation-type mismatch
    - Comparison handling
    - Post-processor logic
    - Intentional conservative decision
    """
    claim = error['claim'].lower()
    expected = error['expected']
    predicted = error['predicted']
    topic = error.get('topic', '')
    path = error.get('path', '')
    
    # False SUPPORTED (predicted SUPPORTED but expected REFUTED)
    if expected == 'REFUTED' and predicted == 'SUPPORTED':
        # Check for comparison claims
        if 'comparison' in topic or any(kw in claim for kw in ['länger', 'größer', 'kleiner', 'mehr', 'weniger', 'höher', 'niedriger']):
            return 'Comparison handling'
        
        # Check for numerical claims
        if any(kw in claim for kw in ['einwohner', 'bevölkerung', 'population', 'mitgliedstaaten']):
            return 'NLI uncertainty / paraphrase sensitivity'
        
        # Check for geo claims
        if 'geography' in topic or any(kw in claim for kw in ['hauptstadt', 'liegt in', 'befindet sich']):
            return 'NLI uncertainty / paraphrase sensitivity'
        
        # Default for false SUPPORTED
        return 'NLI uncertainty / paraphrase sensitivity'
    
    # False REFUTED (predicted REFUTED but expected SUPPORTED)
    if expected == 'SUPPORTED' and predicted == 'REFUTED':
        # Check for comparison claims
        if 'comparison' in topic or any(kw in claim for kw in ['länger', 'größer', 'kleiner', 'mehr', 'weniger', 'höher', 'niedriger']):
            return 'Comparison handling'
        
        # Check for numerical claims
        if any(kw in claim for kw in ['einwohner', 'bevölkerung', 'population', 'mitgliedstaaten']):
            return 'NLI uncertainty / paraphrase sensitivity'
        
        # Check for geo claims
        if 'geography' in topic or any(kw in claim for kw in ['hauptstadt', 'liegt in', 'befindet sich']):
            return 'NLI uncertainty / paraphrase sensitivity'
        
        # Check if evidence is missing (retrieval issue)
        if not error.get('best_evidence_title'):
            return 'Retrieval limitation'
        
        # Default for false REFUTED
        return 'NLI uncertainty / paraphrase sensitivity'
    
    # False NOT_ENOUGH_INFO (predicted NOT_ENOUGH_INFO but expected SUPPORTED/REFUTED)
    if predicted == 'NOT_ENOUGH_INFO':
        # This is intentional conservative behavior
        return 'Intentional conservative decision'
    
    return 'Unknown'

def analyze_errors(cold_file: Path, warm_file: Path) -> Dict[str, Any]:
    """Analyze errors from both cold and warm runs."""
    print("Loading cold run results...")
    cold_results = load_results(cold_file)
    cold_errors = identify_errors(cold_results)
    
    print("Loading warm run results...")
    warm_results = load_results(warm_file)
    warm_errors = identify_errors(warm_results)
    
    # Categorize errors
    cold_categorized = defaultdict(list)
    warm_categorized = defaultdict(list)
    
    for error in cold_errors:
        category = categorize_error(error)
        cold_categorized[category].append(error)
    
    for error in warm_errors:
        category = categorize_error(error)
        warm_categorized[category].append(error)
    
    # Find common errors (same claim ID in both runs)
    cold_error_ids = {e['id'] for e in cold_errors}
    warm_error_ids = {e['id'] for e in warm_errors}
    common_error_ids = cold_error_ids & warm_error_ids
    
    # Find errors that only appear in one run
    cold_only_ids = cold_error_ids - warm_error_ids
    warm_only_ids = warm_error_ids - cold_error_ids
    
    # Analyze error patterns
    cold_by_type = defaultdict(int)
    warm_by_type = defaultdict(int)
    
    for error in cold_errors:
        error_type = f"{error['expected']} -> {error['predicted']}"
        cold_by_type[error_type] += 1
    
    for error in warm_errors:
        error_type = f"{error['expected']} -> {error['predicted']}"
        warm_by_type[error_type] += 1
    
    return {
        'cold_errors': cold_errors,
        'warm_errors': warm_errors,
        'cold_categorized': dict(cold_categorized),
        'warm_categorized': dict(warm_categorized),
        'cold_by_type': dict(cold_by_type),
        'warm_by_type': dict(warm_by_type),
        'common_error_ids': common_error_ids,
        'cold_only_ids': cold_only_ids,
        'warm_only_ids': warm_only_ids,
    }

def print_report(analysis: Dict[str, Any]) -> None:
    """Print error analysis report."""
    print("=" * 80)
    print("PHASE 2.3: ERROR PATTERN CONFIRMATION")
    print("=" * 80)
    
    cold_errors = analysis['cold_errors']
    warm_errors = analysis['warm_errors']
    
    print(f"\nTotal Errors:")
    print(f"  Cold Run: {len(cold_errors)} errors")
    print(f"  Warm Run: {len(warm_errors)} errors")
    print(f"  Common Errors (same claim ID): {len(analysis['common_error_ids'])}")
    print(f"  Cold-Only Errors: {len(analysis['cold_only_ids'])}")
    print(f"  Warm-Only Errors: {len(analysis['warm_only_ids'])}")
    
    print(f"\n{'=' * 80}")
    print("ERROR TYPE DISTRIBUTION")
    print("=" * 80)
    
    print("\nCold Run Error Types:")
    for error_type, count in sorted(analysis['cold_by_type'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type:<40} {count:4d}")
    
    print("\nWarm Run Error Types:")
    for error_type, count in sorted(analysis['warm_by_type'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type:<40} {count:4d}")
    
    print(f"\n{'=' * 80}")
    print("ERROR CATEGORIZATION (Phase 1.1 Taxonomy)")
    print("=" * 80)
    
    print("\nCold Run Categories:")
    for category, errors in sorted(analysis['cold_categorized'].items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {category:<50} {len(errors):4d} errors")
    
    print("\nWarm Run Categories:")
    for category, errors in sorted(analysis['warm_categorized'].items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {category:<50} {len(errors):4d} errors")
    
    # Check for new error patterns
    print(f"\n{'=' * 80}")
    print("NEW ERROR PATTERNS DETECTION")
    print("=" * 80)
    
    # Phase 1.1 categories
    phase1_categories = {
        'Retrieval limitation',
        'NLI uncertainty / paraphrase sensitivity',
        'Relation-type mismatch',
        'Comparison handling',
        'Post-processor logic',
        'Intentional conservative decision',
    }
    
    cold_categories = set(analysis['cold_categorized'].keys())
    warm_categories = set(analysis['warm_categorized'].keys())
    all_categories = cold_categories | warm_categories
    
    new_categories = all_categories - phase1_categories
    if new_categories:
        print(f"\n[!] NEW CATEGORIES DETECTED (not in Phase 1.1 taxonomy):")
        for cat in sorted(new_categories):
            print(f"  - {cat}")
    else:
        print("\n[OK] All error categories match Phase 1.1 taxonomy")
    
    # Sample errors for each category
    print(f"\n{'=' * 80}")
    print("SAMPLE ERRORS BY CATEGORY (Cold Run)")
    print("=" * 80)
    
    for category, errors in sorted(analysis['cold_categorized'].items(), key=lambda x: len(x[1]), reverse=True):
        if errors:
            print(f"\n{category} ({len(errors)} errors):")
            # Show top 3 examples
            for i, error in enumerate(errors[:3], 1):
                print(f"  {i}. [{error['id']}] {error['claim'][:70]}...")
                print(f"     Expected: {error['expected']}, Predicted: {error['predicted']}")
                if error.get('best_evidence_title'):
                    print(f"     Evidence: {error['best_evidence_title']}")
    
    # Compare cold vs warm consistency
    print(f"\n{'=' * 80}")
    print("ERROR CONSISTENCY ANALYSIS")
    print("=" * 80)
    
    # For common errors, check if they have the same predicted label
    common_errors_same = 0
    common_errors_different = 0
    
    cold_error_map = {e['id']: e for e in cold_errors}
    warm_error_map = {e['id']: e for e in warm_errors}
    
    for error_id in analysis['common_error_ids']:
        cold_error = cold_error_map.get(error_id)
        warm_error = warm_error_map.get(error_id)
        
        if cold_error and warm_error:
            if cold_error['predicted'] == warm_error['predicted']:
                common_errors_same += 1
            else:
                common_errors_different += 1
    
    print(f"\nCommon Errors (same claim ID in both runs):")
    print(f"  Same prediction in both runs: {common_errors_same}")
    print(f"  Different prediction: {common_errors_different}")
    if common_errors_same + common_errors_different > 0:
        consistency_rate = common_errors_same / (common_errors_same + common_errors_different) * 100
        print(f"  Consistency Rate: {consistency_rate:.1f}%")
    
    print(f"\n{'=' * 80}")
    print("PHASE 2.3 COMPLETE")
    print("=" * 80)
    print()

if __name__ == "__main__":
    cold_file = Path("results/seed1000_cold.jsonl")
    warm_file = Path("results/seed1000_warm.jsonl")
    
    analysis = analyze_errors(cold_file, warm_file)
    print_report(analysis)
