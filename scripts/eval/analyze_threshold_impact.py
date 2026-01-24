"""
Analyze which NOT_ENOUGH_INFO cases would flip with threshold changes.

For each NOT_ENOUGH_INFO case, test different thresholds and see which would flip.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import httpx


def test_claim_with_scores(claim: str, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Test a claim and return the full API response with scores."""
    try:
        import time
        time.sleep(0.5)  # Small delay to avoid overwhelming the server
        response = httpx.post(
            f"{base_url}/check",
            json={"claim": claim, "max_pages": 3},
            timeout=30.0
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def analyze_threshold_impact(
    entailment: float,
    contradiction: float,
    current_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Analyze which threshold changes would flip the label.
    
    Current logic:
    - SUPPORTED: entailment >= threshold AND entailment > contradiction
    - REFUTED: contradiction >= threshold AND contradiction > entailment
    - NOT_ENOUGH_INFO: otherwise
    """
    result = {
        "current_entailment": entailment,
        "current_contradiction": contradiction,
        "current_threshold": current_threshold,
        "would_be_supported_at": None,
        "would_be_refuted_at": None,
        "should_be_supported": False,  # If entailment >= threshold but got NEI
        "should_be_refuted": False,  # If contradiction >= threshold but got NEI
    }
    
    # Check if it would be SUPPORTED at lower thresholds
    # SUPPORTED requires: entailment >= threshold AND entailment > contradiction
    if entailment > contradiction:
        if entailment >= current_threshold:
            # Already above threshold - should be SUPPORTED but got NOT_ENOUGH_INFO
            # This means post-processors or other logic changed it
            result["should_be_supported"] = True
            result["would_be_supported_at"] = current_threshold  # Already should be
        else:
            # Would be SUPPORTED if threshold was lowered to <= entailment
            result["would_be_supported_at"] = entailment
    
    # Check if it would be REFUTED at lower thresholds
    # REFUTED requires: contradiction >= threshold AND contradiction > entailment
    if contradiction > entailment:
        if contradiction >= current_threshold:
            # Already above threshold - should be REFUTED but got NOT_ENOUGH_INFO
            result["should_be_refuted"] = True
            result["would_be_refuted_at"] = current_threshold  # Already should be
        else:
            # Would be REFUTED if threshold was lowered to <= contradiction
            result["would_be_refuted_at"] = contradiction
    
    # Calculate distance to thresholds
    result["distance_to_support"] = current_threshold - entailment if entailment < current_threshold else 0.0
    result["distance_to_refute"] = current_threshold - contradiction if contradiction < current_threshold else 0.0
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_threshold_impact.py <results.jsonl>")
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
    
    # Find false NOT_ENOUGH_INFO cases
    false_nei = [
        r for r in results
        if r.get('expected') in ['SUPPORTED', 'REFUTED']
        and r.get('label') == 'NOT_ENOUGH_INFO'
        and 'error' not in r
    ]
    
    print("=" * 80)
    print(f"ANALYZING {len(false_nei)} NOT_ENOUGH_INFO CASES")
    print("=" * 80)
    print("\nTesting claims with API to get scores...\n")
    
    base_url = "http://localhost:8000"
    analyses = []
    
    for i, result_item in enumerate(false_nei, 1):
        claim = result_item.get('claim')
        expected = result_item.get('expected')
        item_id = result_item.get('id', f'item_{i}')
        
        print(f"[{i}/{len(false_nei)}] Testing: {item_id} - {claim[:60]}...")
        
        api_result = test_claim_with_scores(claim, base_url)
        
        if 'error' in api_result:
            print(f"  ERROR: {api_result['error']}\n")
            continue
        
        scores = api_result.get('scores', [])
        if not scores:
            print(f"  WARNING: No scores returned (path: {api_result.get('path')})\n")
            analyses.append({
                'id': item_id,
                'claim': claim,
                'expected': expected,
                'path': api_result.get('path'),
                'has_scores': False,
            })
            continue
        
        # Get the best score (first one, or highest entailment)
        best_score = scores[0]
        entailment = best_score.get('entailment', 0.0)
        contradiction = best_score.get('contradiction', 0.0)
        neutral = best_score.get('neutral', 0.0)
        
        analysis = analyze_threshold_impact(entailment, contradiction, current_threshold=0.75)
        analysis['id'] = item_id
        analysis['claim'] = claim
        analysis['expected'] = expected
        analysis['path'] = api_result.get('path')
        analysis['neutral'] = neutral
        analysis['has_scores'] = True
        
        analyses.append(analysis)
        
        print(f"  Scores: E={entailment:.3f}, C={contradiction:.3f}, N={neutral:.3f}")
        if analysis.get('should_be_supported'):
            print(f"  -> SHOULD be SUPPORTED (E={entailment:.3f} >= 0.75) but got NOT_ENOUGH_INFO (post-processor?)")
        elif analysis.get('would_be_supported_at') and not analysis.get('should_be_supported'):
            threshold_needed = analysis['would_be_supported_at']
            if threshold_needed > 0.5:  # Only show if reasonably close
                print(f"  -> Would be SUPPORTED if threshold <= {threshold_needed:.3f}")
        if analysis.get('should_be_refuted'):
            print(f"  -> SHOULD be REFUTED (C={contradiction:.3f} >= 0.75) but got NOT_ENOUGH_INFO (post-processor?)")
        elif analysis.get('would_be_refuted_at') and not analysis.get('should_be_refuted'):
            threshold_needed = analysis['would_be_refuted_at']
            if threshold_needed > 0.5:  # Only show if reasonably close
                print(f"  -> Would be REFUTED if threshold <= {threshold_needed:.3f}")
        print()
    
    # Summary analysis
    print("\n" + "=" * 80)
    print("THRESHOLD IMPACT SUMMARY")
    print("=" * 80)
    
    with_scores = [a for a in analyses if a.get('has_scores')]
    print(f"\nCases with scores: {len(with_scores)}/{len(analyses)}")
    
    # Analyze by expected label
    for expected_label in ['SUPPORTED', 'REFUTED']:
        expected_cases = [a for a in with_scores if a.get('expected') == expected_label]
        if not expected_cases:
            continue
        
        print(f"\n--- Expected {expected_label} ({len(expected_cases)} cases) ---")
        
        if expected_label == 'SUPPORTED':
            # Check which should be SUPPORTED (high entailment but post-processor changed it)
            should_be = [a for a in expected_cases if a.get('should_be_supported')]
            
            # Check which would flip with threshold change (but not already should_be)
            would_flip_threshold = [
                a for a in expected_cases
                if a.get('would_be_supported_at') is not None
                and not a.get('should_be_supported')
                and a.get('current_entailment', 0) > a.get('current_contradiction', 0)
                and a.get('current_entailment', 0) >= 0.60  # Only show if reasonably close
            ]
            
            print(f"Should be SUPPORTED (high score but post-processor changed): {len(should_be)}/{len(expected_cases)}")
            print(f"Would flip to SUPPORTED with lower threshold (>=0.60): {len(would_flip_threshold)}/{len(expected_cases)}")
            
            if should_be:
                print("\nHigh scores but got NOT_ENOUGH_INFO (post-processor issue):")
                for a in sorted(should_be, key=lambda x: x.get('current_entailment', 0), reverse=True):
                    print(f"  {a['id']}: E={a.get('current_entailment', 0):.3f}, C={a.get('current_contradiction', 0):.3f}")
                    print(f"    Claim: {a['claim'][:70]}...")
            
            if would_flip_threshold:
                print("\nWould flip with threshold adjustment:")
                for a in sorted(would_flip_threshold, key=lambda x: x.get('current_entailment', 0), reverse=True):
                    threshold_needed = a.get('current_entailment', 0)
                    distance = a.get('distance_to_support', 0)
                    print(f"  {a['id']}: threshold <= {threshold_needed:.3f} (current: 0.75, distance: {distance:.3f})")
                    print(f"    Claim: {a['claim'][:70]}...")
        
        elif expected_label == 'REFUTED':
            # Check which should be REFUTED (high contradiction but post-processor changed it)
            should_be = [a for a in expected_cases if a.get('should_be_refuted')]
            
            # Check which would flip with threshold change
            would_flip_threshold = [
                a for a in expected_cases
                if a.get('would_be_refuted_at') is not None
                and not a.get('should_be_refuted')
                and a.get('current_contradiction', 0) > a.get('current_entailment', 0)
                and a.get('current_contradiction', 0) >= 0.60  # Only show if reasonably close
            ]
            
            print(f"Should be REFUTED (high score but post-processor changed): {len(should_be)}/{len(expected_cases)}")
            print(f"Would flip to REFUTED with lower threshold (>=0.60): {len(would_flip_threshold)}/{len(expected_cases)}")
            
            if should_be:
                print("\nHigh scores but got NOT_ENOUGH_INFO (post-processor issue):")
                for a in sorted(should_be, key=lambda x: x.get('current_contradiction', 0), reverse=True):
                    print(f"  {a['id']}: E={a.get('current_entailment', 0):.3f}, C={a.get('current_contradiction', 0):.3f}")
                    print(f"    Claim: {a['claim'][:70]}...")
            
            if would_flip_threshold:
                print("\nWould flip with threshold adjustment:")
                for a in sorted(would_flip_threshold, key=lambda x: x.get('current_contradiction', 0), reverse=True):
                    threshold_needed = a.get('current_contradiction', 0)
                    distance = a.get('distance_to_refute', 0)
                    print(f"  {a['id']}: threshold <= {threshold_needed:.3f} (current: 0.75, distance: {distance:.3f})")
                    print(f"    Claim: {a['claim'][:70]}...")
    
    # Overall statistics
    print("\n" + "-" * 80)
    print("OVERALL STATISTICS")
    print("-" * 80)
    
    if with_scores:
        entailments = [a.get('current_entailment', 0) for a in with_scores]
        contradictions = [a.get('current_contradiction', 0) for a in with_scores]
        
        print(f"Mean entailment: {sum(entailments)/len(entailments):.3f}")
        print(f"Mean contradiction: {sum(contradictions)/len(contradictions):.3f}")
        print(f"Max entailment: {max(entailments):.3f}")
        print(f"Max contradiction: {max(contradictions):.3f}")
        
        # Count how many are close to thresholds
        close_to_support = sum(1 for e in entailments if 0.60 <= e < 0.75)
        close_to_refute = sum(1 for c in contradictions if 0.60 <= c < 0.75)
        
        print(f"\nClose to SUPPORT threshold (0.60-0.75): {close_to_support}/{len(with_scores)}")
        print(f"Close to REFUTE threshold (0.60-0.75): {close_to_refute}/{len(with_scores)}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
