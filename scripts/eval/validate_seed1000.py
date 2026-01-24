"""
Step 1: Dataset Sanity Check for seed1000_taxonomy.jsonl

Validates:
- JSONL integrity
- Exactly 1000 rows
- Required fields present
- Labels valid
- Topic distribution
- Label distribution
- Duplicate claims
- Anomalies
"""
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

DATASET_PATH = Path("data/eval/seed1000_taxonomy.jsonl")
REQUIRED_FIELDS = {"id", "topic", "claim", "expected"}
VALID_LABELS = {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}

def validate_dataset() -> Dict[str, Any]:
    """Validate the seed1000 dataset and return statistics."""
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    rows = []
    seen_ids = set()
    seen_claims = {}
    claim_counts = Counter()
    
    # Read and parse JSONL
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    row = json.loads(line)
                    rows.append(row)
                    
                    # Check required fields
                    missing_fields = REQUIRED_FIELDS - set(row.keys())
                    if missing_fields:
                        results["errors"].append(
                            f"Line {line_num}: Missing fields: {missing_fields}"
                        )
                        results["valid"] = False
                    
                    # Check ID uniqueness
                    row_id = row.get("id", "")
                    if row_id in seen_ids:
                        results["warnings"].append(
                            f"Line {line_num}: Duplicate ID: {row_id}"
                        )
                    seen_ids.add(row_id)
                    
                    # Check label validity
                    expected_label = row.get("expected", "")
                    if expected_label not in VALID_LABELS:
                        results["errors"].append(
                            f"Line {line_num}: Invalid label '{expected_label}'"
                        )
                        results["valid"] = False
                    
                    # Track claims for duplicate detection
                    claim = row.get("claim", "")
                    claim_counts[claim] += 1
                    if claim in seen_claims:
                        seen_claims[claim].append(line_num)
                    else:
                        seen_claims[claim] = [line_num]
                    
                except json.JSONDecodeError as e:
                    results["errors"].append(
                        f"Line {line_num}: JSON decode error: {e}"
                    )
                    results["valid"] = False
    
    except FileNotFoundError:
        results["errors"].append(f"Dataset file not found: {DATASET_PATH}")
        results["valid"] = False
        return results
    except Exception as e:
        results["errors"].append(f"Unexpected error reading dataset: {e}")
        results["valid"] = False
        return results
    
    # Check row count
    row_count = len(rows)
    if row_count != 1000:
        results["errors"].append(
            f"Expected 1000 rows, found {row_count}"
        )
        results["valid"] = False
    
    # Calculate statistics
    topics = Counter(row.get("topic", "unknown") for row in rows)
    labels = Counter(row.get("expected", "unknown") for row in rows)
    
    # Find duplicate claims
    duplicate_claims = {
        claim: lines for claim, count in claim_counts.items()
        if count > 1 for lines in [seen_claims[claim]]
    }
    
    # Check for anomalies
    anomalies = []
    
    # Check topic distribution (should match taxonomy)
    expected_topics = {
        'geography_city': 100,
        'geography_physical': 50,
        'politics_office': 50,
        'politics_general': 100,
        'science': 100,
        'sport': 100,
        'biology_medicine': 100,
        'culture': 100,
        'religion': 50,
        'economy': 50,
        'astronomy': 50,
        'comparison': 50,
        'general_knowledge': 100,
    }
    
    for topic, expected_count in expected_topics.items():
        actual_count = topics.get(topic, 0)
        if actual_count != expected_count:
            anomalies.append(
                f"Topic '{topic}': Expected {expected_count}, found {actual_count}"
            )
    
    # Check label distribution (should be approximately 45% SUPPORTED, 35% REFUTED, 20% NOT_ENOUGH_INFO)
    total = sum(labels.values())
    if total > 0:
        supported_pct = (labels.get("SUPPORTED", 0) / total) * 100
        refuted_pct = (labels.get("REFUTED", 0) / total) * 100
        nei_pct = (labels.get("NOT_ENOUGH_INFO", 0) / total) * 100
        
        # Allow some tolerance (±5%)
        if not (40 <= supported_pct <= 50):
            anomalies.append(
                f"SUPPORTED percentage ({supported_pct:.1f}%) outside expected range (40-50%)"
            )
        if not (30 <= refuted_pct <= 40):
            anomalies.append(
                f"REFUTED percentage ({refuted_pct:.1f}%) outside expected range (30-40%)"
            )
        if not (15 <= nei_pct <= 25):
            anomalies.append(
                f"NOT_ENOUGH_INFO percentage ({nei_pct:.1f}%) outside expected range (15-25%)"
            )
    
    # Check for empty or very short claims
    for row in rows:
        claim = row.get("claim", "")
        if not claim or len(claim.strip()) < 5:
            anomalies.append(
                f"Row {row.get('id', 'unknown')}: Claim too short or empty"
            )
        if len(claim) > 500:
            anomalies.append(
                f"Row {row.get('id', 'unknown')}: Claim very long ({len(claim)} chars)"
            )
    
    results["stats"] = {
        "total_rows": row_count,
        "unique_ids": len(seen_ids),
        "unique_claims": len([c for c, count in claim_counts.items() if count == 1]),
        "duplicate_claims": len(duplicate_claims),
        "topic_distribution": dict(topics),
        "label_distribution": dict(labels),
        "duplicate_claim_details": {
            claim: {"count": count, "lines": lines}
            for claim, count in claim_counts.items()
            if count > 1
            for lines in [seen_claims[claim]]
        },
        "anomalies": anomalies,
    }
    
    return results

def print_report(results: Dict[str, Any]) -> None:
    """Print a formatted validation report."""
    print("=" * 80)
    print("STEP 1: DATASET SANITY CHECK")
    print("=" * 80)
    print(f"\nDataset: {DATASET_PATH}")
    print(f"Status: {'[VALID]' if results['valid'] else '[INVALID]'}")
    
    if results["errors"]:
        print(f"\nERRORS ({len(results['errors'])}):")
        for error in results["errors"][:10]:  # Show first 10
            print(f"  - {error}")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")
    
    if results["warnings"]:
        print(f"\nWARNINGS ({len(results['warnings'])}):")
        for warning in results["warnings"][:10]:  # Show first 10
            print(f"  - {warning}")
        if len(results["warnings"]) > 10:
            print(f"  ... and {len(results['warnings']) - 10} more warnings")
    
    stats = results["stats"]
    print(f"\n{'=' * 80}")
    print("STATISTICS")
    print("=" * 80)
    print(f"\nTotal Rows: {stats['total_rows']}")
    print(f"Unique IDs: {stats['unique_ids']}")
    print(f"Unique Claims: {stats['unique_claims']}")
    print(f"Duplicate Claims: {stats['duplicate_claims']}")
    
    if stats['duplicate_claims'] > 0:
        print(f"\nDuplicate Claim Details:")
        for claim, details in list(stats['duplicate_claim_details'].items())[:5]:
            print(f"  Claim: {claim[:60]}...")
            print(f"    Count: {details['count']}")
            print(f"    Lines: {details['lines']}")
        if len(stats['duplicate_claim_details']) > 5:
            print(f"  ... and {len(stats['duplicate_claim_details']) - 5} more duplicates")
    
    print(f"\n{'=' * 80}")
    print("TOPIC DISTRIBUTION")
    print("=" * 80)
    for topic, count in sorted(stats['topic_distribution'].items()):
        print(f"  {topic:25s}: {count:3d}")
    
    print(f"\n{'=' * 80}")
    print("LABEL DISTRIBUTION")
    print("=" * 80)
    total = sum(stats['label_distribution'].values())
    for label, count in sorted(stats['label_distribution'].items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {label:20s}: {count:3d} ({pct:5.1f}%)")
    
    if stats['anomalies']:
        print(f"\n{'=' * 80}")
        print("ANOMALIES")
        print("=" * 80)
        for anomaly in stats['anomalies']:
            print(f"  [WARNING] {anomaly}")
    
    print(f"\n{'=' * 80}")
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\n[RESULT] Dataset is {'VALID' if results['valid'] else 'INVALID'}")
    if results['valid'] and not stats['anomalies']:
        print("[RESULT] No anomalies detected")
    print()

if __name__ == "__main__":
    results = validate_dataset()
    print_report(results)
    
    # Exit with error code if invalid
    exit(0 if results['valid'] else 1)
