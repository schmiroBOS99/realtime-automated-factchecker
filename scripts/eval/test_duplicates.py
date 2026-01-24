"""Test script to understand duplicate issue."""
import json
from collections import Counter
from pathlib import Path

f = Path('data/eval/seed1000_taxonomy.jsonl')
lines = [json.loads(l) for l in f.read_text(encoding='utf-8').strip().split('\n') if l.strip()]

claims = [r['claim'] for r in lines]
dup_counts = Counter(claims)
dups = {k: v for k, v in dup_counts.items() if v > 1}

print(f"Total Claims: {len(claims)}")
print(f"Unique Claims: {len(set(claims))}")
print(f"Duplicates: {len(dups)} Claims appear multiple times")
print(f"\nTop 10 duplicated claims:")
for claim, count in sorted(dups.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {count}x: {claim[:70]}...")

# Check geography_city specifically
geo_city = [r for r in lines if r['topic'] == 'geography_city']
geo_claims = [r['claim'] for r in geo_city]
geo_dup_counts = Counter(geo_claims)
geo_dups = {k: v for k, v in geo_dup_counts.items() if v > 1}

print(f"\nGeography_city: {len(geo_claims)} Claims, {len(set(geo_claims))} unique")
print(f"Duplicates within geography_city: {len(geo_dups)}")
if geo_dups:
    print(f"\nTop 5 duplicates in geography_city:")
    for claim, count in sorted(geo_dups.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {count}x: {claim}")
