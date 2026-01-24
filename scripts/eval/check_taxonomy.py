"""Check taxonomy correctness."""
import json
from collections import Counter
from pathlib import Path

f = Path('data/eval/seed1000_taxonomy.jsonl')
lines = [json.loads(l) for l in f.read_text(encoding='utf-8').strip().split('\n') if l.strip()]

taxonomy_expected = {
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

topics = Counter(r['topic'] for r in lines)
labels = Counter(r['expected'] for r in lines)

print("=" * 70)
print("TAXONOMIE-UBERPRUFUNG")
print("=" * 70)
all_ok = True
for topic in sorted(taxonomy_expected.keys()):
    exp = taxonomy_expected[topic]
    act = topics.get(topic, 0)
    ok = exp == act
    if not ok:
        all_ok = False
    status = 'OK' if ok else 'FEHLER'
    print(f"  {topic:25s}: {exp:3d} -> {act:3d} {status}")

print("=" * 70)
total_exp = sum(taxonomy_expected.values())
total_act = sum(topics.values())
print(f"Gesamt: {total_exp:4d} -> {total_act:4d}")
print(f"Taxonomie: {'KORREKT' if all_ok and total_exp == total_act else 'FEHLER'}")

print("\n" + "=" * 70)
print("LABEL-VERTEILUNG")
print("=" * 70)
for label in sorted(labels.keys()):
    count = labels[label]
    pct = count / len(lines) * 100
    print(f"  {label:20s}: {count:3d} ({pct:5.1f}%)")

claims = [r['claim'] for r in lines]
unique_claims = len(set(claims))
duplicates = len(claims) - unique_claims

print("\n" + "=" * 70)
print("DUPLIKAT-ANALYSE")
print("=" * 70)
print(f"Total Claims: {len(claims)}")
print(f"Unique Claims: {unique_claims}")
print(f"Duplicates: {duplicates}")
print(f"Uniqueness: {unique_claims/len(claims)*100:.1f}%")
