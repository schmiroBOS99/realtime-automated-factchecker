"""Quick script to check dataset statistics."""
import json
from collections import Counter
from pathlib import Path

f = Path('data/eval/seed1000_taxonomy.jsonl')
lines = [json.loads(l) for l in f.read_text(encoding='utf-8').strip().split('\n') if l.strip()]

claims = [r['claim'] for r in lines]
topics = Counter(r['topic'] for r in lines)
labels = Counter(r['expected'] for r in lines)
topic_labels = Counter((r['topic'], r['expected']) for r in lines)

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

print("=" * 70)
print("DATASET-STATISTIKEN")
print("=" * 70)

print(f"\nTotal Claims: {len(claims)}")
print(f"Unique Claims: {len(set(claims))}")
print(f"Duplikate: {len(claims) - len(set(claims))}")

print("\n" + "=" * 70)
print("TAXONOMIE-ÜBERPRÜFUNG")
print("=" * 70)
all_correct = True
for topic, expected in sorted(taxonomy_expected.items()):
    actual = topics.get(topic, 0)
    status = '✓' if actual == expected else '✗'
    if actual != expected:
        all_correct = False
    print(f"{status} {topic:25s}: erwartet={expected:3d}, tatsächlich={actual:3d}")

print("=" * 70)
total_expected = sum(taxonomy_expected.values())
total_actual = sum(topics.values())
print(f"Gesamt: erwartet={total_expected:4d}, tatsächlich={total_actual:4d}")
if all_correct and total_actual == total_expected:
    print("✓ Taxonomie korrekt")
else:
    print("✗ Taxonomie-Fehler")

print("\n" + "=" * 70)
print("LABEL-VERTEILUNG")
print("=" * 70)
for label, count in sorted(labels.items()):
    print(f"  {label:20s}: {count:3d} ({count/len(lines)*100:5.1f}%)")

print("\n" + "=" * 70)
print("TOPIC x LABEL VERTEILUNG (Auszug)")
print("=" * 70)
for (topic, label), count in sorted(topic_labels.items())[:15]:
    print(f"  {topic:20s} {label:20s}: {count:3d}")
