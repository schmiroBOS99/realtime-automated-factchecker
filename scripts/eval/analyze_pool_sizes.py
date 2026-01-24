"""Analyze pool sizes to understand duplicate issue."""
import random
random.seed(42)

# Mix: 45% SUPPORTED, 35% REFUTED, 20% NOT_ENOUGH_INFO
MIX = (0.45, 0.35, 0.20)

def analyze_pool_needs(n):
    n_sup = round(n * MIX[0])
    n_ref = round(n * MIX[1])
    n_nei = n - n_sup - n_ref
    return n_sup, n_ref, n_nei

topics = {
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

# Known pool sizes (approximate)
pool_sizes = {
    'geography_city': {'supported': 75, 'refuted': 150, 'nei': 4},
    'general_knowledge': {'supported': 5, 'refuted': 5, 'nei': 2},
}

print("Pool size analysis:")
print("=" * 80)
for topic, n in topics.items():
    n_sup, n_ref, n_nei = analyze_pool_needs(n)
    print(f"\n{topic} (n={n}):")
    print(f"  Needs: SUPPORTED={n_sup}, REFUTED={n_ref}, NOT_ENOUGH_INFO={n_nei}")
    if topic in pool_sizes:
        sizes = pool_sizes[topic]
        print(f"  Pool sizes: SUPPORTED={sizes.get('supported', '?')}, REFUTED={sizes.get('refuted', '?')}, NOT_ENOUGH_INFO={sizes.get('nei', '?')}")
        if sizes.get('nei', 999) < n_nei:
            print(f"  WARNING: NOT_ENOUGH_INFO pool too small!")
