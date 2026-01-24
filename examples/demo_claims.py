"""
Demo script showing example claims and expected outputs.

This script demonstrates the fact-checking pipeline with 5 representative German claims
across different domains.
"""
import asyncio
import json
from app.pipeline.factchecker import FactChecker


# Example claims covering different domains
DEMO_CLAIMS = [
    {
        "claim": "Berlin ist die Hauptstadt von Deutschland",
        "domain": "geography",
        "expected_label": "SUPPORTED"
    },
    {
        "claim": "Der Nil ist länger als der Rhein",
        "domain": "comparison",
        "expected_label": "SUPPORTED"
    },
    {
        "claim": "Angela Merkel ist Bundeskanzlerin",
        "domain": "politics",
        "expected_label": "REFUTED"  # She was chancellor, but not anymore
    },
    {
        "claim": "Portugal hat 10 Millionen Einwohner",
        "domain": "population",
        "expected_label": "SUPPORTED"  # Approximately correct
    },
    {
        "claim": "Die Erde ist flach",
        "domain": "science",
        "expected_label": "REFUTED"
    }
]


async def main():
    """Run demo claims through the fact-checking pipeline."""
    factchecker = FactChecker()
    await factchecker.start()
    
    print("=" * 80)
    print("German Fact-Checking Service - Demo Claims")
    print("=" * 80)
    print()
    
    results = []
    
    for i, example in enumerate(DEMO_CLAIMS, 1):
        claim = example["claim"]
        domain = example["domain"]
        expected = example["expected_label"]
        
        print(f"\n[{i}/{len(DEMO_CLAIMS)}] Domain: {domain}")
        print(f"Claim: {claim}")
        print(f"Expected: {expected}")
        print("-" * 80)
        
        try:
            result = await factchecker.check(claim, max_pages=3)
            
            predicted = result.get("label", "UNKNOWN")
            path = result.get("path", "UNKNOWN")
            answer = result.get("answer", "No answer provided")
            best_evidence = result.get("best_evidence")
            
            # Check if prediction matches expected
            match = "✓" if predicted == expected else "✗"
            
            print(f"Predicted: {predicted} {match}")
            print(f"Path: {path}")
            print(f"Answer: {answer}")
            
            if best_evidence:
                print(f"Evidence: {best_evidence.get('title', 'N/A')}")
                print(f"  Snippet: {best_evidence.get('snippet', '')[:100]}...")
            
            # Store result
            results.append({
                "claim": claim,
                "domain": domain,
                "expected": expected,
                "predicted": predicted,
                "match": predicted == expected,
                "path": path
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "claim": claim,
                "domain": domain,
                "expected": expected,
                "predicted": "ERROR",
                "match": False,
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    correct = sum(1 for r in results if r.get("match", False))
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"Correct: {correct}/{total} ({accuracy:.1f}%)")
    print()
    
    for r in results:
        status = "✓" if r.get("match") else "✗"
        print(f"{status} {r['claim'][:50]}... -> {r['predicted']}")
    
    await factchecker.close()
    
    # Save results to JSON
    with open("demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to demo_results.json")


if __name__ == "__main__":
    asyncio.run(main())
