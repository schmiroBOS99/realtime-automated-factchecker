"""
Batch claim detection evaluation script.

Processes texts from a JSONL file, calls POST /detect for each text,
and writes results to a JSONL output file with metrics.

Usage:
    python scripts/eval/batch_detect.py --infile data/eval/detect_stress_200.jsonl --out results/detect_stress_200.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import httpx


def load_texts(infile: Path) -> List[Dict[str, Any]]:
    """Load texts from JSONL file."""
    texts = []
    
    with open(infile, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if 'id' not in item:
                    item['id'] = f"line_{line_num}"
                texts.append(item)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping invalid JSONL line {line_num}: {e}", file=sys.stderr)
    
    return texts


def detect_claim(
    client: httpx.Client,
    text: str,
    timeout: float,
) -> tuple[Optional[Dict[str, Any]], float, Optional[str]]:
    """
    Detect a single claim via POST /detect.
    
    Returns:
        (result_dict, latency_ms, error_message)
    """
    start_time = time.perf_counter()
    
    try:
        response = client.post(
            "/detect",
            json={"claim": text},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return result, latency_ms, None
    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return None, latency_ms, "timeout"
    except httpx.HTTPStatusError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return None, latency_ms, f"http_error_{e.response.status_code}"
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return None, latency_ms, str(e)


def main():
    parser = argparse.ArgumentParser(description="Batch claim detection evaluation")
    parser.add_argument("--infile", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000", help="Base URL for API")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Load texts
    print(f"Loading texts from {args.infile}...", file=sys.stderr)
    texts = load_texts(args.infile)
    print(f"Loaded {len(texts)} texts", file=sys.stderr)
    
    # Process texts
    print(f"Processing {len(texts)} texts...", file=sys.stderr)
    print(f"Output: {args.out}", file=sys.stderr)
    
    results = []
    timeout_client = httpx.Timeout(args.timeout)
    
    with httpx.Client(base_url=args.base_url, timeout=timeout_client) as client:
        for i, item in enumerate(texts, 1):
            text = item.get('text', item.get('claim', ''))
            if not text:
                print(f"WARNING: Skipping item {item.get('id', i)} - no text field", file=sys.stderr)
                continue
            
            result, latency_ms, error = detect_claim(client, text, args.timeout)
            
            output_item = {
                'id': item.get('id', f"item_{i}"),
                'text': text,
                'expected_should_detect': item.get('expected', {}).get('should_detect'),
                'expected_should_filter': item.get('expected', {}).get('should_filter'),
                'expected_relation_type': item.get('expected', {}).get('expected_relation_type'),
                'notes': item.get('notes', ''),
                'latency_ms': latency_ms,
            }
            
            if error:
                output_item['error'] = error
            else:
                # Extract relevant fields from API response
                output_item['is_personal_or_subjective'] = result.get('is_personal_or_subjective', False)
                output_item['parsed_claim'] = result.get('parsed_claim', {})
                output_item['category'] = result.get('category', '')
                output_item['relation_type'] = result.get('relation_type', '')
                output_item['is_hard_claim'] = result.get('is_hard_claim', False)
                output_item['subclaims'] = result.get('subclaims', [])
            
            results.append(output_item)
            
            # Progress output
            if i % 10 == 0 or i == len(texts):
                category = result.get('category', 'ERROR') if result else 'ERROR'
                print(f"Progress: {i}/{len(texts)} | id={output_item['id']} | category={category} | latency={latency_ms:.1f}ms", file=sys.stderr)
    
    # Write results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    errors = sum(1 for r in results if 'error' in r)
    print(f"\nCompleted: {len(results)} processed, {errors} errors", file=sys.stderr)
    print(f"Results written to: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
