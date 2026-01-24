"""
Batch fact-checking evaluation script.

Processes claims from a JSONL or TXT file, calls POST /check for each claim,
and writes results to a JSONL output file with metrics.

Usage:
    python scripts/eval/batch_factcheck.py --infile data/eval/smoke_30.jsonl --out results/results.jsonl
    python scripts/eval/batch_factcheck.py --infile data/eval/smoke_30.jsonl --out results/results.jsonl --clear-cache
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import httpx


def load_claims(infile: Path) -> List[Dict[str, Any]]:
    """Load claims from JSONL or TXT file."""
    claims = []
    
    with open(infile, 'r', encoding='utf-8') as f:
        if infile.suffix == '.jsonl':
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # Ensure id field exists
                    if 'id' not in item:
                        item['id'] = f"line_{line_num}"
                    claims.append(item)
                except json.JSONDecodeError as e:
                    print(f"WARNING: Skipping invalid JSONL line {line_num}: {e}", file=sys.stderr)
        else:
            # Plain text file - one claim per line
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                claims.append({
                    'id': f"line_{line_num}",
                    'claim': line,
                })
    
    return claims


def clear_cache(base_url: str, timeout: float) -> bool:
    """Clear the cache via DELETE /cache/clear."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.delete(f"{base_url}/cache/clear")
            response.raise_for_status()
            result = response.json()
            count = result.get('claims_cleared', 0)
            print(f"Cache cleared: {count} claims removed", file=sys.stderr)
            return True
    except Exception as e:
        print(f"ERROR: Failed to clear cache: {e}", file=sys.stderr)
        return False


def factcheck_claim(
    client: httpx.Client,
    claim: str,
    max_pages: int,
    timeout: float,
) -> tuple[Optional[Dict[str, Any]], float, Optional[str]]:
    """
    Fact-check a single claim via POST /check.
    
    Returns:
        (result_dict, latency_ms, error_message)
    """
    start_time = time.perf_counter()
    
    try:
        response = client.post(
            "/check",
            json={"claim": claim, "max_pages": max_pages},
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
        return None, latency_ms, f"error_{str(e)}"


def extract_evidence_titles(result: Dict[str, Any], max_titles: int = 5) -> List[str]:
    """Extract evidence titles from result."""
    titles = []
    
    # Check best_evidence
    best_evidence = result.get('best_evidence')
    if best_evidence and best_evidence.get('title'):
        titles.append(best_evidence['title'])
    
    # Check if there are additional evidences (if result has evidence_count or evidence_top_k)
    # Note: Current API doesn't return all evidences, but we can check for additional fields
    # For now, just return the best evidence title
    
    return titles[:max_titles]


def format_result_line(
    input_item: Dict[str, Any],
    api_result: Optional[Dict[str, Any]],
    latency_ms: float,
    error: Optional[str],
) -> Dict[str, Any]:
    """Format a single result line for output JSONL."""
    output = {
        'id': input_item.get('id', 'unknown'),
        'claim': input_item.get('claim', ''),
    }
    
    # Add expected label if present
    if 'expected' in input_item:
        output['expected'] = input_item['expected']
    
    # Add topic if present
    if 'topic' in input_item:
        output['topic'] = input_item['topic']
    
    # Add error info if present
    if error:
        output['error'] = error
        output['latency_ms'] = round(latency_ms, 2)
        return output
    
    # Add API result fields
    if api_result:
            output['label'] = api_result.get('label', 'UNKNOWN')
            output['confidence'] = api_result.get('confidence', 0.0)
            output['reason'] = api_result.get('reason', '')
            output['path'] = api_result.get('path', 'unknown')
            output['relation_type'] = api_result.get('relation_type', 'unknown')
            
            # Extract NLI scores if available
            scores = api_result.get('scores', [])
            if scores and len(scores) > 0:
                # Get the first (best) score dict
                best_score = scores[0]
                output['entailment_score'] = best_score.get('entailment', 0.0)
                output['contradiction_score'] = best_score.get('contradiction', 0.0)
                output['neutral_score'] = best_score.get('neutral', 0.0)
            
            # Extract evidence info
            best_evidence = api_result.get('best_evidence')
            if best_evidence:
                output['best_evidence_snippet'] = best_evidence.get('snippet', '')
                output['best_evidence_title'] = best_evidence.get('title', '')
                output['best_evidence_url'] = best_evidence.get('url', '')
            
            evidence_titles = extract_evidence_titles(api_result, max_titles=5)
            if evidence_titles:
                output['evidence_titles'] = evidence_titles
            
            # Check for cache hit info
            if 'cache_match' in api_result:
                output['cache_hit'] = True
                output['cache_match_type'] = api_result['cache_match'].get('match_type', '')
                output['cache_match_score'] = api_result['cache_match'].get('match_score', 0.0)
            else:
                output['cache_hit'] = False
    
    output['latency_ms'] = round(latency_ms, 2)
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Batch fact-checking evaluation script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--infile',
        type=Path,
        required=True,
        help='Input file (JSONL or TXT, one claim per line)',
    )
    parser.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Output JSONL file path',
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default='http://localhost:8000',
        help='Base URL of the fact-checking API',
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=3,
        help='Maximum number of evidence pages to retrieve',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        help='HTTP request timeout in seconds',
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cache before running evaluation',
    )
    
    args = parser.parse_args()
    
    # Load claims
    if not args.infile.exists():
        print(f"ERROR: Input file not found: {args.infile}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading claims from {args.infile}...", file=sys.stderr)
    claims = load_claims(args.infile)
    print(f"Loaded {len(claims)} claims", file=sys.stderr)
    
    if not claims:
        print("ERROR: No claims to process", file=sys.stderr)
        sys.exit(1)
    
    # Clear cache if requested
    if args.clear_cache:
        print("Clearing cache...", file=sys.stderr)
        if not clear_cache(args.base_url, args.timeout):
            print("WARNING: Cache clear failed, continuing anyway...", file=sys.stderr)
    
    # Create output directory if needed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Process claims
    print(f"Processing {len(claims)} claims...", file=sys.stderr)
    print(f"Output: {args.out}", file=sys.stderr)
    
    results_processed = 0
    results_with_errors = 0
    
    with open(args.out, 'w', encoding='utf-8') as outfile:
        with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
            for idx, claim_item in enumerate(claims, 1):
                claim_text = claim_item.get('claim', '')
                claim_id = claim_item.get('id', f'item_{idx}')
                
                if not claim_text:
                    print(f"WARNING: Skipping item {idx} (id={claim_id}): empty claim", file=sys.stderr)
                    continue
                
                # Fact-check claim
                result, latency_ms, error = factcheck_claim(
                    client=client,
                    claim=claim_text,
                    max_pages=args.max_pages,
                    timeout=args.timeout,
                )
                
                # Format result
                result_line = format_result_line(claim_item, result, latency_ms, error)
                
                # Write to output
                outfile.write(json.dumps(result_line, ensure_ascii=False) + '\n')
                outfile.flush()
                
                results_processed += 1
                if error:
                    results_with_errors += 1
                
                # Progress logging
                if idx % 10 == 0 or idx == len(claims):
                    status = "ERROR" if error else result_line.get('label', 'UNKNOWN') if result else 'UNKNOWN'
                    print(
                        f"Progress: {idx}/{len(claims)} | "
                        f"id={claim_id} | "
                        f"label={status} | "
                        f"latency={latency_ms:.1f}ms",
                        file=sys.stderr,
                    )
    
    print(
        f"\nCompleted: {results_processed} processed, "
        f"{results_with_errors} errors",
        file=sys.stderr,
    )
    print(f"Results written to: {args.out}", file=sys.stderr)


if __name__ == '__main__':
    main()
