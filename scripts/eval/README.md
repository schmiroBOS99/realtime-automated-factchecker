# Evaluation Harness

Batch evaluation scripts for the fact-checking pipeline.

## Overview

This directory contains scripts for batch evaluation of the fact-checking service:

- **`batch_factcheck.py`**: Processes claims from a JSONL/TXT file, calls the API, and writes results
- **`summarize.py`**: Computes evaluation metrics from results (accuracy, latency, cache stats)

## Prerequisites

1. Start the fact-checking API server:
   ```bash
   uvicorn app.main:app --port 8000
   ```

2. Install dependencies (if not already installed):
   ```bash
   pip install httpx
   ```

## Usage

### 1. Batch Fact-Checking

Process claims from a JSONL or TXT file:

```bash
# Basic usage
python scripts/eval/batch_factcheck.py \
    --infile data/eval/smoke_30.jsonl \
    --out results/results.jsonl

# With cache clearing
python scripts/eval/batch_factcheck.py \
    --infile data/eval/smoke_30.jsonl \
    --out results/results.jsonl \
    --clear-cache

# Custom API URL and max pages
python scripts/eval/batch_factcheck.py \
    --infile data/eval/smoke_30.jsonl \
    --out results/results.jsonl \
    --base-url http://localhost:8000 \
    --max-pages 5 \
    --timeout 60
```

**Input Format (JSONL):**
```json
{"id":"s1","claim":"Berlin ist die Hauptstadt von Deutschland","expected":"SUPPORTED","topic":"geography"}
{"id":"s2","claim":"Paris ist die Hauptstadt von Frankreich","expected":"SUPPORTED","topic":"geography"}
```

**Input Format (TXT):**
One claim per line (will be assigned auto-generated IDs).

**Output Format (JSONL):**
Each line contains:
- `id`, `claim`, `expected` (if provided), `topic` (if provided)
- `label`, `confidence`, `reason`, `path`, `relation_type`
- `latency_ms`
- `evidence_titles` (top 3-5), `best_evidence_snippet`, `best_evidence_title`, `best_evidence_url`
- `cache_hit`, `cache_match_type`, `cache_match_score` (if cache was used)
- `error` (if request failed)

### 2. Summarize Results

Compute evaluation metrics:

```bash
# Human-readable output
python scripts/eval/summarize.py results/results.jsonl

# JSON output
python scripts/eval/summarize.py results/results.jsonl --json
```

**Metrics computed:**
- **Accuracy**: Overall and per-label accuracy (if `expected` labels provided)
- **Label Distribution**: Count and percentage of each label
- **Latency Statistics**: Mean, min, max, p50, p90, p95, p99
- **Cache Statistics**: Hit rate, hits vs misses
- **Path Distribution**: Routes taken (WIKIDATA_FAST_PATH, WIKIPEDIA_RETRIEVAL, etc.)

### 3. Claim Detection Only

Use the `/detect` endpoint to test claim detection/decomposition without fact-checking:

```bash
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: application/json" \
     -d '{"claim": "Berlin ist die Hauptstadt von Deutschland"}'
```

Returns:
- `parsed_claim` (subject, predicate, object, etc.)
- `category`, `relation_type`
- `is_hard_claim`, `subclaims` (if decomposition was used)
- `is_personal_or_subjective` (filter flag)

## Example Workflow

```bash
# 1. Start the API server (in another terminal)
uvicorn app.main:app --port 8000

# 2. Run batch evaluation
python scripts/eval/batch_factcheck.py \
    --infile data/eval/smoke_30.jsonl \
    --out results/smoke_30_results.jsonl \
    --clear-cache

# 3. Summarize results
python scripts/eval/summarize.py results/smoke_30_results.jsonl
```

## Test Dataset

The `data/eval/smoke_30.jsonl` file contains 30 test claims covering:
- Geography (capitals, locations)
- Politics (roles, offices)
- Population (numerical claims)
- Sports (nationality)
- Biology (definition claims)

Each claim includes:
- `id`: Unique identifier
- `claim`: Claim text
- `expected`: Expected label (SUPPORTED, REFUTED, NOT_ENOUGH_INFO)
- `topic`: Claim category

## Notes

- The API must be running before running batch evaluation
- Results are written incrementally (one line per claim) for progress tracking
- Errors are logged but don't stop the batch process
- Cache clearing is optional but recommended for consistent evaluation runs
