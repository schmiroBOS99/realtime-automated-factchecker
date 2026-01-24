# Claim Detector Robustness Evaluation

## Overview

This evaluation focuses specifically on the claim detector (`POST /detect` endpoint) to assess:
- Filtering of non-claims (personal/subjective, questions, vague context)
- Robustness to paraphrases/ASR noise (no punctuation, filler words, number words, short claims)
- Relation-type / subclaims sanity checks

## Dataset

**File:** `data/eval/detect_stress_200.jsonl`

**Composition:**
- 120 cases: `should_detect = true` (factual but messy)
  - Short claims (missing verb/articles)
  - Number words (zwölf, zehn, etc.)
  - ASR filler words (also, äh, hm)
  - No punctuation
  - Comparison claims
  - Nationality claims
  - Various topics (geo, politics, sport, science, etc.)
- 80 cases: `should_filter = true` (opinions, questions, context-only)
  - Personal pronouns (ich, du, deine)
  - Questions (Ist...?, Wusstest du...)
  - Missing context (Er war..., Sie ist...)
  - Subjective opinions
  - Commands
  - Colloquial expressions

## Usage

### Step 1: Start the API server

```bash
uvicorn app.main:app --port 8000
```

### Step 2: Run batch detection

```bash
python scripts/eval/batch_detect.py --infile data/eval/detect_stress_200.jsonl --out results/detect_stress_200.jsonl
```

### Step 3: Summarize results

```bash
python scripts/eval/summarize_detect.py results/detect_stress_200.jsonl
```

Optional: Save summary as JSON:
```bash
python scripts/eval/summarize_detect.py results/detect_stress_200.jsonl --json results/detect_stress_200_summary.json
```

## Metrics

### Filter Metrics (should_filter)
- **True Positives (TP)**: Predicted filter=true, expected filter=true
- **False Positives (FP)**: Predicted filter=true, expected filter=false
- **False Negatives (FN)**: Predicted filter=false, expected filter=true
- **True Negatives (TN)**: Predicted filter=false, expected filter=false
- **Precision, Recall, F1-Score**

### Detection Coverage (should_detect)
- **Recall**: How often detector produced a structured parse (subject OR category exists)
- Measures robustness to messy input (ASR noise, short claims, etc.)

### Relation Type Accuracy
- **Exact match**: Predicted relation_type exactly matches expected
- **Coarse match**: Predicted relation_type contains expected (or vice versa)
- Only computed for cases with `expected_relation_type` field

### Decomposition Sanity
- **Comparison cases**: % with subclaims (should be >0)
- **Definition cases**: % without subclaims (guard should prevent decomposition)

## Expected Output

The summary will show:
1. Filter performance (how well non-claims are filtered)
2. Detection coverage (how well messy claims are detected)
3. Relation-type accuracy (how well claim types are identified)
4. Decomposition sanity (whether comparison/definition guards work)
5. Latency statistics

## Interpretation

- **High Filter Precision**: System correctly identifies non-claims
- **High Filter Recall**: System doesn't miss non-claims that should be filtered
- **High Detection Recall**: System robustly handles messy input (ASR noise, short claims)
- **High Relation-Type Accuracy**: System correctly categorizes claim types
- **Decomposition Sanity**: Comparison claims get subclaims, definition claims don't
