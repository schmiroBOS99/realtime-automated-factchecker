# German Fact-Checking Service

A real-time German fact-checking pipeline using Natural Language Inference (NLI) and hybrid retrieval from Wikipedia and Wikidata.

## 🚀 Features

- **Claim Detection & Filtering**: Identifies factual claims and filters out subjective/opinion statements
- **Hybrid Retrieval**: Wikipedia (sentence-level) + Wikidata (structured data fast-path)
- **Two-Stage Reranking**: Semantic reranking (Stage 1) + Cross-encoder reranking (Stage 2)
- **NLI Verification**: Uses `cross-encoder/nli-deberta-v3-base` for evidence verification
- **Comparison Claims**: Specialized numeric comparison verifier for claims like "Der Nil ist länger als der Rhein"
- **Multi-Evidence Aggregation**: Boosts confidence when multiple evidences agree
- **Multi-Level Caching**: Claim cache, API response cache, embedding cache
- **Real-time Processing**: WebSocket support for live transcription and fact-checking

## 📋 Requirements

- Python 3.9+
- See `app/requirements.txt` for dependencies

## 🛠️ Installation

```bash
# Clone repository
git clone <repository-url>
cd factchecker-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r app/requirements.txt
```

## 🚦 Quick Start

### Start the API server

```bash
uvicorn app.main:app --port 8000
```

### Check a claim

```bash
curl -X POST "http://localhost:8000/check" \
     -H "Content-Type: application/json" \
     -d '{"claim": "Berlin ist die Hauptstadt von Deutschland"}'
```

### Detect claim structure (without verification)

```bash
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: application/json" \
     -d '{"claim": "Angela Merkel ist Bundeskanzlerin"}'
```

## 🏗️ Architecture

The pipeline follows a layered architecture:

- **API Layer**: FastAPI endpoints (`/check`, `/detect`, `/ws/live`)
- **Orchestration Layer**: `FactChecker` coordinates the pipeline
- **Processing Layer**: Claim parsing, retrieval, verification, comparison
- **Data Access Layer**: Wikipedia/Wikidata clients with TTL caching
- **ML Models**: Embedding model + NLI model

See `EVALUATION.md` for detailed architecture diagrams and evaluation results.

## 📁 Project Structure

```
app/
├── main.py                 # FastAPI application
├── pipeline/               # Core pipeline components
│   ├── factchecker.py     # Main orchestrator
│   ├── claim_parser.py    # Claim parsing & decomposition
│   ├── retriever.py       # Wikipedia retrieval
│   ├── verifier.py        # NLI verification
│   ├── comparison_verifier.py  # Numeric comparison
│   ├── claim_matcher.py   # Cache matching
│   ├── semantic_reranker.py   # Stage 1 reranking
│   ├── wikidata_client.py # Wikidata API client
│   ├── wikipedia_client.py # Wikipedia API client
│   └── postprocessors/    # Domain-specific post-processors
└── realtime/              # Real-time claim detection
```

## ⚙️ Configuration

### Enable Cross-Encoder Reranking (Stage 2)

```bash
export FACTCHECK_ENABLE_CROSSENCODER=1
uvicorn app.main:app --port 8000
```

### Cache Configuration

- **Claim Cache**: SQLite database at `./data/claim_cache.db` (auto-created)
- **API Response Cache**: TTL-based in-memory cache (6 hours)
- **Embedding Cache**: In-memory cache for claim matching

## 🔬 Key Components

### Claim Parser
- Extracts structured information (subject, predicate, object)
- Detects hard claims requiring decomposition
- Generates subclaims for complex comparisons

### Retriever
- Sentence-level retrieval from Wikipedia
- Two-stage reranking (semantic + cross-encoder)
- Lexical overlap pruning

### Verifier
- NLI-based evidence verification
- Multi-evidence aggregation with boosting
- Domain-specific post-processors (population, geo, temporal, etc.)

### Comparison Verifier
- Extracts numeric values from evidence
- Performs direct numeric comparisons
- Wikidata fallback for missing values

## 📊 Evaluation

See `EVALUATION.md` for detailed evaluation results including:
- Overall accuracy: 60.10% (standard), 77.29% (NEI=Neutral)
- Topic-specific performance
- Cache performance analysis
- Error taxonomy

## 🔧 API Endpoints

- `GET /ping` - Health check
- `POST /check` - Fact-check a claim
- `POST /detect` - Detect claim structure (without verification)
- `DELETE /cache/clear` - Clear all caches
- `WS /ws/live` - WebSocket for live transcription

## 📝 License

This project is for academic/research purposes.

## 👤 Author

[Maro Rodrigues]
