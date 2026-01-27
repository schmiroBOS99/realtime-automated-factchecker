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
git clone https://github.com/schmiroBOS99/realtime-automated-factchecker.git
cd realtime-automated-factchecker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r app/requirements.txt
```

## 🚦 Quick Start

### Option 1: Docker Compose (Empfohlen - Einfachste Methode)

```bash
# Repository klonen
git clone https://github.com/schmiroBOS99/realtime-automated-factchecker.git
cd realtime-automated-factchecker

# Service starten (lädt automatisch alle Dependencies)
docker-compose up
```

Der Service ist dann unter **http://localhost:8000** erreichbar. Öffne die URL im Browser für die UI.

> 📘 **Für Professoren:** Siehe [ANLEITUNG_FUER_PROFESSOREN.md](ANLEITUNG_FUER_PROFESSOREN.md) für eine detaillierte Schritt-für-Schritt-Anleitung ohne technisches Vorwissen.

### Option 2: Lokale Installation

```bash
# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Dependencies installieren
pip install -r app/requirements.txt

# Server starten
uvicorn app.main:app --port 8000
```

**Für Professoren:** Siehe [ANLEITUNG_FUER_PROFESSOREN.md](ANLEITUNG_FUER_PROFESSOREN.md) für eine einfache Schritt-für-Schritt-Anleitung.

### Check a claim

```bash
curl -X POST "http://localhost:8000/check" \
     -H "Content-Type: application/json" \
     -d '{"claim": "Berlin ist die Hauptstadt von Deutschland"}'
```

**Example Response:**
```json
{
  "claim": "Berlin ist die Hauptstadt von Deutschland",
  "path": "WIKIPEDIA_RETRIEVAL",
  "relation_type": "geo_location",
  "parsed_claim": {
    "subject": "Berlin",
    "predicate": "ist",
    "object": "Deutschland",
    "canonical_claim": "berlin ist hauptstadt deutschland"
  },
  "label": "SUPPORTED",
  "answer": "Die Behauptung ist korrekt. Berlin ist die Hauptstadt von Deutschland.",
  "best_evidence": {
    "title": "Berlin",
    "url": "https://de.wikipedia.org/wiki/Berlin",
    "snippet": "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland.",
    "source": "wikipedia"
  },
  "scores": [
    {
      "entailment": 0.92,
      "contradiction": 0.05,
      "neutral": 0.03
    }
  ],
  "evidence_count": 5
}
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

## 🚀 Quick Deployment

**Für Professoren & Demo:** Siehe [QUICK_START.md](QUICK_START.md) für die einfachste Bereitstellung mit Docker.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Documentation

- [EVALUATION.md](EVALUATION.md) - Detailed evaluation results and methodology
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- API Documentation: Available at `http://localhost:8000/docs` when server is running (FastAPI auto-generated OpenAPI docs)

## 🔗 Model Licenses

- **NLI Model**: `cross-encoder/nli-deberta-v3-base` - Apache 2.0 License
- **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2` - Apache 2.0 License
- Models are automatically downloaded from Hugging Face on first use

## ⚠️ Rate Limiting & API Keys

- **Wikipedia API**: No API key required, but please respect rate limits (see [Wikimedia API Terms](https://www.mediawiki.org/wiki/API:Etiquette))
- **Wikidata API**: No API key required, but please respect rate limits
- **Caching**: All API responses are cached for 6 hours (TTL) to minimize API calls
- **PII Handling**: The system does not store or log personally identifiable information from claims

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details on testing and development.

## 👤 Author

Maro Schmitt Rodrigues - maro.schmitt_rodrigues@student.reutlingen-university.de
