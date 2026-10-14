# BoliBench — Indian Voice AI Crash-Test Lab

> **A diagnostic evaluation lab built for the Great Indian AI Internship Challenge by Gnani AI.**
> BoliBench evaluates how speech-recognition errors alter the meaning of Hindi, Indian English, and Hinglish commands, causing downstream voice agents to perform incorrect actions.

---

## 1. Project Objective & Differentiating Evaluation Engine

Traditional Speech-to-Text (STT) benchmarks rely exclusively on **Word Error Rate (WER)**. While WER measures lexical surface edit distance, it treats all token substitutions equally. In production Indian Voice AI agents (e.g. banking bots, ticketing agents, delivery assistants), acoustic errors are often **task-critical**:

| Failure Category | Example Spoken Input | Misrecognized Output | Downstream Agent Action Failure |
|---|---|---|---|
| **Quantities & Numbers** | "Do packet chahiye." (*Qty: 2*) | "Teen packet chahiye." (*Qty: 3*) | Dispatches wrong inventory count |
| **Financial Magnitudes** | "पाँच लाख रुपये" (*5 Lakh INR*) | "पाँच हज़ार रुपये" (*5 Thousand INR*) | 100x financial value mismatch |
| **Missed Negation** | "नहीं, कैंसिल मत करो" (*Retain*) | "हाँ, कैंसिल कर दो" (*Cancel*) | Unauthorized destructive cancellation |
| **Corrections / Change-of-Mind** | "Flight to Mumbai, no wait, to Delhi" | "Flight to Mumbai" | Books preliminary discarded destination |
| **Product / Brand Names** | "Prisma STT का access चाहिए" | "Krishna STT का access चाहिए" | Wrong software SKU provisioned |
| **Named Entities** | "Send invoice to Rahul Sharma" | "Send invoice to Rohit Sharma" | Wrong customer billed |
| **Command Intent** | "Mera account balance check karo" | "Mera account band kar do" | Destructive account closure triggered |
| **Code-Switching (Hinglish)**| "Order confirm mat karna, cart empty karo" | "Order confirm kar do" | Unauthorized cart checkout |

### The Differentiating Feature: Paired-Contrast Testing

Rather than acting as a generic transcription dashboard, **BoliBench pairs semantically contrasting utterances ($A$ and $B$)** to verify whether the speech recognition and structured interpretation pipeline **preserves the critical distinction**:

```
Test Pair Example (pair_qty_001):
  Utterance A: "Do packet chahiye."   -> Expected Quantity: 2 -> Action: order_2
  Utterance B: "Teen packet chahiye." -> Expected Quantity: 3 -> Action: order_3
  Evaluation Engine: Determines whether the pipeline preserves (2 != 3) or collapses.
```

Full mathematical definitions and scoring taxonomy are documented in [`docs/evaluation_methodology.md`](docs/evaluation_methodology.md).

---

## 2. Verified Technology Stack

- **Speech Recognition Engine:** Gnani Prisma v2.5 STT (REST v3: `https://api.vachana.ai/stt/v3`)
- **Backend:** FastAPI (Python 3.10+) with Pydantic v2 Settings, Schemas, & SQLite Persistence
- **Frontend:** Vanilla HTML5, CSS3, and JavaScript (Zero external CDN dependencies, fully self-contained)
- **Audio Validation:** In-memory container, duration (≤ 60s), size, and signature validation
- **Evaluation Engine:** Deterministic rule-based slot extractors, dynamic programming Levenshtein WER, and paired-contrast scoring
- **Automated Testing:** `pytest` (57 tests passing in < 1 second, 0 network calls, 0 credits spent)

Official Gnani API verification details are documented in [`docs/gnani_api_verification.md`](docs/gnani_api_verification.md).

---

## 3. Directory Layout

```
bolibench/
├── .env.example                     # Sample environment variable template
├── .gitignore                       # Git ignore preventing audio, caches, and secrets leak
├── README.md                        # Project documentation and runbook
├── requirements.txt                 # Pinned dependencies
├── pytest.ini                       # Test configuration
├── docs/
│   ├── gnani_api_verification.md    # Official Gnani Prisma API verification document
│   └── evaluation_methodology.md   # Mathematical formulas, extractors, and failure taxonomy
├── app/
│   ├── __init__.py
│   ├── config.py                    # Environment settings with Pydantic & safe credential masking
│   ├── main.py                      # FastAPI app, static mounting, exception handlers, index route
│   ├── schemas.py                   # Request/response schemas & language metadata
│   ├── evaluation/                  # Paired-Contrast Evaluation Engine
│   │   ├── __init__.py
│   │   ├── schemas.py               # Pydantic schemas for test pairs, metrics, and results
│   │   ├── metrics.py               # WER, text normalization, deterministic extractors, scoring
│   │   ├── dataset.py               # Seed dataset loader and validator
│   │   ├── storage.py               # SQLite persistence, aggregate queries, and CSV/JSON export
│   │   ├── engine.py                # Engine orchestrating mock, custom, and live pair runs
│   │   └── data/
│   │       └── seed_pairs.json      # 12 curated test pairs (24 utterances) across 7 categories
│   ├── routers/
│   │   ├── __init__.py
│   │   └── evaluation.py            # REST endpoints for test cases, scoring, metrics, and export
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_validator.py       # Audio container, signature, size, and duration validation
│   │   └── gnani_client.py          # Gnani Prisma STT REST client with 401/403/429/500/503 handling
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css            # Developer-grade responsive styling and evaluation scorecard
│   │   └── js/
│   │       └── app.js               # Frontend tab switcher, audio telemetry, and evaluation UI
│   └── templates/
│       └── index.html               # Dual-tab UI (Single Audio Lab + Paired-Contrast Crash Lab)
└── tests/
    ├── __init__.py
    ├── conftest.py                  # Pytest fixtures and in-memory WAV audio generators
    ├── test_audio_validator.py      # WAV, MP3, OGG, FLAC, AAC, M4A format and duration tests (12 tests)
    ├── test_gnani_client.py         # Mocked client tests: 200, 400, 401, 403, 429, 500, 503, timeouts (11 tests)
    ├── test_routes.py               # FastAPI route tests: /health, /, /api/transcribe (7 tests)
    ├── test_evaluation_engine.py    # WER, normalization, extractors, contrast scoring, SQLite (19 tests)
    └── test_evaluation_routes.py    # Evaluation API route tests: test cases, mock run, metrics, export (8 tests)
```

---

## 4. Setup & Run Instructions (Windows PowerShell)

### Step 1: Navigate to the Project
```powershell
Set-Location -Path "C:\Users\astru\.gemini\antigravity\scratch\bolibench"
```

### Step 2: Install Pinned Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 3: Configure Credentials (Optional for Mock Mode)
Copy the `.env.example` file to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` in a text editor and insert your Gnani API key:
```ini
GNANI_API_KEY=your_actual_key_from_app_gnani_ai
```
*(If you do not have an active key, leave it unconfigured. BoliBench operates fully in **Mock Mode**, providing deterministic demonstrations of both preserved and crashed distinctions with 0 credit consumption).*

### Step 4: Run the Application
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser to:
- **Interactive UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Health Check API:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Interactive OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 5. Running Automated Tests

Run the full automated test suite:
```powershell
python -m pytest -v
```

### Actual Test Suite Result:
All **57 automated tests** run without internet access using in-memory audio fixtures, HTTP transport mocks, and deterministic rule engines:
- `tests/test_audio_validator.py` (12 tests): Validates container formats (`.wav`, `.mp3`, `.ogg`, `.flac`, `.aac`, `.m4a`), size boundaries, magic header bytes, and duration limits (≤ 60s).
- `tests/test_gnani_client.py` (11 tests): Validates request headers (`X-API-Key-ID`), form-data serialization, 200 responses, 400 Bad Request, 401 Unauthorized, 403 Forbidden, 429 Rate Limits, 500/503 errors, malformed responses, optional model fields, and timeouts.
- `tests/test_routes.py` (7 tests): Validates `/health`, HTML root, mock-mode transcription, language validation, and unmocked missing credential handling.
- `tests/test_evaluation_engine.py` (19 tests): Validates exact mathematical WER, semantic token preservation, deterministic extractors (quantities, negation, corrections, entities, intent), paired-contrast scoring, failure classification, and SQLite storage.
- `tests/test_evaluation_routes.py` (8 tests): Validates `/api/eval/test-cases`, `/api/eval/run-mock/{id}`, `/api/eval/evaluate-transcripts`, `/api/eval/metrics`, and CSV/JSON export.

---

## 6. Evaluation API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/eval/test-cases` | List all 12 seed paired-contrast test cases |
| `GET` | `/api/eval/test-cases/{test_id}` | Retrieve specific test case metadata and expected fields |
| `POST` | `/api/eval/run-mock/{test_id}?scenario=ideal_preserved` | Run deterministic mock evaluation (preserves distinction) |
| `POST` | `/api/eval/run-mock/{test_id}?scenario=crashed_distinction` | Run deterministic mock evaluation (demonstrates acoustic crash) |
| `POST` | `/api/eval/evaluate-transcripts` | Evaluate custom or actual transcripts against expected ground truth |
| `GET` | `/api/eval/results` | List evaluation runs stored in SQLite |
| `GET` | `/api/eval/results/{run_id}` | Retrieve specific evaluation run |
| `GET` | `/api/eval/metrics` | Retrieve aggregate metrics (pass rate, task success rate, average WER) |
| `GET` | `/api/eval/export?format=csv` | Export all evaluation runs as CSV |
| `GET` | `/api/eval/export?format=json` | Export all evaluation runs as JSON |

---

## 7. Security Guarantees & Privacy

- **Zero Credential Exposure:** `GNANI_API_KEY` is loaded on the backend via Pydantic Settings and is **never** sent to the browser.
- **Safe Representation:** `Settings.__repr__` masks credentials to prevent secret leakage in logs or terminal outputs.
- **Privacy by Default:** Uploaded audio is processed in-memory as byte streams and is **never** persisted to disk. SQLite stores only structured evaluation metadata.
- **Credit Protection:** Deterministic mock tests consume 0 Gnani credits. Live requests require explicit user configuration and confirmation.
