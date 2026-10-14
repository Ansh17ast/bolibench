# BoliBench — Indian Voice AI Crash-Test Lab

> **A diagnostic evaluation lab built for the Great Indian AI Internship Challenge by Gnani AI.**
> BoliBench evaluates how speech-recognition errors alter the meaning of Hindi, Indian English, and Hinglish commands, causing downstream voice agents to perform incorrect actions.

---

## 1. Project Objective & Differentiating Evaluation Engine

Traditional Speech-to-Text (STT) benchmarks rely almost exclusively on **Word Error Rate (WER)**. While WER measures lexical surface edit distance, it treats all token substitutions equally. In production Indian Voice AI agents (e.g. banking IVRs, quick-commerce bots, mobility ticketing systems), acoustic errors are often **task-critical**:

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

## 2. System Architecture

```
                     +---------------------------------------+
                     |         Browser UI & API Client       |
                     |  Tab 1: Single Audio Transcription    |
                     |  Tab 2: Paired-Contrast Crash Lab     |
                     +-------------------+-------------------+
                                         |
                                         v HTTP / JSON
+---------------------------------------------------------------------------------+
|                                FastAPI Backend                                  |
|                                                                                 |
|  [Routers]                                                                      |
|  ├── main.py (Health Check, Static Asset Serving, /api/transcribe)              |
|  └── routers/evaluation.py (/api/eval/test-cases, run-mock, metrics, export)    |
|                                                                                 |
|  [Core Services]                                                                |
|  ├── Audio Validator (In-memory container, signature, duration <= 60s, 25MB)   |
|  └── Gnani Prisma Client (REST v3: api.vachana.ai, auth, 400/401/403/429/503)  |
|                                                                                 |
|  [Evaluation Engine]                                                            |
|  ├── Dataset Loader (Curated seed_pairs.json with 12 pairs / 24 utterances)     |
|  ├── Deterministic Extractors (Numbers/Scales, Negation, Corrections, Intent)   |
|  ├── Metric Calculators (DP Levenshtein WER, Field Acc, Distinction Preserved)  |
|  └── Storage Manager (Thread-safe SQLite bolibench_evaluation.db + CSV/JSON)    |
+---------------------------------------------------------------------------------+
```

---

## 3. Key Evaluation Metrics Explained

BoliBench tracks four primary metrics to measure both acoustic accuracy and semantic reliability:

1. **Word Error Rate (WER):**
   Computed using dynamic programming Levenshtein distance on normalized word sequences:
   $$\text{WER} = \frac{S + D + I}{N}$$
   Where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is reference word count. Text normalization preserves semantic markers (numbers, negation tokens) while stripping punctuation and casing.

2. **Critical-Field Accuracy ($Acc_{field}$):**
   Measures whether specific structured business fields (e.g. quantity, amount, destination, negation status) extracted from the transcription match ground-truth values:
   $$Acc_{field} = \frac{\text{Correctly Extracted Fields}}{\text{Total Critical Fields Defined}}$$

3. **Semantic Distinction Accuracy / Preservation Rate ($Preserved(A, B)$):**
   Determines whether the critical distinction between contrasting utterances $A$ and $B$ was maintained:
   $$Preserved(A, B) = \begin{cases}
   1 & \text{if } \hat{v}_{A, k} = v_{A, k}^* \;\land\; \hat{v}_{B, k} = v_{B, k}^* \;\land\; \hat{v}_{A, k} \neq \hat{v}_{B, k} \\
   0 & \text{otherwise}
   \end{cases}$$
   If both utterances collapse to the same extracted value or fail extraction, the distinction has crashed.

4. **Downstream Task Success Rate ($TaskSuccess$):**
   Evaluates whether an automated voice agent receiving the transcription would execute the correct intended business action without triggering an erroneous or destructive command:
   $$TaskSuccess(A, B) = TaskSuccess(A) \;\land\; TaskSuccess(B) \;\land\; Preserved(A, B)$$

---

## 4. Verified Technology Stack

- **Speech Recognition Engine:** Gnani Prisma v2.5 STT (REST v3: `https://api.vachana.ai/stt/v3`)
- **Backend:** FastAPI (Python 3.10+) with Pydantic v2 Settings & Schemas
- **Database:** Thread-safe SQLite persistence (`bolibench_evaluation.db`) with CSV and JSON streaming exports
- **Frontend:** Vanilla HTML5, CSS3, and JavaScript (Zero external CDN dependencies, self-contained)
- **Audio Validation:** In-memory container, duration (≤ 60s), size (≤ 25MB), and header signature validation
- **Evaluation Engine:** Deterministic rule-based slot extractors, dynamic programming Levenshtein WER, and paired-contrast scoring
- **Automated Testing:** `pytest` (57 tests passing in < 1 second, 0 network calls, 0 credits spent)

Official Gnani API verification details are documented in [`docs/gnani_api_verification.md`](docs/gnani_api_verification.md).

---

## 5. Directory Layout

```
bolibench/
├── .env.example                     # Sample environment variable template
├── .gitignore                       # Git ignore preventing audio, caches, logs, and secrets leak
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
│   │   └── __init__.py
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

## 6. Setup & Execution Guide (Windows PowerShell)

### Step 1: Navigate to the Project Root
```powershell
Set-Location -Path "C:\Users\astru\.gemini\antigravity\scratch\bolibench"
```

### Step 2: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 3: Configure Gnani Prisma Securely (Optional for Mock Mode)
Copy the template configuration:
```powershell
Copy-Item .env.example .env
```
Open `.env` and set your key:
```ini
GNANI_API_KEY=your_actual_key_from_app_gnani_ai
```
> **Security Notice:** The `.env` file is excluded in `.gitignore` and must never be committed. The backend never exposes the key to client JavaScript. If no key is configured, the application functions in **Mock Mode** with zero external network requests and zero credit usage.

### Step 4: Launch the Server
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser to:
- **Interactive UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Health Check API:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Interactive OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 7. How to Run Evaluations

### A. Running Mock Evaluations (Zero Credits Spent)

1. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and click the **Crash-Test Evaluation Lab** tab.
2. Select any test pair (e.g. `pair_qty_001: 2 Packets vs 3 Packets`).
3. Click **"Run Ideal (Preserved)"** to evaluate ground-truth transcripts. Notice:
   - WER: 0.0%
   - Critical Field Accuracy: 100%
   - Distinction Preserved: YES
   - Task Success: YES
4. Click **"Run Crashed Distinction"** to simulate an acoustic speech recognition error where the quantity slot collapses. Notice:
   - Failure Classification: `QUANTITY_ERROR`
   - Distinction Preserved: NO
   - Task Success: NO
   - Diagnostic breakdown explaining the downstream business impact.

You can also run evaluations via curl or PowerShell:
```powershell
# Ideal Preserved Run
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/eval/run-mock/pair_qty_001?scenario=ideal_preserved"

# Crashed Distinction Run
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/eval/run-mock/pair_qty_001?scenario=crashed_distinction"
```

### B. Running Live Transcription with Gnani Prisma

1. In the **Single Audio Lab** tab, select the audio language (e.g. `Hindi (hi-IN)` or `English - India (en-IN)`).
2. Choose formatting mode (`Verbatim` or `Transcribe` with Inverse Text Normalization).
3. Toggle the **Mock Mode** switch off to make a live request to `https://api.vachana.ai/stt/v3`.
4. Upload an audio file (`.wav`, `.mp3`, `.ogg`, `.flac`, `.aac`, `.m4a` ≤ 60 seconds).
5. Click **Transcribe Audio**. The client reports provider latency ($T_{\text{provider}}$) and client round-trip latency ($T_{\text{client}}$).

---

## 8. Automated Testing & Verification Status

Run the test suite:
```powershell
python -m pytest -v
```

### Actual Test Suite Result (57 Tests Passing):
```
collected 57 items

tests/test_audio_validator.py ............                               [ 21%]
tests/test_evaluation_engine.py ...................                     [ 54%]
tests/test_evaluation_routes.py ........                                [ 68%]
tests/test_gnani_client.py ...........                                   [ 87%]
tests/test_routes.py .......                                             [100%]

============================= 57 passed in 0.58s ==============================
```

### Verified vs. Pending Status Matrix

| Component | Status | Verification Detail |
|---|---|---|
| **Audio File Validation** | **VERIFIED** | 12 tests: WAV, MP3, OGG, FLAC, AAC, M4A, file limits (25MB), duration limits (60s), magic byte headers. |
| **Gnani REST STT v3 Client** | **VERIFIED** | 11 tests: Verified against official OpenAPI 3.1 specification. Handles 200, 400, 401, 403, 429, 500, 503, timeouts, and malformed responses. |
| **FastAPI Core Routes** | **VERIFIED** | 7 tests: `/health`, `/`, `/api/transcribe` in mock and live modes with validation. |
| **Evaluation Metrics Engine** | **VERIFIED** | 19 tests: Exact Levenshtein WER, slot extractors (quantities, negation, corrections, entities, intent), contrast scoring, and SQLite persistence. |
| **Evaluation REST API** | **VERIFIED** | 8 tests: Test pair catalog, mock execution, custom transcript evaluation, aggregate metrics, and CSV/JSON export. |
| **Live Account Transcription** | **PENDING USER SETUP** | Requires the user to insert their personal `GNANI_API_KEY` into `.env` and execute against their account credits. |

---

## 9. Evaluation REST API Reference

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

## 10. Security Guarantees & Privacy

- **Zero Credential Exposure:** `GNANI_API_KEY` is loaded on the backend via Pydantic Settings and is **never** sent to the browser.
- **Safe Representation:** `Settings.__repr__` masks credentials to prevent secret leakage in logs or terminal outputs.
- **Privacy by Default:** Uploaded audio is processed in-memory as byte streams and is **never** persisted to disk. SQLite stores only structured evaluation metadata.
- **Credit Protection:** Deterministic mock tests consume 0 Gnani credits. Live requests require explicit user configuration and confirmation.

---

## 11. Known Limitations & Future Work

1. **Synthetic Clean Baseline:** The current 12 seed pairs are clean baseline utterances labeled as `synthetic_curated`. BoliBench does not fabricate artificial acoustic noise scores without calibrated ambient multi-speaker audio.
2. **Language Coverage:** The current evaluation engine focuses on Hindi (`hi-IN`) and Indian English (`en-IN`), which cover the majority of code-switching voice agents in India. Future work will expand the paired-contrast catalog to Tamil (`ta-IN`), Telugu (`te-IN`), Kannada (`kn-IN`), and Bengali (`bn-IN`).
3. **No Clinical or Superiority Claims:** BoliBench is a diagnostic evaluation framework designed to benchmark speech-recognition pipelines under paired semantic contrast. It makes no claims of clinical diagnostic validation or unverified benchmark superiority.
4. **Planned Acoustic Benchmark Drive:** Future work includes a crowdsourced recording initiative to collect authentic microphone recordings across diverse Indian regional accents for all 24 utterances.
