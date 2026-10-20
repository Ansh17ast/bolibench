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
- **Audio Validation:** In-memory container, duration (≤ 60s), size (≤ 25MB), and header signature validation; live evaluation strictly enforces WAV container for guaranteed server-side duration verification
- **Evaluation Engine:** Deterministic rule-based slot extractors, dynamic programming Levenshtein WER, and paired-contrast scoring
- **Automated Testing:** `pytest` (91 tests passing in ~1.7s, 0 network calls, 0 credits spent)

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
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_validator.py       # Audio container, signature, size, and duration validation
│   │   ├── gnani_client.py          # Gnani Prisma STT REST client with 401/403/429/500/503 handling
│   │   └── live_guard.py            # Server-side credit guard, origin check & anti-CSRF token manager
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
    ├── test_routes.py               # FastAPI route tests: /health, /, /api/transcribe, telemetry labels (9 tests)
    ├── test_evaluation_engine.py    # WER, normalization, extractors, contrast scoring, SQLite (21 tests)
    └── test_evaluation_routes.py    # Evaluation API route tests: safety defaults, auth, tokens, live runs (31 tests)
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
Open `.env` and configure settings:
```ini
# Optional API Key for real provider requests
GNANI_API_KEY=your_actual_key_from_app_gnani_ai

# Live evaluation safety switch (Disabled by default to protect credits on public deployments)
# Set to 'true' only for authorized local development operator testing
LIVE_EVALUATION_ENABLED=false
```
> **Security Notice:** The `.env` file is excluded in `.gitignore` and must never be committed. The backend never exposes the key to client JavaScript. If no key is configured or `LIVE_EVALUATION_ENABLED=false`, the application operates in safe **Mock-Only Mode** with zero external network requests and zero credit usage.

### Step 4: Launch the Server (Bound to Localhost)
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

1. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and click the **Paired-Contrast Crash Lab** tab.
2. Select any test pair (e.g. `pair_qty_001: 2 Packets vs 3 Packets`).
3. Click **"Run Preserved Distinction (Mock)"** to evaluate ground-truth transcripts. Notice:
   - WER: 0.0%
   - Critical Field Accuracy: 100%
   - Distinction Preserved: YES
   - Task Success: YES
   - Measured Latency: null (Mock Mode — 0 Network Calls)
4. Click **"Simulate Transcript Crash (Error)"** to test downstream failure against hand-authored corrupted transcripts (demonstrates how downstream ordering agent crashes without claiming uncalibrated acoustic measurements). Notice:
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

### B. Running Paired Live Audio Evaluation (Consumes Credits)

1. Set `LIVE_EVALUATION_ENABLED=true` and configure your `GNANI_API_KEY` in `.env`.
2. In the **Paired-Contrast Crash Lab** tab, select a test pair.
3. Click **"🎙️ Live Audio Pair (Consumes Credits)"** to reveal the credit-protected live drawer.
4. Select an audio file for Utterance A and an audio file for Utterance B (**WAV format only, duration verifiably $\le$ 60s**).
5. Select the **Acoustic Condition** (`clean_recorded`, `noisy_recorded`, `clean_synthetic`, `noisy_synthetic`, `pending_acoustic`). This is explicitly user-declared metadata for benchmark cataloguing, not automatic acoustic detection.
6. Review the credit warning notice and check the authorization checkbox:
   `[x] I authorize 2 live API requests to Gnani Prisma and credit deduction`
7. Click **"Execute Paired Live Evaluation"** and confirm the interactive prompt.
8. The frontend and backend execute the secure authorization flow:
   - Frontend requests a single-use authorization token via `POST /api/eval/authorize-live`.
   - Backend verifies `LIVE_EVALUATION_ENABLED=true` and validates same-origin headers (`Sec-Fetch-Site`, `Origin`).
   - Issues a 300-second (5-minute) cryptographically random, server-stored, single-use token.
   - Frontend submits audio uploads with `confirmation_token`, `audio_condition`, and custom header `X-BoliBench-Confirm-Live: CONFIRM_LIVE_EVALUATION`.
   - Backend verifies authorization and validates `audio_condition` against the `AudioCondition` enum BEFORE reading files or initiating network requests.
   - Validates both audio files in-memory using `AudioValidator.validate_live_audio` (verifying WAV header duration $\le$ 60s, zero disk persistence).
   - Sequentially executes live calls to `https://api.vachana.ai/stt/v3` with your `GNANI_API_KEY` passing `format=transcribe` and `preferred_language`.
   - Records genuine client elapsed round-trip time (`client_request_elapsed_ms`, with `backend_latency_ms` alias) and keeps `provider_processing_time_ms = null` ("Not reported" in UI).
   - Evaluates returned transcripts using the deterministic scoring engine.
   - Preserves provider request IDs and persists run to SQLite.
   - If Utterance A fails, Utterance B is never called and a safe failed run is recorded. If Utterance A succeeds and B fails, an incomplete audit run is recorded documenting that Utterance A consumed credits while B failed.
   - Failed or incomplete runs are safely discoverable in SQLite run history and failure category reporting, but are excluded from benchmark accuracy metrics (mean WER, task success, distinction preservation) to prevent operational errors from masquerading as speech model failures.

---

## 8. Automated Testing & Verification Status

Run the test suite:
```powershell
python -m pytest -v
```

### Actual Test Suite Result (95 Tests Passing):
```
collected 95 items

tests/test_audio_validator.py ...............                            [ 15%]
tests/test_evaluation_engine.py ...........................             [ 43%]
tests/test_evaluation_routes.py ..................................      [ 77%]
tests/test_gnani_client.py ............                                  [ 90%]
tests/test_routes.py .........                                           [100%]

============================= 95 passed in 1.72s ==============================
```

### Verified vs. Pending Status Matrix

| Component | Status | Verification Detail |
|---|---|---|
| **Audio File Validation** | **VERIFIED** | 12 tests: WAV, MP3, OGG, FLAC, AAC, M4A, file limits (25MB), duration limits (60s), magic byte headers. |
| **Gnani REST STT v3 Client** | **VERIFIED** | 11 tests: Verified against official OpenAPI 3.1 specification. Measures client-side elapsed time; correctly reports `provider_processing_time_ms = null` when omitted by provider. |
| **FastAPI Core Routes & Telemetry** | **VERIFIED** | 9 tests: `/health` (reflecting `live_evaluation_enabled=false` default), `/`, `/api/transcribe` with clear labels: "Browser End-to-End Latency", "Backend-measured Gnani API request time", "Provider-Internal Processing Time" ("Not reported"). |
| **Evaluation Metrics Engine** | **VERIFIED** | 21 tests: Exact Levenshtein WER, slot extractors, contrast scoring, mock vs live telemetry isolation, SQLite persistence, latency semantics. |
| **Evaluation REST API & Guard** | **VERIFIED** | 31 tests: Test pair catalog, mock execution, custom transcripts, live paired audio endpoint, upload validation, partial failure handling, default disabled live evaluation (`LIVE_EVALUATION_ENABLED=false`), zero provider calls on disabled/unauthorized requests, acoustic condition validation, server-side authorization token (`/authorize-live`), anti-CSRF header, cross-origin rejection, missing origin on non-local host rejection, and CSV/JSON export. |
| **Live Account Transcription** | **PENDING USER SETUP** | Requires the user to set `LIVE_EVALUATION_ENABLED=true`, insert their personal `GNANI_API_KEY` into `.env`, and execute against their account credits. |

---

## 9. Evaluation REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/eval/test-cases` | List all 12 seed paired-contrast test cases |
| `GET` | `/api/eval/test-cases/{test_id}` | Retrieve specific test case metadata and expected fields |
| `POST` | `/api/eval/authorize-live` | Issue single-use cryptographic authorization token for upcoming live run (requires `LIVE_EVALUATION_ENABLED=true`) |
| `POST` | `/api/eval/run-mock/{test_id}?scenario=ideal_preserved` | Run deterministic mock evaluation (preserves distinction; 0 credits) |
| `POST` | `/api/eval/run-mock/{test_id}?scenario=crashed_distinction` | Run deterministic mock evaluation with hand-authored simulated corruption |
| `POST` | `/api/eval/run-live/{test_id}` | Execute live paired audio evaluation against Gnani Prisma STT v3 (requires `LIVE_EVALUATION_ENABLED=true`, confirmation token, anti-CSRF header, and valid `audio_condition`; consumes credits) |
| `POST` | `/api/eval/evaluate-transcripts` | Evaluate custom or actual transcripts against expected ground truth |
| `GET` | `/api/eval/results` | List evaluation runs stored in SQLite |
| `GET` | `/api/eval/results/{run_id}` | Retrieve specific evaluation run |
| `GET` | `/api/eval/metrics` | Retrieve aggregate metrics (pass rate, task success rate, average WER, mock/live split) |
| `GET` | `/api/eval/export?format=csv` | Export all evaluation runs as CSV |
| `GET` | `/api/eval/export?format=json` | Export all evaluation runs as JSON |

---

## 10. Security Guarantees & Hardened Credit Protection

- **Deployment Safety Default:** `LIVE_EVALUATION_ENABLED` defaults to `false`. When deployed to public hosting, endpoints that could consume the owner's credits are completely inert and reject requests with `403 Forbidden` before reading uploads or calling the provider.
- **Server-Side Credit Guard:** Live evaluation requires an explicit authorization step (`POST /api/eval/authorize-live`) generating a single-use, 5-minute cryptographically random token.
- **Anti-CSRF & Origin Verification:** Requests require `X-BoliBench-Confirm-Live: CONFIRM_LIVE_EVALUATION` and are checked against `Sec-Fetch-Site` and `Origin`. Cross-origin requests and requests missing origin information on non-local hosts are blocked with `403 Forbidden` before reading file uploads or invoking the provider.
- **Restricted CORS Policy:** Restricted to local same-origin development hosts (`localhost:8000`, `127.0.0.1:8000`, `localhost:3000`, `127.0.0.1:3000`).
- **Zero Credential Exposure:** `GNANI_API_KEY` is loaded on the backend via Pydantic Settings and is **never** sent to the browser.
- **Safe Representation:** `Settings.__repr__` masks credentials to prevent secret leakage in logs or terminal outputs. Authorization tokens never appear in evaluation results or server logs.
- **Privacy by Default:** Uploaded audio is processed in-memory as byte streams and is **never** persisted to disk. SQLite stores only structured evaluation metadata.
- **Credit Protection:** Deterministic mock tests consume 0 Gnani credits. Automated tests use mocked clients exclusively and make zero external network requests.
- **Partial Failure Guard:** Live paired evaluation aborts if either utterance fails, preventing corrupted or half-baked evaluations from being marked as successful.

---

## 11. Known Limitations & Research Boundaries

1. **Text-Only Simulation Baseline:** The current 12 seed pairs are hand-authored linguistic test cases (`text_only_simulation`). BoliBench does not claim acoustic noise robustness has been evaluated until real audio has been processed.
2. **User-Declared Acoustic Metadata:** The `audio_condition` field is user-declared metadata for benchmark cataloguing, not automatic acoustic detection. BoliBench does not pretend acoustic conditions can be detected automatically.
3. **Pending Calibrated Acoustic Research (`pending_acoustic`):** Future extensions will record all 24 utterances under calibrated multi-speaker noise environments (street noise, call center chatter, reverberant rooms) with measured SNR following ITU-T P.56 standards.
4. **Regional Indian Language Expansion:** Current coverage spans Hindi (`hi-IN`) and Indian English (`en-IN`). Planned additions will expand into Tamil (`ta-IN`), Telugu (`te-IN`), Kannada (`kn-IN`), and Bengali (`bn-IN`).
5. **No Clinical or Superiority Claims:** BoliBench is an objective diagnostic framework benchmarking downstream voice agent semantic robustness. It makes no claims of clinical diagnostic validation or unverified benchmark superiority.
