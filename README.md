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
                     +-------------------------------------------------------+
                     |                Browser UI & Audio Client              |
                     |  Tab 1: Single Audio Transcription                    |
                     |  Tab 2: Paired-Contrast Crash Lab                     |
                     |  Tab 3: Human Recording Studio (/studio)              |
                     |         - Client-Side 16kHz WAV Converter             |
                     |         - Dual-Utterance Contrast Recording           |
                     |         - In-Browser Audio Diagnostics Preview        |
                     +---------------------------+---------------------------+
                                                 | HTTP / JSON / Multipart
                                                 v
+-----------------------------------------------------------------------------------------+
|                                    FastAPI Backend                                      |
|                                                                                         |
|  [Routers]                                                                              |
|  ├── main.py (Health Check, Static Mounting, /api/transcribe)                           |
|  ├── routers/evaluation.py (/api/eval/test-cases, run-mock, run-live, resume-live,       |
|  │                          human-pairs, metrics, export, derived-assessments)          |
|  └── routers/recording.py (/api/studio/upload, recordings, pairs, verify,               |
|                           trusted-speakers, config)                                     |
|                                                                                         |
|  [Core Services]                                                                        |
|  ├── Audio Validator (In-memory container, signature, duration <= 60s, 25MB)           |
|  ├── Human Audio Processor & Diagnostics (16kHz PCM standardization, silence, clipping) |
|  ├── Recording Storage Manager (Idempotent SHA-256 deduplication, atomic writes,        |
|  │                             fail-closed trusted-speaker registry)                    |
|  ├── Live Credit Guard (300s single-use tokens, CSRF origin verification, flags)        |
|  └── Gnani Prisma Client (REST v3: api.vachana.ai, auth, 400/401/403/429/500/503)       |
|                                                                                         |
|  [Evaluation & Resumption Engine]                                                       |
|  ├── Dataset Loader & Manifest Resolver (seed_pairs.json, benchmark_manifest_v2.0.0)    |
|  ├── Evaluation Resumption Engine (Resumes partial failures without re-charging credits) |
|  ├── Dataset Classification Cohorts (human_verified, synthetic, test_artifact, unverified) |
|  ├── Deterministic Extractors (Numbers/Scales, Negation, Corrections, Intent)           |
|  ├── Metric Calculators (DP Levenshtein WER, Cross-Script Normalizer, Distinction Rate) |
|  └── Storage Manager (Thread-safe SQLite in WAL mode with busy timeout + CSV/JSON)      |
+-----------------------------------------------------------------------------------------+
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

5. **Supplementary Cross-Script Normalized WER ($WER_{norm}$):**
   In Indian voice AI evaluation, reference transcripts are frequently recorded in Romanized Hindi (Latin script, e.g. *"Do packet chahiye"*), while production Indian STT models (such as Gnani Prisma STT v3) output native Devanagari script and Hindu-Arabic numerals (e.g. *"2 पैकेट चाहिए"*).
   - **Why Raw WER is 1.0 (100%):** Dynamic programming Levenshtein distance compares surface characters. Because Latin letters share zero token overlap with Devanagari characters, all words are scored as substitutions ($S = 3, D = 0, I = 0 \rightarrow \text{WER}_{raw} = 1.0$).
   - **Why Normalized WER is 0.0 (0%):** BoliBench's `canonicalize_cross_script_text()` deterministically canonicalizes supported domain equivalences (`"Do"` before quantified nouns $\rightarrow$ `"2"`, `"Teen"` $\rightarrow$ `"3"`, `"packet"` $\leftrightarrow$ `"पैकेट"`, `"chahiye"` $\leftrightarrow$ `"चाहिए"`), matching the canonical tokens exactly ($WER_{norm} = 0.0$).
   - **Task-Specific Scope & Disambiguation:** This normalizer is strictly scoped to supported benchmark vocabulary and is **not** a universal Hindi transliteration engine. It contextually disambiguates ambiguous tokens (e.g. Hindi `"do"` / `"दो"` is preserved as a verb when preceded by auxiliary stems like `"kar"`, `"bhej"`, or `"cancel"`, and is only converted to `"2"` when preceding a quantified noun).
   - **Offline Derived Re-Scoring:** While the original live measurement remains immutable in SQLite (`wer_a = 1.0`, `wer_b = 1.0`, `task_success = False`, `primary_failure_category = intent_error`), BoliBench computes and exports an offline derived re-score (`rescore_task_success = True`, `normalized_wer = 0.0`) with provenance `offline_deterministic_rescore`, consuming **0 external provider calls** and **0 credits**.

---

## 4. Verified Technology Stack

- **Speech Recognition Engine:** Gnani Prisma v2.5 STT (REST v3: `https://api.vachana.ai/stt/v3`)
- **Backend:** FastAPI (Python 3.10+) with Pydantic v2 Settings & Schemas
- **Database:** Thread-safe SQLite persistence (`bolibench_evaluation.db`) in WAL mode (`journal_mode=WAL`, `busy_timeout=5000ms`) with CSV and JSON streaming exports, dedicated `derived_assessments` table for offline re-scores, and atomic schema migration tracking.
- **Frontend & Human Studio:** Vanilla HTML5, CSS3, and JavaScript (Zero external CDN dependencies, self-contained) featuring:
  - Dual-tab evaluation UI with judge-ready side-by-side contrast audit modal.
  - Dedicated **Human Recording Studio** (`/studio`) with client-side 16kHz linear PCM WAV converter (`audio_converter.js`), microphone lifecycle management, and session state preservation.
- **Audio Validation & Processing:** In-memory container, duration (≤ 60s), size (≤ 25MB), and header signature validation; client and server WAV standardization; acoustic diagnostics (RMS energy, digital clipping, silence detection).
- **Storage & Upload Reliability:** SHA-256 idempotency locks, atomic write-replace file staging, race-condition mitigation, and fail-closed trusted-speaker registry.
- **Evaluation Engine:** Deterministic rule-based slot extractors, dynamic programming Levenshtein WER, cross-script canonicalization, paired-contrast scoring, evaluation resumption, and dataset classification cohorts.
- **Automated Testing:**
  - `pytest`: **262 Python unit and integration tests passing in ~29s** (0 live network calls, 0 credits spent, temporary isolated filesystem and SQLite database fixtures).
  - Node.js test runner: **39 frontend unit tests passing in <1s** (`test_audio_converter.mjs`, `test_frontend_selection.mjs`, `test_studio_remediation.mjs`).
  - **Total verified suite: 301 tests passing.**

Official Gnani API verification details are documented in [`docs/gnani_api_verification.md`](docs/gnani_api_verification.md).

---

## 5. Directory Layout

```
bolibench/
├── .env.example                     # Sample environment variable template
├── .gitignore                       # Git ignore preventing audio, caches, logs, WAL files, and secrets leak
├── README.md                        # Project documentation and runbook
├── requirements.txt                 # Pinned dependencies
├── pytest.ini                       # Test configuration
├── docs/
│   ├── gnani_api_verification.md    # Official Gnani Prisma API verification document
│   ├── evaluation_methodology.md   # Mathematical formulas, extractors, and failure taxonomy
│   └── hinglish_transcription_policy.md # Cross-script transcription normalization rules
├── app/
│   ├── __init__.py
│   ├── config.py                    # Environment settings with Pydantic & safe credential masking
│   ├── main.py                      # FastAPI app, static mounting, router inclusion, health route
│   ├── schemas.py                   # Request/response schemas & language metadata
│   ├── recording_schemas.py         # Schemas for recordings, pairs, diagnostics, and classifications
│   ├── evaluation/                  # Paired-Contrast Evaluation Engine
│   │   ├── __init__.py
│   │   ├── schemas.py               # Pydantic schemas for test pairs, metrics, and results
│   │   ├── metrics.py               # WER, text normalization, deterministic extractors, scoring
│   │   ├── dataset.py               # Seed dataset loader and validator
│   │   ├── storage.py               # SQLite persistence (WAL mode), aggregate queries, CSV/JSON export
│   │   ├── engine.py                # Engine orchestrating mock, custom, live, and resumed pair runs
│   │   ├── classifier.py            # Dataset classification cohorts (human_verified, synthetic, etc.)
│   │   ├── batch_runner.py          # Batch evaluation execution orchestrator
│   │   └── data/
│   │       └── seed_pairs.json      # 12 curated test pairs (24 utterances) across 7 categories
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── evaluation.py            # REST endpoints for test cases, live runs, resumption, and metrics
│   │   └── recording.py             # Human Recording Studio endpoints (/api/studio/*)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── admin_guard.py           # Administrative authorization guard (ADMIN_API_KEY Bearer/header)
│   │   ├── audio_validator.py       # Audio container, signature, size, and duration validation
│   │   ├── gnani_client.py          # Gnani Prisma STT REST client with robust status code handling
│   │   ├── live_guard.py            # Server-side credit guard, origin check & anti-CSRF token manager
│   │   ├── recording_storage.py     # Recording storage manager with idempotency and trusted speaker gate
│   │   └── recording_processor.py   # Human audio processor, acoustic diagnostics, and WAV conversion
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css            # Responsive styling, scorecard components, and studio layouts
│   │   └── js/
│   │       ├── app.js               # Dual-tab evaluation UI, telemetry, and live modal controller
│   │       ├── studio.js            # Human Recording Studio recording and workflow controller
│   │       └── audio_converter.js   # Client-side 16kHz mono linear PCM WAV encoder
│   └── templates/
│       ├── index.html               # Dual-tab UI (Single Audio Lab + Paired-Contrast Crash Lab)
│       └── studio.html              # Human Recording Studio interface
└── tests/
    ├── conftest.py                  # Pytest fixtures and isolated temporary storage/database redirections
    ├── test_audio_validator.py      # Audio format, duration, and limit tests (15 tests)
    ├── test_batch_runner.py         # Batch runner orchestration tests (27 tests)
    ├── test_benchmark_v2_manifest.py# Benchmark v2 manifest schema and validation tests (5 tests)
    ├── test_dataset_integrity.py    # Classification and cohort segregation tests (6 tests)
    ├── test_evaluation_engine.py    # WER, extractors, contrast scoring, and SQLite tests (37 tests)
    ├── test_evaluation_quality_phase4.py # End-to-end evaluation quality tests (8 tests)
    ├── test_evaluation_resumption.py # Partial evaluation resumption tests (5 tests)
    ├── test_evaluation_routes.py    # Evaluation API route tests (50 tests)
    ├── test_frontend_experience.py  # Frontend navigation and modal tests (15 tests)
    ├── test_gnani_client.py         # Mocked Gnani STT client tests (12 tests)
    ├── test_human_recording_studio.py # Human Recording Studio backend tests (26 tests)
    ├── test_human_workflow_acceptance.py # End-to-end studio workflow acceptance tests (3 tests)
    ├── test_live_transcripts_regression.py # Live transcript regression tests (4 tests)
    ├── test_recording_isolation_regression.py # Storage isolation regression tests (2 tests)
    ├── test_routes.py               # Core FastAPI route tests (19 tests)
    ├── test_storage_concurrency_and_migrations.py # SQLite concurrency and migration tests (4 tests)
    ├── test_storage_isolation_regression.py # Storage redirect isolation tests (4 tests)
    ├── test_trusted_speakers_auth.py # Administrative authorization guard tests (9 tests)
    ├── test_trusted_speakers_fail_closed.py # Fail-closed trusted-speaker gate tests (7 tests)
    ├── test_upload_idempotency.py   # Upload idempotency and concurrency tests (4 tests)
    ├── test_audio_converter.mjs     # Node.js WAV encoder unit tests (11 tests)
    ├── test_frontend_selection.mjs  # Node.js frontend selection tests (17 tests)
    └── test_studio_remediation.mjs  # Node.js studio UI remediation tests (11 tests)
```

---

## 6. Setup & Execution Guide (Windows PowerShell)

### Step 1: Navigate to the Project Root
```powershell
Set-Location -Path "<path-to-bolibench>"
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

# Administrative secret for trusted speaker onboarding
ADMIN_API_KEY=your_admin_secret_key_here

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
- **Human Recording Studio:** [http://127.0.0.1:8000/studio](http://127.0.0.1:8000/studio)
- **Health Check API:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Interactive OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 7. How to Run Evaluations & Recording Studio

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

### B. Human Recording Studio (`/studio`) Workflow

BoliBench includes a dedicated **Human Recording Studio** for capturing, inspecting, and standardizing genuine spoken contrastive pairs:

1. Open [http://127.0.0.1:8000/studio](http://127.0.0.1:8000/studio).
2. Select a scenario template from 6 domain categories (Quantities, Negation, Corrections, Financial Magnitudes, Named Entities, Code-Switching) or define custom contrast transcripts.
3. Grant microphone permission and record **Utterance A** and **Utterance B**.
4. The client-side WAV converter (`audio_converter.js`) automatically encodes the browser audio into 16kHz mono linear PCM WAV before transmission.
5. Ingested recordings are stored with SHA-256 idempotency protection in `human_recordings_v1/`.
6. **Curated Human Quality Gate:** To qualify a pair for official human-verified benchmark evaluation:
   - Both utterances must have operator-verified transcripts.
   - Explicit speaker consent must be confirmed.
   - Audio diagnostics must verify clean decoding, absence of digital clipping, and non-silent signal.
   - Both speakers must be registered in the **Trusted Speaker Registry** (`POST /api/studio/trusted-speakers`).
   - The operator must explicitly verify and seal the contrast via `POST /api/studio/pairs/{pair_id}/verify`.

### C. Running Paired Live Audio Evaluation (Consumes Credits)

1. Set `LIVE_EVALUATION_ENABLED=true` and configure your `GNANI_API_KEY` in `.env`.
2. In the **Paired-Contrast Crash Lab** tab, select a test pair (synthetic seed pair or verified human pair).
3. Click **"🎙️ Live Audio Pair (Consumes Credits)"** to reveal the credit-protected live drawer.
4. Select an audio file for Utterance A and an audio file for Utterance B (**WAV format only, duration verifiably $\le$ 60s**).
5. Select the **Acoustic Condition** (`clean_recorded`, `noisy_recorded`, `clean_synthetic`, `noisy_synthetic`, `pending_acoustic`). This is user-declared metadata for benchmark cataloguing.
6. Review the credit warning notice and check the authorization checkbox:
   `[x] I authorize 2 live API requests to Gnani Prisma and credit deduction`
7. Click **"Execute Paired Live Evaluation"** and confirm the interactive prompt.
8. The system executes the secure authorization flow via single-use 300s token, checks same-origin headers, validates WAV audio in memory, sequentially invokes Gnani Prisma STT v3, and stores results.

### D. Evaluation Resumption (Credit-Saving Interrupted Run Recovery)

If a live paired evaluation is interrupted (e.g., Utterance A succeeded but Utterance B encountered a network timeout):
- BoliBench records the partial result with `resumable: true` and preserves the provider's Utterance A transcript and request ID.
- Operators can query interrupted runs via `GET /api/eval/resumable-runs`.
- Resuming via `POST /api/eval/resume-live/{run_id}` executes **only the missing utterance**, reusing the historical transcript for the completed utterance and consuming **exactly 1 API credit** instead of re-charging for both.

---

## 8. Automated Testing & Verification Status

Run the complete test suite:

```powershell
# 1. Run all Python unit and integration tests (262 tests)
python -m pytest tests -q

# 2. Run all Node.js frontend and converter tests (39 tests)
node --test tests/test_audio_converter.mjs tests/test_frontend_selection.mjs tests/test_studio_remediation.mjs
```

### Actual Test Suite Result (301 Tests Passing):

```
============================= 262 passed in ~29.3s =============================
TAP version 13
# tests 39
# suites 0
# pass 39
# fail 0
# cancelled 0
# skipped 0
# todo 0
# duration_ms 350.9631
```

### Verified Component Test Matrix

| Component / Test Suite | Test Count | Verification Scope |
|---|---|---|
| **Audio File Validation** (`test_audio_validator.py`) | 15 tests | WAV, MP3, OGG, FLAC, AAC, M4A container validation, duration checks (≤ 60s), 25MB limits, magic byte headers, linear PCM inspection. |
| **Batch Runner & Resumption** (`test_batch_runner.py`, `test_evaluation_resumption.py`) | 32 tests | Batch manifest execution, dry-run reports, partial provider recovery, resumption idempotency, run ID stability. |
| **Benchmark Manifest v2** (`test_benchmark_v2_manifest.py`) | 5 tests | Manifest schema adherence, 4 batches (48 pairs / 96 utterances), within-pair distinctness, subscription non-renewal annotations. |
| **Dataset Classification & Cohorts** (`test_dataset_integrity.py`) | 6 tests | Logical segregation of `human_verified`, `synthetic`, `test_artifact`, and `unverified` cohorts; query filtering (`curated_only`, `exclude_test_artifacts`). |
| **Evaluation Metrics Engine** (`test_evaluation_engine.py`) | 37 tests | Exact Levenshtein WER, cross-script canonicalization (`do` vs `2`), slot extractors, contrast scoring, derived offline re-scores. |
| **Evaluation Quality Gate** (`test_evaluation_quality_phase4.py`) | 8 tests | End-to-end evaluation quality, transcript verification gating, explicit consent enforcement, acoustic diagnostics enforcement, contrast preservation. |
| **Evaluation REST API & Credit Guard** (`test_evaluation_routes.py`) | 50 tests | Test catalog, mock execution, custom transcripts, live paired endpoint, upload validation, disabled defaults, token validation, CSRF origin rejection. |
| **Frontend Experience & Modals** (`test_frontend_experience.py`) | 15 tests | Multi-tab navigation, mode-separated dashboards, crash lab summaries, audio preview controls, history filtering, side-by-side contrast modal. |
| **Gnani REST STT v3 Client** (`test_gnani_client.py`) | 12 tests | Official OpenAPI 3.1 compliance; 200, 400, 401, 403, 429, 500, 503 handling; client elapsed time measurement; null latency handling. |
| **Human Recording Studio** (`test_human_recording_studio.py`, `test_human_workflow_acceptance.py`) | 29 tests | Microphone recording, scenario switching, transcript preservation, client WAV conversion, double-click protection, pair verification workflow. |
| **Live Transcripts Regression** (`test_live_transcripts_regression.py`) | 4 tests | Named entities, command intent, corrections, and completed pairs distinction preservation across historical live transcripts. |
| **FastAPI Core Routes & Telemetry** (`test_routes.py`) | 19 tests | `/health`, `/`, `/api/transcribe` with explicit telemetry labels, mock mode, disabled defaults, latency separation, environment overrides. |
| **Storage Concurrency & WAL** (`test_storage_concurrency_and_migrations.py`) | 4 tests | SQLite WAL mode, 5000ms busy timeout, concurrent reads/writes without table lock errors, atomic migration markers. |
| **Test & Storage Isolation** (`test_storage_isolation_regression.py`, `test_recording_isolation_regression.py`) | 6 tests | Guarantees test suites redirect to isolated temporary directories; production SQLite DB and recordings remain 100% unmutated. |
| **Admin Authorization Guard** (`test_trusted_speakers_auth.py`) | 9 tests | Fail-closed unconfigured server rejection (403), missing/empty credentials rejection (401), invalid credentials rejection (403), Bearer and X-Admin-Token authorization, test prefix rejection, privacy masking. |
| **Trusted Speaker Gate** (`test_trusted_speakers_fail_closed.py`) | 7 tests | Missing/corrupt registry fails closed; empty registry fails closed; unknown speakers untrusted; test prefixes strictly rejected; pair verification gate. |
| **Upload Idempotency** (`test_upload_idempotency.py`) | 4 tests | SHA-256 payload hashing, duplicate upload detection, race-condition mitigation, and payload collision detection. |
| **Client-Side WAV Converter** (`test_audio_converter.mjs`) | 11 tests | Node.js unit tests for RIFF/WAVE header assembly, 16kHz linear PCM encoding, 16-bit signed LE integer clamping, duration checks. |
| **Frontend Selection Suite** (`test_frontend_selection.mjs`) | 17 tests | Node.js DOM tests for dynamic test case selection, category filtering, hash change routing, and modal state isolation. |
| **Studio UI Remediation** (`test_studio_remediation.mjs`) | 11 tests | Node.js tests for audio device switching, session recovery, transcript preservation on decline, and save button de-bouncing. |
| **Total Test Suite** | **301 tests** | **100% PASS** (262 Python + 39 Node.js) with 0 live provider calls and 0 credits spent. |

---

## 9. REST API Reference

### Evaluation & Diagnostic Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/eval/test-cases` | List all 12 seed paired-contrast test cases |
| `GET` | `/api/eval/test-cases/{test_id}` | Retrieve specific test case metadata and expected fields |
| `POST` | `/api/eval/authorize-live` | Issue single-use cryptographic authorization token for upcoming live run (requires `LIVE_EVALUATION_ENABLED=true`) |
| `POST` | `/api/eval/run-mock/{test_id}?scenario=ideal_preserved` | Run deterministic mock evaluation (preserves distinction; 0 credits) |
| `POST` | `/api/eval/run-mock/{test_id}?scenario=crashed_distinction` | Run deterministic mock evaluation with hand-authored simulated corruption |
| `POST` | `/api/eval/run-live/{test_id}` | Execute live paired audio evaluation against Gnani Prisma STT v3 (requires token, anti-CSRF header, and valid `audio_condition`; consumes credits) |
| `GET` | `/api/eval/human-pairs` | List human-recorded contrast pairs with query filters (`classification`, `curated_only`, `exclude_test_artifacts`, `category`, `status`) |
| `POST` | `/api/eval/run-human-pair-live/{pair_id}` | Execute live paired evaluation for a human recording pair against Gnani Prisma STT v3 |
| `POST` | `/api/eval/run-human-pairs-batch-live` | Execute batch live evaluation across multiple human recording pairs |
| `GET` | `/api/eval/resumable-runs` | List interrupted evaluation runs eligible for resumption |
| `POST` | `/api/eval/resume-live/{run_id}` | Resume an interrupted evaluation run (executes only the missing utterance, reusing historical transcript) |
| `POST` | `/api/eval/evaluate-transcripts` | Evaluate custom or actual transcripts against expected ground truth |
| `GET` | `/api/eval/results` | List evaluation runs stored in SQLite |
| `GET` | `/api/eval/results/{run_id}` | Retrieve specific evaluation run |
| `GET` | `/api/eval/runs/{run_id}/derived-rescore` | Retrieve offline deterministic derived assessment for a specific run without mutating source |
| `GET` | `/api/eval/derived-assessments` | List all offline derived assessments |
| `GET` | `/api/eval/derived-assessments/export?format=csv` | Export all offline derived assessments as CSV |
| `GET` | `/api/eval/derived-assessments/export?format=json` | Export all offline derived assessments as JSON |
| `GET` | `/api/eval/metrics` | Retrieve aggregate metrics (pass rate, task success rate, average WER, cohort segregation) |
| `GET` | `/api/eval/export?format=csv` | Export all evaluation runs as CSV |
| `GET` | `/api/eval/export?format=json` | Export all evaluation runs as JSON |

### Human Recording Studio Endpoints (`/api/studio/*`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/studio/config` | Returns studio configuration, capabilities, 6 categories, and scenario templates |
| `POST` | `/api/studio/upload` | Upload audio recording with idempotency protection, 16kHz WAV standardization, and acoustic diagnostics |
| `GET` | `/api/studio/recordings` | List stored human recordings with optional category, speaker, or classification filters |
| `GET` | `/api/studio/recordings/{id}/metadata` | Retrieve complete metadata and acoustic diagnostics for an individual recording |
| `GET` | `/api/studio/recordings/{id}/audio` | Stream the standardized 16kHz WAV audio file for browser playback |
| `GET` | `/api/studio/pairs` | List candidate and completed contrast pairs with utterance associations and verification statuses |
| `POST` | `/api/studio/pairs/create` | Create a matched contrast pair container linking Utterance A and Utterance B |
| `POST` | `/api/studio/pairs/{pair_id}/verify` | Formally verify and seal a contrast pair (requires manual transcripts, consent, diagnostics, and trusted speakers) |
| `GET` | `/api/studio/trusted-speakers` | List registered trusted human speakers |
| `POST` | `/api/studio/trusted-speakers` | Register a verified human speaker in the trusted speaker registry (requires administrative authorization via `ADMIN_API_KEY` Bearer or `X-Admin-Token`; rejects test/mock prefixes) |

---

## 10. Security Guarantees & Reliability Controls

- **Deployment Safety Default:** `LIVE_EVALUATION_ENABLED` defaults to `false`. When deployed to public hosting, endpoints that could consume credits are completely inert and reject requests with `403 Forbidden` before reading uploads or calling the provider.
- **Server-Side Credit Guard:** Live evaluation requires an explicit authorization step (`POST /api/eval/authorize-live`) generating a single-use, 5-minute cryptographically random token.
- **Anti-CSRF & Origin Verification:** Requests require `X-BoliBench-Confirm-Live: CONFIRM_LIVE_EVALUATION` and are checked against `Sec-Fetch-Site` and `Origin`. Cross-origin requests and requests missing origin information on non-local hosts are blocked with `403 Forbidden`.
- **Restricted CORS Policy:** Restricted to local same-origin development hosts (`localhost:8000`, `127.0.0.1:8000`, `localhost:3000`, `127.0.0.1:3000`).
- **Zero Credential Exposure:** `GNANI_API_KEY` is loaded on the backend via Pydantic Settings and is **never** sent to the browser.
- **Safe Representation:** `Settings.__repr__` masks credentials to prevent secret leakage in logs or terminal outputs. Authorization tokens never appear in evaluation results or server logs.
- **Upload Idempotency:** The recording pipeline computes SHA-256 hashes of upload payloads and enforces per-process/thread locks to prevent duplicate file creation or race conditions on rapid double-clicks.
- **Administrative Authorization Gate:** `POST /api/studio/trusted-speakers` enforces server-side administrative authorization (`ADMIN_API_KEY`). Requests without credentials are rejected with `401 Unauthorized`; invalid keys are rejected with `403 Forbidden` via constant-time comparison (`secrets.compare_digest`). If `ADMIN_API_KEY` is not configured on the server, the endpoint fails closed (`403 Forbidden`). Credentials are never logged.
- **Fail-Closed Trusted Speaker Gate:** If `trusted_speakers.json` is missing or corrupted, the system fails closed (empty registry). Unknown speakers and test/mock prefixes (`spk_test`, `spk_mock`, `spk_fail`, etc.) can never be verified as human-verified benchmark pairs.
- **SQLite WAL Mode & Concurrency:** Evaluation database operates with `journal_mode=WAL` and `busy_timeout=5000ms`, enabling concurrent non-blocking reads and writes without table locking errors.
- **Test Isolation Fixtures:** Test suites redirect storage and database paths to isolated temporary folders, ensuring production databases, recording manifests, and benchmark artifacts remain completely unmodified.

---

## 11. Known Limitations, Release Blockers & Research Boundaries

1. **RESOLVED — Administrative Trusted-Speaker Authorization:**
   - `POST /api/studio/trusted-speakers` is protected by a server-side fail-closed authorization guard (`verify_admin_authorization`).
   - Requires administrative authorization via `Authorization: Bearer <token>` or `X-Admin-Token: <token>` matching `ADMIN_API_KEY`.
   - Rejects unauthenticated requests with `401 Unauthorized` and invalid credentials with `403 Forbidden` using constant-time comparison. Fails closed with `403 Forbidden` if `ADMIN_API_KEY` is unconfigured on the server.
2. **Scientific Boundary of Real Human Benchmark Data:**
   - BoliBench provides the end-to-end infrastructure, client-side standardization, acoustic diagnostics, and verification gating for human paired-contrast testing.
   - However, raw historical recordings stored in `human_recordings_v1/` remain classified as `unverified` until an operator manually verifies transcripts, confirms explicit consent, and seals the contrast pair.
   - BoliBench strictly does **not** claim an off-the-shelf validated human dataset is included without verified provenance.
3. **Text-Only Simulation Baseline:**
   - The 12 seed pairs in `seed_pairs.json` are hand-authored linguistic test cases (`text_only_simulation`). BoliBench does not claim acoustic noise robustness has been evaluated on these pairs without real audio execution.
4. **User-Declared Acoustic Metadata:**
   - The `audio_condition` field is user-declared metadata for benchmark cataloguing (`clean_recorded`, `noisy_recorded`, `clean_synthetic`, `noisy_synthetic`), not automatic acoustic detection. BoliBench does not pretend acoustic noise profiles can be detected automatically from single audio clips.
5. **Regional Indian Language Expansion:**
   - Current verified test cases and normalizers span Hindi (`hi-IN`) and Hinglish/Indian English (`en-IN`). Planned expansions include Tamil (`ta-IN`), Telugu (`te-IN`), Kannada (`kn-IN`), and Bengali (`bn-IN`).
6. **No Clinical or Superiority Claims:**
   - BoliBench is an objective diagnostic framework benchmarking downstream voice agent semantic robustness. It makes no claims of clinical diagnostic validation or unverified benchmark superiority.
