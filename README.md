# BoliBench — Indian Voice AI Crash-Test Lab

> **A portfolio-quality diagnostic lab built for the Great Indian AI Internship Challenge by Gnani AI.**  
> BoliBench evaluates how speech-recognition errors affect the actions of downstream voice agents, with primary focus on Hindi, Indian English, and Hinglish.

---

## 1. Project Objective & Differentiator

Traditional Speech-to-Text (STT) benchmarks rely heavily on **Word Error Rate (WER)**. While WER measures lexical surface distance, it treats all word errors equally. In real-world Indian Voice AI agents (e.g. banking bots, ticketing agents, delivery assistants), acoustic errors are often **task-critical**:

| Error Category | Example Spoken Input | Misrecognized Output | Downstream Agent Action Failure |
|---|---|---|---|
| **Incorrect Quantities** | "पाँच लाख रुपये" (*5 Lakh INR*) | "पाँच हज़ार रुपये" (*5 Thousand INR*) | Transferred wrong financial amount |
| **Missed Negations** | "नहीं, कैंसिल मत करो" (*No, do not cancel*) | "हाँ, कैंसिल कर दो" (*Yes, cancel it*) | Unauthorized deletion / booking cancellation |
| **Entity / Product Names** | "Prisma STT का access चाहिए" | "Krishna STT का access चाहिए" | Wrong product license provisioned |
| **Misunderstood Corrections** | "Flight to Mumbai, no wait, to Delhi" | "Flight to Mumbai" | Booked destination contrary to corrected intent |
| **Entity ITN Formatting** | "नौ आठ सात छह..." (*Phone number*) | "9 8 7 6..." (spaced string) | Schema validation rejection in CRM |

**BoliBench is designed to expose and evaluate these exact crash points.**

---

## 2. Verified Gnani AI Technology Stack

- **Speech Recognition Engine:** Gnani Prisma v2.5 STT (REST v3 endpoint: `https://api.vachana.ai/stt/v3`)
- **Backend:** FastAPI (Python 3.10+) with Pydantic v2 Settings & Schemas
- **Frontend:** Vanilla HTML5, CSS3, and JavaScript (Zero external CDN dependencies, fully self-contained)
- **Validation Engine:** In-memory audio container & duration validator (`AudioValidator`)
- **Automated Testing:** `pytest` and `pytest-asyncio` with deterministic mock transports

Official documentation verification details are documented in [`docs/gnani_api_verification.md`](docs/gnani_api_verification.md).

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
│   └── gnani_api_verification.md    # Official Gnani Prisma API verification document
├── app/
│   ├── __init__.py
│   ├── config.py                    # Environment settings via Pydantic
│   ├── main.py                      # FastAPI routes, error handlers, static mounting
│   ├── schemas.py                   # Pydantic request/response schemas & language metadata
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_validator.py       # Container, signature, size, and duration validation
│   │   └── gnani_client.py          # Gnani Prisma STT v3 REST client with latency telemetry
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css            # Developer-grade responsive styling
│   │   └── js/
│   │       └── app.js               # Upload UI, audio preview, API fetch, and telemetry
│   └── templates/
│       └── index.html               # Interactive crash-test lab interface
└── tests/
    ├── __init__.py
    ├── conftest.py                  # Pytest fixtures and in-memory WAV audio generators
    ├── test_audio_validator.py      # Format, extension, duration, and signature unit tests
    ├── test_gnani_client.py         # Mocked client tests (200, 400, 403, 429, 500, timeouts)
    └── test_routes.py               # FastAPI route tests (/health, /, /api/transcribe)
```

---

## 4. Setup & Run Instructions (Windows PowerShell)

### Step 1: Clone or Navigate to the Project
```powershell
Set-Location -Path "C:\Users\astru\.gemini\antigravity\scratch\bolibench"
```

### Step 2: (Optional) Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Pinned Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 4: Configure Credentials
Copy the `.env.example` file to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` in your text editor and insert your Gnani API key:
```ini
GNANI_API_KEY=your_actual_key_from_app_gnani_ai
```
*(If you do not have an active key yet, leave it blank; BoliBench has a built-in **Mock Mode** allowing full local UI and integration demonstration).*

### Step 5: Run the Application
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
- **Interactive UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Health Check API:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Interactive OpenAPI Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 5. Running Automated Tests

Run the full automated test suite:
```powershell
python -m pytest -v
```

All 25 automated tests run without network calls using in-memory audio fixtures and HTTP transport mocks:
- `tests/test_audio_validator.py` (10 tests): Validates container formats (`.wav`, `.mp3`, `.ogg`, `.flac`), size boundaries, header magic bytes, and duration limits (<= 60s).
- `tests/test_gnani_client.py` (8 tests): Validates Gnani Prisma v3 request headers (`X-API-Key-ID`), form-data serialization, 200 responses, 400 Bad Request, 403 Forbidden, 429 Rate Limits, 500/503 errors, and timeouts.
- `tests/test_routes.py` (7 tests): Validates the `/health` endpoint, HTML rendering, mock-mode transcription, language validation, and missing credential error responses.

---

## 6. Security Guarantees

- **No Hardcoded Secrets:** Credentials are read strictly from `.env` via `Pydantic Settings`.
- **Frontend Isolation:** The `GNANI_API_KEY` is kept on the backend and is **never** sent or accessible to frontend JavaScript.
- **Privacy by Default:** Uploaded audio clips are processed in-memory as byte streams and are **never** persisted to disk.
- **`.gitignore` Enforced:** `.env`, `.venv`, and audio files are blocked from version control commits.
