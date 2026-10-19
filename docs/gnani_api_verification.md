# Gnani AI Prisma STT API Verification Report

**Project:** BoliBench — Indian Voice AI Crash-Test Lab  
**Date of Verification:** October 14, 2026  
**Document Status:** Complete & Verified against Official Gnani AI Documentation  

---

## 1. Documentation Sources Consulted

The following official Gnani AI endpoints and documentation portals were inspected:
- **Gnani AI Documentation Portal:** [https://docs.gnani.ai/](https://docs.gnani.ai/)
- **Documentation Index (Mintlify LLMS):** [https://docs.gnani.ai/llms.txt](https://docs.gnani.ai/llms.txt)
- **Gnani Speech-to-Text REST v3 Docs:** [https://docs.gnani.ai/api/STT/speech-to-text.md](https://docs.gnani.ai/api/STT/speech-to-text.md)
- **Gnani Quick Start Guide:** [https://docs.gnani.ai/api/introduction/quick-start.md](https://docs.gnani.ai/api/introduction/quick-start.md)
- **Voice Platform Console:** [https://app.gnani.ai/voice](https://app.gnani.ai/voice)
- **Great Indian AI Internship Challenge Details:** [https://www.gnani.ai/internship](https://www.gnani.ai/internship)

---

## 2. Confirmed API Contract (Prisma STT REST v3)

### 2.1 Endpoint & Method
- **Method:** `POST`
- **URL:** `https://api.vachana.ai/stt/v3`
- **Content-Type:** `multipart/form-data`

*(Note: In earlier references, older endpoints or domains may have been cited. The verified production v3 endpoint is explicitly `https://api.vachana.ai/stt/v3`.)*

### 2.2 Authentication Mechanism
- **Header:** `X-API-Key-ID`
- **Type:** String (API Key obtained from [https://app.gnani.ai/voice](https://app.gnani.ai/voice))
- **Environment Variable in BoliBench:** `GNANI_API_KEY`

### 2.3 Required Request Parameters
Sent via `multipart/form-data`:
| Field | Type | Description |
|---|---|---|
| `audio_file` | Binary File | Audio file to transcribe. Supported containers: WAV, MP3, OGG, FLAC, AAC, M4A. Max duration: 60s (ideal ≤ 30s). |
| `language_code` | String | BCP-47 language tag. Must be one of the 10 supported Indian language codes. |

### 2.4 Supported Languages (10 Indian Languages)
| Language | Code | Native Script |
|---|---|---|
| Bengali | `bn-IN` | Bengali (বাংলা) |
| English (India) | `en-IN` | English (Latin) / Hinglish mixed audio accepted |
| Gujarati | `gu-IN` | Gujarati (ગુજરાતી) |
| Hindi | `hi-IN` | Devanagari (हिन्दी) |
| Kannada | `kn-IN` | Kannada (ಕನ್ನಡ) |
| Malayalam | `ml-IN` | Malayalam (മലയാളം) |
| Marathi | `mr-IN` | Devanagari (मराठी) |
| Punjabi | `pa-IN` | Gurmukhi (ਪੰਜਾਬੀ) |
| Tamil | `ta-IN` | Tamil (தமிழ்) |
| Telugu | `te-IN` | Telugu (తెలుగు) |

### 2.5 Optional Request Parameters
| Parameter | Type | Default | Description |
|---|---|---|---|
| `format` | String enum | `verbatim` | `verbatim` (raw spoken ASR text) or `transcribe` (Inverse Text Normalization enabled: numbers, currency, dates, phone numbers formatted). |
| `itn_native_numerals` | Boolean | `false` | When `format=transcribe`, outputs digits in the target language's native script (e.g. `₹५,०००` instead of `₹5,000`). |
| `enable_substitution`| Boolean | `false` | Enables client-defined find-and-replace rules post-transcription. |
| `substitution_map`   | String (JSON) | `null` | JSON string of array of rules (up to 10 rules): `[{"patterns": ["rs"], "replacement": "INR"}]`. |
| `bias_list`          | String | `null` | Words to boost during recognition (up to 100 single words of letters only), as JSON array `["Word1","Word2"]` or comma-separated. |
| `bias_score`         | Float enum | `0.0` | Boost strength: `0.0`, `0.5`, `1.0`, `1.5`, or `2.0`. |

### 2.6 Audio Constraints
- **Containers/Extensions:** `.wav`, `.mp3`, `.ogg`, `.flac`, `.aac`, `.m4a`
- **Duration:** Maximum 60 seconds (ideal ≤ 30 seconds). Audio exceeding 60 seconds triggers a `400 Bad Request`.
- **Sample Rate:** 8 kHz to 44.1 kHz accepted (automatically normalized to 16 kHz mono on the server).
- **Client Size Guardrail:** BoliBench implements a 25 MB safety limit for uploads to prevent memory exhaustion and DoS.

### 2.7 Response Schema (HTTP 200 OK)
```json
{
  "success": true,
  "request_id": "req_abc123",
  "timestamp": "20261014_233000.123",
  "transcript": "नमस्ते, आप कैसे हैं?",
  "model": "gnani-prisma-v2.5"
}
```
*(Engineering Note on Provider Latency: The official OpenAPI specification for Gnani STT REST v3 does not include a provider-side processing time field in its response. Consequently, BoliBench records client-measured HTTP elapsed time as `client_request_elapsed_ms` (with `backend_latency_ms` maintained as a backwards-compatible alias) and preserves `provider_processing_time_ms = null` (rendered as "Not reported"). Client elapsed time is never substituted or fabricated as provider processing time.)*

### 2.8 Documented Error Responses
| HTTP Status | Error Type (`error.type`) | Typical Reason | Client Handling |
|---|---|---|---|
| `400` | `INVALID_REQUEST_ERROR` | Audio exceeds 60 seconds, invalid language code, corrupt audio file. | `GnaniBadRequestError` (400) |
| `401` | `UNAUTHORIZED` | Invalid API key format or unauthorized request. | `GnaniAuthenticationError` (401) |
| `403` | `FORBIDDEN` | Missing/invalid `X-API-Key-ID`, expired credentials, or exhausted credits. | `GnaniAuthenticationError` (403) |
| `429` | `RATE_LIMIT_ERROR` | Concurrency or rate limit threshold reached. | `GnaniRateLimitError` (429) |
| `500` | `API_ERROR` | Internal server or processing error on Gnani backend. | `GnaniServerError` (500) |
| `502` | `MALFORMED_RESPONSE` | Upstream returned 200 OK without a `transcript` field. | `GnaniServerError` (502) |
| `503` | `SERVICE_UNAVAILABLE` | Speech recognition cluster temporarily down or restarting. | `GnaniServiceUnavailableError` (503) |
| `504` | `GATEWAY_TIMEOUT` | Request timed out exceeding configured timeout window. | `GnaniTimeoutError` (504) |

Error JSON Structure:
```json
{
  "success": false,
  "error": {
    "type": "INVALID_REQUEST_ERROR",
    "message": "Audio duration exceeds maximum limit of 60 seconds, while the ideal duration is 30 seconds."
  }
}
```

---

## 3. Confirmed vs. Unverified Matrix

| Item | Status | Verification Source | Notes |
|---|---|---|---|
| **Base Endpoint URL** | **CONFIRMED** | Gnani Official REST STT v3 OpenAPI Spec | `https://api.vachana.ai/stt/v3` |
| **Auth Header** | **CONFIRMED** | Gnani Official REST STT v3 OpenAPI Spec | `X-API-Key-ID: <key>` |
| **Request Form Encoding** | **CONFIRMED** | Documentation & cURL samples | `multipart/form-data` |
| **Supported File Formats** | **CONFIRMED** | OpenAPI Specification `audio_file` field | WAV, MP3, OGG, FLAC, AAC, M4A verified |
| **Max Duration Constraint** | **CONFIRMED** | OpenAPI Documentation | 60 seconds max |
| **10 Language Codes** | **CONFIRMED** | OpenAPI Spec `language_code` enum | `bn-IN`, `en-IN`, `gu-IN`, `hi-IN`, `kn-IN`, `ml-IN`, `mr-IN`, `pa-IN`, `ta-IN`, `te-IN` |
| **Response Format** | **CONFIRMED** | OpenAPI Spec `STTResponse` schema | `success`, `transcript`, `timestamp`, `request_id` (`model` is optional; no server processing duration reported) |
| **Telemetry Semantics** | **CONFIRMED & HARDENED** | BoliBench Telemetry Separation | Browser end-to-end latency vs backend-measured Gnani API request time vs provider processing time ("Not reported"). |
| **Error Format (400, 401, 403, 429, 500, 503)** | **CONFIRMED** | OpenAPI Spec `StandardErrorResponse` schema | `success: false`, `error: {type, message}` |
| **Deployment Safety Setting** | **CONFIRMED & IMPLEMENTED** | `LIVE_EVALUATION_ENABLED=false` | Server defaults to disabled live evaluation to protect credits on public deployments. |
| **Paired Live Audio Route** | **CONFIRMED & TESTED** | `POST /api/eval/run-live/{test_id}` | Protected by server authorization token (`/api/eval/authorize-live`), anti-CSRF header, same-origin check, acoustic condition enum validation, and atomic partial failure abortion. |
| **Live Account Authentication** | **PENDING USER SETUP** | User's Gnani Console Account | Live request pending user-supplied `GNANI_API_KEY` in `.env`. |
| **Credit Protection** | **CONFIRMED & HARDENED** | BoliBench Safety Guardrails | Automated tests consume 0 credits. UI and API require server-side authorization token, confirmation header, and user consent before live calls. |

---

## 4. Live API Test Status & Credit Protection Notice

> **IMPORTANT SCIENTIFIC & SECURITY DECLARATION:**
> In accordance with project instructions, **no live API call is claimed to have succeeded** until an actual request with real credentials is executed by the user against their personal quota.
>
> **Credit Protection Guarantees:**
> 1. The automated test suite (`python -m pytest -v`) runs exclusively against deterministic mocked provider responses. Zero live network calls are made during tests, consuming zero programme credits.
> 2. The web application never triggers live requests automatically on page load or when switching tabs.
> 3. The server defaults to `LIVE_EVALUATION_ENABLED=false` so public or untrusted deployments cannot trigger billable calls without operator authorization.
> 4. The paired live evaluation endpoint (`POST /api/eval/run-live/{test_id}`) is guarded by an explicit authorization step (`POST /api/eval/authorize-live`), requiring a single-use token, custom anti-CSRF header, and same-origin verification.
> 5. API keys and confirmation secrets are managed strictly on the server backend and are never leaked to client responses or application logs.
