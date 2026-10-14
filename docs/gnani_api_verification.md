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

### 2.8 Documented Error Responses
| HTTP Status | Error Type (`error.type`) | Typical Reason |
|---|---|---|
| `400` | `INVALID_REQUEST_ERROR` | Audio exceeds 60 seconds, invalid language code, corrupt audio file. |
| `403` | `FORBIDDEN` | Missing/invalid `X-API-Key-ID`, expired credentials, or exhausted credits. |
| `429` | `RATE_LIMIT_ERROR` | Concurrency or rate limit threshold reached. |
| `500` | `API_ERROR` | Internal server or processing error on Gnani backend. |
| `503` | `SERVICE_UNAVAILABLE` | Speech recognition cluster temporarily down or restarting. |

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
| **Supported File Formats** | **CONFIRMED** | OpenAPI Specification `audio_file` field | WAV, MP3, OGG, FLAC, AAC, M4A |
| **Max Duration Constraint** | **CONFIRMED** | OpenAPI Documentation | 60 seconds max |
| **10 Language Codes** | **CONFIRMED** | OpenAPI Spec `language_code` enum | `bn-IN`, `en-IN`, `gu-IN`, `hi-IN`, `kn-IN`, `ml-IN`, `mr-IN`, `pa-IN`, `ta-IN`, `te-IN` |
| **Response Format** | **CONFIRMED** | OpenAPI Spec `STTResponse` schema | `success`, `transcript`, `timestamp`, `request_id` |
| **Error Format** | **CONFIRMED** | OpenAPI Spec `StandardErrorResponse` schema | `success: false`, `error: {type, message}` |
| **Live Account Authentication** | **REQUIRES USER TEST** | User's Gnani Console Account | Live request has **not** been run yet because the user must provide their `GNANI_API_KEY` in `.env`. |
| **Account Credits / Quotas** | **REQUIRES USER TEST** | User's Gnani Dashboard | 5,000 challenge credits apply per registration. |

---

## 4. Live API Test Status Notice

> **IMPORTANT DECLARATION:**  
> In accordance with project instructions, **no live API call is claimed to have succeeded** until an actual request with real credentials is executed by the user.  
> The backend client is built with strict adherence to the verified official OpenAPI 3.1 specification. Automated test suites validate all client methods, file constraints, timeouts, and error handling paths using deterministic mocked responses.
