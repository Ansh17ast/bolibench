# BoliBench — Paired-Contrast Evaluation Methodology

**Author:** BoliBench Engineering & Speech-AI Evaluation Research  
**Version:** 1.0.0  
**Status:** Implemented & Verified in Automated Test Suite  

---

## 1. Executive Summary & Core Philosophy

Traditional Automated Speech Recognition (ASR) benchmarking evaluates transcript quality using **Word Error Rate (WER)**. While WER provides an objective lexical distance, it possesses a severe fundamental limitation for conversational voice agents: **it treats all token substitutions equally**.

In production voice agents (e.g., banking IVRs, quick-commerce bots, mobility ticketing systems), acoustic errors are **asymmetrical and task-critical**:

```
Example 1 (Low WER, Catastrophic Agent Failure):
  Reference:  "Do packet chahiye."       (Expected quantity: 2)
  Hypothesis: "Teen packet chahiye."     (Recognized quantity: 3)
  WER:        33.3% (1 substitution / 3 words)
  Downstream: E-commerce order agent dispatches 3 packets instead of 2. 
              The financial and inventory intent is 100% violated.

Example 2 (Low WER, Catastrophic Account Action):
  Reference:  "Nahi subscription cancel mat karo."  (Expected action: retain)
  Hypothesis: "Haan subscription cancel kar do."    (Recognized action: cancel)
  WER:        50.0% (2 substitutions / 4 words)
  Downstream: Unrecoverable account deletion against user explicit direction.
```

**BoliBench replaces surface-only dashboard metrics with Paired-Contrast Evaluation:**
Instead of testing utterances in isolation, BoliBench pairs semantically contrasting utterances ($A$ and $B$) and evaluates whether the speech-to-text pipeline and downstream interpretation **preserve the critical distinction** between them.

---

## 2. Mathematical Metric Definitions

### 2.1 Word Error Rate (WER)

WER is computed using standard dynamic programming Levenshtein distance on normalized word sequences:

$$\text{WER} = \frac{S + D + I}{N}$$

Where:
- $S$ = Number of word substitutions
- $D$ = Number of word deletions
- $I$ = Number of word insertions
- $N$ = Number of words in the ground-truth reference transcript

#### Boundary Conditions & Edge Cases:
- **Empty Reference & Empty Hypothesis:** $\text{WER} = 0.0$
- **Empty Reference & Non-Empty Hypothesis ($M$ words):** $\text{WER} = 1.0$, with $I = M$
- **Non-Empty Reference ($N$ words) & Empty Hypothesis:** $\text{WER} = 1.0$, with $D = N$

### 2.2 Text Normalization Rules

Text normalization is executed prior to Levenshtein distance computation to eliminate cosmetic formatting discrepancies while strictly safeguarding semantic tokens:

1. **Punctuation Stripping:** Removes ASCII punctuation (`.,!?;:"'()[]{}-`) and Devanagari danda characters (`।` and `॥`).
2. **Case Normalization:** Lowercases Latin characters; leaves Devanagari and Indic scripts unaffected.
3. **Whitespace Normalization:** Strips leading/trailing whitespace and compresses consecutive spaces into a single space.
4. **Preservation Invariant:** Normalization **never** strips or normalizes away cardinal numbers, scale markers, negation tokens, or intent words. Specifically:
   - `"do"` is never converted or normalized into `"teen"`.
   - `"karna"` is never normalized into `"mat karna"`.
   - In Hindi Romanized transcripts, lexical tokens are preserved verbatim.

### 2.3 Critical-Field Accuracy ($Acc_{field}$)

Critical fields represent structured domain slots extracted from the spoken utterance (e.g. quantity, amount, negation status, destination, entity).

$$Acc_{field} = \frac{\sum_{i \in \{A, B\}} \sum_{f \in F_i} \mathbb{I}\left(\hat{v}_{i, f} = v_{i, f}^*\right)}{\sum_{i \in \{A, B\}} |F_i|}$$

Where:
- $F_i$ = Set of required critical fields defined for utterance $i$.
- $v_{i, f}^*$ = Manually specified ground-truth value for field $f$ in utterance $i$.
- $\hat{v}_{i, f}$ = Deterministically extracted value for field $f$ from the transcription.
- $\mathbb{I}(\cdot)$ = Indicator function (1 if match, 0 if mismatch). Numeric equality supports float tolerance; string equality is case-insensitive.

### 2.4 Semantic Distinction Preservation ($Preserved(A, B)$)

Let $k$ be the specified primary distinction field of the test case pair (e.g., `quantity`, `is_negated`, `final_target`, `person_name`, `product_name`, `intent`):

$$Preserved(A, B) = \begin{cases} 
1 & \text{if } \hat{v}_{A, k} = v_{A, k}^* \;\land\; \hat{v}_{B, k} = v_{B, k}^* \;\land\; \hat{v}_{A, k} \neq \hat{v}_{B, k} \\
0 & \text{otherwise} 
\end{cases}$$

If either utterance fails to extract the expected value, or if both utterances collapse into the identical extracted value, the semantic distinction has collapsed ($Preserved(A, B) = 0$).

### 2.5 Downstream Task Success ($TaskSuccess$)

An utterance pair achieves Downstream Task Success if and only if both utterances would cause the voice agent to trigger the correct downstream action, and the distinction between them was preserved:

$$TaskSuccess(A, B) = TaskSuccess(A) \;\land\; TaskSuccess(B) \;\land\; Preserved(A, B)$$

Where:
$$TaskSuccess(i) = \mathbb{I}\left(\forall f \in F_i, \hat{v}_{i, f} = v_{i, f}^*\right) \;\land\; \mathbb{I}\left(\widehat{Intent}_i = Intent_i^*\right)$$

### 2.6 Aggregate Benchmark Metrics

- **Semantic Distinction Preservation Rate (DPR):**
  $$\text{DPR} = \frac{\sum_{r=1}^R Preserved_r(A, B)}{R} \times 100\%$$

- **Downstream Task Success Rate (TSR):**
  $$\text{TSR} = \frac{\sum_{r=1}^R TaskSuccess_r(A, B)}{R} \times 100\%$$

- **Mean Word Error Rate ($\overline{\text{WER}}$):**
  $$\overline{\text{WER}} = \frac{1}{R} \sum_{r=1}^R \frac{\text{WER}_{r, A} + \text{WER}_{r, B}}{2}$$

---

## 3. Deterministic Extraction Engine Architecture

BoliBench uses **deterministic, rule-based extractors** rather than prompting an LLM during scoring. This provides two critical scientific guarantees:
1. Ground truth is 100% reproducible and tamper-proof.
2. Scoring requires zero third-party LLM inference latency or cost.

### 3.1 Supported Extraction Modules

| Module | Target Categories | Key Linguistic Rules & Token Sets |
|---|---|---|
| **Numeric & Scale Extractor** | Quantities, Financial amounts | - Romanized Hindi numbers (`ek`, `do`, `teen`, `chaar`, `paanch`, `chhah`, `saat`, `aath`, `nau`, `das`, `bees`, `pachaas`, etc.)<br>- Devanagari script words (`एक`, `दो`, `तीन`, `चार`, `पाँच`, etc.) and numerals (`०`-`९`)<br>- Scale multipliers (`sau`/`100`, `hazaar`/`1,000`, `lakh`/`100,000`, `crore`/`10,000,000`)<br>- Currency symbols (`₹500`, `₹5,000`) |
| **Negation Extractor** | Account retention, payment gates | - Explicit Negation tokens: `nahi`, `nahin`, `नहीं`, `mat`, `मत`, `na`, `न`, `ruko`, `रुको`, `no`, `not`, `never`, `dont`, `stop`<br>- Affirmative tokens: `haan`, `हाँ`, `yes`, `karo`, `करो`, `ok`, `okay`, `confirm`, `proceed`<br>- Contextual verb resolution: `"cancel mat karo"` $\rightarrow$ `keep_booking`; `"cancel kar do"` $\rightarrow$ `cancel_booking` |
| **Correction & Mind-Change Extractor** | Travel destinations, item counts | - Detects cue phrases: `"no wait"`, `"nahi"`, `"actually"`, `"change to"`, `"nahi ruko"`<br>- Resolves initial discarded value vs final intended target<br>- Prevents downstream agents from booking initial value |
| **Entity & Product Extractor** | SKUs, personnel, cities | - Named entities: `Rahul Sharma` vs `Rohit Sharma`<br>- Products: `Prisma STT` vs `Krishna STT`, `Timbre` vs `Timber`<br>- Cities: `Delhi` vs `Mumbai` |
| **Command Intent Classifier** | Banking, e-commerce | - `CHECK_BALANCE` vs `CLOSE_ACCOUNT`<br>- `TRANSFER_FUNDS` vs `REQUEST_FUNDS`<br>- `EMPTY_CART` vs `CONFIRM_ORDER` |

---

## 4. Failure Classification Taxonomy

When an evaluation run fails, the engine assigns an explicit diagnostic classification:

| Failure Category | Trigger Condition | Real-World Agent Impact |
|---|---|---|
| `QUANTITY_ERROR` | `distinction_field = quantity` collapsed or mismatched | Downstream agent orders incorrect count or transfers wrong amount. |
| `MISSED_NEGATION` | Negation token dropped or affirmative token misrecognized | Downstream agent executes an irreversible action contrary to user instruction (e.g. cancels booking instead of retaining). |
| `CORRECTION_ERROR` | Final intended value after correction cue is lost | Agent acts on preliminary discarded value rather than user's final intent. |
| `ENTITY_SUBSTITUTION` | Person name or product brand misheard | Agent invoices wrong customer or provisions wrong software SKU. |
| `INTENT_ERROR` | Command intent handler misclassified | High-risk workflow triggered (e.g. account closure instead of balance inquiry). |
| `HIGH_WER_CORRUPTION` | WER > 0.60 or transcript empty | Severe acoustic dropout prevented any structured field extraction. |
| `EXTRACTION_FAILURE` | Structured parser could not locate required slot | Output format failed schema validation. |
| `NONE` | Distinction preserved and all tasks succeeded | Voice agent executed user intent cleanly. |

---

## 5. Seed Dataset Specifications & Simulation Semantics

- **Total Pairs:** 12 curated contrast pairs (24 unique utterances).
- **Languages Covered:**
  - `hi-IN` (Hindi): Monolingual Hindi & Hinglish code-mixed commands.
  - `en-IN` (English - India): Indian English accent & syntax patterns.
- **Dataset Classification (`audio_condition: text_only_simulation`):**
  All 12 seed pairs represent **hand-authored text specifications**. In accordance with scientific integrity guidelines, BoliBench explicitly categorizes them as text-only simulations (`text_only_simulation`, provenance `hand_authored_text_simulation`).
- **Terminology & Boundary Guarantees:**
  - **Hand-authored simulated transcript corruption:** Predetermined text corruptions injected for deterministic crash demonstration (e.g. demonstrating how a quantity substitution crashes an order agent). These are **not** reported as measured model errors or acoustic noise tests.
  - **Actual recorded audio processed by Gnani Prisma:** Genuine spoken audio recordings evaluated via `POST /api/eval/run-live/{test_id}` or `POST /api/transcribe`.
  - **Actual recorded audio tested under a measured noisy condition:** Testing under calibrated ambient noise (e.g. street noise, call center babble). Marked as `pending_acoustic` until authentic multi-speaker audio with calibrated SNR is collected.
  - **Synthetic or generated audio:** Spoken audio produced by TTS engines (e.g. Gnani Timbre), only classified as synthetic audio when genuine audio files exist.
- **Privacy & PII:** Contains zero personal data, telephone numbers, or real customer call audio (`text_simulation_no_pii`).
- **Storage:** Persisted as versioned JSON in `app/evaluation/data/seed_pairs.json`.

---

## 6. Execution Modes, Telemetry & Hardened Credit Protection

BoliBench strictly isolates execution semantics, telemetry, and credit protection between mock and live modes:

- **Mock Execution Mode (`mock`):**
  - Executes deterministically using reference transcripts or hand-authored simulated crash transcripts.
  - Consumes **0 Gnani Prisma credits** and executes **0 external network requests**.
  - **Telemetry Semantics:** Fabricated latency numbers are strictly prohibited. `client_request_elapsed_ms`, `client_measured_latency_ms`, and `provider_processing_time_ms` remain strictly `null`. If latency is simulated for demonstration, it is recorded exclusively in `simulated_latency_ms` and clearly labeled as simulated in the UI and exports.
- **Live Execution Mode (`live`):**
  - Available via the paired live-audio endpoint `POST /api/eval/run-live/{test_id}`.
  - **Server-Side Authorization & Credit Guard:**
    - Live execution requires an explicit authorization handshake via `POST /api/eval/authorize-live` before any provider calls are permitted.
    - The server issues a cryptographically secure, time-limited (5-minute), single-use authorization token mapped to the specific test case.
    - Requests must include the custom anti-CSRF header `X-BoliBench-Confirm-Live: CONFIRM_LIVE_EVALUATION`.
    - Same-origin headers (`Sec-Fetch-Site`, `Origin`, `Referer`) are strictly validated against trusted local development origins (`localhost:8000`, `127.0.0.1:8000`, `localhost:3000`, `127.0.0.1:3000`). Untrusted or cross-origin requests are rejected with `403 Forbidden` before reading file uploads or invoking the provider.
    - Single-use tokens cannot be replayed or reused for multiple evaluation runs.
  - **Audio Validation:** Validates both audio uploads in-memory against format, duration (≤ 60s), size (≤ 25MB), and container header signatures.
  - **Client & Provider Latency Semantics:**
    - Measures genuine client-side round-trip elapsed time ($T_{\text{client}}$) via `time.perf_counter()` around the HTTP request and records it as `client_request_elapsed_ms` (and `client_measured_latency_ms`).
    - Inspects the official Gnani response schema for provider-reported processing duration ($T_{\text{provider}}$). Because the official Gnani STT REST v3 API does not report server processing duration in its response payload, `provider_processing_time_ms` remains strictly `null`. Client-side elapsed time is **never** substituted for provider latency.
  - **Partial Failure Safeguard:** If either provider request fails or times out, the evaluation is aborted immediately. An incomplete pair is never persisted or presented as a successful evaluation.
  - **Credit Protection:** Live evaluation requires an explicit user confirmation checkbox, confirmation dialog, and server authorization token. Automated tests always use deterministic mocked client responses and never consume programme credits.

---

## 7. Current Implementation Status & Pending Acoustic Research

1. **Implemented & Verified:**
   - Deterministic paired-contrast evaluation scoring engine.
   - Exact dynamic programming Levenshtein Word Error Rate (WER).
   - Domain-specific critical field extractors (quantities, negation, corrections, entities, intent).
   - Thread-safe SQLite persistence with JSON and CSV exports preserving latency semantics.
   - Telemetry separation ensuring mock runs never report fabricated measured latency.
   - Server-side credit guard and anti-CSRF authorization manager (`app/services/live_guard.py`).
   - Live paired audio REST API (`POST /api/eval/run-live/{test_id}`) protected by `POST /api/eval/authorize-live`.
   - Comprehensive test suite (78 tests passing with 0 network calls and 0 credits consumed).
2. **Pending Acoustic Research (`pending_acoustic`):**
   - Calibrated acoustic noise testing with multi-speaker recordings under measured signal-to-noise ratios (street noise, cafeteria babble, reverberation).
   - Crowdsourced audio dataset recording all 24 seed utterances across diverse regional Indian accents.
3. **Regional Language Expansion:**
   - Future planned paired-contrast catalogs for Tamil (`ta-IN`), Telugu (`te-IN`), Kannada (`kn-IN`), and Bengali (`bn-IN`).
