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

### 2.7 Cross-Script Normalization & Supplementary Normalized WER ($WER_{norm}$)

Acoustic evaluations on Indian language speech AI face a widespread orthographic challenge: **reference transcripts often use Romanized Hindi (Latin script) while enterprise ASR models (e.g. Gnani Prisma STT v3) output Devanagari script and Hindu-Arabic numerals**.

#### 2.7.1 Why Raw WER is 1.0 (100%) Under Script Mismatches
Dynamic programming Levenshtein distance compares surface token representations. For example:
- **Reference:** `"Do packet chahiye"` (3 words: `Do`, `packet`, `chahiye`)
- **Hypothesis:** `"2 पैकेट चाहिए"` (3 words: `2`, `पैकेट`, `चाहिए`)

Because Latin characters share zero token overlap with Devanagari tokens, all 3 words are scored as substitutions ($S = 3, D = 0, I = 0, N = 3$):
$$\text{WER}_{raw} = \frac{3 + 0 + 0}{3} = 1.0 \;(100.0\%)$$
Raw WER is strictly orthographic; it accurately reflects the unmapped script divergence, but conceals that the acoustic recognition was phonetically and semantically correct.

#### 2.7.2 Supplementary Cross-Script Normalizer
To evaluate phonetic and semantic accuracy independently of script encoding, BoliBench provides a supplementary deterministic normalizer:
`canonicalize_cross_script_text(text)`

This function canonicalizes supported numerical and lexical equivalences:
- Numerals: `"do"` (when modifying quantified nouns) $\rightarrow$ `"2"`, `"teen"` $\rightarrow$ `"3"`, `"ek"` $\rightarrow$ `"1"`, `"chaar"` $\rightarrow$ `"4"`
- Vocabulary: `"packet"` $\leftrightarrow$ `"पैकेट"`, `"chahiye"` $\leftrightarrow$ `"चाहिए"`, `"order"` $\leftrightarrow$ `"ऑर्डर"`
- Output: Both reference and hypothesis canonicalize to `"2 पैकेट चाहिए"`, producing:
$$\text{WER}_{norm} = \frac{0 + 0 + 0}{3} = 0.0 \;(0.0\%)$$

#### 2.7.3 Scope Limitations & Disambiguation Rules
- **Task-Specific, Not Universal:** Cross-script normalization is strictly restricted to supported benchmark domain equivalences. It is **not** a general-purpose Hindi transliteration or translation model.
- **Context-Aware Disambiguation of Ambiguous Tokens:** Romanized `"do"` and Devanagari `"दो"` are polysemous in Hindi: they can denote the cardinal number $2$ or the imperative verb *"give"* / verbal auxiliary (e.g. *"kar do"*, *"bhej do"*, *"rok do"*, *"cancel kar do"*).
  - When preceded by verbal auxiliary stems (`kar`, `bhej`, `de`, `rok`, `rakh`, `mat`, `cancel`), `"do"` / `"दो"` is preserved as the verbal token `"do"` and is **never** converted to numeral `"2"`.
  - When followed by quantified noun stems (`packet`, `plate`, `kilo`, `sau`, `rupaye`, `chahiye`) or appearing as a standalone count, it canonicalizes to `"2"`.
- **Integrity Rule:** Raw WER is **always** preserved verbatim in benchmark logs and exports. Supplementary normalized WER is never reported as raw WER.

### 2.8 Offline Derived Re-Scoring & Audit Isolation

When upstream STT models return valid acoustic transcripts that initially fail downstream task execution due to unmapped downstream regex patterns (e.g. `UNKNOWN_INTENT` when Devanagari `"चाहिए"` was not yet recognized by `extract_command_intent()`), BoliBench supports **Offline Deterministic Re-Scoring**:
1. **Source Run Immutability:** The original live record in SQLite (`evaluation_runs`) remains strictly untouched, preserving the original measurement, raw WER, provider request IDs, client latency, and initial failure category.
2. **Dedicated Storage:** Derived assessments are stored in a separate table (`derived_assessments`) with provenance `offline_deterministic_rescore`.
3. **Credit & Network Invariant:** Re-scoring operates entirely offline on previously stored transcripts. It makes **0 external provider calls** and consumes **0 credits**.
4. **Metric Isolation:** Derived assessments are never counted as additional live runs in aggregate metrics (`get_aggregate_metrics()`), preventing artificial inflation of completed experiment counts.

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
- **Acoustic Condition Metadata & Non-Detection Principle:**
  - `audio_condition` is strictly **user-declared metadata** for benchmark cataloguing, validated against the `AudioCondition` enum (`clean_recorded`, `noisy_recorded`, `clean_synthetic`, `noisy_synthetic`, `pending_acoustic`).
  - BoliBench **does not pretend acoustic conditions can be detected automatically**. It does not perform unverified SNR or acoustic scene classification.
  - A recording must **never** be labeled "noisy" merely because noise testing is planned.
  - `text_only_simulation` is strictly restricted to text-based evaluation and cannot be selected for live audio uploads.
- **Protocol for Producing Calibrated Noisy Recordings (Future Work):**
  When noisy acoustic evaluations are eventually performed, the following rigorous procedure must be executed and documented:
  1. **Noise Sources:** Standardized acoustic noise profiles (e.g. Demand dataset, cafeteria babble, urban traffic noise, Indian call center ambient chatter).
  2. **Calibrated Mixing:** Active speech levels must be calibrated in dBov according to ITU-T P.56 speech voltmeter standards.
  3. **Target Signal-to-Noise Ratio (SNR):** Additive mixing at calibrated SNR tiers (+15 dB, +10 dB, +5 dB, 0 dB).
  4. **Provenance Documentation:** Audio metadata must record the noise source file, mixing script, target SNR, and room reverberation ($RT_{60}$) parameters.
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
  - **Deployment Safety Policy (`LIVE_EVALUATION_ENABLED`):**
    - The server defaults to `LIVE_EVALUATION_ENABLED=false` to protect the operator's Gnani API credits when deployed to public hosting or demo environments.
    - When disabled, both `POST /api/eval/authorize-live` and `POST /api/eval/run-live/{test_id}` reject requests with `403 Forbidden` before reading uploads into memory or initiating any provider calls.
    - Live execution is available only for authorized local development and operator testing when explicitly enabled via `LIVE_EVALUATION_ENABLED=true` in `.env`.
  - **Server-Side Authorization & Credit Guard:**
    - Live execution requires an explicit authorization handshake via `POST /api/eval/authorize-live` before any provider calls are permitted.
    - The server issues a cryptographically random (256-bit `secrets.token_hex`), server-stored, time-limited (300 seconds / 5 minutes), single-use authorization token mapped to the specific test case.
    - The token is stored statefully in the server memory; it is not a signed JWT.
    - Requests must include the custom anti-CSRF header `X-BoliBench-Confirm-Live: CONFIRM_LIVE_EVALUATION`.
    - Same-origin headers (`Sec-Fetch-Site`, `Origin`, `Referer`) are strictly validated against explicit complete local development origins (`http://localhost:8000`, `http://127.0.0.1:8000`, `http://localhost:3000`, `http://127.0.0.1:3000`, `http://test`, `http://testserver`). Untrusted or cross-origin requests are rejected with `403 Forbidden`. Missing origin information is rejected on non-local hosts under a documented development exception.
    - Single-use tokens cannot be replayed or reused for multiple evaluation runs.
  - **Audio Validation:**
    - Single-audio transcription validates container magic bytes and file size across standard formats (`.wav`, `.mp3`, `.ogg`, `.flac`, `.aac`, `.m4a`).
    - Paired live evaluation strictly restricts uploads to formats with server-side verifiable duration (`.wav` $\le$ 60s via wave header parsing) before initiating provider requests to safeguard owner credits from unvalidated long audio clips.
  - **Client & Provider Latency Semantics:**
    - Measures genuine client-side round-trip elapsed time ($T_{\text{client}}$) via `time.perf_counter()` around the HTTP request and records it as `client_request_elapsed_ms` (and `client_measured_latency_ms`).
    - The legacy `backend_latency_ms` field is documented as a backwards-compatible alias for `client_request_elapsed_ms`.
    - Inspects the official Gnani response schema for provider-reported processing duration ($T_{\text{provider}}$). Because the official Gnani STT REST v3 API does not report server processing duration in its response payload, `provider_processing_time_ms` remains strictly `null` (displayed as "Not reported" in the UI). Client-side elapsed time is **never** substituted for provider latency.
  - **Partial Failure Safeguard & Metric Aggregation Isolation:**
    - If Utterance A fails, Utterance B is never called (zero unnecessary provider requests), and a safe failed audit run (`overall_status="failed"`) is persisted with error details and timings.
    - If Utterance A succeeds but Utterance B fails or times out, a safe incomplete audit record (`overall_status="incomplete"`) is persisted preserving Utterance A's real transcript and provider request ID, documenting that Utterance A consumed credits while Utterance B failed.
    - Incomplete or failed runs are safely audited in SQLite and discoverable in run history (`/api/eval/results`) and failure reporting (`failure_category_breakdown`), but are strictly excluded from benchmark accuracy metrics (mean WER, task success rate, distinction preservation rate, and critical-field accuracy) so operational provider errors never contaminate transcription or semantic performance measurements.
    - Empty completed datasets return null rather than invented scores (e.g. no fabricated 0.0% or 100% metrics).
    - No automatic retries are performed.
  - **Credit Protection:** Live evaluation requires an explicit user confirmation checkbox, confirmation dialog, and server authorization token. Automated tests always use deterministic mocked client responses and never consume programme credits.

### 6.1 Taxonomy of Evaluation Artifact Classes

BoliBench establishes a strict separation across three distinct evaluation artifact classes:

| Artifact Class | Upstream Provider Interaction | External Network & Credit Consumption | Storage Table | Aggregate Metric Inclusion | Primary Purpose |
|---|---|---|---|---|---|
| **Provider-Backed Live Runs (`execution_mode='live'`)** | Genuine Gnani Prisma STT v3 REST API invocation | Real network requests; consumes programme credits | `evaluation_runs` | Yes (included in live aggregates) | Empirically validate real-world speech recognition, measure client round-trip latency, record provider request IDs, and test downstream agent execution on live transcripts. |
| **Deterministic Mock Runs (`execution_mode='mock'`)** | Hand-authored ground-truth or simulated corrupted transcripts | 0 network calls; 0 credits spent | `evaluation_runs` | Yes (isolated under mock aggregates) | Continuous integration, unit testing, and reproducible regression testing without external dependencies. |
| **Derived Offline Assessments (`provenance='offline_deterministic_rescore'`)** | None (pure local deterministic re-score of already stored transcripts) | 0 network calls; 0 credits spent | `derived_assessments` | **No** (strictly excluded from aggregate run counts) | Re-evaluate historical transcripts under updated normalization or intent rules without rewriting the original live measurement or inflating run counts. |

### 6.2 Live Evidence Scope & Honest Scientific Boundaries

1. **Current Empirical Evidence Scope:**
   The live evaluation evidence currently consists of **one paired experiment with two utterances**:
   - Run ID: `eval_live_758277ca81d8`
   - Test Case: `pair_qty_001` (Numeral contrast: Utterance A *"Do packet chahiye"* vs Utterance B *"Teen packet chahiye"*)
   - Audio Condition: `clean_recorded` (User-declared metadata)
   - Genuine Request IDs: `01a15b46-a360-7cff-9e6d-8fcb69c7da6b` (Utterance A) and `01a15b46-a6b5-78bf-baf9-10d531921fe0` (Utterance B)
2. **Strict Non-Generalization Principle:**
   BoliBench **does not claim broad model accuracy, statistically significant findings, or general downstream voice agent failure** from this single paired experiment. This run serves as an empirical proof of the paired-contrast crash-testing methodology, demonstrating that:
   - The upstream provider correctly recognized both spoken numbers in Devanagari (`"2 पैकेट चाहिए"` and `"3 पैकेट चाहिए"`), preserving the paired numeral contrast ($2 \neq 3$).
   - Script-sensitive raw WER was 100.0% due to Latin reference transcripts vs Devanagari provider output.
   - Initial downstream task execution failed because the domain intent parser initially lacked support for Devanagari `"चाहिए"`.
3. **Latency Reporting Transparency:**
   The client-measured round-trip elapsed time was **2027.77 ms** (Utterance A: 1157.92 ms, Utterance B: 869.85 ms). Because the Gnani Prisma STT v3 REST API response payload does not report internal processing duration, provider-internal processing latency is explicitly recorded as `null` and displayed as *"Not reported"* rather than substituting client-measured values.

---

## 7. Current Implementation Status & Pending Acoustic Research

1. **Implemented & Verified:**
   - Deterministic paired-contrast evaluation scoring engine.
   - Exact dynamic programming Levenshtein Word Error Rate (WER) and supplementary cross-script normalized WER ($WER_{norm}$).
   - Context-aware token disambiguation (preserving verbal auxiliary *"do"* vs cardinal numeral *"2"*).
   - Domain-specific critical field extractors (quantities, negation, corrections, entities, intent).
   - Thread-safe SQLite persistence with JSON and CSV exports preserving latency semantics and execution status.
   - Dedicated `derived_assessments` table and API endpoints for reproducible offline re-scoring.
   - Strict telemetry separation ensuring mock runs never report fabricated measured latency.
   - Single-audio telemetry cleanly distinguishing browser end-to-end latency, backend-measured Gnani API request time, and provider-internal processing time ("Not reported").
   - Server-side credit guard, origin validation, and anti-CSRF authorization manager (`app/services/live_guard.py`).
   - Default disabled live evaluation deployment safety guard (`LIVE_EVALUATION_ENABLED=false`).
   - Live paired audio REST API (`POST /api/eval/run-live/{test_id}`) with strict acoustic condition enum validation and format restriction to WAV (verifiably $\le 60$s).
   - Backward-compatible SQLite migration and metric aggregation isolation ensuring operational errors and derived re-scores never contaminate benchmark performance averages.
   - Comprehensive test suite (**122 tests passing with 0 network calls and 0 credits consumed**).
2. **Pending Acoustic Research (`pending_acoustic`):**
   - Calibrated acoustic noise testing with multi-speaker recordings under measured signal-to-noise ratios (street noise, cafeteria babble, reverberation).
   - Crowdsourced audio dataset recording all 24 seed utterances across diverse regional Indian accents.
3. **Regional Language Expansion:**
   - Future planned paired-contrast catalogs for Tamil (`ta-IN`), Telugu (`te-IN`), Kannada (`kn-IN`), and Bengali (`bn-IN`).
