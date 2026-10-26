# BoliBench Hinglish Transcription and Normalization Policy

**Version:** 1.0.0
**Status:** Approved Standard
**Target Domain:** Enterprise Conversational Voice AI (Banking, E-Commerce, Logistics, Ticketing)
**Applicable Locales:** `hi-IN` / `en-IN` Mixed-Code (Hinglish)

---

## 1. Executive Summary & Purpose

Modern Indian voice AI systems operate predominantly in **Hinglish**—code-mixed speech combining Hindi grammar, vocabulary, and syntax with English terminology and loan words. Benchmarking Automatic Speech Recognition (ASR) engines (such as Gnani Prisma, Google Speech-to-Text, Whisper, and Azure Speech) on conversational Hinglish presents unique evaluation challenges:

1. **Orthographic Inconsistency:** Romanized Hindi lacks a single standardized phonetic spelling (e.g., *chahiye* vs. *chahie*, *pachaas* vs. *pachas*).
2. **Script Divergence:** ASR providers frequently output transcripts in Devanagari script (`५०० रुपये`), Latin script (`500 rupaye`), or mixed scripts (`500 रुपये transfer kar do`), even when evaluating identical audio.
3. **Numeral Formatting:** Numbers may be transcribed as digits (`2`), English words (`two`), Hindi words in Roman script (`do`), or Hindi words in Devanagari (`दो`).
4. **Loan Word Transliteration:** Common English technical and transactional words are frequently transliterated phonetically into Devanagari (`ट्रांसफर`, `कैंसिल`, `बुकिंग`) rather than output in English script.

The purpose of this policy is to establish a rigorous, auditable standard for:
- Ground-truth transcription of human recordings.
- Cross-script text normalization during benchmark evaluation.
- Methodological demarcation between true ASR recognition errors and surface normalization artifacts.
- The comparative suitability of Word Error Rate (WER) versus Character Error Rate (CER) in code-mixed evaluations.
- An absolute zero-tolerance principle for silent rewrites that could compromise semantic distinction or critical transaction parameters.

---

## 2. Orthography and Script Standards

### 2.1 Ground-Truth Reference Standard
All curated benchmark contrast pairs in BoliBench maintain ground-truth references using **standard Roman-script Hinglish** (conversational Latin orthography), supplemented with canonical metadata for expected entity slots:

- **Reference Script:** Latin/Roman script (`UTF-8`).
- **Devanagari Dual Representation:** Where paired Devanagari reference transcripts exist, they must be stored in parallel metadata fields without overwriting or displacing the primary Roman reference.
- **Language Tagging:** Utterances are explicitly tagged with `hi-IN` (Indic code-mixed context).

### 2.2 Standard Romanized Hindi Spelling Conventions
To eliminate arbitrary orthographic variance in benchmark creation, ground-truth transcriptions must adhere to standard phonemic conventions:

| Category | Standard Form | Discouraged / Non-Canonical Variants | Rationale |
| :--- | :--- | :--- | :--- |
| **Aspirated Consonants** | `haan`, `nahin`, `chhe` | `han`, `nahi`, `che`, `chhah` | Retains standard Indic phonetic transliteration |
| **Auxiliary Verbs** | `chahiye` | `chahie`, `chahiyea` | Standardized verbal inflection |
| **Vowel Length** | `paanch`, `teen`, `saat`, `aath` | `panch`, `tin`, `sat`, `ath` | Preserves long vowel phoneme markers for numerals |
| **Tens & Multiples** | `chaalis`, `pachaas`, `saath` | `chalis`, `pachas`, `sat` | Disambiguates numbers from other words (`saath` [60] vs `saath` [together]) |
| **Large Scales** | `hazaar`, `lakh`, `crore` | `hazar`, `lac`, `karod` | Standard Indian English / Romanized banking terminology |

---

## 3. Punctuation, Casing, and Numeral Tokens

### 3.1 Punctuation Stripping
Spoken conversational utterances in enterprise voice AI do not carry acoustic punctuation tokens. ASR evaluation must therefore strip punctuation prior to computing Levenshtein alignment:
- **Stripped Punctuation Pattern:** `[\.,!?;:\"'—\(\)\[\]\{\}«»“”‘’।॥\-]`
- **Indic Dandam Marks:** Devanagari dandas (`।` and `॥`) are treated identically to commas/periods and stripped.
- **Hyphens:** Hyphens between compound words (e.g., `pre-paid`, `check-in`) are replaced with a single space.
- **Whitespace:** Consecutive whitespace characters are collapsed into a single ASCII space `\x20`, and leading/trailing whitespace is trimmed.

### 3.2 Casing Rules
Case distinctions do not exist in spoken phonetics or Indic scripts.
- All Latin characters are converted to lowercase prior to WER alignment.
- Devanagari characters are Unicode NFC-normalized (`unicodedata.normalize("NFC", text)`), leaving glyphs unmodified by case folding.

### 3.3 Numeral Token Normalization & Disambiguation
Numeral representation is the single highest source of spurious WER in Indian voice AI. A customer saying *"Mujhe do ticket chahiye"* may be transcribed as:
- `Mujhe 2 ticket chahiye` (digit)
- `Mujhe do ticket chahiye` (Roman Hindi)
- `मुझे २ टिकट चाहिए` (Devanagari digit)
- `मुझे दो टिकट चाहिए` (Devanagari text)

#### The `do` / `दो` Polysemy Rule
In Hindi, the word **`do`** / **`दो`** has two fundamentally distinct grammatical meanings:
1. **Cardinal Numeral `2`:** *"Do packet chahiye"* / *"Do sau rupaye"*.
2. **Imperative / Auxiliary Light Verb ("give" / "perform"):** *"Kar do"* / *"Bhej do"* / *"Band kar do"* / *"Cancel kar do"*.

**Normalization Policy:**
- `do` / `दो` MUST NOT be indiscriminately mapped to digit `2`.
- When preceded by verbal roots (`kar`, `bhej`, `de`, `rok`, `rakh`, `bata`, `hata`, `daal`, `khol`, `chhod`, `mat`, `nahi`), `do` is preserved as the verbal token `do`.
- When followed by quantified nouns, units, scales, or ordering predicates (`packet`, `plate`, `rupaye`, `sau`, `ticket`, `chahiye`), `do` is canonicalized to digit `2`.
- Isolated or ambiguous occurrences default conservatively to verbal token `do`.

#### Scale Token Equivalence
The following equivalences are mapped deterministically during Normalized WER calculation:
- `ek` / `एक` / `one` / `१` $\to$ `1`
- `do` / `दो` (quantifier context) / `two` / `२` $\to$ `2`
- `teen` / `tin` / `तीन` / `three` / `३` $\to$ `3`
- `paanch` / `panch` / `पाँच` / `five` / `५` $\to$ `5`
- `sau` / `सौ` / `hundred` $\to$ `100`
- `hazaar` / `हज़ार` / `thousand` $\to$ `1000`
- `lakh` / `lac` / `लाख` $\to$ `100000`
- `crore` / `करोड़` $\to$ `10000000`

---

## 4. English Loan Words vs. Native Hindi Transliterations

### 4.1 Enterprise Domain Code-Mixing
Hinglish in transactional voice systems uses standard English words for business entities. ASR engines frequently produce valid phonetic Devanagari transliterations. BoliBench treats these as orthographic equivalents during Normalized WER calculation:

| Domain | Standard Roman Lexeme | Devanagari Transliteration Equivalence |
| :--- | :--- | :--- |
| **Banking / Payments** | `transfer` | `ट्रांसफर` |
| | `account` | `अकाउंट`, `खाता` |
| | `balance` | `बैलेंस` |
| | `rupaye` | `rupees`, `rs`, `रुपये`, `रुपए` |
| **E-Commerce / Food** | `packet` | `packets`, `पैकेट`, `पेकेट` |
| | `order` | `ऑर्डर` |
| | `cart` | `कार्ट` |
| | `delivery` | `डिलीवरी` |
| | `pizza` | `पिज़्ज़ा`, `पिज्जा` |
| | `biryani` | `बिरयानी` |
| **Customer Support** | `cancel` | `कैंसिल`, `रद्द` |
| | `confirm` | `कन्फर्म`, `पुष्टि` |
| | `booking` | `बुकिंग` |
| | `return` | `रिटर्न` |

### 4.2 Bounded Domain Vocabulary
- Equivalence maps are restricted to a verified, domain-scoped whitelist.
- Broad, unconstrained phonetic transliteration fuzzy matching is strictly prohibited to prevent false positives and semantic drift.

---

## 5. Demarcating True ASR Errors vs. Normalization Artifacts

BoliBench establishes a rigorous separation between genuine speech recognition failures and benign surface discrepancies:

```
+--------------------------------------------------------------------------+
|                        CLASSIFICATION TAXONOMY                           |
+--------------------------------------------------------------------------+
| [TRUE RECOGNITION FAILURES]                [NORMALIZATION ARTIFACTS]      |
|                                                                          |
| * Entity Value Mutation:                   * Cross-Script Divergence:    |
|   "500" -> "5000" (critical failure)         "sau" -> "100" -> "सौ"      |
|                                                                          |
| * Polarity Flip:                           * Transliterated Loan Word:   |
|   "transfer kar do" -> "mat karo"            "cancel" -> "कैंसिल"        |
|                                                                          |
| * Intent Verb Omission:                    * Phonetic Roman Variant:     |
|   "ticket book karo" -> "ticket status"      "chahiye" -> "chahie"       |
|                                                                          |
| * Acoustic Substitution:                   * Punctuation / Capitalization|
|   "Rohit" -> "Rahul" (target distinction)    "Order!" -> "order"         |
+--------------------------------------------------------------------------+
```

### 5.1 True ASR Recognition Failures
A difference is classified as a genuine speech recognition failure if it satisfies any of the following:
1. **Critical Entity Mutation:** Alteration of dates, quantities, account identifiers, currency amounts, or named beneficiaries (e.g., reference: `500 rupaye`, hypothesis: `50 rupaye`).
2. **Intent Polarity Inversion:** Alteration of affirmative vs. negative intent (e.g., reference: `haan confirm karo`, hypothesis: `nahi confirm karo`).
3. **Contrast Distinction Collapse:** Failure to preserve the acoustic or semantic difference between paired utterances $A$ and $B$ (e.g., Utterance A asks for `2 packet` and Utterance B asks for `3 packet`, but the engine predicts `2 packet` for both).

### 5.2 Normalization Artifacts
A difference is classified as a normalization artifact if:
1. The acoustic phonemes correspond to the intended lexeme, but the ASR provider rendered it in Devanagari while the reference was in Latin script.
2. The word is an agreed English loan word rendered phonetically into Indic script.
3. The number was expressed in verbal words rather than Arabic numerals.

---

## 6. Metric Suitability: WER vs. CER in Mixed-Code Speech

### 6.1 Limitations of Raw Word Error Rate (WER)
Word Error Rate is computed via token-level Levenshtein distance:
$$\text{WER} = \frac{S + D + I}{N}$$

In code-mixed Hinglish:
- If a model transcribes `Do packet chahiye` as `2 पैकेट चाहिए`, raw WER is **100%** (3 substitutions out of 3 tokens).
- Relying solely on raw WER severely misrepresents the real-world business utility of the ASR engine for downstream intent extraction.

### 6.2 Hazards of Character Error Rate (CER)
Character Error Rate evaluates Levenshtein distance at the Unicode character level:
$$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$

While CER is useful for purely phonetic languages, in Hinglish evaluation:
- **Dangerously Low Penalty for Semantic Inversion:** The difference between `do` (cardinal 2) and `ho` (verb) is 1 character substitution out of 2 (50% word-level impact, but negligible CER impact across a 30-character sentence).
- **Masking Polarity Changes:** The difference between `karo` and `mat karo` is only 4 characters, yet it completely flips transactional execution.
- **Cross-Script Incompatibility:** Comparing Latin `packet` (6 characters) with Devanagari `पैकेट` (4 Unicode codepoints) yields a meaningless character Levenshtein distance across disjoint Unicode blocks.

### 6.3 BoliBench's Multi-Tiered Evaluation Paradigm
To solve this dilemma, BoliBench rejects both unadjusted raw WER and character-level smoothing in favor of a **four-tier evaluation pipeline**:

1. **Tier 1: Raw Word Error Rate (`wer`):**
   Literal word-level Levenshtein distance after basic punctuation stripping and lowercasing. Provides an unvarnished audit of textual match.
2. **Tier 2: Cross-Script Normalized WER (`normalized_wer`):**
   Word-level Levenshtein distance computed after domain-scoped numeral and script canonicalization. Measures phonetic and semantic transcript fidelity without script bias.
3. **Tier 3: Critical Field Accuracy (`critical_field_accuracy`):**
   Deterministic verification that key entity slots (quantities, amounts, destinations, dates) extracted from the actual transcript match the expected values.
4. **Tier 4: Contrast Distinction Preservation (`distinction_preserved` & `task_success`):**
   Pair-level verification that the ASR engine successfully distinguished between paired contrast utterances $A$ and $B$, avoiding conversational collapse.

---

## 7. Zero-Tolerance Principle for Silent Rewrites

To maintain complete scientific integrity and auditability, BoliBench enforces the following **Zero-Tolerance Principles**:

1. **Immutable Ground Truth:**
   Reference transcripts, once verified in a curated cohort (`human_verified`), must NEVER be modified automatically or silently overwritten. Any correction requires an explicit, audited pull request with changelog notes.
2. **Deterministic, Reversible Normalization:**
   Text normalizers must be pure functions without external network calls, non-deterministic heuristics, or stochastic LLM rewriting.
3. **No Loss of Distinctions:**
   A normalization rule must never map two semantically distinct tokens to the same canonical representation. Specifically:
   - Different numbers must never be equated (`2` $\ne$ `3`, `50` $\ne$ `500`).
   - Affirmative and negative particles must never be equated (`haan` $\ne$ `nahi`).
   - Different beneficiary names or locations must never be merged.
4. **Dual Reporting Requirement:**
   Normalized metrics must ALWAYS be presented alongside Raw metrics in API responses, database persistence, and benchmark reports. Systems must never report Normalized WER without disclosing the Raw WER.

---

## 8. Human Dataset Curation & Verification Standards

For an audio recording and pair to attain **`human_verified`** benchmark status:

1. **Recording Quality Checklist:**
   - Sample rate: 16,000 Hz, 16-bit PCM Mono WAV format.
   - Signal-to-Noise Ratio (SNR) $\ge 20\text{ dB}$.
   - No digital clipping or excessive acoustic echo.
   - Natural native speaker pronunciation matching the target Indian demographic.
2. **Transcript Verification:**
   - Independent verification by a native speaker that the audio verbatim matches the Roman Hinglish reference.
   - Discrepancies between spoken audio and written reference disqualify the pair until re-recorded.
3. **Cohort Segregation:**
   - Curated benchmark metrics must be generated exclusively from `human_verified` pairs (`curated_only=True`).
   - Test artifacts (`test_artifact`) and unverified uploads (`unverified`) are segregated logically and physically excluded from official benchmark scores.
