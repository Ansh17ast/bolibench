"""Evaluation metrics, text normalizer, deterministic field extractors, and contrast scoring."""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from app.evaluation.schemas import (
    FailureCategory,
    PairEvaluationResult,
    TestCasePair,
    TestCategory,
    UtteranceEvaluationResult,
    UtteranceSpec,
)


# -----------------------------------------------------------------------------
# 1. Text Normalization for Evaluation (Preserving Semantic Distinctions)
# -----------------------------------------------------------------------------

# Characters to strip during normalization (punctuation & quotes)
PUNCTUATION_PATTERN = re.compile(r"[\.,!?;:\"'—\(\)\[\]\{\}«»“”‘’।॥\-]")


def normalize_text(text: Optional[str], language_code: str = "hi-IN") -> str:
    """Normalizes text for Word Error Rate (WER) and field extraction.

    Guarantees:
    - Strips punctuation and excessive whitespace.
    - Lowers case for Latin characters.
    - Preserves Unicode Devanagari characters and Hindi/English number tokens.
    - NEVER normalizes away meaningful distinctions (e.g. 'do' vs 'teen',
      'karna' vs 'mat karna', or corrected amounts).
    """
    if not text:
        return ""

    # Normalize unicode composition (NFC)
    normalized = unicodedata.normalize("NFC", text.strip())

    # Lowercase Latin script characters (does not affect Devanagari/Indic scripts)
    normalized = normalized.lower()

    # Remove standard punctuation and Devanagari dandas (। ॥)
    normalized = PUNCTUATION_PATTERN.sub(" ", normalized)

    # Collapse multiple spaces into a single space
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


# -----------------------------------------------------------------------------
# 2. Word Error Rate (WER) Calculation
# -----------------------------------------------------------------------------
# 2. Word Error Rate (WER) & Cross-Script Normalized WER Calculations
# -----------------------------------------------------------------------------

# Vocabulary equivalence mapping across Devanagari, Romanized Hindi, and numerals
CROSS_SCRIPT_MAP: Dict[str, str] = {
    # Quantities & Numbers: Map to canonical digit strings (never conflating different numbers)
    "ek": "1", "एक": "1", "1": "1", "१": "1", "one": "1",
    # 'do' and 'दो' are contextually disambiguated in canonicalize_cross_script_text()
    "2": "2", "२": "2", "two": "2",
    "teen": "3", "tin": "3", "तीन": "3", "3": "3", "३": "3", "three": "3",
    "char": "4", "chaar": "4", "चार": "4", "4": "4", "४": "4", "four": "4",
    "paanch": "5", "panch": "5", "पाँच": "5", "पांच": "5", "5": "5", "५": "5", "five": "5",
    "chhah": "6", "che": "6", "chhe": "6", "छह": "6", "6": "6", "६": "6", "six": "6",
    "saat": "7", "सात": "7", "7": "7", "७": "7", "seven": "7",
    "aath": "8", "ath": "8", "आठ": "8", "8": "8", "८": "8", "eight": "8",
    "nau": "9", "नौ": "9", "9": "9", "९": "9", "nine": "9",
    "das": "10", "दस": "10", "10": "10", "१०": "10", "ten": "10",
    "gyarah": "11", "ग्यारह": "11", "11": "11",
    "barah": "12", "बारह": "12", "12": "12",
    "terah": "13", "तेरह": "13", "13": "13",
    "chaudah": "14", "चौदह": "14", "14": "14",
    "pandrah": "15", "पंद्रह": "15", "15": "15",
    "solah": "16", "सोलह": "16", "16": "16",
    "satrah": "17", "सत्रह": "17", "17": "17",
    "atharah": "18", "अठारह": "18", "18": "18",
    "unnis": "19", "उन्नीस": "19", "19": "19",
    "bees": "20", "बीस": "20", "20": "20",
    "pachees": "25", "पच्चीस": "25", "25": "25",
    "tees": "30", "तीस": "30", "30": "30",
    "chaalis": "40", "चालीस": "40", "40": "40",
    "pachaas": "50", "pachas": "50", "पचास": "50", "50": "50",
    "saath": "60", "साठ": "60", "60": "60",
    "sattar": "70", "सत्तर": "70", "70": "70",
    "assi": "80", "अस्सी": "80", "80": "80",
    "nabbe": "90", "नब्बे": "90", "90": "90",
    "sau": "100", "सौ": "100", "100": "100",
    "hazaar": "1000", "hazar": "1000", "हज़ार": "1000", "हजार": "1000", "1000": "1000",
    "lakh": "100000", "lac": "100000", "लाख": "100000", "100000": "100000",
    "crore": "10000000", "करोड़": "10000000", "10000000": "10000000",

    # Common vocabulary in seed evaluations
    "packet": "packet", "packets": "packet", "पैकेट": "packet", "पेकेट": "packet",
    "chahiye": "chahiye", "चाहिए": "chahiye", "चाहिये": "chahiye",
    "rupaye": "rupaye", "rupees": "rupaye", "रुपये": "rupaye", "रुपए": "rupaye", "rs": "rupaye",
    "kar": "kar", "कर": "kar", "karo": "karo", "करो": "karo",
    "bhejo": "bhejo", "भेजो": "bhejo",
    "transfer": "transfer", "ट्रांसफर": "transfer",
    "nahi": "nahi", "nahin": "nahi", "नहीं": "nahi", "ना": "nahi",
    "mat": "mat", "मत": "mat",
    "haan": "haan", "हाँ": "haan", "हां": "haan",
    "cancel": "cancel", "कैंसिल": "cancel", "रद्द": "cancel",
    "booking": "booking", "बुकिंग": "booking",
    "balance": "balance", "बैलेंस": "balance",
    "account": "account", "अकाउंट": "account", "खाता": "account",
    "band": "band", "बंद": "band",
    "cart": "cart", "कार्ट": "cart",
    "khali": "khali", "खाली": "khali",
    "confirm": "confirm", "कन्फर्म": "confirm", "पुष्टि": "confirm",
    "order": "order", "ऑर्डर": "order",
    "plate": "plate", "प्लेट": "plate",
    "biryani": "biryani", "बिरयानी": "biryani",
    "pizza": "pizza", "पिज़्ज़ा": "pizza", "पिज्जा": "pizza",
    "mumbai": "mumbai", "मुंबई": "mumbai",
    "delhi": "delhi", "दिल्ली": "delhi",
    "flight": "flight", "फ्लाइट": "flight",
}

# Auxiliary verbal triggers where 'do' / 'दो' functions as the imperative/auxiliary verb ("give" / "perform")
VERBAL_AUXILIARY_PRECEDING_TOKENS: Set[str] = {
    "kar", "कर", "karo", "करो", "karna", "करना",
    "bhej", "भेज", "bhejo", "भेजो",
    "de", "दे", "dijiye", "दीजिए", "dekh", "देख",
    "le", "ले", "lo", "लो",
    "rok", "रोक", "roko", "रोको",
    "rakh", "रख", "rakho", "रखो",
    "bata", "बता", "batao", "बताओ",
    "hata", "हटा", "hatao", "हटाओ",
    "daal", "डाल", "daalo", "डालो",
    "khol", "खोल", "kholo", "खोलो",
    "chhod", "छोड़", "chhodo", "छोड़ो",
    "mat", "मत", "nahi", "nahin", "नहीं",
    "paise", "पैसे", "kitab", "किताब",
}

# Tokens following 'do' / 'दो' that indicate it is functioning as the cardinal NUMERAL 2
NUMERAL_QUANTIFIED_NOUN_TOKENS: Set[str] = {
    "packet", "packets", "पैकेट", "पेकेट",
    "plate", "plates", "प्लेट",
    "rupaye", "rupees", "रुपये", "रुपए", "rs",
    "sau", "सौ", "hazaar", "हज़ार", "hazar", "हजार", "lakh", "लाख", "crore", "करोड़",
    "item", "items", "चीज", "चीजें",
    "ticket", "tickets", "टिकट",
    "piece", "pieces", "पीस",
    "kilo", "kg", "gram", "किलो", "ग्राम",
    "cup", "glass", "bottle", "bottles", "कप", "ग्लास", "बोतल",
    "saal", "mahine", "din", "ghante", "साल", "महीने", "दिन", "घंटे",
    "log", "लोग", "dost", "दोस्त",
    "chahiye", "चाहिए", "चाहिये",
    "aur", "और",
    "pizza", "पिज़्ज़ा", "पिज्जा", "biryani", "बिरयानी",
    "thousand", "hundred",
}


def canonicalize_cross_script_text(text: Optional[str], language_code: str = "hi-IN") -> str:
    """Normalizes script and numeric representations across Hindi and Romanized Hindi for evaluation.

    Scope Limitation & Domain Specificity:
    - This is a targeted, domain-scoped supplementary normalization procedure for Indian voice AI
      benchmarking (specifically e-commerce orders, banking transactions, ticketing, and numbers).
    - It is NOT a universal arbitrary Hindi-English machine transliteration engine.
    - Raw Word Error Rate (WER) and its Levenshtein operations (S/D/I) are strictly preserved and
      never masked by this metric.

    Context-Aware Disambiguation for Romanized 'do' / Devanagari 'दो':
    - In Hindi, 'do' / 'दो' is polysemous: it represents either the cardinal numeral 2 (e.g. 'Do packet')
      or the imperative / auxiliary light verb 'to give/perform' (e.g. 'kar do', 'bhej do', 'band kar do').
    - To prevent silent semantic conflation, 'do' / 'दो' is contextually classified:
        1. When preceded by verbal roots (e.g. 'kar', 'bhej', 'de', 'rok', 'rakh', 'mat'), it is preserved
           as the verbal token 'do', avoiding spurious conversion to numeral '2'.
        2. When followed by quantified nouns, units, scales, or ordering predicates (e.g. 'packet', 'plate',
           'rupaye', 'sau', 'chahiye'), it is canonicalized to digit '2'.
        3. Devanagari 'दो' without verbal trigger defaults to numeral '2', but when preceded by a verbal
           auxiliary (e.g. 'कर दो'), it preserves the verbal token 'do'.
        4. Standalone or ambiguous occurrences without quantifier context default conservatively to 'do'.

    Guarantees:
    - Cardinal numbers remain strictly distinct (e.g. '2' and '3' are never conflated).
    - Semantic contrasts such as negation ('nahi' vs 'haan') and imperative actions are preserved.
    - Equivalences for supported benchmark pairs (e.g. 'Do packet chahiye' vs '2 पैकेट चाहिए',
      'Teen packet chahiye' vs '3 पैकेट चाहिए') evaluate cleanly to 0.0 normalized WER.
    """
    if not text:
        return ""

    norm = normalize_text(text, language_code)
    words = norm.split()
    canonical_tokens: List[str] = []

    for i, w in enumerate(words):
        if w in ("do", "दो"):
            is_verbal = (i > 0 and words[i - 1] in VERBAL_AUXILIARY_PRECEDING_TOKENS)
            is_numeral = (i + 1 < len(words) and words[i + 1] in NUMERAL_QUANTIFIED_NOUN_TOKENS)

            if is_verbal:
                canonical_tokens.append("do")
            elif is_numeral:
                canonical_tokens.append("2")
            elif w == "दो" and not is_verbal:
                canonical_tokens.append("2")
            else:
                canonical_tokens.append("do")
        else:
            canonical_tokens.append(CROSS_SCRIPT_MAP.get(w, w))

    return " ".join(canonical_tokens)


def _levenshtein_distance_words(
    norm_ref: str,
    norm_hyp: str,
) -> Tuple[float, int, int, int, int]:
    """Internal dynamic programming Levenshtein distance calculator over word tokens."""
    ref_words = norm_ref.split() if norm_ref else []
    hyp_words = norm_hyp.split() if norm_hyp else []

    r_len = len(ref_words)
    h_len = len(hyp_words)

    if r_len == 0 and h_len == 0:
        return 0.0, 0, 0, 0, 0
    if r_len == 0:
        return 1.0, 0, 0, h_len, 0
    if h_len == 0:
        return 1.0, 0, r_len, 0, r_len

    # Distance matrix (r_len + 1) x (h_len + 1)
    d = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                deletion = d[i - 1][j] + 1
                insertion = d[i][j - 1] + 1
                d[i][j] = min(substitution, deletion, insertion)

    # Backtrace to count individual S, D, I operations
    substitutions = 0
    deletions = 0
    insertions = 0

    i, j = r_len, h_len
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            i = max(0, i - 1)
            j = max(0, j - 1)

    wer = round(float(substitutions + deletions + insertions) / float(r_len), 4)
    return wer, substitutions, deletions, insertions, r_len


def calculate_wer(
    reference: str,
    hypothesis: str,
    language_code: str = "hi-IN",
) -> Tuple[float, int, int, int, int]:
    """Calculates standard raw Word Error Rate using Levenshtein distance on words.

    Preserves exact script and character distinctions without cross-script transliteration.

    Returns:
        Tuple of (wer, substitutions, deletions, insertions, reference_word_count)
    """
    norm_ref = normalize_text(reference, language_code)
    norm_hyp = normalize_text(hypothesis, language_code)
    return _levenshtein_distance_words(norm_ref, norm_hyp)


def calculate_cross_script_normalized_wer(
    reference: str,
    hypothesis: str,
    language_code: str = "hi-IN",
) -> Tuple[float, int, int, int, int]:
    """Calculates supplementary cross-script-normalized Word Error Rate (WER).

    Resolves Hindi / Romanized-Hindi and numeral representations to canonical tokens
    while strictly preserving semantic distinctions (e.g. distinct numbers, negation).

    Returns:
        Tuple of (normalized_wer, substitutions, deletions, insertions, reference_word_count)
    """
    canon_ref = canonicalize_cross_script_text(reference, language_code)
    canon_hyp = canonicalize_cross_script_text(hypothesis, language_code)
    return _levenshtein_distance_words(canon_ref, canon_hyp)


# -----------------------------------------------------------------------------
# 3. Deterministic Field Extractors
# -----------------------------------------------------------------------------

# Number mappings
INDIC_DEVANAGARI_DIGITS = {"०": 0, "१": 1, "२": 2, "३": 3, "४": 4, "५": 5, "६": 6, "७": 7, "८": 8, "९": 9}

NUMBER_WORDS: Dict[str, int] = {
    # Hindi romanized
    "ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5,
    "chhah": 6, "che": 6, "chhe": 6, "saat": 7, "aath": 8, "ath": 8, "nau": 9, "das": 10,
    "gyarah": 11, "barah": 12, "terah": 13, "chaudah": 14, "pandrah": 15, "solah": 16,
    "satrah": 17, "atharah": 18, "unnis": 19, "bees": 20, "pachees": 25, "tees": 30,
    "chaalis": 40, "pachaas": 50, "pachas": 50, "saath": 60, "sattar": 70, "assi": 80, "nabbe": 90,
    "sau": 100, "hazaar": 1000, "hazar": 1000, "lakh": 100000, "lac": 100000, "crore": 10000000,
    # Devanagari script words
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "बीस": 20, "पच्चीस": 25, "तीस": 30, "चालीस": 40, "पचास": 50,
    "सौ": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 100000, "करोड़": 10000000,
    # English words
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "twenty-five": 25,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}

SCALE_MULTIPLIERS: Dict[str, int] = {
    "sau": 100, "सौ": 100, "hundred": 100,
    "hazaar": 1000, "hazar": 1000, "हज़ार": 1000, "हजार": 1000, "thousand": 1000, "k": 1000,
    "lakh": 100000, "lac": 100000, "लाख": 100000,
    "crore": 10000000, "करोड़": 10000000,
}

NEGATION_TOKENS: Set[str] = {
    "nahi", "nahin", "नहीं", "mat", "मत", "na", "न", "ruko", "रुको",
    "no", "not", "dont", "don't", "never", "stopp", "stop",
}

AFFIRMATIVE_TOKENS: Set[str] = {
    "haan", "haa", "हाँ", "yes", "sure", "karo", "करो", "ok", "okay",
    "confirm", "proceed", "transfer", "send", "book", "checkout", "order",
}


def extract_quantities(text: str) -> Dict[str, Any]:
    """Extracts numeric quantities, scales, and currency amounts deterministically."""
    norm = normalize_text(text)
    words = norm.split()
    results: Dict[str, Any] = {}

    # 1. Check for currency expressions (e.g. ₹500, ₹5,000, ₹5,20,000)
    currency_match = re.search(r"₹\s*([\d,]+(?:\.\d+)?)", text)
    if currency_match:
        val_str = currency_match.group(1).replace(",", "")
        try:
            results["amount"] = float(val_str) if "." in val_str else int(val_str)
            results["quantity"] = results["amount"]
        except ValueError:
            pass

    # 2. Look for compound quantity with scale (e.g., "paanch lakh", "5 hazaar", "पाँच हज़ार")
    for i, word in enumerate(words):
        base_val = None
        if word in NUMBER_WORDS:
            base_val = NUMBER_WORDS[word]
        elif word.isdigit():
            base_val = int(word)

        if base_val is not None:
            # Check next word for scale multiplier
            if i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in SCALE_MULTIPLIERS:
                    scale = SCALE_MULTIPLIERS[next_word]
                    total = base_val * scale
                    results["quantity"] = total
                    results["scale"] = scale
                    results["base_number"] = base_val
                    return results

            # Single number found (if not already set)
            if "quantity" not in results:
                results["quantity"] = base_val

    # 3. Direct digit extraction from text
    if "quantity" not in results:
        digits = re.findall(r"\b\d+\b", norm)
        if digits:
            results["quantity"] = int(digits[0])

    return results


def extract_negation(text: str) -> Dict[str, Any]:
    """Extracts negation status and affirmative/negative intent deterministically."""
    norm = normalize_text(text)
    words = norm.split()

    is_negated = any(word in NEGATION_TOKENS for word in words)

    # Detect specific imperative combinations (e.g. "cancel mat karo" vs "cancel kar do")
    action = "proceed"
    if "cancel" in words:
        if is_negated:
            # "cancel mat karo" -> keep / do not cancel
            action = "keep_booking"
        else:
            # "cancel kar do" -> perform cancellation
            action = "cancel_booking"
    elif "confirm" in words or "checkout" in words:
        if is_negated:
            action = "hold_cart"
        else:
            action = "confirm_order"
    elif "transfer" in words:
        if is_negated:
            action = "block_transfer"
        else:
            action = "execute_transfer"
    else:
        action = "denied" if is_negated else "confirmed"

    return {
        "is_negated": is_negated,
        "action": action,
    }


def extract_corrections(text: str) -> Dict[str, Any]:
    """Extracts change-of-mind corrections, identifying initial vs final intended target."""
    norm = normalize_text(text)

    # Correction cue phrases
    cues = [
        "no wait", "no wait to", "nahi", "nahin", "nahi ruko",
        "nahi pehle", "actually", "change to", "not", "make it",
    ]

    for cue in cues:
        pattern = rf"\b{re.escape(cue)}\b"
        match = re.search(pattern, norm)
        if match:
            before_text = norm[:match.start()].strip()
            after_text = norm[match.end():].strip()

            # Extract final target from after_text
            after_qty = extract_quantities(after_text)
            before_qty = extract_quantities(before_text)

            # Check destinations (e.g. "flight to Mumbai, no wait, to Delhi")
            dest_pattern = r"(?:to|flight to|jaana hai)\s+([a-zA-Z]+)"
            before_dest = re.findall(dest_pattern, before_text)
            after_dest = re.findall(r"(?:to|flight to|se|)\s*([a-zA-Z]+)", after_text)

            final_destination = after_dest[-1] if after_dest else (after_text.split()[-1] if after_text else None)
            initial_destination = before_dest[-1] if before_dest else (before_text.split()[-1] if before_text else None)

            return {
                "correction_detected": True,
                "initial_value": before_qty.get("quantity") or initial_destination,
                "final_target": after_qty.get("quantity") or final_destination,
                "quantity": after_qty.get("quantity"),
                "destination": final_destination,
            }

    # If no correction cue is found, the statement is an uncorrected direct command
    direct_qty = extract_quantities(norm)
    destinations = re.findall(r"(?:to|flight to)\s+([a-zA-Z]+)", norm)
    final_dest = destinations[-1] if destinations else None

    return {
        "correction_detected": False,
        "initial_value": direct_qty.get("quantity") or final_dest,
        "final_target": direct_qty.get("quantity") or final_dest,
        "quantity": direct_qty.get("quantity"),
        "destination": final_dest,
    }


def extract_entities(text: str) -> Dict[str, Any]:
    """Extracts named entities and product names deterministically."""
    norm = normalize_text(text)

    results: Dict[str, Any] = {}

    # Person names
    if "rahul sharma" in norm or "rahul" in norm:
        results["person_name"] = "Rahul Sharma"
    elif "rohit sharma" in norm or "rohit" in norm:
        results["person_name"] = "Rohit Sharma"

    # Product names (Prisma vs Krishna, Timbre vs Timber)
    if "prisma" in norm:
        results["product_name"] = "Prisma STT"
    elif "krishna" in norm:
        results["product_name"] = "Krishna STT"
    elif "timbre" in norm:
        results["product_name"] = "Timbre"
    elif "timber" in norm:
        results["product_name"] = "Timber"

    # Cities
    if "mumbai" in norm:
        results["destination"] = "Mumbai"
    elif "delhi" in norm:
        results["destination"] = "Delhi"

    return results


def extract_command_intent(text: str) -> str:
    """Classifies spoken utterance command intent supporting Latin, Romanized Hindi, and Devanagari."""
    norm = normalize_text(text)
    words = norm.split()

    # Negation tokens for intent checks
    negation_set = {"mat", "nahi", "nahin", "no", "not", "मत", "नहीं", "ना"}

    if "balance" in norm or "बैलेंस" in norm:
        return "CHECK_BALANCE"
    if "band" in norm or "बंद" in norm or "close" in norm or "terminate" in norm:
        return "CLOSE_ACCOUNT"
    if ("cancel" in norm or "कैंसिल" in norm or "रद्द" in norm) and any(n in words for n in negation_set):
        return "KEEP_BOOKING"
    if "cancel" in norm or "कैंसिल" in norm or "रद्द" in norm:
        return "CANCEL_BOOKING"
    if "empty" in norm or "khali" in norm or "खाली" in norm:
        return "EMPTY_CART"
    if "confirm" in norm or "checkout" in norm or "कन्फर्म" in norm or "पुष्टि" in norm:
        return "CONFIRM_ORDER"
    if "pay" in norm or "bhejo" in norm or "भेजो" in norm or "transfer" in norm or "ट्रांसफर" in norm:
        return "TRANSFER_FUNDS"
    if "request" in norm or "maang" in norm or "मांग" in norm or "माँग" in norm:
        return "REQUEST_FUNDS"
    if "order" in norm or "ऑर्डर" in norm or "chahiye" in norm or "चाहिए" in norm or "चाहिये" in norm or "plate" in norm or "प्लेट" in norm:
        return "ORDER_PRODUCT"

    return "UNKNOWN_INTENT"


def extract_all_critical_fields(
    transcript: str,
    category: TestCategory,
    language_code: str = "hi-IN",
) -> Dict[str, Any]:
    """Combines deterministic extractors based on test category."""
    fields: Dict[str, Any] = {}

    if category == TestCategory.QUANTITIES_AND_NUMBERS:
        fields.update(extract_quantities(transcript))
    elif category == TestCategory.NEGATION:
        fields.update(extract_negation(transcript))
    elif category == TestCategory.CORRECTIONS_AND_MIND_CHANGES:
        fields.update(extract_corrections(transcript))
    elif category in (TestCategory.NAMED_ENTITIES, TestCategory.PRODUCT_NAMES):
        fields.update(extract_entities(transcript))
    elif category == TestCategory.COMMAND_INTENT:
        fields["intent"] = extract_command_intent(transcript)
        fields.update(extract_quantities(transcript))
    elif category == TestCategory.HINDI_ENGLISH_CODE_SWITCHING:
        fields.update(extract_negation(transcript))
        fields.update(extract_quantities(transcript))
        fields.update(extract_entities(transcript))

    return fields


# -----------------------------------------------------------------------------
# 4. Single Utterance Evaluator
# -----------------------------------------------------------------------------

def evaluate_utterance(
    utterance_spec: UtteranceSpec,
    actual_transcript: str,
    category: TestCategory,
    language_code: str = "hi-IN",
    latency_ms: Optional[float] = None,
    client_request_elapsed_ms: Optional[float] = None,
    provider_processing_time_ms: Optional[float] = None,
    simulated_latency_ms: Optional[float] = None,
    request_id: Optional[str] = None,
) -> UtteranceEvaluationResult:
    """Evaluates a single utterance transcript against its ground truth specification."""
    wer, s, d, i, r_len = calculate_wer(
        reference=utterance_spec.reference_transcript,
        hypothesis=actual_transcript,
        language_code=language_code,
    )
    norm_wer, _, _, _, _ = calculate_cross_script_normalized_wer(
        reference=utterance_spec.reference_transcript,
        hypothesis=actual_transcript,
        language_code=language_code,
    )

    extracted_fields = extract_all_critical_fields(
        transcript=actual_transcript,
        category=category,
        language_code=language_code,
    )

    detected_intent = extract_command_intent(actual_transcript)
    intent_match = (detected_intent == utterance_spec.expected_intent)

    # Evaluate critical fields match
    field_matches: Dict[str, bool] = {}
    expected_fields = utterance_spec.expected_critical_fields

    for field_name, expected_val in expected_fields.items():
        actual_val = extracted_fields.get(field_name)
        if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
            field_matches[field_name] = (round(float(expected_val), 2) == round(float(actual_val), 2))
        elif isinstance(expected_val, str) and isinstance(actual_val, str):
            field_matches[field_name] = (expected_val.strip().lower() == actual_val.strip().lower())
        else:
            field_matches[field_name] = (expected_val == actual_val)

    critical_fields_match = (len(field_matches) > 0 and all(field_matches.values())) if expected_fields else True

    # Task success requires both critical fields and intent match
    task_success = critical_fields_match and intent_match

    effective_client_elapsed = client_request_elapsed_ms if client_request_elapsed_ms is not None else latency_ms

    return UtteranceEvaluationResult(
        utterance_id=utterance_spec.id,
        reference_transcript=utterance_spec.reference_transcript,
        actual_transcript=actual_transcript,
        wer=wer,
        substitutions=s,
        deletions=d,
        insertions=i,
        reference_length=r_len,
        extracted_fields=extracted_fields,
        critical_fields_match=critical_fields_match,
        field_matches=field_matches,
        detected_intent=detected_intent,
        intent_match=intent_match,
        task_success=task_success,
        latency_ms=effective_client_elapsed,
        client_request_elapsed_ms=effective_client_elapsed,
        provider_processing_time_ms=provider_processing_time_ms,
        simulated_latency_ms=simulated_latency_ms,
        request_id=request_id,
        normalized_wer=norm_wer,
    )


# -----------------------------------------------------------------------------
# 5. Paired-Contrast Evaluator & Failure Classifier
# -----------------------------------------------------------------------------

def classify_failure(
    test_case: TestCasePair,
    result_a: UtteranceEvaluationResult,
    result_b: UtteranceEvaluationResult,
    distinction_preserved: bool,
) -> Tuple[FailureCategory, str]:
    """Classifies speech recognition failure and explains downstream impact."""
    if result_a.task_success and result_b.task_success and distinction_preserved:
        return FailureCategory.NONE, "Semantic distinction preserved; voice agent would execute intended actions."

    category = test_case.category
    dist_field = test_case.distinction_field

    val_a = result_a.extracted_fields.get(dist_field)
    val_b = result_b.extracted_fields.get(dist_field)

    exp_a = test_case.utterance_a.expected_critical_fields.get(dist_field)
    exp_b = test_case.utterance_b.expected_critical_fields.get(dist_field)

    # 1. High WER corruption
    if (result_a.wer > 0.6 and not result_a.actual_transcript.strip()) or (result_b.wer > 0.6 and not result_b.actual_transcript.strip()):
        return (
            FailureCategory.HIGH_WER_CORRUPTION,
            "Severe transcription dropout or simulated corruption prevented field extraction.",
        )

    # 2. Negation Failure
    if category == TestCategory.NEGATION or dist_field == "is_negated":
        if not distinction_preserved or not (result_a.critical_fields_match and result_b.critical_fields_match):
            return (
                FailureCategory.MISSED_NEGATION,
                f"Voice agent missed critical negation (Expected distinction: {exp_a} vs {exp_b}, got {val_a} vs {val_b}). "
                "Downstream agent would perform an irreversible action contrary to user instruction (e.g. cancel instead of retain).",
            )

    # 3. Quantity / Number Error
    if category == TestCategory.QUANTITIES_AND_NUMBERS or dist_field in ("quantity", "amount", "scale"):
        if not distinction_preserved or not (result_a.critical_fields_match and result_b.critical_fields_match):
            return (
                FailureCategory.QUANTITY_ERROR,
                f"Numeric distinction collapsed (Expected: {exp_a} vs {exp_b}, extracted: {val_a} vs {val_b}). "
                "Downstream ordering or transaction agent would execute the wrong quantity/amount.",
            )

    # 4. Correction / Change-of-mind Error
    if category == TestCategory.CORRECTIONS_AND_MIND_CHANGES or dist_field in ("final_target", "destination"):
        if not distinction_preserved or not (result_a.critical_fields_match and result_b.critical_fields_match):
            return (
                FailureCategory.CORRECTION_ERROR,
                f"Agent failed to resolve change-of-mind correction (Expected final target: {exp_a} vs {exp_b}, got {val_a} vs {val_b}). "
                "Downstream agent would book initial discarded value rather than corrected user intent.",
            )

    # 5. Entity Substitution (Named Entities & Products)
    if category in (TestCategory.NAMED_ENTITIES, TestCategory.PRODUCT_NAMES) or dist_field in ("product_name", "person_name"):
        if not distinction_preserved or not (result_a.critical_fields_match and result_b.critical_fields_match):
            return (
                FailureCategory.ENTITY_SUBSTITUTION,
                f"Entity substitution error (Expected: {exp_a} vs {exp_b}, recognized: {val_a} vs {val_b}). "
                "Downstream voice bot routed to wrong person or requested wrong product SKU.",
            )

    # 6. Intent Error
    if category == TestCategory.COMMAND_INTENT or not (result_a.intent_match and result_b.intent_match):
        return (
            FailureCategory.INTENT_ERROR,
            f"Command intent misclassified (Utterance A: {result_a.detected_intent}, Utterance B: {result_b.detected_intent}). "
            "Downstream workflow triggered incorrect business logic handler.",
        )

    # 7. Default extraction failure
    return (
        FailureCategory.EXTRACTION_FAILURE,
        "Failed to extract structured critical fields from one or both transcripts.",
    )


def score_pair(
    run_id: str,
    test_case: TestCasePair,
    transcript_a: str,
    transcript_b: str,
    execution_mode: str = "mock",
    audio_condition: str = "text_only_simulation",
    timestamp: str = "",
    client_latency_ms: Optional[float] = None,
    client_request_elapsed_ms: Optional[float] = None,
    provider_latency_ms: Optional[float] = None,
    simulated_latency_ms: Optional[float] = None,
    latency_a_ms: Optional[float] = None,
    latency_b_ms: Optional[float] = None,
    client_elapsed_a_ms: Optional[float] = None,
    client_elapsed_b_ms: Optional[float] = None,
    provider_time_a_ms: Optional[float] = None,
    provider_time_b_ms: Optional[float] = None,
    request_id_a: Optional[str] = None,
    request_id_b: Optional[str] = None,
) -> PairEvaluationResult:
    """Scores a paired contrast test deterministically.

    Guarantees:
    - Mock execution mode NEVER populates client_measured_latency_ms or provider_processing_time_ms.
    - If latency is simulated for demonstration, it is recorded in simulated_latency_ms.
    - Live execution mode calculates client_measured_latency_ms strictly from elapsed provider requests.
    - provider_processing_time_ms remains null unless genuinely reported by provider.
    """
    is_live = (execution_mode == "live")

    # Strictly enforce mock vs live latency fields
    if is_live:
        total_client = client_request_elapsed_ms if client_request_elapsed_ms is not None else client_latency_ms
        final_client_latency = total_client
        final_provider_latency = provider_latency_ms
        final_latency_a = client_elapsed_a_ms if client_elapsed_a_ms is not None else latency_a_ms
        final_latency_b = client_elapsed_b_ms if client_elapsed_b_ms is not None else latency_b_ms
        final_provider_a = provider_time_a_ms
        final_provider_b = provider_time_b_ms
        final_simulated_latency = None
        final_request_id_a = request_id_a
        final_request_id_b = request_id_b
    else:
        # Mock mode: Measured latencies must remain null
        final_client_latency = None
        final_provider_latency = None
        final_latency_a = None
        final_latency_b = None
        final_provider_a = None
        final_provider_b = None
        final_simulated_latency = simulated_latency_ms
        final_request_id_a = None
        final_request_id_b = None

    # 1. Evaluate Utterance A
    res_a = evaluate_utterance(
        utterance_spec=test_case.utterance_a,
        actual_transcript=transcript_a,
        category=test_case.category,
        language_code=test_case.language_code,
        latency_ms=final_latency_a,
        client_request_elapsed_ms=final_latency_a,
        provider_processing_time_ms=final_provider_a,
        simulated_latency_ms=simulated_latency_ms if not is_live else None,
        request_id=final_request_id_a,
    )

    # 2. Evaluate Utterance B
    res_b = evaluate_utterance(
        utterance_spec=test_case.utterance_b,
        actual_transcript=transcript_b,
        category=test_case.category,
        language_code=test_case.language_code,
        latency_ms=final_latency_b,
        client_request_elapsed_ms=final_latency_b,
        provider_processing_time_ms=final_provider_b,
        simulated_latency_ms=simulated_latency_ms if not is_live else None,
        request_id=final_request_id_b,
    )

    # 3. Check Semantic Distinction Preservation
    dist_field = test_case.distinction_field
    val_a = res_a.extracted_fields.get(dist_field)
    val_b = res_b.extracted_fields.get(dist_field)

    exp_a = test_case.utterance_a.expected_critical_fields.get(dist_field)
    exp_b = test_case.utterance_b.expected_critical_fields.get(dist_field)

    distinction_preserved = (
        val_a is not None and val_b is not None
        and val_a == exp_a and val_b == exp_b
        and val_a != val_b
    )

    # 4. Critical-field accuracy across all expected fields in both utterances
    total_fields = len(test_case.utterance_a.expected_critical_fields) + len(test_case.utterance_b.expected_critical_fields)
    matched_fields = sum(1 for m in res_a.field_matches.values() if m) + sum(1 for m in res_b.field_matches.values() if m)
    critical_field_accuracy = round(float(matched_fields) / float(total_fields), 4) if total_fields > 0 else 1.0

    # 5. Downstream Task Success
    task_success = res_a.task_success and res_b.task_success and distinction_preserved

    # 6. Failure Classification
    failure_category, failure_explanation = classify_failure(
        test_case=test_case,
        result_a=res_a,
        result_b=res_b,
        distinction_preserved=distinction_preserved,
    )

    avg_wer = round((res_a.wer + res_b.wer) / 2.0, 4)
    avg_norm_wer = None
    if res_a.normalized_wer is not None and res_b.normalized_wer is not None:
        avg_norm_wer = round((res_a.normalized_wer + res_b.normalized_wer) / 2.0, 4)

    return PairEvaluationResult(
        run_id=run_id,
        test_case_id=test_case.id,
        category=test_case.category.value if hasattr(test_case.category, "value") else str(test_case.category),
        language_code=test_case.language_code,
        execution_mode=execution_mode,  # type: ignore[arg-type]
        audio_condition=audio_condition,
        timestamp=timestamp,
        utterance_a=res_a,
        utterance_b=res_b,
        wer_a=res_a.wer,
        wer_b=res_b.wer,
        average_wer=avg_wer,
        normalized_wer_a=res_a.normalized_wer,
        normalized_wer_b=res_b.normalized_wer,
        average_normalized_wer=avg_norm_wer,
        critical_field_accuracy=critical_field_accuracy,
        semantic_distinction_preserved=distinction_preserved,
        task_success=task_success,
        primary_failure_category=failure_category,
        failure_explanation=failure_explanation,
        client_measured_latency_ms=final_client_latency,
        client_request_elapsed_ms=final_client_latency,
        provider_processing_time_ms=final_provider_latency,
        simulated_latency_ms=final_simulated_latency,
    )
