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
    "pack": "pack", "पैक": "pack",
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
    "amount": "amount", "अमाउंट": "amount",
    "aur": "aur", "और": "aur",
    "me": "me", "में": "me",
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
    "rahul": "rahul", "राहुल": "rahul",
    "rohit": "rohit", "रोहित": "rohit",
    "sharma": "sharma", "शर्मा": "sharma",
    "return": "return", "रिटर्न": "return",
    "badalna": "badalna", "बदलना": "badalna", "badal": "badalna", "बदल": "badalna",
    "delivery": "delivery", "डिलीवरी": "delivery",
    "subah": "subah", "सुबह": "subah",
    "baje": "baje", "बजे": "baje",
    "naam": "naam", "नाम": "naam",
    "par": "par", "पर": "par",
    "hai": "hai", "है": "hai",
    "mera": "mera", "मेरा": "mera",
    "mujhe": "mujhe", "मुझे": "mujhe",
    "apna": "apna", "अपना": "apna",
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


def _extract_all_numeric_candidates(text: str) -> List[int]:
    """Helper returning all integer quantities mentioned in text in sequence."""
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text)
    cleaned = re.sub(r"[₹$€£]", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,2}:\d{2}\b", " ", cleaned)
    cleaned = re.sub(
        r"(?:सुबह|subah|शाम|shaam|at)?\s*(?:\d{1,2}|[a-zA-Z\u0900-\u097F]+)\s*(?:बजे|baje|am|pm|clock|o'clock)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    norm = normalize_text(cleaned)
    words = norm.split()
    found: List[int] = []
    i = 0
    while i < len(words):
        w = words[i]
        val = None
        if w in NUMBER_WORDS:
            if w in ("do", "दो"):
                if (i > 0 and words[i - 1] in VERBAL_AUXILIARY_PRECEDING_TOKENS) or (
                    i + 1 < len(words)
                    and words[i + 1] in {"not", "n't", "dont", "you", "they", "we", "it", "so", "nothing", "cancel"}
                ):
                    i += 1
                    continue
            val = NUMBER_WORDS[w]
        elif w.isdigit():
            val = int(w)
        if val is not None:
            if i + 1 < len(words) and words[i + 1] in SCALE_MULTIPLIERS:
                val = val * SCALE_MULTIPLIERS[words[i + 1]]
                i += 1
            found.append(val)
        i += 1
    return found


def extract_quantities(text: str) -> Dict[str, Any]:
    """Extracts numeric quantities, scales, and currency amounts deterministically."""
    # Mask clock time expressions (e.g. "09:00", "सुबह 9 बजे", "10:00 am") so hours are never extracted as order quantities
    cleaned_for_qty = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    cleaned_for_qty = re.sub(
        r"(?:सुबह|subah|शाम|shaam|at)?\s*(?:\d{1,2}|[a-zA-Z\u0900-\u097F]+)\s*(?:बजे|baje|am|pm|clock|o'clock)\b",
        " ",
        cleaned_for_qty,
        flags=re.IGNORECASE,
    )
    norm = normalize_text(cleaned_for_qty)
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
            # Context-aware check for polysemous Hindi 'do' / 'दो':
            # Preceded by verbal root (kar do, bhej do) -> imperative auxiliary verb
            if word in ("do", "दो") and i > 0 and words[i - 1] in VERBAL_AUXILIARY_PRECEDING_TOKENS:
                continue
            # Followed by negative or English auxiliary pronoun (do not, do you, don't) -> English verb
            if word in ("do", "दो") and i + 1 < len(words) and words[i + 1] in {"not", "n't", "dont", "you", "they", "we", "it", "so", "nothing", "cancel"}:
                continue
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
    norm = unicodedata.normalize("NFC", text.lower())
    norm_clean = re.sub(r"[\.,!?;:\"'—\(\)\[\]\{\}«»“”‘’।॥\-]", " ", norm)
    words = norm_clean.split()

    neg_tokens = {
        "nahi", "nahin", "नहीं", "mat", "मत", "na", "ना", "न", "ruko", "रुको",
        "no", "not", "dont", "don't", "never", "stop", "stopp", "neither", "nor", "none",
    }
    is_negated = any(w in neg_tokens for w in words)
    if not is_negated:
        is_negated = bool(
            re.search(
                r"\b(don'?t|can'?t|won'?t|didn'?t|doesn'?t|isn'?t|aren'?t|wasn'?t|weren'?t|never|not|no|nahi|nahin|नहीं|मत|mat|na|ना|रुको|ruko)\b",
                norm,
                re.IGNORECASE,
            )
        )

    # Detect specific imperative combinations
    action = "proceed"
    has_cancel = bool(re.search(r"\b(cancel|कैंसल|कैंसिल|रद्द)\b", norm))
    has_order = bool(re.search(r"\b(order|ऑर्डर)\b", norm))

    if has_cancel:
        if is_negated:
            action = "keep_order" if has_order else "keep_booking"
        else:
            action = "cancel_order" if has_order else "cancel_booking"
    elif bool(re.search(r"\b(confirm|checkout|कन्फर्म|पुष्टि)\b", norm)):
        action = "hold_cart" if is_negated else "confirm_order"
    elif bool(re.search(r"\b(transfer|ट्रांसफर|bhejo|भेजो)\b", norm)):
        action = "block_transfer" if is_negated else "execute_transfer"
    elif bool(re.search(r"\b(invoice|ईमेल|email)\b", norm)):
        action = "suppress_email" if is_negated else "email_invoice"
    elif bool(re.search(r"\b(renew|रिन्यू|subscription)\b", norm)):
        action = "do_not_renew_subscription" if is_negated else "renew_subscription"
    elif bool(re.search(r"\b(खाना|food|dish)\b", norm)):
        action = "reject_dish" if is_negated else "order_dish"
    else:
        action = "denied" if is_negated else "confirmed"

    return {
        "is_negated": is_negated,
        "negation_detected": is_negated,
        "action": action,
    }


def extract_corrections(text: str) -> Dict[str, Any]:
    """Extracts change-of-mind corrections, identifying initial vs final intended target."""
    cleaned_num_text = re.sub(r"(?<=\d),(?=\d)", "", text)
    cleaned_num_text = re.sub(r"[₹$€£]", " ", cleaned_num_text)
    norm = normalize_text(cleaned_num_text)

    # 1. Check for item / color / destination retractions (e.g. "लाल नहीं नीला", "पार्सल दिल्ली नहीं मुंबई")
    known_colors = {
        "लाल", "नीला", "नीली", "पीला", "पीली", "हरा", "हरी", "काला", "काली", "सफ़ेद", "गुलाबी", "बैंगनी", "नारंगी", "आसमानी", "भूरा", "भूरी",
        "red", "blue", "green", "black", "white", "yellow", "pink", "purple", "orange", "brown",
    }
    known_cities = {
        "दिल्ली", "मुंबई", "जयपुर", "उदयपुर", "बैंगलोर", "बेंगलुरु", "हैदराबाद", "चेन्नई", "कोलकाता", "पुणे", "अहमदाबाद",
        "delhi", "mumbai", "jaipur", "udaipur", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata", "pune",
    }
    m_item = re.search(r"([A-Za-z\u0900-\u097F]+)\s+(?:नहीं|nahi|not)\s*,?\s*([A-Za-z\u0900-\u097F]+)", norm)
    if m_item:
        w_init = m_item.group(1).strip()
        w_fin = m_item.group(2).strip()
        if (w_init in known_colors and w_fin in known_colors) or (w_init in known_cities and w_fin in known_cities):
            is_dest = w_init in known_cities or w_fin in known_cities
            return {
                "correction_detected": True,
                "initial_value": w_init,
                "final_target": w_fin,
                "quantity": None,
                "destination": w_fin if is_dest else None,
                "final_destination": w_fin if is_dest else None,
                "initial_destination": w_init if is_dest else None,
                "initial_item": None if is_dest else w_init,
                "final_item": None if is_dest else w_fin,
                "color": None if is_dest else w_fin,
            }

    # 2. Explicit multi-token retraction cues
    explicit_cues = [
        "no wait to", "no wait", "wait make that", "make that", "wait make it", "wait sorry",
        "nahi ruko", "nahi pehle", "nahi nahi", "नहीं नहीं", "नहीं रुको", "actually", "change to", "make it", "sorry",
    ]

    for cue in explicit_cues:
        pattern = rf"(?:^|\s){re.escape(cue)}(?:\s|$)"
        match = re.search(pattern, norm)
        if match:
            before_text = norm[:match.start()].strip()
            after_text = norm[match.end():].strip()

            after_candidates = _extract_all_numeric_candidates(after_text)
            before_candidates = _extract_all_numeric_candidates(before_text)

            dest_pattern = r"\b(?:flight to|courier to|jaana hai|it to|to)\s+([a-zA-Z]+)"
            before_dest = re.findall(dest_pattern, before_text)
            after_dest = re.findall(r"\b(?:flight to|courier to|it to|se|to)\s+([a-zA-Z]+)", after_text)
            if not after_dest and before_dest and cue.endswith("to"):
                m_direct = re.match(r"^([a-zA-Z]+)", after_text)
                if m_direct:
                    after_dest = [m_direct.group(1)]

            final_destination = after_dest[-1].lower() if after_dest else None
            initial_destination = before_dest[-1].lower() if before_dest else None

            # Time candidates in before/after text
            before_times = re.findall(r"\b(\d{1,2})\s*(?:am|pm)?\b", before_text)
            after_times = re.findall(r"\b(\d{1,2})\s*(?:am|pm)?\b", after_text)
            init_time = f"{int(before_times[-1]):02d}:00" if before_times else None
            fin_time = f"{int(after_times[-1]):02d}:00" if after_times else None

            final_val = after_candidates[-1] if after_candidates else (final_destination or fin_time)
            initial_val = (
                before_candidates[-1]
                if before_candidates
                else (after_candidates[0] if len(after_candidates) > 1 else (initial_destination or init_time))
            )

            return {
                "correction_detected": True,
                "initial_value": initial_val,
                "final_target": final_val,
                "quantity": after_candidates[-1] if after_candidates else None,
                "destination": final_destination,
                "final_destination": final_destination,
                "initial_destination": initial_destination,
                "initial_time": init_time,
                "final_time": fin_time,
                "time": fin_time,
            }

    # 3. Numeric retractions: e.g. "two packets, not two—three packets" or "दो पैकेट नहीं दो, तीन पैकेट"
    cand_list = _extract_all_numeric_candidates(cleaned_num_text)
    if any(k in norm.split() for k in ("not", "nahi", "nahin", "नहीं", "no", "ना")):
        if len(cand_list) >= 2 and cand_list[0] != cand_list[-1]:
            return {
                "correction_detected": True,
                "initial_value": cand_list[0],
                "final_target": cand_list[-1],
                "quantity": cand_list[-1],
                "destination": None,
                "final_destination": None,
            }

    # 4. If no correction cue is found, direct uncorrected command
    direct_qty = extract_quantities(cleaned_num_text)
    destinations = re.findall(r"(?:to|flight to|courier to)\s+([a-zA-Z]+)", norm)
    final_dest = destinations[-1].lower() if destinations else None

    # Check for direct color/item
    direct_item = None
    for item in ("लाल", "नीला", "पीला", "हरा", "काला", "सफ़ेद", "red", "blue", "green", "black", "white"):
        if item in norm:
            direct_item = item
            break

    # Check for direct time
    direct_times = re.findall(r"\b(\d{1,2})\s*(?:am|pm)?\b", norm)
    direct_time = f"{int(direct_times[0]):02d}:00" if (direct_times and any(w in norm for w in ("am", "pm", "meeting", "बजे", "baje"))) else None

    direct_val = (
        direct_qty.get("quantity")
        if direct_qty.get("quantity") is not None
        else (final_dest or direct_item or direct_time)
    )

    return {
        "correction_detected": False,
        "initial_value": direct_val,
        "final_target": direct_val,
        "quantity": direct_qty.get("quantity"),
        "destination": final_dest,
        "final_destination": final_dest,
        "initial_item": direct_item,
        "final_item": direct_item,
        "color": direct_item,
        "initial_time": direct_time,
        "final_time": direct_time,
        "time": direct_time,
    }


def extract_entities(text: str) -> Dict[str, Any]:
    """Extracts named entities, person names, and products deterministically across English & Devanagari."""
    norm = unicodedata.normalize("NFC", text.strip())
    results: Dict[str, Any] = {}

    # 1. Product names
    if "prisma" in norm.lower() or "प्रिज्मा" in norm:
        results["product_name"] = "Prisma STT"
    elif "krishna" in norm.lower() or "कृष्णा" in norm:
        results["product_name"] = "Krishna STT"
    elif "timbre" in norm.lower():
        results["product_name"] = "Timbre"
    elif "timber" in norm.lower():
        results["product_name"] = "Timber"

    # 2. Cities
    if "mumbai" in norm.lower() or "मुंबई" in norm:
        results["destination"] = "Mumbai"
    elif "delhi" in norm.lower() or "दिल्ली" in norm:
        results["destination"] = "Delhi"

    # 3. Explicit Anchored Person Names (Rahul Sharma vs Rohit Sharma)
    if any(p in norm.lower() for p in ("rahul sharma", "राहुल शर्मा")):
        results["person_name"] = "Rahul Sharma"
        results["name"] = "Rahul Sharma"
        results["recipient"] = "Rahul Sharma"
    elif any(p in norm.lower() for p in ("rohit sharma", "रोहित शर्मा")):
        results["person_name"] = "Rohit Sharma"
        results["name"] = "Rohit Sharma"
        results["recipient"] = "Rohit Sharma"

    # 4. English Name Patterns: "payment to <Name>", "transfer money to <Name>", "assign ticket to <Name>", "call <Name>"
    pronoun_exclusions = {
        "delhi", "mumbai", "my", "me", "the", "a", "an", "this", "that", "these", "those",
        "you", "everyone", "someone", "anyone", "nobody", "all", "us", "them", "him", "her",
        "it", "here", "there", "work", "home", "office", "school", "hospital", "bank",
        "today", "tomorrow", "morning", "evening", "night", "now",
    }
    m_en = re.search(
        r"\b(?:assign ticket to|deliver (?:the )?package to|payment to|money to|transfer (?:money )?to|send (?:payment |money )?to|call)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b",
        norm,
        re.IGNORECASE,
    )
    if m_en and "name" not in results:
        raw_name = m_en.group(1).title()
        if raw_name.lower() not in pronoun_exclusions:
            results["name"] = raw_name
            results["recipient"] = raw_name
            results["contact"] = raw_name
            results["assignee"] = raw_name
            results["person_name"] = raw_name

    # 5. Hindi Devanagari Name Patterns:
    m_hi_ko = re.search(r"([^\s]+)\s+को(?:\s+|$)", norm)
    if m_hi_ko and "name" not in results:
        raw_name = m_hi_ko.group(1).strip()
        if raw_name not in {"मुझ", "मुझे", "हम", "हमें", "सब", "कस्टमर", "शाम"}:
            results["name"] = raw_name
            results["recipient"] = raw_name
            results["contact"] = raw_name
            results["assignee"] = raw_name
            results["person_name"] = raw_name

    m_hi_khata = re.search(r"([^\s]+)\s+के\s+खाते\s+में(?:\s+|$)", norm)
    if m_hi_khata and "name" not in results:
        raw_name = m_hi_khata.group(1).strip()
        results["name"] = raw_name
        results["recipient"] = raw_name
        results["contact"] = raw_name
        results["assignee"] = raw_name

    m_hi_no = re.search(r"([^\s]+)\s+का\s+नंबर(?:\s+|$)", norm)
    if m_hi_no and "name" not in results:
        raw_name = m_hi_no.group(1).strip()
        results["name"] = raw_name
        results["recipient"] = raw_name
        results["contact"] = raw_name

    # Single-name fallbacks
    if "name" not in results:
        if "rahul" in norm.lower() or "राहुल" in norm:
            results["name"] = "Rahul"
            results["recipient"] = "Rahul"
            results["person_name"] = "Rahul Sharma"
        elif "rohit" in norm.lower() or "रोहित" in norm:
            results["name"] = "Rohit"
            results["recipient"] = "Rohit"
            results["person_name"] = "Rohit Sharma"
        elif "rohan" in norm.lower() or "रोहन" in norm:
            results["name"] = "Rohan"
            results["recipient"] = "Rohan"
            results["person_name"] = "Rohan"

    return results


HINDI_TIME_WORDS: Dict[str, int] = {
    "ek": 1, "एक": 1, "1": 1,
    "do": 2, "दो": 2, "2": 2,
    "teen": 3, "तीन": 3, "3": 3,
    "char": 4, "chaar": 4, "चार": 4, "4": 4,
    "paanch": 5, "panch": 5, "पाँच": 5, "पांच": 5, "5": 5,
    "chhah": 6, "che": 6, "छह": 6, "6": 6,
    "saat": 7, "सात": 7, "7": 7,
    "aath": 8, "ath": 8, "आठ": 8, "8": 8,
    "nau": 9, "नौ": 9, "9": 9,
    "das": 10, "दस": 10, "10": 10,
    "gyarah": 11, "ग्यारह": 11, "11": 11,
    "barah": 12, "बारह": 12, "12": 12,
}

EN_TIME_WORDS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def extract_time(text: str) -> Dict[str, Any]:
    """Extracts scheduled time, hour, and period deterministically across Hindi and English."""
    norm = unicodedata.normalize("NFC", text.lower())
    results: Dict[str, Any] = {}

    has_pm_word = ("दोपहर" in norm and "12" not in norm) or "शाम" in norm or "रात" in norm or bool(re.search(r"\bpm\b", norm))
    has_am_word = "सुबह" in norm or bool(re.search(r"\bam\b", norm))

    # 1. Digital format like "09:00", "19:30", "02:00", "04:00"
    m_digital = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if m_digital:
        h = int(m_digital.group(1))
        m = int(m_digital.group(2))
        h_24 = h
        if has_pm_word and h < 12:
            h_24 = h + 12
        elif has_am_word and h == 12:
            h_24 = 0
        results["hour"] = h_24
        results["minute"] = m
        results["time"] = f"{h_24:02d}:{m:02d}"
        results["delivery_time"] = f"{h:02d}:{m:02d}"
        results["meridiem"] = "PM" if has_pm_word else ("AM" if has_am_word else ("PM" if h_24 >= 12 else "AM"))
        results["period"] = results["meridiem"]
        return results

    # 2. English hour + AM/PM: e.g. "five PM", "8 PM", "six AM", "six PM", "seven PM", "nine PM"
    m_en = re.search(r"\b([a-zA-Z]+|\d{1,2})\s*(am|pm)\b", norm)
    if m_en:
        h_token = m_en.group(1)
        period = m_en.group(2).upper()
        h = int(h_token) if h_token.isdigit() else EN_TIME_WORDS.get(h_token)
        if h is not None:
            h_24 = (h + 12) if (period == "PM" and h < 12) else (0 if (period == "AM" and h == 12) else h)
            results["hour"] = h_24
            results["time"] = f"{h_24:02d}:00"
            results["delivery_time"] = f"{h:02d}:00"
            results["meridiem"] = period
            results["period"] = period
            return results

    # 3. English compound words: e.g. "ten thirty", "twelve thirty"
    m_comp = re.search(
        r"\b(ten|twelve|one|two|three|four|five|six|seven|eight|nine|eleven)\s+(thirty|fifteen|forty five)\b",
        norm,
    )
    if m_comp:
        h_w = m_comp.group(1)
        m_w = m_comp.group(2)
        h = EN_TIME_WORDS.get(h_w, 0)
        m = 30 if m_w == "thirty" else (15 if m_w == "fifteen" else 45)
        results["hour"] = h
        results["minute"] = m
        results["time"] = f"{h:02d}:{m:02d}"
        results["delivery_time"] = f"{h:02d}:{m:02d}"
        return results

    # 4. Hindi half-past: "साढ़े सात बजे", "साढ़े आठ बजे"
    m_half = re.search(r"(?:साढ़े|साढ़े)\s+([^\s]+)\s*बजे", norm)
    if m_half:
        tok = m_half.group(1)
        h = int(tok) if tok.isdigit() else HINDI_TIME_WORDS.get(tok)
        if h is not None:
            h_24 = (h + 12) if (has_pm_word and h < 12) else h
            results["hour"] = h_24
            results["minute"] = 30
            results["time"] = f"{h_24:02d}:30"
            results["delivery_time"] = f"{h:02d}:30"
            results["meridiem"] = "PM" if has_pm_word else "AM"
            results["period"] = results["meridiem"]
            return results

    # 5. Hindi hour + बजे: "नौ बजे", "ग्यारह बजे", "दो बजे", "चार बजे", "दस बजे"
    m_baje = re.search(r"([^\s]+)\s*बजे", norm)
    if m_baje:
        tok = m_baje.group(1)
        h = int(tok) if tok.isdigit() else HINDI_TIME_WORDS.get(tok)
        if h is not None:
            h_24 = (h + 12) if (has_pm_word and h < 12) else h
            results["hour"] = h_24
            results["time"] = f"{h_24:02d}:00"
            results["delivery_time"] = f"{h:02d}:00"
            results["meridiem"] = "PM" if has_pm_word else ("AM" if has_am_word else None)
            results["period"] = results["meridiem"]
            return results

    # 6. Direct hour match ONLY if explicit time cue ("सुबह", "subah", "शाम", "shaam") is present
    if "सुबह" in norm or "subah" in norm or "शाम" in norm or "shaam" in norm:
        words = norm.split()
        for w in words:
            if w in HINDI_TIME_WORDS:
                h = HINDI_TIME_WORDS[w]
                h_24 = (h + 12) if (has_pm_word and h < 12) else h
                results["hour"] = h_24
                results["time"] = f"{h_24:02d}:00"
                results["delivery_time"] = f"{h:02d}:00"
                results["period"] = "AM" if has_am_word else "PM"
                results["meridiem"] = results["period"]
                return results

    return results


def normalize_intent_alias(intent: Optional[str]) -> Optional[str]:
    """Normalizes equivalent intent identifiers to canonical benchmark representations."""
    if not intent:
        return intent
    aliases = {
        "RETURN_ORDER": "RETURN_PRODUCT",
        "RETURN_PRODUCT": "RETURN_PRODUCT",
        "CHANGE_ADDRESS": "MODIFY_ADDRESS",
        "MODIFY_ADDRESS": "MODIFY_ADDRESS",
        "REFUND_PAYMENT": "REFUND_PAYMENT",
        "CHECK_REFUND": "CHECK_REFUND",
        "BOOK_CAB": "BOOK_CAB",
        "CANCEL_CAB": "CANCEL_CAB",
        "CANCEL_MANDATE": "CANCEL_MANDATE",
        "REPEAT_TRANSACTION": "REPEAT_TRANSACTION",
        "TRACK_PACKAGE": "TRACK_PACKAGE",
        "EXCHANGE_ITEM": "EXCHANGE_ITEM",
        "ESCALATE_HUMAN_AGENT": "ESCALATE_HUMAN_AGENT",
        "FILE_COMPLAINT": "FILE_COMPLAINT",
        "CANCEL_ORDER": "CANCEL_ORDER",
        "MODIFY_ORDER": "MODIFY_ORDER",
    }
    return aliases.get(intent.strip(), intent.strip())


def extract_command_intent(text: str) -> str:
    """Classifies spoken utterance command intent supporting Latin, Romanized Hindi, and Devanagari."""
    norm = unicodedata.normalize("NFC", text.lower())

    # Negation tokens for intent checks
    negation_set = {"mat", "nahi", "nahin", "no", "not", "मत", "नहीं", "ना"}
    is_neg = any(n in norm.split() for n in negation_set)

    # 1. Cab booking vs cancellation
    if any(k in norm for k in ("गाड़ी", "गाड़ी", "cab", "taxi", "ride", "टैक्सी")):
        if any(c in norm for c in ("cancel", "कैंसिल", "कैंसल", "रद्द")):
            return "CANCEL_CAB"
        if any(b in norm for b in ("book", "बुक", "dispatch")):
            return "BOOK_CAB"

    # 2. Address modification (MUST precede delivery inquiries and generic modify)
    if any(a in norm for a in ("address", "एड्रेस", "पता")) and any(c in norm for c in ("change", "modify", "badal", "बदल", "update", "अपडेट")):
        return "MODIFY_ADDRESS"

    # 3. Order / Package Tracking & Delivery Inquiry
    if any(t in norm for t in ("track", "tracking", "ट्रैक")):
        return "TRACK_PACKAGE"
    if "when will" in norm and any(p in norm for p in ("package", "order", "delivery", "arrive")):
        return "DELIVERY_INQUIRY"

    # 4. Refund status vs refund request
    if any(r in norm for r in ("refund", "रिफंड")):
        if any(s in norm for s in ("status", "स्टेटस", "बताओ", "check")):
            return "CHECK_REFUND"
        return "REFUND_PAYMENT"

    # 5. Exchange item
    if any(e in norm for e in ("exchange", "एक्सचेंज")):
        return "EXCHANGE_ITEM"

    # 6. Escalation / Customer Support
    if any(c in norm for c in ("customer care", "कस्टमर केयर", "human", "बात करवाओ", "agent", "support")):
        return "ESCALATE_HUMAN_AGENT"

    # 7. Complaint
    if any(c in norm for c in ("complaint", "शिकायत", "complain")):
        return "FILE_COMPLAINT"

    # 8. Payment Repetition vs Cancel Mandate
    if any(m in norm for m in ("recurring", "mandate", "ऑटोपे", "ऑटो पे")) or (any(s in norm for s in ("stop", "रोक", "कैंसिल", "cancel")) and any(p in norm for p in ("payment", "deduction", "पेमेंट"))):
        return "CANCEL_MANDATE"
    if any(r in norm for r in ("repeat", "again", "फिर से", "दोबारा")) and any(p in norm for p in ("payment", "पेमेंट", "transaction", "ट्रांजेक्शन", "रुपये")):
        return "REPEAT_TRANSACTION"

    # 9. Request funds
    if any(k in norm for k in ("request", "maang", "मांग", "माँग")):
        return "REQUEST_FUNDS"

    # 10. Post-order return / modify intents (must precede generic order intent)
    if any(r in norm for r in ("return", "रिटर्न", "वापस")):
        return "RETURN_ORDER"
    if any(m in norm for m in ("modify", "मॉडिफाई", "badal", "बदल", "change", "edit")):
        return "MODIFY_ORDER"

    # 11. Cancellation of order / package / booking
    if any(c in norm for c in ("cancel", "कैंसल", "कैंसिल", "रद्द")):
        has_pkg = any(p in norm for p in ("package", "delivery", "डिलीवरी", "order", "ऑर्डर"))
        if is_neg:
            return "KEEP_ORDER" if has_pkg else "KEEP_BOOKING"
        return "CANCEL_ORDER" if has_pkg else "CANCEL_BOOKING"

    # 12. Delivery scheduling / inquiry
    if "डिलीवरी" in norm or "delivery" in norm:
        if any(t in norm for t in ("बजे", "baje", "time", "सुबह", "subah", "शाम", "shaam")):
            return "SCHEDULE_DELIVERY"
        return "DELIVERY_INQUIRY"

    # 13. Banking / Balance
    if "balance" in norm or "बैलेंस" in norm:
        return "CHECK_BALANCE"
    if any(k in norm for k in ("band", "बंद", "close", "terminate")):
        return "CLOSE_ACCOUNT"

    # 14. Generic Payments
    if any(k in norm for k in ("pay", "bhejo", "भेजो", "transfer", "ट्रांसफर")):
        return "TRANSFER_FUNDS"

    # 15. Cart and Confirmation
    if any(k in norm for k in ("empty", "khali", "खाली")):
        return "EMPTY_CART"
    if any(c in norm for c in ("confirm", "checkout", "कन्फर्म", "पुष्टि")):
        return "CONFIRM_ORDER"

    # 16. General ordering
    if any(k in norm for k in ("order", "ऑर्डर", "chahiye", "चाहिए", "plate", "प्लेट")):
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
    elif category == TestCategory.TIME_EXTRACTION:
        fields.update(extract_time(transcript))
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
    if utterance_spec.expected_intent is not None:
        intent_match = (
            normalize_intent_alias(detected_intent) == normalize_intent_alias(utterance_spec.expected_intent)
        )
    else:
        intent_match = True

    # Evaluate critical fields match
    field_matches: Dict[str, bool] = {}
    expected_fields = utterance_spec.expected_critical_fields

    for field_name, expected_val in expected_fields.items():
        actual_val = extracted_fields.get(field_name)
        if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
            field_matches[field_name] = (round(float(expected_val), 2) == round(float(actual_val), 2))
        elif isinstance(expected_val, str) and isinstance(actual_val, str):
            if field_name == "intent":
                field_matches[field_name] = (
                    normalize_intent_alias(expected_val) == normalize_intent_alias(actual_val)
                )
            else:
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

    # 6. Time Extraction Error
    if category == TestCategory.TIME_EXTRACTION or dist_field in ("delivery_time", "time", "hour"):
        if not distinction_preserved or not (result_a.critical_fields_match and result_b.critical_fields_match):
            return (
                FailureCategory.TIME_ERROR,
                f"Time distinction collapsed (Expected: {exp_a} vs {exp_b}, extracted: {val_a} vs {val_b}). "
                "Downstream scheduling or delivery agent would book or route at the wrong time.",
            )

    # 7. Intent Error
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

    def _critical_values_match(actual: Any, expected: Any) -> bool:
        if actual is None or expected is None:
            return actual is expected
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return round(float(actual), 2) == round(float(expected), 2)
        if isinstance(actual, str) and isinstance(expected, str):
            if dist_field == "intent":
                return normalize_intent_alias(actual) == normalize_intent_alias(expected)
            return actual.strip().lower() == expected.strip().lower()
        return actual == expected

    distinction_preserved = (
        val_a is not None and val_b is not None
        and _critical_values_match(val_a, exp_a)
        and _critical_values_match(val_b, exp_b)
        and not _critical_values_match(val_a, val_b)
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
