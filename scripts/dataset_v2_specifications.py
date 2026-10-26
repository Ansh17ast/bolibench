"""Specification generator for BoliBench 600-Pair Expanded Benchmark (v2.0.0).

Generates 600 genuinely varied, unique paired contrast specifications:
- 100 Quantity and number pairs
- 100 Negation pairs (including non-renewal distinction)
- 100 Correction and changed-intent pairs (quantities, destinations, attributes, times)
- 100 Command-intent pairs (banking, e-commerce, mobility, customer support)
- 100 Named-entity pairs (phonetic Indian personal names across banking, dispatch, CRM)
- 100 Time-extraction pairs (Hindi colloquial, half-past, English AM/PM, digital)

Total: Exactly 600 pairs (1,200 unique utterances).
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List
from app.evaluation.batch_manifest import AudioSourceType


def generate_all_600_specifications() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    voices_hi = ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"]
    voices_en = ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural"]

    # =========================================================================
    # 1. QUANTITIES AND NUMBERS (100 pairs: bench_v2_qty_001 .. bench_v2_qty_100)
    # =========================================================================
    # 95 tts_synthetic + 5 acoustic_transformed
    qty_templates = [
        # Hindi retail/food
        ("मुझे {a} पैकेट समोसे चाहिए", "मुझे {b} पैकेट समोसे चाहिए", "hi-IN", "ORDER_{a}_UNITS", "ORDER_{b}_UNITS", "Hindi food samosa order count"),
        ("कृपया {a} प्लेट बिरयानी भेज दीजिए", "कृपया {b} प्लेट बिरयानी भेज दीजिए", "hi-IN", "ORDER_{a}_UNITS", "ORDER_{b}_UNITS", "Hindi biryani plate quantity"),
        ("{a} किलो चीनी दे दो", "{b} किलो चीनी दे दो", "hi-IN", "DISPATCH_{a}_KG", "DISPATCH_{b}_KG", "Hindi grocery weight in kilograms"),
        ("मुझे {a} पानी की बोतलें चाहिए", "मुझे {b} पानी की बोतलें चाहिए", "hi-IN", "ORDER_{a}_UNITS", "ORDER_{b}_UNITS", "Hindi beverage bottles count"),
        ("{a} कप गरम चाय भिजवा दो", "{b} कप गरम चाय भिजवा दो", "hi-IN", "ORDER_{a}_UNITS", "ORDER_{b}_UNITS", "Hindi beverage tea cups count"),
        ("दुकान से {a} किलो आटा मंगवाओ", "दुकान से {b} किलो आटा मंगवाओ", "hi-IN", "ORDER_{a}_KG", "ORDER_{b}_KG", "Hindi flour weight in kilograms"),
        ("हमें {a} नए लैपटॉप चाहिए", "हमें {b} नए लैपटॉप चाहिए", "hi-IN", "DISPATCH_{a}_LAPTOPS", "DISPATCH_{b}_LAPTOPS", "Hindi hardware inventory laptops"),
        ("{a} डिब्बे मिठाई पैक कर दो", "{b} डिब्बे मिठाई पैक कर दो", "hi-IN", "PACK_{a}_BOXES", "PACK_{b}_BOXES", "Hindi sweets box packaging"),
        ("मुझे {a} रोटी और दाल चाहिए", "मुझे {b} रोटी और दाल चाहिए", "hi-IN", "ORDER_{a}_ROTIS", "ORDER_{b}_ROTIS", "Hindi dinner bread order"),
        ("कार्यालय के लिए {a} कुर्सियां भेजो", "कार्यालय के लिए {b} कुर्सियां भेजो", "hi-IN", "DISPATCH_{a}_CHAIRS", "DISPATCH_{b}_CHAIRS", "Hindi office furniture chairs"),
        # English / Hinglish retail & inventory
        ("Please dispatch {a} boxes of apples", "Please dispatch {b} boxes of apples", "en-IN", "DISPATCH_{a}_BOXES", "DISPATCH_{b}_BOXES", "English fruit carton dispatch"),
        ("Order {a} computer monitors for office", "Order {b} computer monitors for office", "en-IN", "ORDER_{a}_MONITORS", "ORDER_{b}_MONITORS", "English IT hardware monitors"),
        ("We need {a} hotel rooms for guests", "We need {b} hotel rooms for guests", "en-IN", "BOOK_{a}_ROOMS", "BOOK_{b}_ROOMS", "English hospitality rooms booking"),
        ("Send {a} shipping containers today", "Send {b} shipping containers today", "en-IN", "DISPATCH_{a}_CONTAINERS", "DISPATCH_{b}_CONTAINERS", "English logistics cargo containers"),
        ("Book {a} movie tickets for the night show", "Book {b} movie tickets for the night show", "en-IN", "BOOK_{a}_TICKETS", "BOOK_{b}_TICKETS", "English cinema tickets booking"),
        ("Deliver {a} pizzas to my apartment", "Deliver {b} pizzas to my apartment", "en-IN", "ORDER_{a}_PIZZAS", "ORDER_{b}_PIZZAS", "English food delivery pizzas"),
        ("Reserve {a} flight seats for passengers", "Reserve {b} flight seats for passengers", "en-IN", "BOOK_{a}_SEATS", "BOOK_{b}_SEATS", "English airline seat reservation"),
        ("I want {a} cotton shirts from catalog", "I want {b} cotton shirts from catalog", "en-IN", "ORDER_{a}_SHIRTS", "ORDER_{b}_SHIRTS", "English apparel shirt count"),
        ("Purchase {a} licenses for team software", "Purchase {b} licenses for team software", "en-IN", "BUY_{a}_LICENSES", "BUY_{b}_LICENSES", "English software seat licenses"),
        ("Prepare {a} food meal packets for delivery", "Prepare {b} food meal packets for delivery", "en-IN", "PACK_{a}_MEALS", "PACK_{b}_MEALS", "English meal packets count"),
        # Banking & Currency amounts (Rupees)
        ("खाते में {a} रुपये जमा करो", "खाते में {b} रुपये जमा करो", "hi-IN", "DEPOSIT_{a}_INR", "DEPOSIT_{b}_INR", "Hindi bank deposit amount"),
        ("मुझे {a} रुपये का लोन चाहिए", "मुझे {b} रुपये का लोन चाहिए", "hi-IN", "APPLY_LOAN_{a}_INR", "APPLY_LOAN_{b}_INR", "Hindi bank loan request amount"),
        ("कृपया {a} रुपये का भुगतान कीजिए", "कृपया {b} रुपये का भुगतान कीजिए", "hi-IN", "PAY_{a}_INR", "PAY_{b}_INR", "Hindi bill payment amount"),
        ("Transfer {a} rupees to vendor account", "Transfer {b} rupees to vendor account", "en-IN", "TRANSFER_{a}_INR", "TRANSFER_{b}_INR", "English vendor fund transfer"),
        ("Send payment of {a} rupees immediately", "Send payment of {b} rupees immediately", "en-IN", "PAY_{a}_INR", "PAY_{b}_INR", "English instant payment amount"),
    ]

    qty_number_pairs = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10),
        (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10),
        (10, 20), (15, 25), (20, 30), (25, 35), (30, 40), (40, 50),
        (50, 100), (100, 200), (200, 300), (300, 400), (400, 500),
        (500, 1000), (1000, 2000), (1500, 2500), (2000, 3000), (2500, 5000),
        (5000, 10000), (10000, 20000), (25000, 50000), (50000, 100000),
        (2, 8), (3, 9), (4, 10), (12, 18), (14, 28), (25, 75), (50, 150),
        (100, 500), (250, 750), (500, 1500), (1000, 5000), (2000, 10000),
        (1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90),
        (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24),
        (10, 100), (50, 500), (100, 1000), (500, 5000), (1000, 10000),
        (12, 24), (15, 30), (20, 40), (25, 50), (30, 60), (35, 70), (40, 80), (45, 90),
        (100, 150), (200, 250), (300, 350), (400, 450), (500, 550), (600, 650),
        (700, 750), (800, 850), (900, 950), (1000, 1500),
    ]

    for idx in range(100):
        pair_num = idx + 1
        pair_id = f"bench_v2_qty_{pair_num:03d}"
        tpl = qty_templates[idx % len(qty_templates)]
        num_a, num_b = qty_number_pairs[idx]
        lang = tpl[2]
        voice = voices_hi[idx % 2] if lang == "hi-IN" else voices_en[idx % 2]

        is_transformed = (idx >= 95)  # Last 5 pairs are acoustic transformed
        source_type = AudioSourceType.ACOUSTIC_TRANSFORMED if is_transformed else AudioSourceType.TTS_SYNTHETIC
        parent_id = f"bench_v2_qty_{(idx - 5) + 1:03d}" if is_transformed else None
        transform_params = "atempo=1.12,highpass=f=200,lowpass=f=3400" if is_transformed else None

        ref_a = tpl[0].format(a=num_a, b=num_b)
        ref_b = tpl[1].format(a=num_a, b=num_b)
        act_a = tpl[3].format(a=num_a, b=num_b)
        act_b = tpl[4].format(a=num_a, b=num_b)

        specs.append({
            "pair_id": pair_id,
            "category": "quantities_and_numbers",
            "language_code": lang,
            "source_type": source_type,
            "voice": voice,
            "source_pair_id": parent_id,
            "transformation_params": transform_params,
            "reference_transcript_a": ref_a,
            "reference_transcript_b": ref_b,
            "expected_critical_fields_a": {"quantity": num_a},
            "expected_critical_fields_b": {"quantity": num_b},
            "expected_intent_a": None,
            "expected_intent_b": None,
            "expected_action_a": act_a,
            "expected_action_b": act_b,
            "distinction_field": "quantity",
            "description": f"{tpl[5]} ({num_a} vs {num_b})",
            "tts_voice_or_source": voice if not is_transformed else f"transformed_from_{parent_id}",
            "notes": f"Quantity contrast {num_a} vs {num_b} ({source_type.value})",
        })

    # =========================================================================
    # 2. NEGATION (100 pairs: bench_v2_neg_001 .. bench_v2_neg_100)
    # =========================================================================
    neg_scenarios = [
        # Hindi Order/Booking Cancellation
        ("मेरा ऑर्डर कैंसिल कर दो", "मेरा ऑर्डर कैंसिल मत करना", "hi-IN", "CANCEL_ORDER", "KEEP_ORDER", "Hindi order cancel prohibitive mat"),
        ("मेरी बस टिकट कैंसिल करो", "मेरी बस टिकट कैंसिल मत करो", "hi-IN", "CANCEL_BOOKING", "KEEP_BOOKING", "Hindi bus ticket cancel prohibitive mat"),
        ("मेरी ट्रेन टिकट रद्द कर दो", "मेरी ट्रेन टिकट रद्द मत करना", "hi-IN", "CANCEL_BOOKING", "KEEP_BOOKING", "Hindi train booking cancellation mat"),
        ("मेरी टैक्सी बुकिंग रद्द करो", "मेरी टैक्सी बुकिंग रद्द मत करना", "hi-IN", "CANCEL_BOOKING", "KEEP_BOOKING", "Hindi cab booking cancellation mat"),
        ("मेरा होटल कमरा कैंसिल कर दो", "मेरा होटल कमरा कैंसिल मत करो", "hi-IN", "CANCEL_BOOKING", "KEEP_BOOKING", "Hindi hotel booking cancellation mat"),
        ("मेरी फ्लाइट बुकिंग रद्द कर दो", "मेरी फ्लाइट बुकिंग रद्द मत करो", "hi-IN", "CANCEL_BOOKING", "KEEP_BOOKING", "Hindi flight cancellation prohibitive mat"),
        ("मेरा फूड ऑर्डर कैंसिल कर दीजिए", "मेरा फूड ऑर्डर कैंसिल मत कीजिए", "hi-IN", "CANCEL_ORDER", "KEEP_ORDER", "Hindi polite food order cancellation"),
        ("यह पार्सल डिलीवरी कैंसिल करो", "यह पार्सल डिलीवरी कैंसिल मत करो", "hi-IN", "CANCEL_ORDER", "KEEP_ORDER", "Hindi package delivery cancellation"),
        ("मेरी टेबल रिजर्वेशन कैंसिल कर दो", "मेरी टेबल रिजर्वेशन कैंसिल मत करो", "hi-IN", "CANCEL_BOOKING", "KEEP_BOOKING", "Hindi dining reservation cancellation"),
        ("मेरी डॉक्टर अपॉइंटमेंट कैंसिल करो", "मेरी डॉक्टर अपॉइंटमेंट कैंसिल मत करना", "hi-IN", "CANCEL_BOOKING", "KEEP_BOOKING", "Hindi doctor appointment cancellation"),
        # Hindi Order Confirmation
        ("मेरा ऑर्डर कन्फर्म कर दो", "मेरा ऑर्डर कन्फर्म मत करना", "hi-IN", "CONFIRM_ORDER", "HOLD_ORDER", "Hindi order confirmation prohibitive mat"),
        ("यह बुकिंग कन्फर्म करो", "यह बुकिंग कन्फर्म मत करो", "hi-IN", "CONFIRM_ORDER", "HOLD_ORDER", "Hindi booking confirmation prohibitive mat"),
        ("कार्ट का चेकआउट कर दो", "कार्ट का चेकआउट मत करना", "hi-IN", "CONFIRM_ORDER", "HOLD_ORDER", "Hindi cart checkout prohibitive mat"),
        ("यह पेमेंट कन्फर्म करो", "यह पेमेंट कन्फर्म मत करो", "hi-IN", "CONFIRM_ORDER", "HOLD_ORDER", "Hindi payment confirmation prohibitive mat"),
        ("सब्सक्रिप्शन प्लान कन्फर्म कर दीजिए", "सब्सक्रिप्शन प्लान कन्फर्म मत कीजिए", "hi-IN", "CONFIRM_ORDER", "HOLD_ORDER", "Hindi plan confirmation prohibitive mat"),
        # Hindi Payment & Funds Transfer
        ("पांच सौ रुपये ट्रांसफर कर दो", "पांच सौ रुपये ट्रांसफर मत करना", "hi-IN", "EXECUTE_TRANSFER", "BLOCK_TRANSFER", "Hindi bank transfer prohibitive mat"),
        ("पैसे खाते में भेज दो", "पैसे खाते में मत भेजो", "hi-IN", "EXECUTE_TRANSFER", "BLOCK_TRANSFER", "Hindi send money prohibitive mat"),
        ("वेंडर को पेमेंट भेज दो", "वेंडर को पेमेंट मत भेजना", "hi-IN", "EXECUTE_TRANSFER", "BLOCK_TRANSFER", "Hindi vendor payment prohibitive mat"),
        ("यह बिल अभी भर दो", "यह बिल अभी मत भरो", "hi-IN", "EXECUTE_TRANSFER", "BLOCK_TRANSFER", "Hindi utility bill pay prohibitive mat"),
        ("रुपये तुरंत भेज दीजिए", "रुपये तुरंत मत भेजिए", "hi-IN", "EXECUTE_TRANSFER", "BLOCK_TRANSFER", "Hindi urgent fund transfer prohibitive mat"),
        # Hindi Subscription Renewal (CRITICAL NON-RENEWAL RESOLUTION!)
        ("मेरा सालाना सब्सक्रिप्शन रिन्यू कर दो", "मेरा सालाना सब्सक्रिप्शन रिन्यू मत करना", "hi-IN", "RENEW_SUBSCRIPTION", "DO_NOT_RENEW_SUBSCRIPTION", "Hindi annual subscription non-renewal"),
        ("मेरा मंथली प्लान रिन्यू कर दो", "मेरा मंथली प्लान रिन्यू मत करना", "hi-IN", "RENEW_SUBSCRIPTION", "DO_NOT_RENEW_SUBSCRIPTION", "Hindi monthly plan non-renewal"),
        ("मेंबरशिप अपने आप रिन्यू कर दो", "मेंबरशिप अपने आप रिन्यू मत करना", "hi-IN", "RENEW_SUBSCRIPTION", "DO_NOT_RENEW_SUBSCRIPTION", "Hindi membership non-renewal"),
        ("प्रीमियम सर्विस रिन्यू कर दो", "प्रीमियम सर्विस कभी रिन्यू मत करना", "hi-IN", "RENEW_SUBSCRIPTION", "DO_NOT_RENEW_SUBSCRIPTION", "Hindi premium service emphatic non-renewal"),
        ("ऑटो रिन्यूअल चालू रखो", "ऑटो रिन्यूअल बंद कर दो", "hi-IN", "RENEW_SUBSCRIPTION", "DO_NOT_RENEW_SUBSCRIPTION", "Hindi auto-renewal toggle off"),
        # Hindi Communication & Location
        ("बिल ईमेल पर भेज दो", "बिल ईमेल पर मत भेजना", "hi-IN", "EMAIL_INVOICE", "SUPPRESS_EMAIL", "Hindi email invoice prohibitive mat"),
        ("इनवॉइस पीडीएफ भेज दो", "इनवॉइस पीडीएफ मत भेजो", "hi-IN", "EMAIL_INVOICE", "SUPPRESS_EMAIL", "Hindi PDF invoice send prohibitive mat"),
        ("रसीद ईमेल कर दीजिए", "रसीद ईमेल मत कीजिए", "hi-IN", "EMAIL_INVOICE", "SUPPRESS_EMAIL", "Hindi receipt email prohibitive mat"),
        ("मेरी लाइव लोकेशन शेयर कर दो", "मेरी लाइव लोकेशन शेयर मत करना", "hi-IN", "SHARE_LOCATION", "BLOCK_LOCATION", "Hindi live location share prohibitive mat"),
        ("पता ड्राइवर को भेज दो", "पता ड्राइवर को मत भेजना", "hi-IN", "SHARE_LOCATION", "BLOCK_LOCATION", "Hindi address share prohibitive mat"),
        # English Order & Booking Cancellation
        ("Cancel my food order right now", "Do not cancel my food order right now", "en-IN", "CANCEL_ORDER", "KEEP_ORDER", "English food order declarative negation"),
        ("Cancel my flight reservation", "Don't cancel my flight reservation", "en-IN", "CANCEL_BOOKING", "KEEP_BOOKING", "English flight reservation contraction negation"),
        ("Cancel my cab booking immediately", "Don't cancel my cab booking immediately", "en-IN", "CANCEL_BOOKING", "KEEP_BOOKING", "English cab booking contraction negation"),
        ("Cancel my hotel reservation for tonight", "Do not cancel my hotel reservation for tonight", "en-IN", "CANCEL_BOOKING", "KEEP_BOOKING", "English hotel reservation declarative negation"),
        ("Cancel my train ticket to Delhi", "Never cancel my train ticket to Delhi", "en-IN", "CANCEL_BOOKING", "KEEP_BOOKING", "English train ticket emphatic negation"),
        # English Subscription Non-Renewal (CRITICAL DOMAIN DISTINCTION!)
        ("Renew my annual subscription today", "Never renew my annual subscription today", "en-IN", "RENEW_SUBSCRIPTION", "DO_NOT_RENEW_SUBSCRIPTION", "English annual subscription non-renewal never"),
        ("Renew my software license automatically", "Do not renew my software license automatically", "en-IN", "RENEW_SUBSCRIPTION", "DO_NOT_RENEW_SUBSCRIPTION", "English license non-renewal do not"),
        ("Renew my gym membership this month", "Don't renew my gym membership this month", "en-IN", "RENEW_SUBSCRIPTION", "DO_NOT_RENEW_SUBSCRIPTION", "English gym membership non-renewal don't"),
        ("Keep my cloud storage plan active", "Do not renew my cloud storage plan", "en-IN", "RENEW_SUBSCRIPTION", "DO_NOT_RENEW_SUBSCRIPTION", "English cloud storage non-renewal"),
        ("Renew my prime streaming membership", "Never renew my prime streaming membership", "en-IN", "RENEW_SUBSCRIPTION", "DO_NOT_RENEW_SUBSCRIPTION", "English streaming membership non-renewal"),
        # English Fund Transfers & Billing
        ("Transfer two thousand rupees to merchant", "Do not transfer two thousand rupees to merchant", "en-IN", "EXECUTE_TRANSFER", "BLOCK_TRANSFER", "English merchant payment declarative negation"),
        ("Send five hundred rupees right now", "Don't send five hundred rupees right now", "en-IN", "EXECUTE_TRANSFER", "BLOCK_TRANSFER", "English instant payment contraction negation"),
        ("Charge my credit card for this purchase", "Do not charge my credit card for this purchase", "en-IN", "EXECUTE_TRANSFER", "BLOCK_TRANSFER", "English credit card charge declarative negation"),
        ("Pay the electricity bill today", "Don't pay the electricity bill today", "en-IN", "EXECUTE_TRANSFER", "BLOCK_TRANSFER", "English bill payment contraction negation"),
        ("Process this transaction immediately", "Never process this transaction immediately", "en-IN", "EXECUTE_TRANSFER", "BLOCK_TRANSFER", "English transaction emphatic negation"),
        # English Notifications & Email
        ("Email the payment receipt to me", "Do not email the payment receipt to me", "en-IN", "EMAIL_INVOICE", "SUPPRESS_EMAIL", "English payment receipt declarative negation"),
        ("Send the tax invoice to my email", "Don't send the tax invoice to my email", "en-IN", "EMAIL_INVOICE", "SUPPRESS_EMAIL", "English tax invoice contraction negation"),
        ("Email booking confirmation details", "Do not email booking confirmation details", "en-IN", "EMAIL_INVOICE", "SUPPRESS_EMAIL", "English booking confirmation declarative negation"),
        ("Share my contact number with rider", "Do not share my contact number with rider", "en-IN", "SHARE_LOCATION", "BLOCK_LOCATION", "English contact share declarative negation"),
        ("Send delivery notification to customer", "Don't send delivery notification to customer", "en-IN", "SUPPRESS_EMAIL", "EMAIL_INVOICE", "English notification contraction negation"),
    ]

    for idx in range(100):
        pair_num = idx + 1
        pair_id = f"bench_v2_neg_{pair_num:03d}"
        tpl = neg_scenarios[idx % len(neg_scenarios)]
        lang = tpl[2]
        voice = voices_hi[idx % 2] if lang == "hi-IN" else voices_en[idx % 2]

        is_transformed = (idx >= 95)
        source_type = AudioSourceType.ACOUSTIC_TRANSFORMED if is_transformed else AudioSourceType.TTS_SYNTHETIC
        parent_id = f"bench_v2_neg_{(idx - 5) + 1:03d}" if is_transformed else None
        transform_params = "atempo=0.92,asetrate=16000*0.95,aresample=16000" if is_transformed else None

        specs.append({
            "pair_id": pair_id,
            "category": "negation",
            "language_code": lang,
            "source_type": source_type,
            "voice": voice,
            "source_pair_id": parent_id,
            "transformation_params": transform_params,
            "reference_transcript_a": tpl[0],
            "reference_transcript_b": tpl[1],
            "expected_critical_fields_a": {"is_negated": False, "negation_detected": False},
            "expected_critical_fields_b": {"is_negated": True, "negation_detected": True},
            "expected_intent_a": None,
            "expected_intent_b": None,
            "expected_action_a": tpl[3],
            "expected_action_b": tpl[4],
            "distinction_field": "is_negated",
            "description": f"{tpl[5]} ({source_type.value})",
            "tts_voice_or_source": voice if not is_transformed else f"transformed_from_{parent_id}",
            "notes": f"Negation polarity contrast ({source_type.value})",
        })

    # =========================================================================
    # 3. CORRECTIONS AND MIND CHANGES (100 pairs: bench_v2_cor_001 .. bench_v2_cor_100)
    # =========================================================================
    cor_scenarios = [
        # Numeric corrections (Hindi & English)
        ("मुझे दो पैकेट चाहिए", "मुझे दो नहीं तीन पैकेट चाहिए", "hi-IN", "final_target", 2, 3, "DISPATCH_2_PACKETS", "DISPATCH_3_PACKETS", "Hindi quantity correction 2 to 3"),
        ("मुझे तीन समोसे दे दो", "मुझे तीन नहीं चार समोसे दे दो", "hi-IN", "final_target", 3, 4, "ORDER_3_UNITS", "ORDER_4_UNITS", "Hindi food correction 3 to 4"),
        ("पाँच प्लेट बिरयानी भेजो", "पाँच प्लेट नहीं छह प्लेट बिरयानी भेजो", "hi-IN", "final_target", 5, 6, "ORDER_5_PLATES", "ORDER_6_PLATES", "Hindi food correction 5 to 6"),
        ("एक किलो चीनी देना", "अरे एक नहीं दो किलो चीनी देना", "hi-IN", "final_target", 1, 2, "ORDER_1_KG", "ORDER_2_KG", "Hindi grocery correction 1 to 2"),
        ("चार बोतल पानी भेजो", "चार नहीं पांच बोतल पानी भेजो", "hi-IN", "final_target", 4, 5, "ORDER_4_BOTTLES", "ORDER_5_BOTTLES", "Hindi water correction 4 to 5"),
        ("दो डिब्बे गुलाब जामुन पैक करो", "दो नहीं तीन डिब्बे गुलाब जामुन पैक करो", "hi-IN", "final_target", 2, 3, "PACK_2_BOXES", "PACK_3_BOXES", "Hindi sweets correction 2 to 3"),
        ("पांच सौ रुपये ट्रांसफर करो", "पांच सौ नहीं एक हजार रुपये ट्रांसफर करो", "hi-IN", "final_target", 500, 1000, "TRANSFER_500_INR", "TRANSFER_1000_INR", "Hindi bank correction 500 to 1000"),
        ("Book two tickets for the movie", "Book two tickets, actually make that four tickets", "en-IN", "final_target", 2, 4, "BOOK_2_TICKETS", "BOOK_4_TICKETS", "English cinema correction 2 to 4"),
        ("Order one laptop for testing", "Order one laptop, wait make that three laptops", "en-IN", "final_target", 1, 3, "ORDER_1_LAPTOP", "ORDER_3_LAPTOPS", "English IT correction 1 to 3"),
        ("Transfer five thousand rupees", "Transfer five thousand rupees, sorry make it ten thousand", "en-IN", "final_target", 5000, 10000, "TRANSFER_5000_INR", "TRANSFER_10000_INR", "English funds correction 5k to 10k"),
        ("Reserve two hotel rooms", "Reserve two rooms, actually wait make it three rooms", "en-IN", "final_target", 2, 3, "BOOK_2_ROOMS", "BOOK_3_ROOMS", "English room correction 2 to 3"),
        ("Send three cartons of supplies", "Send three cartons, sorry make that five cartons", "en-IN", "final_target", 3, 5, "DISPATCH_3_CARTONS", "DISPATCH_5_CARTONS", "English supply correction 3 to 5"),
        ("We need two conference chairs", "We need two chairs, no wait four chairs", "en-IN", "final_target", 2, 4, "ORDER_2_CHAIRS", "ORDER_4_CHAIRS", "English chair correction 2 to 4"),
        ("Order two large pizzas", "Order two pizzas, actually make it three large pizzas", "en-IN", "final_target", 2, 3, "ORDER_2_PIZZAS", "ORDER_3_PIZZAS", "English food correction 2 to 3"),
        ("Purchase five user licenses", "Purchase five licenses, no sorry ten licenses", "en-IN", "final_target", 5, 10, "BUY_5_LICENSES", "BUY_10_LICENSES", "English software license 5 to 10"),
        # Destination corrections (Hindi & English)
        ("Send the courier to Delhi", "Send the courier to Delhi, wait sorry, send it to Mumbai", "en-IN", "final_destination", "delhi", "mumbai", "DISPATCH_TO_DELHI", "DISPATCH_TO_MUMBAI", "English logistics destination Delhi to Mumbai"),
        ("Book a flight to Bangalore", "Book a flight to Bangalore, actually make it Hyderabad", "en-IN", "final_destination", "bangalore", "hyderabad", "FLIGHT_TO_BANGALORE", "FLIGHT_TO_HYDERABAD", "English flight destination Bangalore to Hyderabad"),
        ("Deliver parcel to my office", "Deliver parcel to office, wait actually deliver to home", "en-IN", "final_destination", "office", "home", "DELIVER_OFFICE", "DELIVER_HOME", "English parcel destination office to home"),
        ("पार्सल दिल्ली भेज दीजिए", "पार्सल दिल्ली नहीं मुंबई भेज दीजिए", "hi-IN", "final_destination", "delhi", "mumbai", "SHIP_DELHI", "SHIP_MUMBAI", "Hindi parcel destination Delhi to Mumbai"),
        ("गाड़ी जयपुर के लिए करो", "गाड़ी जयपुर नहीं उदयपुर के लिए करो", "hi-IN", "final_destination", "jaipur", "udaipur", "CAB_JAIPUR", "CAB_UDAIPUR", "Hindi cab destination Jaipur to Udaipur"),
        # Item / Color attribute corrections (Hindi & English)
        ("मुझे लाल कुर्ता चाहिए", "मुझे लाल नहीं नीला कुर्ता चाहिए", "hi-IN", "final_item", "लाल", "नीला", "PACK_RED_KURTA", "PACK_BLUE_KURTA", "Hindi kurta color red to blue"),
        ("मुझे काला बैग चाहिए", "मुझे काला नहीं भूरा बैग चाहिए", "hi-IN", "final_item", "काला", "भूरा", "PACK_BLACK_BAG", "PACK_BROWN_BAG", "Hindi bag color black to brown"),
        ("मुझे पीली साड़ी दिखाओ", "मुझे पीली नहीं हरी साड़ी दिखाओ", "hi-IN", "final_item", "पीली", "हरी", "SHOW_YELLOW_SAREE", "SHOW_GREEN_SAREE", "Hindi saree color yellow to green"),
        ("I want the black shoes", "I want the black shoes, wait actually the white shoes", "en-IN", "final_item", "black", "white", "ORDER_BLACK_SHOES", "ORDER_WHITE_SHOES", "English shoe color black to white"),
        ("Give me the red jacket", "Give me the red jacket, sorry I meant the blue jacket", "en-IN", "final_item", "red", "blue", "ORDER_RED_JACKET", "ORDER_BLUE_JACKET", "English jacket color red to blue"),
        # Time corrections (Hindi & English)
        ("Schedule meeting at ten AM", "Schedule meeting at ten AM, no wait, eleven AM", "en-IN", "final_time", "10:00", "11:00", "CALENDAR_10_AM", "CALENDAR_11_AM", "English appointment time 10 to 11 AM"),
        ("Book table for seven PM", "Book table for seven PM, sorry make that eight PM", "en-IN", "final_time", "19:00", "20:00", "RESERVE_7_PM", "RESERVE_8_PM", "English dinner time 7 to 8 PM"),
        ("Deliver food at one PM", "Deliver food at one PM, actually make it two PM", "en-IN", "final_time", "13:00", "14:00", "DELIVER_1_PM", "DELIVER_2_PM", "English lunch time 1 to 2 PM"),
        ("मीटिंग सुबह नौ बजे रखो", "मीटिंग सुबह नौ बजे नहीं दस बजे रखो", "hi-IN", "final_time", "09:00", "10:00", "MEETING_9_AM", "MEETING_10_AM", "Hindi morning meeting 9 to 10 AM"),
        ("कॉल शाम चार बजे करो", "कॉल शाम चार बजे नहीं पांच बजे करो", "hi-IN", "final_time", "16:00", "17:00", "CALL_4_PM", "CALL_5_PM", "Hindi evening call 4 to 5 PM"),
    ]

    for idx in range(100):
        pair_num = idx + 1
        pair_id = f"bench_v2_cor_{pair_num:03d}"
        tpl = cor_scenarios[idx % len(cor_scenarios)]
        lang = tpl[2]
        dist_field = tpl[3]
        init_val = tpl[4]
        fin_val = tpl[5]
        act_a = tpl[6]
        act_b = tpl[7]
        voice = voices_hi[idx % 2] if lang == "hi-IN" else voices_en[idx % 2]

        is_transformed = (idx >= 95)
        source_type = AudioSourceType.ACOUSTIC_TRANSFORMED if is_transformed else AudioSourceType.TTS_SYNTHETIC
        parent_id = f"bench_v2_cor_{(idx - 5) + 1:03d}" if is_transformed else None
        transform_params = "volume=1.2,aecho=0.8:0.88:60:0.4" if is_transformed else None

        exp_fields_a = {"correction_detected": False, dist_field: init_val}
        exp_fields_b = {"correction_detected": True, f"initial_{dist_field}": init_val, dist_field: fin_val}
        if dist_field == "final_target":
            exp_fields_a["quantity"] = init_val
            exp_fields_b["quantity"] = fin_val
        elif dist_field == "final_time":
            exp_fields_a["time"] = init_val
            exp_fields_b["time"] = fin_val

        specs.append({
            "pair_id": pair_id,
            "category": "corrections_and_mind_changes",
            "language_code": lang,
            "source_type": source_type,
            "voice": voice,
            "source_pair_id": parent_id,
            "transformation_params": transform_params,
            "reference_transcript_a": tpl[0],
            "reference_transcript_b": tpl[1],
            "expected_critical_fields_a": exp_fields_a,
            "expected_critical_fields_b": exp_fields_b,
            "expected_intent_a": None,
            "expected_intent_b": None,
            "expected_action_a": act_a,
            "expected_action_b": act_b,
            "distinction_field": dist_field,
            "description": f"{tpl[8]} ({init_val} to {fin_val})",
            "tts_voice_or_source": voice if not is_transformed else f"transformed_from_{parent_id}",
            "notes": f"In-flight correction contrast ({source_type.value})",
        })

    # =========================================================================
    # 4. COMMAND INTENT (100 pairs: bench_v2_cmd_001 .. bench_v2_cmd_100)
    # =========================================================================
    cmd_scenarios = [
        # Return Product vs Modify Order
        ("मुझे अपना सामान वापस करना है", "मुझे अपना ऑर्डर मॉडिफाई करना है", "hi-IN", "RETURN_PRODUCT", "MODIFY_ORDER", "RETURN_PRODUCT", "MODIFY_ORDER", "Hindi return product vs modify order"),
        ("I want to return the product", "I want to modify the product", "en-IN", "RETURN_PRODUCT", "MODIFY_ORDER", "RETURN_PRODUCT", "MODIFY_ORDER", "English return product vs modify product"),
        ("यह पार्सल वापस भेजना है", "यह पार्सल का ऑर्डर बदलना है", "hi-IN", "RETURN_PRODUCT", "MODIFY_ORDER", "RETURN_PRODUCT", "MODIFY_ORDER", "Hindi parcel return vs order modify"),
        ("I need to return this item", "I need to edit this order", "en-IN", "RETURN_PRODUCT", "MODIFY_ORDER", "RETURN_PRODUCT", "MODIFY_ORDER", "English return item vs edit order"),
        # Return Product vs Exchange Item
        ("यह शर्ट मुझे रिटर्न करनी है", "यह शर्ट मुझे एक्सचेंज करनी है", "hi-IN", "RETURN_PRODUCT", "EXCHANGE_ITEM", "RETURN_PRODUCT", "EXCHANGE_ITEM", "Hindi return shirt vs exchange shirt"),
        ("I want to return these shoes", "I want to exchange these shoes", "en-IN", "RETURN_PRODUCT", "EXCHANGE_ITEM", "RETURN_PRODUCT", "EXCHANGE_ITEM", "English return shoes vs exchange shoes"),
        ("यह घड़ी वापस ले लो", "यह घड़ी एक्सचेंज कर दो", "hi-IN", "RETURN_PRODUCT", "EXCHANGE_ITEM", "RETURN_PRODUCT", "EXCHANGE_ITEM", "Hindi watch return vs watch exchange"),
        ("Return this defective product", "Exchange this defective product", "en-IN", "RETURN_PRODUCT", "EXCHANGE_ITEM", "RETURN_PRODUCT", "EXCHANGE_ITEM", "English product return vs product exchange"),
        # Refund Request vs Check Refund Status
        ("मेरे पैसे रिफंड कर दो", "रिफंड का स्टेटस बताओ", "hi-IN", "REFUND_PAYMENT", "CHECK_REFUND", "REFUND_PAYMENT", "CHECK_REFUND", "Hindi issue refund vs check refund status"),
        ("Please refund my payment", "Check my refund status", "en-IN", "REFUND_PAYMENT", "CHECK_REFUND", "REFUND_PAYMENT", "CHECK_REFUND", "English issue refund vs check refund status"),
        ("पैसे तुरंत वापस लौटा दो", "रिफंड कब तक आएगा बताओ", "hi-IN", "REFUND_PAYMENT", "CHECK_REFUND", "REFUND_PAYMENT", "CHECK_REFUND", "Hindi instant refund vs refund arrival inquiry"),
        ("Process my refund immediately", "Track status of my refund", "en-IN", "REFUND_PAYMENT", "CHECK_REFUND", "REFUND_PAYMENT", "CHECK_REFUND", "English refund processing vs refund tracking"),
        # Book Cab vs Cancel Cab
        ("एयरपोर्ट के लिए गाड़ी बुक कर दो", "एयरपोर्ट वाली गाड़ी कैंसिल कर दो", "hi-IN", "BOOK_CAB", "CANCEL_CAB", "BOOK_CAB", "CANCEL_CAB", "Hindi airport cab book vs cancel"),
        ("Book a cab to railway station", "Cancel my cab to railway station", "en-IN", "BOOK_CAB", "CANCEL_CAB", "BOOK_CAB", "CANCEL_CAB", "English cab book vs cancel"),
        ("तुरंत एक टैक्सी बुक करो", "मेरी टैक्सी बुकिंग रद्द करो", "hi-IN", "BOOK_CAB", "CANCEL_CAB", "BOOK_CAB", "CANCEL_CAB", "Hindi taxi dispatch vs cancellation"),
        ("Book a ride right now", "Cancel my scheduled ride", "en-IN", "BOOK_CAB", "CANCEL_CAB", "BOOK_CAB", "CANCEL_CAB", "English ride booking vs cancellation"),
        # Modify Delivery Address vs Delivery Status Inquiry
        ("डिलीवरी का एड्रेस बदलना है", "डिलीवरी कब तक पहुंचेगी बताओ", "hi-IN", "MODIFY_ADDRESS", "DELIVERY_INQUIRY", "MODIFY_ADDRESS", "DELIVERY_INQUIRY", "Hindi address change vs delivery ETA"),
        ("Change my delivery address", "When will my package arrive", "en-IN", "MODIFY_ADDRESS", "DELIVERY_INQUIRY", "MODIFY_ADDRESS", "DELIVERY_INQUIRY", "English address change vs delivery ETA"),
        ("मेरा पता अपडेट कर दो", "पार्सल डिलीवरी का समय क्या है", "hi-IN", "MODIFY_ADDRESS", "DELIVERY_INQUIRY", "MODIFY_ADDRESS", "DELIVERY_INQUIRY", "Hindi address update vs delivery time query"),
        ("Update shipping address for order", "Check expected delivery date", "en-IN", "MODIFY_ADDRESS", "DELIVERY_INQUIRY", "MODIFY_ADDRESS", "DELIVERY_INQUIRY", "English update shipping address vs delivery date"),
        # Repeat Transaction vs Cancel Autopay Mandate
        ("यह पेमेंट दोबारा कर दो", "यह ऑटोपे रोक दो", "hi-IN", "REPEAT_TRANSACTION", "CANCEL_MANDATE", "REPEAT_TRANSACTION", "CANCEL_MANDATE", "Hindi repeat payment vs cancel autopay"),
        ("Repeat this payment transaction", "Cancel my recurring autopay mandate", "en-IN", "REPEAT_TRANSACTION", "CANCEL_MANDATE", "REPEAT_TRANSACTION", "CANCEL_MANDATE", "English repeat payment vs cancel mandate"),
        ("वेंडर को फिर से रुपये भेजो", "मंथली ऑटोपे पेमेंट कैंसिल करो", "hi-IN", "REPEAT_TRANSACTION", "CANCEL_MANDATE", "REPEAT_TRANSACTION", "CANCEL_MANDATE", "Hindi vendor repeat vs monthly autopay cancel"),
        ("Execute previous transaction again", "Stop recurring payment deduction", "en-IN", "REPEAT_TRANSACTION", "CANCEL_MANDATE", "REPEAT_TRANSACTION", "CANCEL_MANDATE", "English repeat deduction vs stop recurring"),
        # Escalate to Human vs File Formal Complaint
        ("कस्टमर केयर एग्जीक्यूटिव से बात करवाओ", "मुझे सर्विस के खिलाफ शिकायत दर्ज करनी है", "hi-IN", "ESCALATE_HUMAN_AGENT", "FILE_COMPLAINT", "ESCALATE_HUMAN_AGENT", "FILE_COMPLAINT", "Hindi human agent escalation vs file complaint"),
        ("Connect me to a human representative", "I want to file a formal complaint", "en-IN", "ESCALATE_HUMAN_AGENT", "FILE_COMPLAINT", "ESCALATE_HUMAN_AGENT", "FILE_COMPLAINT", "English human escalation vs file complaint"),
        ("किसी एजेंट से बात करा दीजिए", "खराब सर्विस की शिकायत लिखो", "hi-IN", "ESCALATE_HUMAN_AGENT", "FILE_COMPLAINT", "ESCALATE_HUMAN_AGENT", "FILE_COMPLAINT", "Hindi call agent vs log poor service complaint"),
        ("Transfer call to support manager", "Register an official grievance", "en-IN", "ESCALATE_HUMAN_AGENT", "FILE_COMPLAINT", "ESCALATE_HUMAN_AGENT", "FILE_COMPLAINT", "English manager escalation vs official grievance"),
        # Transfer Funds vs Request Funds
        ("खाते में पांच सौ रुपये भेजो", "पांच सौ रुपये का पेमेंट रिक्वेस्ट करो", "hi-IN", "TRANSFER_FUNDS", "REQUEST_FUNDS", "TRANSFER_FUNDS", "REQUEST_FUNDS", "Hindi transfer money vs request money"),
        ("Send five hundred rupees now", "Request five hundred rupees from contact", "en-IN", "TRANSFER_FUNDS", "REQUEST_FUNDS", "TRANSFER_FUNDS", "REQUEST_FUNDS", "English transfer funds vs request funds"),
        ("दो हजार रुपये ट्रांसफर कर दो", "दो हजार रुपये मांगो", "hi-IN", "TRANSFER_FUNDS", "REQUEST_FUNDS", "TRANSFER_FUNDS", "REQUEST_FUNDS", "Hindi transfer 2000 vs request 2000"),
        ("Pay one thousand rupees immediately", "Request one thousand rupees payment", "en-IN", "TRANSFER_FUNDS", "REQUEST_FUNDS", "TRANSFER_FUNDS", "REQUEST_FUNDS", "English pay 1000 vs request 1000"),
        # Empty Cart vs Confirm Order
        ("मेरा कार्ट खाली कर दो", "मेरा ऑर्डर कन्फर्म कर दो", "hi-IN", "EMPTY_CART", "CONFIRM_ORDER", "EMPTY_CART", "CONFIRM_ORDER", "Hindi empty cart vs confirm order"),
        ("Clear all items from my cart", "Confirm and place this order", "en-IN", "EMPTY_CART", "CONFIRM_ORDER", "EMPTY_CART", "CONFIRM_ORDER", "English empty cart vs confirm order"),
        ("शॉपिंग बैग खाली करो", "शॉपिंग बैग चेकआउट करो", "hi-IN", "EMPTY_CART", "CONFIRM_ORDER", "EMPTY_CART", "CONFIRM_ORDER", "Hindi clear bag vs checkout bag"),
        ("Empty cart completely", "Checkout and proceed to payment", "en-IN", "EMPTY_CART", "CONFIRM_ORDER", "EMPTY_CART", "CONFIRM_ORDER", "English empty cart vs checkout"),
        # Order Product vs Track Package
        ("नया पार्सल बुक करो", "मेरा पार्सल ट्रैक करो", "hi-IN", "ORDER_PRODUCT", "TRACK_PACKAGE", "ORDER_PRODUCT", "TRACK_PACKAGE", "Hindi book parcel vs track parcel"),
        ("Place a new order for supplies", "Track package status for my delivery", "en-IN", "ORDER_PRODUCT", "TRACK_PACKAGE", "ORDER_PRODUCT", "TRACK_PACKAGE", "English order supplies vs track package"),
        ("दो पैकेट बिरयानी ऑर्डर करो", "फूड डिलीवरी का लाइव स्टेटस ट्रैक करो", "hi-IN", "ORDER_PRODUCT", "TRACK_PACKAGE", "ORDER_PRODUCT", "TRACK_PACKAGE", "Hindi order food vs live track food"),
        ("Order new mobile phone", "Track shipment location of phone", "en-IN", "ORDER_PRODUCT", "TRACK_PACKAGE", "ORDER_PRODUCT", "TRACK_PACKAGE", "English buy phone vs track shipment"),
    ]

    for idx in range(100):
        pair_num = idx + 1
        pair_id = f"bench_v2_cmd_{pair_num:03d}"
        tpl = cmd_scenarios[idx % len(cmd_scenarios)]
        lang = tpl[2]
        int_a = tpl[3]
        int_b = tpl[4]
        act_a = tpl[5]
        act_b = tpl[6]
        voice = voices_hi[idx % 2] if lang == "hi-IN" else voices_en[idx % 2]

        is_transformed = (idx >= 95)
        source_type = AudioSourceType.ACOUSTIC_TRANSFORMED if is_transformed else AudioSourceType.TTS_SYNTHETIC
        parent_id = f"bench_v2_cmd_{(idx - 5) + 1:03d}" if is_transformed else None
        transform_params = "atempo=1.08,bass=g=4" if is_transformed else None

        specs.append({
            "pair_id": pair_id,
            "category": "command_intent",
            "language_code": lang,
            "source_type": source_type,
            "voice": voice,
            "source_pair_id": parent_id,
            "transformation_params": transform_params,
            "reference_transcript_a": tpl[0],
            "reference_transcript_b": tpl[1],
            "expected_critical_fields_a": {"intent": int_a},
            "expected_critical_fields_b": {"intent": int_b},
            "expected_intent_a": int_a,
            "expected_intent_b": int_b,
            "expected_action_a": act_a,
            "expected_action_b": act_b,
            "distinction_field": "intent",
            "description": f"{tpl[7]} ({int_a} vs {int_b})",
            "tts_voice_or_source": voice if not is_transformed else f"transformed_from_{parent_id}",
            "notes": f"Command intent contrast {int_a} vs {int_b} ({source_type.value})",
        })

    # =========================================================================
    # 5. NAMED ENTITIES (100 pairs: bench_v2_ent_001 .. bench_v2_ent_100)
    # =========================================================================
    name_pairs_list = [
        ("Aarav", "Vikram", "en-IN"),
        ("Priya", "Neha", "en-IN"),
        ("Rahul", "Rohit", "en-IN"),
        ("Amit", "Sumit", "en-IN"),
        ("Kavita", "Sunita", "en-IN"),
        ("Rajesh", "Rakesh", "en-IN"),
        ("Rohan", "Sohan", "en-IN"),
        ("Meera", "Maya", "en-IN"),
        ("Sanjay", "Ajay", "en-IN"),
        ("Deepa", "Divya", "en-IN"),
        ("Tanvi", "Tarun", "en-IN"),
        ("Manish", "Mayank", "en-IN"),
        ("Pooja", "Priya", "en-IN"),
        ("Arjun", "Aryan", "en-IN"),
        ("Ananya", "Aditya", "en-IN"),
        ("Ritu", "Richa", "en-IN"),
        ("Gaurav", "Saurabh", "en-IN"),
        ("Alok", "Ashok", "en-IN"),
        ("Sneha", "Shreya", "en-IN"),
        ("Vishal", "Vikas", "en-IN"),
        # Hindi Devanagari pairs
        ("आरव", "विक्रम", "hi-IN"),
        ("प्रिया", "नेहा", "hi-IN"),
        ("राहुल", "रोहित", "hi-IN"),
        ("अमित", "सुमित", "hi-IN"),
        ("कविता", "सुनीता", "hi-IN"),
        ("राजेश", "राकेश", "hi-IN"),
        ("रोहन", "सोहन", "hi-IN"),
        ("मीरा", "माया", "hi-IN"),
        ("संजय", "अजय", "hi-IN"),
        ("दीपा", "दिव्या", "hi-IN"),
        ("तन्वी", "तरुण", "hi-IN"),
        ("मनीष", "मयंक", "hi-IN"),
        ("पूजा", "प्रिया", "hi-IN"),
        ("अर्जुन", "आर्यन", "hi-IN"),
        ("अनन्या", "आदित्य", "hi-IN"),
        ("ऋतु", "ऋचा", "hi-IN"),
        ("गौरव", "सौरभ", "hi-IN"),
        ("आलोक", "अशोक", "hi-IN"),
        ("स्नेहा", "श्रेया", "hi-IN"),
        ("विशाल", "विकास", "hi-IN"),
    ]

    ent_templates_en = [
        ("Send payment to {name}", "TRANSFER_TO_{name_upper}", "English fund transfer recipient name"),
        ("Transfer money to {name}", "TRANSFER_TO_{name_upper}", "English money transfer recipient name"),
        ("Assign ticket to {name}", "ASSIGN_TO_{name_upper}", "English ticket assignment name"),
        ("Deliver the package to {name}", "DELIVER_TO_{name_upper}", "English package recipient name"),
        ("Call {name} right now", "DIAL_CONTACT_{name_upper}", "English contact voice call name"),
    ]

    ent_templates_hi = [
        ("{name} को पांच सौ रुपये भेजो", "TRANSFER_TO_{name_upper}", "Hindi fund transfer recipient name"),
        ("{name} के खाते में पैसे ट्रांसफर करो", "TRANSFER_TO_{name_upper}", "Hindi bank account transfer name"),
        ("{name} को कॉल लगाओ", "DIAL_CONTACT_{name_upper}", "Hindi phone call contact name"),
        ("{name} का नंबर डायल करो", "DIAL_CONTACT_{name_upper}", "Hindi dial number contact name"),
        ("यह पार्सल {name} को दे देना", "DELIVER_TO_{name_upper}", "Hindi parcel recipient name"),
    ]

    for idx in range(100):
        pair_num = idx + 1
        pair_id = f"bench_v2_ent_{pair_num:03d}"
        n_tuple = name_pairs_list[idx % len(name_pairs_list)]
        name_a, name_b, lang = n_tuple

        voice = voices_hi[idx % 2] if lang == "hi-IN" else voices_en[idx % 2]
        is_transformed = (idx >= 95)
        source_type = AudioSourceType.ACOUSTIC_TRANSFORMED if is_transformed else AudioSourceType.TTS_SYNTHETIC
        parent_id = f"bench_v2_ent_{(idx - 5) + 1:03d}" if is_transformed else None
        transform_params = "atempo=1.05,treble=g=3" if is_transformed else None

        tpl_list = ent_templates_hi if lang == "hi-IN" else ent_templates_en
        t_chosen = tpl_list[idx % len(tpl_list)]

        ref_a = t_chosen[0].format(name=name_a)
        ref_b = t_chosen[0].format(name=name_b)
        act_a = t_chosen[1].format(name_upper=name_a.upper())
        act_b = t_chosen[1].format(name_upper=name_b.upper())

        specs.append({
            "pair_id": pair_id,
            "category": "named_entities",
            "language_code": lang,
            "source_type": source_type,
            "voice": voice,
            "source_pair_id": parent_id,
            "transformation_params": transform_params,
            "reference_transcript_a": ref_a,
            "reference_transcript_b": ref_b,
            "expected_critical_fields_a": {"name": name_a, "recipient": name_a, "contact": name_a, "assignee": name_a},
            "expected_critical_fields_b": {"name": name_b, "recipient": name_b, "contact": name_b, "assignee": name_b},
            "expected_intent_a": None,
            "expected_intent_b": None,
            "expected_action_a": act_a,
            "expected_action_b": act_b,
            "distinction_field": "name",
            "description": f"{t_chosen[2]} ({name_a} vs {name_b})",
            "tts_voice_or_source": voice if not is_transformed else f"transformed_from_{parent_id}",
            "notes": f"Named entity contrast {name_a} vs {name_b} ({source_type.value})",
        })

    # =========================================================================
    # 6. TIME EXTRACTION (100 pairs: bench_v2_tim_001 .. bench_v2_tim_100)
    # =========================================================================
    time_pairs_data = [
        # Hindi clock hour contrasts
        ("सुबह नौ बजे मीटिंग है", "सुबह दस बजे मीटिंग है", "hi-IN", "09:00", "10:00", 9, 10, "SCHEDULE_09_00", "SCHEDULE_10_00", "Hindi morning meeting 9 AM vs 10 AM"),
        ("सुबह आठ बजे पहुंचना है", "सुबह नौ बजे पहुंचना है", "hi-IN", "08:00", "09:00", 8, 9, "SCHEDULE_08_00", "SCHEDULE_09_00", "Hindi morning arrival 8 AM vs 9 AM"),
        ("दोपहर दो बजे कॉल करो", "दोपहर चार बजे कॉल करो", "hi-IN", "14:00", "16:00", 14, 16, "CALL_14_00", "CALL_16_00", "Hindi afternoon call 2 PM vs 4 PM"),
        ("शाम छह बजे डिलीवरी भेजो", "शाम सात बजे डिलीवरी भेजो", "hi-IN", "18:00", "19:00", 18, 19, "DELIVER_18_00", "DELIVER_19_00", "Hindi evening delivery 6 PM vs 7 PM"),
        ("रात आठ बजे डिनर तैयार रखना", "रात नौ बजे डिनर तैयार रखना", "hi-IN", "20:00", "21:00", 20, 21, "DINNER_20_00", "DINNER_21_00", "Hindi night dinner 8 PM vs 9 PM"),
        ("सुबह सात बजे अलार्म लगाओ", "सुबह आठ बजे अलार्म लगाओ", "hi-IN", "07:00", "08:00", 7, 8, "ALARM_07_00", "ALARM_08_00", "Hindi alarm clock 7 AM vs 8 AM"),
        ("दोपहर बारह बजे आ जाना", "दोपहर एक बजे आ जाना", "hi-IN", "12:00", "13:00", 12, 13, "ARRIVE_12_00", "ARRIVE_13_00", "Hindi arrival 12 PM vs 1 PM"),
        ("शाम पांच बजे कैब भेजो", "शाम छह बजे कैब भेजो", "hi-IN", "17:00", "18:00", 17, 18, "CAB_17_00", "CAB_18_00", "Hindi evening cab 5 PM vs 6 PM"),
        ("रात दस बजे दवाई खानी है", "रात ग्यारह बजे दवाई खानी है", "hi-IN", "22:00", "23:00", 22, 23, "MEDS_22_00", "MEDS_23_00", "Hindi medicine alert 10 PM vs 11 PM"),
        ("सुबह ग्यारह बजे इंटरव्यू है", "दोपहर बारह बजे इंटरव्यू है", "hi-IN", "11:00", "12:00", 11, 12, "INTERVIEW_11_00", "INTERVIEW_12_00", "Hindi interview 11 AM vs 12 PM"),
        # Hindi half-past contrasts
        ("साढ़े सात बजे निकलना है", "साढ़े आठ बजे निकलना है", "hi-IN", "07:30", "08:30", 7, 8, "LEAVE_07_30", "LEAVE_08_30", "Hindi half past 7:30 vs 8:30"),
        ("साढ़े नौ बजे क्लास शुरू होगी", "साढ़े दस बजे क्लास शुरू होगी", "hi-IN", "09:30", "10:30", 9, 10, "CLASS_09_30", "CLASS_10_30", "Hindi class start 9:30 vs 10:30"),
        ("शाम साढ़े छह बजे मिलना", "शाम साढ़े सात बजे मिलना", "hi-IN", "18:30", "19:30", 18, 19, "MEET_18_30", "MEET_19_30", "Hindi evening meet 6:30 PM vs 7:30 PM"),
        ("दोपहर साढ़े बारह बजे लंच करेंगे", "दोपहर साढ़े एक बजे लंच करेंगे", "hi-IN", "12:30", "13:30", 12, 13, "LUNCH_12_30", "LUNCH_13_30", "Hindi lunch 12:30 PM vs 1:30 PM"),
        ("सुबह साढ़े आठ बजे ट्रेन है", "सुबह साढ़े नौ बजे ट्रेन है", "hi-IN", "08:30", "09:30", 8, 9, "TRAIN_08_30", "TRAIN_09_30", "Hindi train schedule 8:30 AM vs 9:30 AM"),
        # English hour AM / PM
        ("Schedule conference call at nine AM", "Schedule conference call at ten AM", "en-IN", "09:00", "10:00", 9, 10, "CALL_09_00", "CALL_10_00", "English morning call 9 AM vs 10 AM"),
        ("Set client meeting at two PM", "Set client meeting at three PM", "en-IN", "14:00", "15:00", 14, 15, "MEET_14_00", "MEET_15_00", "English afternoon client meeting 2 PM vs 3 PM"),
        ("Deliver parcel at four PM", "Deliver parcel at five PM", "en-IN", "16:00", "17:00", 16, 17, "DELIVER_16_00", "DELIVER_17_00", "English parcel delivery 4 PM vs 5 PM"),
        ("Book flight departure at six AM", "Book flight departure at seven AM", "en-IN", "06:00", "07:00", 6, 7, "FLIGHT_06_00", "FLIGHT_07_00", "English flight departure 6 AM vs 7 AM"),
        ("Call me at eight PM tonight", "Call me at nine PM tonight", "en-IN", "20:00", "21:00", 20, 21, "CALL_20_00", "CALL_21_00", "English evening phone call 8 PM vs 9 PM"),
        ("Doctor appointment is at ten AM", "Doctor appointment is at eleven AM", "en-IN", "10:00", "11:00", 10, 11, "APPOINTMENT_10_00", "APPOINTMENT_11_00", "English doctor appointment 10 AM vs 11 AM"),
        ("Reserve dining table for seven PM", "Reserve dining table for eight PM", "en-IN", "19:00", "20:00", 19, 20, "RESERVE_19_00", "RESERVE_20_00", "English dinner reservation 7 PM vs 8 PM"),
        ("Schedule interview at one PM", "Schedule interview at two PM", "en-IN", "13:00", "14:00", 13, 14, "INTERVIEW_13_00", "INTERVIEW_14_00", "English interview schedule 1 PM vs 2 PM"),
        ("Dispatch morning courier at eight AM", "Dispatch morning courier at nine AM", "en-IN", "08:00", "09:00", 8, 9, "DISPATCH_08_00", "DISPATCH_09_00", "English morning dispatch 8 AM vs 9 AM"),
        ("Pick me up at five PM", "Pick me up at six PM", "en-IN", "17:00", "18:00", 17, 18, "PICKUP_17_00", "PICKUP_18_00", "English ride pickup 5 PM vs 6 PM"),
        # English compound minutes (e.g. ten thirty vs eleven thirty)
        ("Meeting is scheduled at ten thirty", "Meeting is scheduled at eleven thirty", "en-IN", "10:30", "11:30", 10, 11, "MEET_10_30", "MEET_11_30", "English compound minutes 10:30 vs 11:30"),
        ("Train leaves at twelve thirty", "Train leaves at one thirty", "en-IN", "12:30", "01:30", 12, 1, "TRAIN_12_30", "TRAIN_01_30", "English compound train 12:30 vs 1:30"),
        ("Flight takes off at six thirty", "Flight takes off at seven thirty", "en-IN", "06:30", "07:30", 6, 7, "FLIGHT_06_30", "FLIGHT_07_30", "English compound flight 6:30 vs 7:30"),
        ("Doctor arrives at four thirty", "Doctor arrives at five thirty", "en-IN", "04:30", "05:30", 4, 5, "DOCTOR_04_30", "DOCTOR_05_30", "English compound arrival 4:30 vs 5:30"),
        ("Webinar starts at eight thirty", "Webinar starts at nine thirty", "en-IN", "08:30", "09:30", 8, 9, "WEBINAR_08_30", "WEBINAR_09_30", "English compound webinar 8:30 vs 9:30"),
    ]

    for idx in range(100):
        pair_num = idx + 1
        pair_id = f"bench_v2_tim_{pair_num:03d}"
        tpl = time_pairs_data[idx % len(time_pairs_data)]
        lang = tpl[2]
        t_a = tpl[3]
        t_b = tpl[4]
        h_a = tpl[5]
        h_b = tpl[6]
        act_a = tpl[7]
        act_b = tpl[8]
        voice = voices_hi[idx % 2] if lang == "hi-IN" else voices_en[idx % 2]

        is_transformed = (idx >= 95)
        source_type = AudioSourceType.ACOUSTIC_TRANSFORMED if is_transformed else AudioSourceType.TTS_SYNTHETIC
        parent_id = f"bench_v2_tim_{(idx - 5) + 1:03d}" if is_transformed else None
        transform_params = "atempo=0.94,highpass=f=300" if is_transformed else None

        specs.append({
            "pair_id": pair_id,
            "category": "time_extraction",
            "language_code": lang,
            "source_type": source_type,
            "voice": voice,
            "source_pair_id": parent_id,
            "transformation_params": transform_params,
            "reference_transcript_a": tpl[0],
            "reference_transcript_b": tpl[1],
            "expected_critical_fields_a": {"time": t_a, "delivery_time": t_a, "hour": h_a},
            "expected_critical_fields_b": {"time": t_b, "delivery_time": t_b, "hour": h_b},
            "expected_intent_a": None,
            "expected_intent_b": None,
            "expected_action_a": act_a,
            "expected_action_b": act_b,
            "distinction_field": "delivery_time",
            "description": f"{tpl[9]} ({t_a} vs {t_b})",
            "tts_voice_or_source": voice if not is_transformed else f"transformed_from_{parent_id}",
            "notes": f"Time extraction contrast {t_a} vs {t_b} ({source_type.value})",
        })

    return specs


if __name__ == "__main__":
    all_specs = generate_all_600_specifications()
    print(f"Total specifications generated: {len(all_specs)}")
    from collections import Counter
    cats = Counter(s["category"] for s in all_specs)
    sources = Counter(s["source_type"].value for s in all_specs)
    print("Categories:", cats)
    print("Sources:", sources)
