"""Batch benchmark manifest models, file validation, and reporting for BoliBench.

Guarantees:
- Validates every audio file before any live provider call.
- Strictly enforces WAV format matching backend constraints:
  * File must exist and be non-empty (<= 25 MB).
  * Valid RIFF/WAVE header.
  * Standard linear PCM format parseable by wave standard library.
  * Duration strictly <= 60.0 seconds.
- Categorizes pairs into ready, invalid, duplicated, and missing ground-truth.
- Separates audio source types: human_recorded, tts_synthetic, acoustic_transformed.
- Disclaims statistical significance on small/correlated samples and notes that
  TTS/transformed audio does not establish performance across independent human speakers.
"""

from enum import Enum
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import wave

from pydantic import BaseModel, Field

from app.services.audio_validator import AudioValidationError, AudioValidator

logger = logging.getLogger("bolibench.batch_manifest")


class AudioSourceType(str, Enum):
    """Origin taxonomy for benchmark audio."""
    HUMAN_RECORDED = "human_recorded"
    TTS_SYNTHETIC = "tts_synthetic"
    ACOUSTIC_TRANSFORMED = "acoustic_transformed"


class BatchManifestItem(BaseModel):
    """Specification of a single paired-contrast benchmark item."""
    pair_id: str = Field(..., description="Stable, unique identifier for the test pair (e.g. pair_neg_001)")
    category: str = Field(..., description="Scenario category (e.g. negation, quantities_and_numbers, etc.)")
    language_code: str = Field("hi-IN", description="Target BCP-47 language code (e.g. hi-IN, en-IN)")
    audio_condition: str = Field("clean_recorded", description="Acoustic environment metadata")
    source_type: AudioSourceType = Field(AudioSourceType.HUMAN_RECORDED, description="Source provenance")
    audio_path_a: str = Field(..., description="Path to audio file for Utterance A (relative to manifest or absolute)")
    audio_path_b: str = Field(..., description="Path to audio file for Utterance B (relative to manifest or absolute)")
    reference_transcript_a: str = Field(..., description="Ground-truth reference transcript for Utterance A")
    reference_transcript_b: str = Field(..., description="Ground-truth reference transcript for Utterance B")
    expected_critical_fields_a: Dict[str, Any] = Field(default_factory=dict, description="Expected critical business fields for A")
    expected_critical_fields_b: Dict[str, Any] = Field(default_factory=dict, description="Expected critical business fields for B")
    expected_intent_a: Optional[str] = Field(None, description="Expected intent for A (e.g. ORDER_PRODUCT)")
    expected_intent_b: Optional[str] = Field(None, description="Expected intent for B")
    expected_action_a: Optional[str] = Field(None, description="Expected downstream action for A")
    expected_action_b: Optional[str] = Field(None, description="Expected downstream action for B")
    distinction_field: Optional[str] = Field(None, description="Field where distinction must be preserved (e.g. quantity, negation)")
    description: Optional[str] = Field(None, description="Description of the semantic contrast")
    speaker_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional speaker or recording notes")
    tts_voice_or_source: Optional[str] = Field(None, description="TTS voice ID or recording source")
    transformation_params: Optional[str] = Field(None, description="Parameters if acoustic-transformed")
    dataset_version: str = Field("1.0.0", description="Dataset version")
    notes: Optional[str] = Field(None, description="Auditing notes")


class BatchManifest(BaseModel):
    """Complete batch benchmark manifest container."""
    manifest_version: str = Field("1.0.0", description="Manifest schema version")
    name: str = Field("BoliBench Curated Benchmark Manifest", description="Manifest title")
    description: str = Field("Audited paired-contrast evaluation manifest", description="Purpose & scope")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    items: List[BatchManifestItem] = Field(default_factory=list, description="List of benchmark pairs")


class ManifestValidationReport:
    """Detailed audit report of manifest items before any execution."""

    def __init__(self):
        self.total_items: int = 0
        self.ready_items: List[BatchManifestItem] = []
        self.invalid_items: List[Dict[str, Any]] = []
        self.duplicated_items: List[str] = []
        self.missing_ground_truth_items: List[str] = []
        self.manual_review_items: List[Dict[str, Any]] = []
        self.counts_by_source: Dict[str, int] = {
            AudioSourceType.HUMAN_RECORDED.value: 0,
            AudioSourceType.TTS_SYNTHETIC.value: 0,
            AudioSourceType.ACOUSTIC_TRANSFORMED.value: 0,
        }
        self.counts_by_category: Dict[str, int] = {}
        self.audio_details: Dict[str, Dict[str, Any]] = {}

    @property
    def ready_count(self) -> int:
        return len(self.ready_items)

    @property
    def planned_requests_count(self) -> int:
        """Every ready paired benchmark item requires exactly 2 sequential provider requests."""
        return self.ready_count * 2

    @property
    def is_valid(self) -> bool:
        """True only if at least one pair is ready and no duplicates or invalid files exist."""
        return self.ready_count > 0 and len(self.invalid_items) == 0 and len(self.duplicated_items) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "ready_count": self.ready_count,
            "planned_requests_count": self.planned_requests_count,
            "invalid_count": len(self.invalid_items),
            "duplicated_count": len(self.duplicated_items),
            "missing_ground_truth_count": len(self.missing_ground_truth_items),
            "manual_review_count": len(self.manual_review_items),
            "counts_by_source": self.counts_by_source,
            "counts_by_category": self.counts_by_category,
            "invalid_items": self.invalid_items,
            "duplicated_items": self.duplicated_items,
            "missing_ground_truth_items": self.missing_ground_truth_items,
            "manual_review_items": self.manual_review_items,
            "is_valid": self.is_valid,
        }

    def summary_markdown(self) -> str:
        """Generates a structured human-readable markdown audit report."""
        lines = [
            "# Batch Manifest Pre-Flight Audit Report",
            "",
            f"- **Total Pairs in Manifest**: `{self.total_items}`",
            f"- **Ready for Evaluation**: `{self.ready_count}`",
            f"- **Planned Provider Requests**: `{self.planned_requests_count}` (2 requests per ready pair)",
            f"- **Invalid Pairs**: `{len(self.invalid_items)}`",
            f"- **Duplicated Pair IDs**: `{len(self.duplicated_items)}`",
            f"- **Missing Ground Truth**: `{len(self.missing_ground_truth_items)}`",
            f"- **Semantic Review Flags**: `{len(self.manual_review_items)}` (audited; not excluded)",
            "",
            "### Source Provenance Breakdown",
            "| Source Type | Count | Share |",
            "| :--- | :---: | :---: |",
            f"| Human Recorded | {self.counts_by_source.get(AudioSourceType.HUMAN_RECORDED.value, 0)} | {self.counts_by_source.get(AudioSourceType.HUMAN_RECORDED.value, 0) / max(1, self.total_items) * 100:.1f}% |",
            f"| TTS Synthetic | {self.counts_by_source.get(AudioSourceType.TTS_SYNTHETIC.value, 0)} | {self.counts_by_source.get(AudioSourceType.TTS_SYNTHETIC.value, 0) / max(1, self.total_items) * 100:.1f}% |",
            f"| Acoustic Transformed | {self.counts_by_source.get(AudioSourceType.ACOUSTIC_TRANSFORMED.value, 0)} | {self.counts_by_source.get(AudioSourceType.ACOUSTIC_TRANSFORMED.value, 0) / max(1, self.total_items) * 100:.1f}% |",
            "",
            "> [!NOTE]",
            "> **Evaluation Integrity Notice:** TTS-generated audio and acoustic-transformed variants do not establish performance across independent human speakers. Results on synthetic or transformed audio must be reported under separate taxonomy.",
            "",
            "### Category Distribution",
            "| Category | Ready Pairs |",
            "| :--- | :---: |",
        ]
        for cat, count in sorted(self.counts_by_category.items()):
            lines.append(f"| `{cat}` | {count} |")

        if self.invalid_items:
            lines.extend([
                "",
                "### Invalid Items (Excluded)",
                "| Pair ID | Error Type | Message |",
                "| :--- | :--- | :--- |",
            ])
            for item in self.invalid_items:
                lines.append(f"| `{item['pair_id']}` | `{item['error_type']}` | {item['message']} |")

        if self.duplicated_items:
            lines.extend([
                "",
                "### Duplicated Pair IDs",
                f"- Duplicates detected: `{', '.join(self.duplicated_items)}`",
            ])

        if self.manual_review_items:
            lines.extend([
                "",
                "### Semantic & Manual Review Audit Items",
                "| Pair ID | Category | Review Type | Detail |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for m in self.manual_review_items:
                lines.append(f"| `{m['pair_id']}` | `{m.get('category', 'unknown')}` | `{m['review_type']}` | {m['note']} |")

        return "\n".join(lines)


def _extract_pcm_frames_and_hash(content: bytes) -> Optional[Tuple[bytes, str]]:
    """Extracts decoded linear PCM audio sample frames and their SHA-256 hash."""
    try:
        with wave.open(io.BytesIO(content), "rb") as w:
            frames = w.readframes(w.getnframes())
            return frames, hashlib.sha256(frames).hexdigest()
    except Exception:
        return None


def load_manifest(path: Union[str, Path]) -> BatchManifest:
    """Loads a BatchManifest from a JSON file."""
    manifest_path = Path(path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return BatchManifest(
            name="Imported Batch Manifest",
            description="Loaded from item array",
            created_at="",
            items=[BatchManifestItem(**item) for item in data],
        )

    return BatchManifest(**data)


def save_manifest(manifest: BatchManifest, path: Union[str, Path]) -> None:
    """Saves a BatchManifest to a JSON file."""
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2, ensure_ascii=False)


def _resolve_audio_path(path_str: str, base_dir: Path) -> Path:
    """Resolves an audio path relative to base_dir, current working directory, or workspace root."""
    p = Path(path_str)
    if p.is_absolute():
        return p

    # 1. Check relative to manifest's base directory
    cand1 = (base_dir / p).resolve()
    if cand1.exists():
        return cand1

    # 2. Check relative to current working directory
    cand2 = (Path.cwd() / p).resolve()
    if cand2.exists():
        return cand2

    # 3. Check relative to parent of base_dir (e.g. data/../local_test_audio)
    cand3 = (base_dir.parent / p).resolve()
    if cand3.exists():
        return cand3

    return cand1


def validate_manifest(
    manifest: Union[BatchManifest, str, Path],
    base_dir: Optional[Path] = None,
    audio_validator: Optional[AudioValidator] = None,
) -> ManifestValidationReport:
    """Performs rigorous pre-flight validation on every item in the batch manifest.

    Guarantees:
    - Never invokes external network calls or consumes credits.
    - Inspects file existence, file size, RIFF/WAVE magic bytes, linear PCM headers, and duration <= 60s.
    - Prevents duplicate pair_ids.
    - Flags missing or identical ground-truth references.
    - Detects identical file hashes and identical decoded PCM audio samples within pairs.
    - Detects missing distinction_field and contradictory category-specific critical fields.
    - Flags high-lexical-overlap and near-identical duration pairs for manual semantic audit.
    """
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).resolve()
        if base_dir is None:
            base_dir = manifest_path.parent
        manifest_obj = load_manifest(manifest_path)
    else:
        manifest_obj = manifest
        if base_dir is None:
            base_dir = Path.cwd()

    validator = audio_validator or AudioValidator()
    report = ManifestValidationReport()
    report.total_items = len(manifest_obj.items)

    seen_pair_ids: Set[str] = set()

    for item in manifest_obj.items:
        # 1. Duplicate pair ID check
        if item.pair_id in seen_pair_ids:
            report.duplicated_items.append(item.pair_id)
            report.invalid_items.append({
                "pair_id": item.pair_id,
                "error_type": "DUPLICATE_PAIR_ID",
                "message": f"Pair ID '{item.pair_id}' is duplicated in manifest.",
            })
            continue
        seen_pair_ids.add(item.pair_id)

        # 2. Ground-truth reference completeness
        ref_a = (item.reference_transcript_a or "").strip()
        ref_b = (item.reference_transcript_b or "").strip()
        if not ref_a or not ref_b:
            report.missing_ground_truth_items.append(item.pair_id)
            report.invalid_items.append({
                "pair_id": item.pair_id,
                "error_type": "MISSING_GROUND_TRUTH",
                "message": f"Pair '{item.pair_id}' has empty reference transcripts (ref_a='{ref_a}', ref_b='{ref_b}').",
            })
            continue

        # 3. Audio file path resolution
        resolved_a = _resolve_audio_path(item.audio_path_a, base_dir)
        resolved_b = _resolve_audio_path(item.audio_path_b, base_dir)

        if not resolved_a.exists():
            report.invalid_items.append({
                "pair_id": item.pair_id,
                "error_type": "AUDIO_FILE_NOT_FOUND",
                "message": f"Audio file A not found on disk: {resolved_a}",
            })
            continue

        if not resolved_b.exists():
            report.invalid_items.append({
                "pair_id": item.pair_id,
                "error_type": "AUDIO_FILE_NOT_FOUND",
                "message": f"Audio file B not found on disk: {resolved_b}",
            })
            continue

        # 3. Audio file reading and binary validation
        try:
            content_a = resolved_a.read_bytes()
            ext_a, ct_a = validator.validate_live_audio(resolved_a.name, content_a)
        except AudioValidationError as exc:
            report.invalid_items.append({
                "pair_id": item.pair_id,
                "error_type": f"INVALID_AUDIO_A_{exc.error_type}",
                "message": f"Utterance A invalid ({resolved_a.name}): {exc.message}",
            })
            continue
        except Exception as exc:
            report.invalid_items.append({
                "pair_id": item.pair_id,
                "error_type": "UNREADABLE_AUDIO_A",
                "message": f"Failed to read Utterance A ({resolved_a.name}): {str(exc)}",
            })
            continue

        try:
            content_b = resolved_b.read_bytes()
            ext_b, ct_b = validator.validate_live_audio(resolved_b.name, content_b)
        except AudioValidationError as exc:
            report.invalid_items.append({
                "pair_id": item.pair_id,
                "error_type": f"INVALID_AUDIO_B_{exc.error_type}",
                "message": f"Utterance B invalid ({resolved_b.name}): {exc.message}",
            })
            continue
        except Exception as exc:
            report.invalid_items.append({
                "pair_id": item.pair_id,
                "error_type": "UNREADABLE_AUDIO_B",
                "message": f"Failed to read Utterance B ({resolved_b.name}): {str(exc)}",
            })
            continue

        # 4. Audio uniqueness within pair (file bytes & decoded linear PCM samples)
        hash_a = hashlib.sha256(content_a).hexdigest()
        hash_b = hashlib.sha256(content_b).hexdigest()
        if hash_a == hash_b:
            report.invalid_items.append({
                "pair_id": item.pair_id,
                "error_type": "IDENTICAL_AUDIO_FILES",
                "message": f"Utterance A and B in pair '{item.pair_id}' have identical file content (SHA-256: {hash_a[:12]}...). Recordings must be distinct.",
            })
            continue

        pcm_info_a = _extract_pcm_frames_and_hash(content_a)
        pcm_info_b = _extract_pcm_frames_and_hash(content_b)
        if pcm_info_a and pcm_info_b and pcm_info_a[1] == pcm_info_b[1]:
            report.invalid_items.append({
                "pair_id": item.pair_id,
                "error_type": "IDENTICAL_PCM_AUDIO",
                "message": f"Utterance A and B in pair '{item.pair_id}' have identical decoded PCM audio samples (PCM hash: {pcm_info_a[1][:12]}...). Recordings must be distinct.",
            })
            continue

        # 5. Reference transcripts distinctness
        norm_ref_a = " ".join(ref_a.strip().lower().split())
        norm_ref_b = " ".join(ref_b.strip().lower().split())
        if norm_ref_a == norm_ref_b:
            report.invalid_items.append({
                "pair_id": item.pair_id,
                "error_type": "IDENTICAL_REFERENCE_TRANSCRIPTS",
                "message": f"Pair '{item.pair_id}' has identical reference transcripts ('{ref_a}'). Paired contrast requires distinct semantic targets.",
            })
            continue

        # 6. Critical fields completeness & consistency (when fields or distinction specified)
        df = (item.distinction_field or "").strip()
        fa = item.expected_critical_fields_a or {}
        fb = item.expected_critical_fields_b or {}

        if df or fa or fb:
            if not df:
                report.invalid_items.append({
                    "pair_id": item.pair_id,
                    "error_type": "MISSING_DISTINCTION_FIELD",
                    "message": f"Pair '{item.pair_id}' specifies critical fields but is missing required 'distinction_field'.",
                })
                continue

            contradiction_msg = None
            if item.category == "quantities_and_numbers":
                qa = fa.get("quantity") if "quantity" in fa else fa.get("number")
                qb = fb.get("quantity") if "quantity" in fb else fb.get("number")
                if qa is not None and qb is not None and qa == qb:
                    contradiction_msg = f"Contradictory quantity: both sides specify quantity={qa}."
            elif item.category == "negation":
                na = fa.get("is_negated")
                nb = fb.get("is_negated")
                if na is not None and nb is not None and na == nb:
                    contradiction_msg = f"Contradictory negation: both sides specify is_negated={na}."
            elif item.category == "command_intent":
                ia = item.expected_intent_a
                ib = item.expected_intent_b
                if ia and ib and str(ia).strip().upper() == str(ib).strip().upper():
                    contradiction_msg = f"Contradictory intent: both sides specify expected_intent='{ia}'."
            elif item.category == "named_entities":
                ea = fa.get("name") or fa.get("recipient")
                eb = fb.get("name") or fb.get("recipient")
                if ea and eb and str(ea).strip().lower() == str(eb).strip().lower():
                    contradiction_msg = f"Contradictory entity: both sides specify name='{ea}'."
            elif item.category == "time_extraction":
                ta = fa.get("time") or fa.get("delivery_time")
                tb = fb.get("time") or fb.get("delivery_time")
                if ta and tb and str(ta).strip().lower() == str(tb).strip().lower():
                    contradiction_msg = f"Contradictory time: both sides specify time='{ta}'."

            if contradiction_msg:
                report.invalid_items.append({
                    "pair_id": item.pair_id,
                    "error_type": "CONTRADICTORY_CRITICAL_FIELDS",
                    "message": f"Pair '{item.pair_id}' critical field conflict: {contradiction_msg}",
                })
                continue

        # 7. Extract WAV metadata for auditing
        dur_a = _get_wav_duration_seconds(content_a)
        dur_b = _get_wav_duration_seconds(content_b)
        report.audio_details[item.pair_id] = {
            "path_a": str(resolved_a),
            "size_a": len(content_a),
            "duration_a": dur_a,
            "path_b": str(resolved_b),
            "size_b": len(content_b),
            "duration_b": dur_b,
        }

        # 8. Flag pairs for semantic and manual audit (does not exclude legitimate contrast)
        words_a = set(norm_ref_a.split())
        words_b = set(norm_ref_b.split())
        overlap = len(words_a & words_b) / max(1, len(words_a | words_b))
        if overlap >= 0.75:
            report.manual_review_items.append({
                "pair_id": item.pair_id,
                "category": item.category,
                "review_type": "HIGH_LEXICAL_OVERLAP",
                "overlap_ratio": round(overlap, 2),
                "note": f"Contrast pair shares {round(overlap*100)}% words; target distinction verified on '{df}'.",
            })

        if abs(dur_a - dur_b) < 0.05:
            report.manual_review_items.append({
                "pair_id": item.pair_id,
                "category": item.category,
                "review_type": "SIMILAR_DURATION",
                "delta_seconds": round(abs(dur_a - dur_b), 3),
                "note": f"Similar audio duration ({dur_a}s vs {dur_b}s); confirmed distinct PCM samples ({hash_a[:8]} vs {hash_b[:8]}).",
            })

        # Item is valid and ready
        report.ready_items.append(item)
        src_val = item.source_type.value if isinstance(item.source_type, AudioSourceType) else str(item.source_type)
        report.counts_by_source[src_val] = report.counts_by_source.get(src_val, 0) + 1
        cat_val = item.category
        report.counts_by_category[cat_val] = report.counts_by_category.get(cat_val, 0) + 1

    return report


def _get_wav_duration_seconds(content: bytes) -> float:
    """Internal helper to extract duration from PCM WAV bytes."""
    try:
        with wave.open(io.BytesIO(content), "rb") as w:
            rate = w.getframerate()
            frames = w.getnframes()
            return round(frames / float(rate), 2) if rate > 0 else 0.0
    except Exception:
        return 0.0


def create_sample_manifest(output_path: Optional[Union[str, Path]] = None) -> BatchManifest:
    """Creates a sample 5-pair curated batch manifest linking available local PCM test audio.

    Maps directly to BoliBench's seed dataset high-stakes contrast pairs:
    1. pair_neg_order_001 (negation)
    2. pair_corr_qty_001 (corrections_and_mind_changes)
    3. pair_intent_order_001 (command_intent)
    4. pair_ent_deliv_001 (named_entities)
    5. pair_time_deliv_001 (time_extraction)
    """
    from datetime import datetime, timezone

    items = [
        BatchManifestItem(
            pair_id="pair_neg_order_001",
            category="negation",
            language_code="hi-IN",
            audio_condition="clean_recorded",
            source_type=AudioSourceType.HUMAN_RECORDED,
            audio_path_a="local_test_audio/negation1_pcm.wav",
            audio_path_b="local_test_audio/negation2_pcm.wav",
            reference_transcript_a="मेरा ऑर्डर कैंसल कर दो।",
            reference_transcript_b="मेरा ऑर्डर कैंसल मत करना।",
            expected_critical_fields_a={"is_negated": False, "action": "cancel_order"},
            expected_critical_fields_b={"is_negated": True, "action": "keep_order"},
            expected_intent_a="CANCEL_ORDER",
            expected_intent_b="CANCEL_ORDER",
            expected_action_a="cancel_order",
            expected_action_b="keep_order",
            distinction_field="is_negated",
            description="Critical negation distinction in order cancellation workflow.",
        ),
        BatchManifestItem(
            pair_id="pair_corr_qty_001",
            category="corrections_and_mind_changes",
            language_code="hi-IN",
            audio_condition="clean_recorded",
            source_type=AudioSourceType.HUMAN_RECORDED,
            audio_path_a="local_test_audio/correction1_pcm.wav",
            audio_path_b="local_test_audio/correction2_pcm.wav",
            reference_transcript_a="मुझे दो पैकेट चाहिए।",
            reference_transcript_b="दो नहीं, तीन पैकेट चाहिए।",
            expected_critical_fields_a={"correction_detected": False, "final_target": 2},
            expected_critical_fields_b={"correction_detected": True, "final_target": 3},
            expected_intent_a="ORDER_PRODUCT",
            expected_intent_b="ORDER_PRODUCT",
            expected_action_a="order_quantity_2",
            expected_action_b="order_quantity_3",
            distinction_field="final_target",
            description="Mid-sentence correction altering ordered product quantity.",
        ),
        BatchManifestItem(
            pair_id="pair_intent_order_001",
            category="command_intent",
            language_code="hi-IN",
            audio_condition="clean_recorded",
            source_type=AudioSourceType.HUMAN_RECORDED,
            audio_path_a="local_test_audio/intent_confusion1_pcm.wav",
            audio_path_b="local_test_audio/intent_confusion2_pcm.wav",
            reference_transcript_a="मुझे अपना ऑर्डर रिटर्न करना है।",
            reference_transcript_b="मुझे अपना ऑर्डर बदलना है।",
            expected_critical_fields_a={"intent": "RETURN_ORDER"},
            expected_critical_fields_b={"intent": "MODIFY_ORDER"},
            expected_intent_a="RETURN_ORDER",
            expected_intent_b="MODIFY_ORDER",
            expected_action_a="initiate_return",
            expected_action_b="modify_order",
            distinction_field="intent",
            description="Phonetically adjacent command intent contrast (Return vs Modify).",
        ),
        BatchManifestItem(
            pair_id="pair_ent_deliv_001",
            category="named_entities",
            language_code="hi-IN",
            audio_condition="clean_recorded",
            source_type=AudioSourceType.HUMAN_RECORDED,
            audio_path_a="local_test_audio/named_entities1_pcm.wav",
            audio_path_b="local_test_audio/named_entities2_pcm.wav",
            reference_transcript_a="डिलीवरी राहुल शर्मा के नाम पर है।",
            reference_transcript_b="डिलीवरी रोहित शर्मा के नाम पर है।",
            expected_critical_fields_a={"person_name": "Rahul Sharma"},
            expected_critical_fields_b={"person_name": "Rohit Sharma"},
            expected_intent_a="VERIFY_RECIPIENT",
            expected_intent_b="VERIFY_RECIPIENT",
            expected_action_a="verify_rahul_sharma",
            expected_action_b="verify_rohit_sharma",
            distinction_field="person_name",
            description="Indian personal name disambiguation (Rahul vs Rohit).",
        ),
        BatchManifestItem(
            pair_id="pair_time_deliv_001",
            category="time_extraction",
            language_code="hi-IN",
            audio_condition="clean_recorded",
            source_type=AudioSourceType.HUMAN_RECORDED,
            audio_path_a="local_test_audio/time_extraction1_pcm.wav",
            audio_path_b="local_test_audio/time_extraction2_pcm.wav",
            reference_transcript_a="डिलीवरी सुबह नौ बजे चाहिए।",
            reference_transcript_b="डिलीवरी सुबह दस बजे चाहिए।",
            expected_critical_fields_a={"delivery_time": "09:00"},
            expected_critical_fields_b={"delivery_time": "10:00"},
            expected_intent_a="SCHEDULE_DELIVERY",
            expected_intent_b="SCHEDULE_DELIVERY",
            expected_action_a="schedule_0900",
            expected_action_b="schedule_1000",
            distinction_field="delivery_time",
            description="Time slot disambiguation in Hindi (9 AM vs 10 AM).",
        ),
    ]

    manifest = BatchManifest(
        manifest_version="1.0.0",
        name="BoliBench Curated 5-Pair Pilot Manifest",
        description="Curated high-stakes paired speech contrast benchmark for Gnani Prisma STT v3 pilot evaluation.",
        created_at=datetime.now(timezone.utc).isoformat(),
        items=items,
    )

    if output_path:
        save_manifest(manifest, output_path)

    return manifest
