"""Storage manager and integrity service for BoliBench Human Recording Studio.

Maintains dedicated storage in human_recordings_v1/:
- Preserves raw uploaded recordings unmodified.
- Stores standardized 16kHz mono PCM WAV copies.
- Persists JSON metadata records and global manifest.
- Provides atomic writes, collision avoidance, and path-traversal prevention.
- Manages contrastive A/B pairs with mandatory semantic distinction verification.
- Completely isolated from synthetic benchmark manifests and SQLite evaluation runs.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from app.recording_schemas import (
    AudioDiagnostics,
    BenchmarkCategory,
    ContrastPairVerification,
    DatasetClassification,
    RecordingLanguage,
    RecordingMetadata,
    UtteranceRole,
    ValidationStatus,
)
from app.services.recording_processor import AudioProcessingError, HumanAudioProcessor

logger = logging.getLogger("bolibench.recording_storage")

SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")
ALLOWED_EXTENSIONS = {".webm", ".ogg", ".wav", ".mp3", ".m4a", ".flac", ".aac"}

TEST_SPEAKER_PREFIXES = (
    "spk_mock",
    "spk_test",
    "spk_fail",
    "spk_duration_test",
    "spk_acc_",
    "spk_isolated",
    "spk_enterprise_01",
    "spk_trans_sep",
)

TEST_PAIR_PREFIXES = (
    "human_test_",
    "human_mock_",
    "human_eval_mock_",
    "human_acc_pair_",
    "human_bus_test_",
    "human_live_fail_",
)


def determine_recording_classification(
    speaker_id: Any,
    pair_id: Optional[str] = None,
    is_transcript_verified: bool = False,
    consent_given: bool = True,
    diagnostics: Optional[AudioDiagnostics] = None,
    explicit_classification: Optional[str] = None,
    trusted_speakers: Optional[Set[str]] = None,
) -> str:
    """Classifies a recording as human_verified, synthetic, test_artifact, or unverified.

    Rules:
    - If explicit_classification is provided and valid, it is honored.
    - If speaker_id or pair_id matches automated test prefixes, classified as test_artifact.
    - If trusted_speakers is configured, speaker must be registered in the trusted registry.
    - If consent is given, transcript is manually verified, and audio diagnostics show clean decoding
      without silence or severe clipping, classified as human_verified.
    - Otherwise classified as unverified (awaiting verification or incomplete).
    """
    if hasattr(speaker_id, "speaker_id"):
        meta = speaker_id
        speaker_id_val = meta.speaker_id
        pair_id = pair_id or meta.pair_id
        is_transcript_verified = is_transcript_verified or meta.is_transcript_verified
        consent_given = consent_given if consent_given is not True else meta.consent_given
        diagnostics = diagnostics or meta.diagnostics
        explicit_classification = explicit_classification or getattr(meta, "classification", None)
    elif isinstance(speaker_id, dict):
        d = speaker_id
        speaker_id_val = d.get("speaker_id", "")
        pair_id = pair_id or d.get("pair_id")
        is_transcript_verified = is_transcript_verified or d.get("is_transcript_verified", False)
        consent_given = consent_given if consent_given is not True else d.get("consent_given", True)
        diagnostics = diagnostics or d.get("diagnostics")
        explicit_classification = explicit_classification or d.get("classification")
    else:
        speaker_id_val = str(speaker_id or "")

    if explicit_classification and explicit_classification in {c.value for c in DatasetClassification}:
        if explicit_classification != DatasetClassification.UNVERIFIED.value:
            return explicit_classification

    clean_speaker = speaker_id_val.lower()
    clean_pair = (pair_id or "").lower()

    if any(clean_speaker.startswith(p) for p in TEST_SPEAKER_PREFIXES) or any(clean_pair.startswith(p) for p in TEST_PAIR_PREFIXES):
        return DatasetClassification.TEST_ARTIFACT.value

    # Trusted speaker check if registry is active
    if trusted_speakers is not None and clean_speaker not in {s.lower() for s in trusted_speakers}:
        return DatasetClassification.UNVERIFIED.value

    decoding_ok = diagnostics.decoding_successful if diagnostics else True
    not_silent = not diagnostics.is_silent if diagnostics else True
    not_clipped = not diagnostics.is_clipped if diagnostics else True

    if is_transcript_verified and consent_given and decoding_ok and not_silent and not_clipped:
        return DatasetClassification.HUMAN_VERIFIED.value

    return DatasetClassification.UNVERIFIED.value


def determine_pair_classification(
    pair_data: Dict[str, Any],
    rec_a_classification: Optional[str] = None,
    rec_b_classification: Optional[str] = None,
    explicit_classification: Optional[str] = None,
    trusted_speakers: Optional[Set[str]] = None,
) -> str:
    """Classifies a contrastive pair as human_verified, synthetic, test_artifact, or unverified."""
    if explicit_classification and explicit_classification in {c.value for c in DatasetClassification}:
        return explicit_classification

    pid = (pair_data.get("pair_id") or "").lower()
    spk = (pair_data.get("speaker_id") or "").lower()

    if (
        any(pid.startswith(p) for p in TEST_PAIR_PREFIXES)
        or any(spk.startswith(p) for p in TEST_SPEAKER_PREFIXES)
        or rec_a_classification == DatasetClassification.TEST_ARTIFACT.value
        or rec_b_classification == DatasetClassification.TEST_ARTIFACT.value
    ):
        return DatasetClassification.TEST_ARTIFACT.value

    if trusted_speakers is not None and spk and spk not in {s.lower() for s in trusted_speakers}:
        return DatasetClassification.UNVERIFIED.value

    is_verified_pair = pair_data.get("is_verified_pair", False) or pair_data.get("verified", False)
    confirmed_dist = pair_data.get("confirmed_semantic_distinction", False) or pair_data.get("verified", False)
    has_both = bool(pair_data.get("utterance_a_recording_id") and pair_data.get("utterance_b_recording_id")) or pair_data.get("status") == "ready" or pair_data.get("verified", False)

    recs_ok = True
    if rec_a_classification is not None or rec_b_classification is not None:
        recs_ok = (
            rec_a_classification == DatasetClassification.HUMAN_VERIFIED.value
            and rec_b_classification == DatasetClassification.HUMAN_VERIFIED.value
        )

    if has_both and is_verified_pair and confirmed_dist and recs_ok:
        return DatasetClassification.HUMAN_VERIFIED.value

    return DatasetClassification.UNVERIFIED.value


class StorageSecurityError(Exception):
    """Raised when an invalid identifier or path traversal is detected."""
    pass


class StorageCollisionError(Exception):
    """Raised when a recording ID already exists."""
    pass


class StorageValidationError(Exception):
    """Raised when metadata or upload requirements fail validation."""
    pass


DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().parent.parent.parent / "human_recordings_v1"


class RecordingStorage:
    """Thread-safe storage manager for human voice recordings and contrast pairs."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # Default to human_recordings_v1 in project root
            self.base_dir = Path(DEFAULT_RECORDINGS_DIR).resolve()
        else:
            self.base_dir = Path(base_dir).resolve()

        self.raw_dir = self.base_dir / "raw"
        self.standardized_dir = self.base_dir / "standardized_16k"
        self.metadata_dir = self.base_dir / "metadata"
        self.pairs_dir = self.base_dir / "pairs"
        self.idempotency_dir = self.base_dir / "idempotency"
        self.manifest_file = self.base_dir / "manifest.json"
        self.trusted_speakers_file = self.base_dir / "trusted_speakers.json"

        self._ensure_directories()
        self._lock = threading.RLock()
        self.processor = HumanAudioProcessor()

        self._trusted_speakers: Set[str] = set()
        if self.trusted_speakers_file.exists():
            try:
                with open(self.trusted_speakers_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._trusted_speakers = set(data)
                elif isinstance(data, dict):
                    self._trusted_speakers = set(data.get("speakers", []))
                else:
                    logger.warning("trusted_speakers.json has unexpected structure; defaulting to empty registry.")
                    self._trusted_speakers = set()
            except Exception as e:
                logger.warning("Could not load trusted_speakers.json (%s); failing closed to empty registry.", str(e))
                self._trusted_speakers = set()

    def register_trusted_speaker(self, speaker_id: str) -> None:
        """Registers a verified human speaker in the trusted speaker registry."""
        clean_speaker = self.sanitize_id(speaker_id, "speaker_id")
        if any(clean_speaker.lower().startswith(p) for p in TEST_SPEAKER_PREFIXES):
            raise StorageValidationError(
                f"Cannot register test/mock speaker '{clean_speaker}' as a trusted human speaker."
            )
        with self._lock:
            self._trusted_speakers.add(clean_speaker)
            with open(self.trusted_speakers_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "speakers": sorted(list(self._trusted_speakers)),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    def is_speaker_trusted(self, speaker_id: str) -> bool:
        """Checks if a speaker is registered in the trusted speaker registry.
        Fails closed: returns False if registry is empty, or if speaker is unknown,
        empty, or matches test/mock prefixes.
        """
        clean_speaker = (speaker_id or "").strip()
        if not clean_speaker:
            return False
        if any(clean_speaker.lower().startswith(p) for p in TEST_SPEAKER_PREFIXES):
            return False
        return clean_speaker in self._trusted_speakers

    def list_trusted_speakers(self) -> List[str]:
        """Lists all registered trusted speakers."""
        if self._trusted_speakers is None:
            return []
        return sorted(list(self._trusted_speakers))

    def _ensure_directories(self) -> None:
        """Creates necessary storage directory hierarchy."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.standardized_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.pairs_dir.mkdir(parents=True, exist_ok=True)
        self.idempotency_dir.mkdir(parents=True, exist_ok=True)

        if not self.manifest_file.exists():
            initial_manifest = {
                "format_version": "1.0.0",
                "storage_id": "human_recordings_v1",
                "description": "BoliBench Human Speech Evaluation Dataset (Acoustic Manifest)",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "total_recordings": 0,
                "total_valid_50s": 0,
                "recordings": [],
                "pairs": [],
            }
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump(initial_manifest, f, indent=2, ensure_ascii=False)

    def sanitize_id(self, identifier: str, field_name: str = "ID") -> str:
        """Sanitizes and enforces path safety on identifiers."""
        if not identifier or not isinstance(identifier, str):
            raise StorageSecurityError(f"{field_name} must be a non-empty string.")

        clean_id = identifier.strip()
        if not SAFE_ID_REGEX.match(clean_id):
            raise StorageSecurityError(
                f"Invalid {field_name} '{identifier}'. Must contain only alphanumeric characters, dashes, and underscores."
            )

        if len(clean_id) < 3 or len(clean_id) > 128:
            raise StorageSecurityError(f"{field_name} length must be between 3 and 128 characters.")

        return clean_id

    def generate_recording_id(self, speaker_id: str) -> str:
        """Generates a collision-resistant unique recording ID."""
        import uuid
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        short_uuid = uuid.uuid4().hex[:8]
        clean_speaker = re.sub(r"[^a-zA-Z0-9]", "", speaker_id)[:12] or "anon"
        candidate = f"rec_{date_str}_{clean_speaker}_{short_uuid}"

        # Ensure no collision
        if (self.metadata_dir / f"{candidate}.json").exists():
            return f"{candidate}_{uuid.uuid4().hex[:4]}"
        return candidate

    def save_recording(
        self,
        raw_bytes: bytes,
        original_filename: str,
        speaker_id: str,
        language: str,
        category: str,
        reference_transcript: str,
        consent_given: bool,
        consent_statement: str,
        recording_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        pair_id: Optional[str] = None,
        utterance_role: str = "standalone",
        verified_transcript: Optional[str] = None,
        is_transcript_verified: bool = False,
        client_duration_seconds: Optional[float] = None,
        idempotency_key: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> RecordingMetadata:
        """Ingests, standardizes, validates, and stores a human audio recording.

        Safety:
        - Atomic staging prevents corrupt file writes from interrupted uploads.
        - Enforces explicit consent.
        - Preserves original raw audio unmodified.
        - Generates 16kHz mono PCM WAV.
        - Performs acoustic validation (silence, clipping, duration, decoding).
        - Enforces upload idempotency using callers' idempotency keys.
        """
        # 1. Enforce explicit consent
        if not consent_given:
            raise StorageValidationError("Explicit consent is required to record and store audio for evaluation.")

        with self._lock:
            # 2. Validate input strings and identifiers
            clean_speaker_id = self.sanitize_id(speaker_id, "speaker_id")
            rec_id = self.sanitize_id(recording_id, "recording_id") if recording_id else self.generate_recording_id(clean_speaker_id)
            clean_scenario_id = self.sanitize_id(scenario_id, "scenario_id") if scenario_id else None
            clean_pair_id = self.sanitize_id(pair_id, "pair_id") if pair_id else None
            clean_idempotency_key = self.sanitize_id(idempotency_key, "idempotency_key") if idempotency_key else None

            # Validate category
            valid_cats = {c.value for c in BenchmarkCategory}
            if category not in valid_cats:
                raise StorageValidationError(
                    f"Invalid benchmark category '{category}'. Must be one of: {', '.join(sorted(valid_cats))}"
                )

            # Validate language
            valid_langs = {l.value for l in RecordingLanguage}
            if language not in valid_langs:
                raise StorageValidationError(
                    f"Invalid language '{language}'. Must be one of: {', '.join(sorted(valid_langs))}"
                )

            # Validate audio content
            if not raw_bytes or len(raw_bytes) < 64:
                raise StorageValidationError("Audio payload is empty or too short.")

            # Compute content SHA-256 and payload signature for idempotency
            content_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            payload_elements = [
                content_sha256,
                clean_speaker_id,
                language,
                category,
                reference_transcript.strip(),
                clean_pair_id or "",
                utterance_role,
            ]
            payload_signature = hashlib.sha256(":".join(payload_elements).encode("utf-8")).hexdigest()

            # Check idempotency record if key was provided
            if clean_idempotency_key:
                idem_file = self.idempotency_dir / f"{clean_idempotency_key}.json"
                if idem_file.exists():
                    try:
                        with open(idem_file, "r", encoding="utf-8") as f:
                            idem_data = json.load(f)
                        if idem_data.get("payload_hash") == payload_signature:
                            existing_rec_id = idem_data.get("recording_id")
                            existing_rec = self.get_recording(existing_rec_id)
                            if existing_rec:
                                logger.info(
                                    "Idempotent hit for key '%s' -> returning existing recording '%s'",
                                    clean_idempotency_key,
                                    existing_rec_id,
                                )
                                return existing_rec
                        # Payload mismatch with existing key -> conflict
                        raise StorageCollisionError(
                            f"Idempotency key '{clean_idempotency_key}' was previously used with a different upload payload or parameters."
                        )
                    except StorageCollisionError:
                        raise
                    except Exception as e:
                        logger.error("Error reading idempotency record for %s: %s", clean_idempotency_key, str(e))

            # Idempotency & collision check on recording_id:
            if (self.metadata_dir / f"{rec_id}.json").exists():
                try:
                    with open(self.metadata_dir / f"{rec_id}.json", "r", encoding="utf-8") as f:
                        existing_meta = json.load(f)
                    if clean_pair_id and existing_meta.get("pair_id") == clean_pair_id and existing_meta.get("utterance_role") == utterance_role:
                        logger.info("Idempotent update for recording '%s' (pair '%s', role '%s')", rec_id, clean_pair_id, utterance_role)
                    elif not recording_id:
                        import uuid
                        rec_id = f"{rec_id}_{uuid.uuid4().hex[:4]}"
                    else:
                        raise StorageCollisionError(f"Recording ID '{rec_id}' already exists in human_recordings_v1.")
                except StorageCollisionError:
                    raise
                except Exception:
                    raise StorageCollisionError(f"Recording ID '{rec_id}' already exists in human_recordings_v1.")

            # Extract and validate file extension
            raw_ext = Path(original_filename).suffix.lower() if original_filename else ".webm"
            if raw_ext not in ALLOWED_EXTENSIONS:
                raw_ext = ".webm"

            raw_filename = f"{rec_id}_raw{raw_ext}"
            std_filename = f"{rec_id}.wav"

            raw_path = self.raw_dir / raw_filename
            std_path = self.standardized_dir / std_filename
            temp_raw_path = self.base_dir / f".tmp_{rec_id}{raw_ext}"

            # 3. Atomic upload staging (prevents corrupt files on interrupted upload)
            try:
                with open(temp_raw_path, "wb") as f:
                    f.write(raw_bytes)

                # Move atomically to raw directory
                temp_raw_path.replace(raw_path)
            except Exception as e:
                if temp_raw_path.exists():
                    temp_raw_path.unlink()
                raise StorageValidationError(f"Failed to write audio payload to disk: {str(e)}")

            # 4. Standardize audio to 16 kHz mono PCM WAV
            try:
                self.processor.convert_to_standard_wav(input_path=raw_path, output_wav_path=std_path)
            except AudioProcessingError as e:
                # Clean up raw file on conversion failure
                if raw_path.exists():
                    raw_path.unlink()
                raise StorageValidationError(f"Audio normalization failed: {str(e)}")

            # 5. Acoustic & Duration Validation
            try:
                diagnostics, validation_status = self.processor.analyze_standard_wav(
                    wav_path=std_path, client_duration=client_duration_seconds
                )
            except Exception as e:
                diagnostics = None
                validation_status = ValidationStatus.INVALID_FORMAT
                logger.error("Acoustic validation failed for %s: %s", rec_id, str(e))

            now_iso = datetime.now(timezone.utc).isoformat()

            # Parse role
            role_enum = UtteranceRole.STANDALONE
            if utterance_role == "utterance_a":
                role_enum = UtteranceRole.UTTERANCE_A
            elif utterance_role == "utterance_b":
                role_enum = UtteranceRole.UTTERANCE_B

            rec_classification = determine_recording_classification(
                speaker_id=clean_speaker_id,
                pair_id=clean_pair_id,
                is_transcript_verified=bool(is_transcript_verified),
                consent_given=True,
                diagnostics=diagnostics,
                explicit_classification=classification,
                trusted_speakers=self._trusted_speakers,
            )

            # 6. Assemble Metadata
            metadata = RecordingMetadata(
                recording_id=rec_id,
                speaker_id=clean_speaker_id,
                language=language,
                category=category,
                scenario_id=clean_scenario_id,
                pair_id=clean_pair_id,
                utterance_role=role_enum,
                reference_transcript=reference_transcript.strip(),
                verified_transcript=verified_transcript.strip() if verified_transcript else None,
                is_transcript_verified=bool(is_transcript_verified),
                consent_given=True,
                consent_statement=consent_statement.strip(),
                consent_timestamp=now_iso,
                created_at=now_iso,
                client_duration_seconds=client_duration_seconds,
                diagnostics=diagnostics,
                validation_status=validation_status,
                raw_audio_filename=raw_filename,
                standardized_audio_filename=std_filename,
                raw_audio_path=f"raw/{raw_filename}",
                standardized_audio_path=f"standardized_16k/{std_filename}",
                idempotency_key=clean_idempotency_key,
                content_sha256=content_sha256,
                classification=rec_classification,
            )

            # 7. Write metadata JSON
            meta_file = self.metadata_dir / f"{rec_id}.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)

            # 8. Thread-safe manifest update
            self._update_manifest(metadata)

            # 9. Sync recording into pairs directory
            if clean_pair_id:
                self._sync_recording_to_pair(metadata)

            # 10. Persist idempotency mapping if key was provided
            if clean_idempotency_key:
                idem_file = self.idempotency_dir / f"{clean_idempotency_key}.json"
                tmp_idem = self.idempotency_dir / f".tmp_{clean_idempotency_key}.json"
                idem_record = {
                    "idempotency_key": clean_idempotency_key,
                    "payload_hash": payload_signature,
                    "content_sha256": content_sha256,
                    "recording_id": rec_id,
                    "created_at": now_iso,
                }
                with open(tmp_idem, "w", encoding="utf-8") as f:
                    json.dump(idem_record, f, indent=2)
                tmp_idem.replace(idem_file)

            return metadata

    def _sync_recording_to_pair(self, metadata: RecordingMetadata) -> None:
        """Synchronizes an uploaded recording into its contrastive pair record in pairs/."""
        if not metadata.pair_id:
            return
        pid = metadata.pair_id
        pair_file = self.pairs_dir / f"{pid}.json"
        now_iso = datetime.now(timezone.utc).isoformat()

        pair_data = {}
        if pair_file.exists():
            try:
                with open(pair_file, "r", encoding="utf-8") as f:
                    pair_data = json.load(f)
            except Exception:
                pair_data = {}

        pair_data.setdefault("pair_id", pid)
        pair_data.setdefault("category", metadata.category)
        pair_data.setdefault("language", metadata.language)
        pair_data.setdefault("speaker_id", metadata.speaker_id)
        pair_data.setdefault("created_at", metadata.created_at)
        pair_data["updated_at"] = now_iso

        transcript = metadata.verified_transcript or metadata.reference_transcript
        if metadata.utterance_role == UtteranceRole.UTTERANCE_A:
            pair_data["utterance_a_recording_id"] = metadata.recording_id
            pair_data["utterance_a_transcript"] = transcript
        elif metadata.utterance_role == UtteranceRole.UTTERANCE_B:
            pair_data["utterance_b_recording_id"] = metadata.recording_id
            pair_data["utterance_b_transcript"] = transcript

        has_a = bool(pair_data.get("utterance_a_recording_id"))
        has_b = bool(pair_data.get("utterance_b_recording_id"))

        # Check transcript verification
        rec_a = self.get_recording(pair_data["utterance_a_recording_id"]) if has_a else None
        rec_b = self.get_recording(pair_data["utterance_b_recording_id"]) if has_b else None
        a_verified = rec_a.is_transcript_verified if rec_a else False
        b_verified = rec_b.is_transcript_verified if rec_b else False

        if has_a and has_b:
            txt_a = (pair_data.get("utterance_a_transcript") or "").strip().lower()
            txt_b = (pair_data.get("utterance_b_transcript") or "").strip().lower()
            if a_verified and b_verified:
                if txt_a != txt_b and txt_a and txt_b:
                    pair_data["is_verified_pair"] = True
                    pair_data["confirmed_semantic_distinction"] = True
                    if not pair_data.get("is_evaluated"):
                        pair_data["status"] = "ready_for_evaluation"
                else:
                    pair_data["status"] = "awaiting_verification"
            else:
                pair_data["status"] = "awaiting_verification"
        else:
            pair_data["status"] = "incomplete"

        # Determine and persist pair classification
        rec_a_class = rec_a.classification if rec_a else None
        rec_b_class = rec_b.classification if rec_b else None
        pair_data["classification"] = determine_pair_classification(
            pair_data=pair_data,
            rec_a_classification=rec_a_class,
            rec_b_classification=rec_b_class,
        )

        with open(pair_file, "w", encoding="utf-8") as f:
            json.dump(pair_data, f, indent=2, ensure_ascii=False)

    def _update_manifest(self, metadata: RecordingMetadata) -> None:
        """Appends new recording record to global manifest with thread lock."""
        with self._lock:
            manifest_data = self._read_manifest_unsafe()
            recordings_list = manifest_data.get("recordings", [])

            # Check if record already exists in list (update if so)
            existing_idx = next((i for i, r in enumerate(recordings_list) if r.get("recording_id") == metadata.recording_id), -1)
            summary_entry = {
                "recording_id": metadata.recording_id,
                "speaker_id": metadata.speaker_id,
                "language": metadata.language,
                "category": metadata.category,
                "pair_id": metadata.pair_id,
                "utterance_role": metadata.utterance_role.value,
                "reference_transcript": metadata.reference_transcript,
                "verified_transcript": metadata.verified_transcript,
                "duration_seconds": metadata.diagnostics.duration_seconds if metadata.diagnostics else 0.0,
                "is_complete_50s": metadata.diagnostics.is_complete_50s if metadata.diagnostics else False,
                "validation_status": metadata.validation_status.value,
                "is_transcript_verified": metadata.is_transcript_verified,
                "classification": metadata.classification,
                "created_at": metadata.created_at,
            }

            if existing_idx >= 0:
                recordings_list[existing_idx] = summary_entry
            else:
                recordings_list.append(summary_entry)

            manifest_data["recordings"] = recordings_list
            manifest_data["total_recordings"] = len(recordings_list)
            manifest_data["total_valid_50s"] = sum(1 for r in recordings_list if r.get("is_complete_50s"))
            manifest_data["updated_at"] = datetime.now(timezone.utc).isoformat()

            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    def _read_manifest_unsafe(self) -> Dict[str, Any]:
        """Reads manifest file without locking (caller must hold self._lock)."""
        if not self.manifest_file.exists():
            self._ensure_directories()
        with open(self.manifest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_recording(self, recording_id: str) -> Optional[RecordingMetadata]:
        """Retrieves complete metadata for an individual recording."""
        clean_id = self.sanitize_id(recording_id, "recording_id")
        meta_file = self.metadata_dir / f"{clean_id}.json"
        if not meta_file.exists():
            return None
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "classification" not in data:
            data["classification"] = determine_recording_classification(
                speaker_id=data.get("speaker_id", ""),
                pair_id=data.get("pair_id"),
                is_transcript_verified=data.get("is_transcript_verified", False),
                consent_given=data.get("consent_given", True),
                diagnostics=AudioDiagnostics(**data["diagnostics"]) if data.get("diagnostics") else None,
                trusted_speakers=self._trusted_speakers,
            )
        return RecordingMetadata(**data)

    def list_recordings(
        self,
        category: Optional[str] = None,
        speaker_id: Optional[str] = None,
        pair_id: Optional[str] = None,
        is_complete_50s: Optional[bool] = None,
        classification: Optional[str] = None,
        exclude_test_artifacts: bool = False,
    ) -> List[Dict[str, Any]]:
        """Queries recorded human audio library with optional filters."""
        with self._lock:
            manifest_data = self._read_manifest_unsafe()
            records = manifest_data.get("recordings", [])

        filtered = []
        for r in records:
            if "classification" not in r:
                r["classification"] = determine_recording_classification(
                    speaker_id=r.get("speaker_id", ""),
                    pair_id=r.get("pair_id"),
                    is_transcript_verified=r.get("is_transcript_verified", False),
                    consent_given=r.get("consent_given", True),
                    trusted_speakers=self._trusted_speakers,
                )
            if classification and r.get("classification") != classification:
                continue
            if exclude_test_artifacts and r.get("classification") == DatasetClassification.TEST_ARTIFACT.value:
                continue
            if category and r.get("category") != category:
                continue
            if speaker_id and r.get("speaker_id") != speaker_id:
                continue
            if pair_id and r.get("pair_id") != pair_id:
                continue
            if is_complete_50s is not None and r.get("is_complete_50s") != is_complete_50s:
                continue
            # Backfill transcripts from metadata json if older manifest entry lacked them
            if not r.get("reference_transcript"):
                rec_meta = self.get_recording(r.get("recording_id", ""))
                if rec_meta:
                    r["reference_transcript"] = rec_meta.reference_transcript
                    r["verified_transcript"] = rec_meta.verified_transcript
            filtered.append(r)

        # Return latest recordings first
        filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return filtered

    def get_audio_file_path(self, recording_id: str, raw: bool = False) -> Path:
        """Returns the absolute path to the requested audio file with path traversal check."""
        clean_id = self.sanitize_id(recording_id, "recording_id")
        meta = self.get_recording(clean_id)
        if not meta:
            raise FileNotFoundError(f"Recording '{recording_id}' not found.")

        if raw:
            target_path = (self.base_dir / meta.raw_audio_path).resolve()
        else:
            target_path = (self.base_dir / meta.standardized_audio_path).resolve()

        # Enforce that target_path strictly stays inside self.base_dir
        if not str(target_path).startswith(str(self.base_dir)):
            raise StorageSecurityError("Illegal path traversal detected.")

        if not target_path.exists():
            raise FileNotFoundError(f"Audio file not found at {target_path}")

        return target_path

    def create_pair_definition(
        self,
        pair_id: str,
        category: str,
        language: str,
        speaker_id: str,
        scenario_description: Optional[str] = None,
        expected_distinction: Optional[str] = None,
        distinction_field: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explicitly initializes or creates a candidate contrast pair definition in pairs/."""
        clean_pair_id = self.sanitize_id(pair_id, "pair_id")
        clean_speaker_id = self.sanitize_id(speaker_id, "speaker_id")
        now_iso = datetime.now(timezone.utc).isoformat()

        pair_file = self.pairs_dir / f"{clean_pair_id}.json"
        existing = {}
        if pair_file.exists():
            try:
                with open(pair_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        pair_data = {
            "pair_id": clean_pair_id,
            "category": category,
            "language": language,
            "speaker_id": clean_speaker_id,
            "scenario_description": scenario_description.strip() if scenario_description else existing.get("scenario_description"),
            "expected_distinction_desc": expected_distinction.strip() if expected_distinction else existing.get("expected_distinction_desc"),
            "distinction_field": distinction_field or existing.get("distinction_field"),
            "utterance_a_recording_id": existing.get("utterance_a_recording_id", ""),
            "utterance_b_recording_id": existing.get("utterance_b_recording_id", ""),
            "utterance_a_transcript": existing.get("utterance_a_transcript", ""),
            "utterance_b_transcript": existing.get("utterance_b_transcript", ""),
            "is_verified_pair": existing.get("is_verified_pair", False),
            "confirmed_semantic_distinction": existing.get("confirmed_semantic_distinction", False),
            "status": existing.get("status", "incomplete"),
            "last_run_id": existing.get("last_run_id"),
            "is_evaluated": existing.get("is_evaluated", False),
            "created_at": existing.get("created_at", now_iso),
            "updated_at": now_iso,
        }
        with open(pair_file, "w", encoding="utf-8") as f:
            json.dump(pair_data, f, indent=2, ensure_ascii=False)
        return pair_data

    def get_pair(self, pair_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete metadata, transcripts, and audio paths for a contrastive pair."""
        clean_pair_id = self.sanitize_id(pair_id, "pair_id")
        all_pairs = self.list_candidate_pairs()
        for p in all_pairs:
            if p.get("pair_id") == clean_pair_id:
                return p
        return None

    def update_pair_evaluation_status(self, pair_id: str, run_id: str, overall_status: str) -> None:
        """Updates evaluation status and references on a pair record."""
        clean_pair_id = self.sanitize_id(pair_id, "pair_id")
        pair_file = self.pairs_dir / f"{clean_pair_id}.json"
        now_iso = datetime.now(timezone.utc).isoformat()
        pair_data = {}
        if pair_file.exists():
            try:
                with open(pair_file, "r", encoding="utf-8") as f:
                    pair_data = json.load(f)
            except Exception:
                pair_data = {}

        pair_data["last_run_id"] = run_id
        pair_data["is_evaluated"] = True
        pair_data["status"] = "completed" if overall_status == "completed" else "provider_failure"
        pair_data["evaluated_at"] = now_iso
        pair_data["updated_at"] = now_iso

        with open(pair_file, "w", encoding="utf-8") as f:
            json.dump(pair_data, f, indent=2, ensure_ascii=False)

    def list_candidate_pairs(
        self,
        classification: Optional[str] = None,
        exclude_test_artifacts: bool = False,
        curated_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Lists contrastive pairs, aggregating pair definitions and audio recordings with optional classification filtering."""
        recordings = self.list_recordings()
        pairs_dict: Dict[str, Dict[str, Any]] = {}

        # 1. Discover pairs from pairs/ directory
        if self.pairs_dir.exists():
            for pfile in self.pairs_dir.glob("*.json"):
                try:
                    with open(pfile, "r", encoding="utf-8") as f:
                        pdata = json.load(f)
                        pid = pdata.get("pair_id")
                        if pid:
                            pairs_dict[pid] = {
                                "pair_id": pid,
                                "category": pdata.get("category"),
                                "language": pdata.get("language"),
                                "speaker_id": pdata.get("speaker_id"),
                                "scenario_description": pdata.get("scenario_description"),
                                "expected_distinction_desc": pdata.get("expected_distinction_desc"),
                                "distinction_field": pdata.get("distinction_field"),
                                "utterance_a": None,
                                "utterance_b": None,
                                "status": pdata.get("status", "incomplete"),
                                "last_run_id": pdata.get("last_run_id"),
                                "is_evaluated": pdata.get("is_evaluated", False),
                                "is_verified_pair": pdata.get("is_verified_pair", False),
                                "confirmed_semantic_distinction": pdata.get("confirmed_semantic_distinction", False),
                                "classification": pdata.get("classification"),
                                "verification": pdata,
                                "created_at": pdata.get("created_at"),
                            }
                except Exception:
                    pass

        # 2. Correlate recordings with pairs
        for r in recordings:
            pid = r.get("pair_id")
            if not pid:
                continue

            if pid not in pairs_dict:
                pairs_dict[pid] = {
                    "pair_id": pid,
                    "category": r.get("category"),
                    "language": r.get("language"),
                    "speaker_id": r.get("speaker_id"),
                    "scenario_description": None,
                    "expected_distinction_desc": None,
                    "distinction_field": None,
                    "utterance_a": None,
                    "utterance_b": None,
                    "status": "incomplete",
                    "last_run_id": None,
                    "is_evaluated": False,
                    "is_verified_pair": False,
                    "confirmed_semantic_distinction": False,
                    "classification": None,
                    "verification": None,
                    "created_at": r.get("created_at"),
                }

            role = r.get("utterance_role")
            rec_id = r.get("recording_id")
            rec_meta = self.get_recording(rec_id) if rec_id else None

            utt_data = {
                "recording_id": rec_id,
                "reference_transcript": rec_meta.reference_transcript if rec_meta else "",
                "verified_transcript": rec_meta.verified_transcript if rec_meta else "",
                "is_transcript_verified": rec_meta.is_transcript_verified if rec_meta else False,
                "duration_seconds": rec_meta.diagnostics.duration_seconds if rec_meta and rec_meta.diagnostics else r.get("duration_seconds", 0.0),
                "is_complete_50s": rec_meta.diagnostics.is_complete_50s if rec_meta and rec_meta.diagnostics else r.get("is_complete_50s", False),
                "classification": rec_meta.classification if rec_meta else r.get("classification", DatasetClassification.UNVERIFIED.value),
                "audio_url": f"/api/studio/recordings/{rec_id}/audio" if rec_id else "",
            }

            if role == "utterance_a" and pairs_dict[pid]["utterance_a"] is None:
                pairs_dict[pid]["utterance_a"] = utt_data
            elif role == "utterance_b" and pairs_dict[pid]["utterance_b"] is None:
                pairs_dict[pid]["utterance_b"] = utt_data

        # Fallback check: if pair JSON directly recorded utterance recording IDs, ensure they are populated
        for pid, pair_info in pairs_dict.items():
            pdata = pair_info.get("verification") or {}
            rec_a_id = pdata.get("utterance_a_recording_id")
            if pair_info["utterance_a"] is None and rec_a_id:
                rec_meta_a = self.get_recording(rec_a_id)
                if rec_meta_a:
                    pair_info["utterance_a"] = {
                        "recording_id": rec_meta_a.recording_id,
                        "reference_transcript": rec_meta_a.reference_transcript,
                        "verified_transcript": rec_meta_a.verified_transcript,
                        "is_transcript_verified": rec_meta_a.is_transcript_verified,
                        "duration_seconds": rec_meta_a.diagnostics.duration_seconds if rec_meta_a.diagnostics else 0.0,
                        "is_complete_50s": rec_meta_a.diagnostics.is_complete_50s if rec_meta_a.diagnostics else False,
                        "classification": rec_meta_a.classification,
                        "audio_url": f"/api/studio/recordings/{rec_meta_a.recording_id}/audio",
                    }
            rec_b_id = pdata.get("utterance_b_recording_id")
            if pair_info["utterance_b"] is None and rec_b_id:
                rec_meta_b = self.get_recording(rec_b_id)
                if rec_meta_b:
                    pair_info["utterance_b"] = {
                        "recording_id": rec_meta_b.recording_id,
                        "reference_transcript": rec_meta_b.reference_transcript,
                        "verified_transcript": rec_meta_b.verified_transcript,
                        "is_transcript_verified": rec_meta_b.is_transcript_verified,
                        "duration_seconds": rec_meta_b.diagnostics.duration_seconds if rec_meta_b.diagnostics else 0.0,
                        "is_complete_50s": rec_meta_b.diagnostics.is_complete_50s if rec_meta_b.diagnostics else False,
                        "classification": rec_meta_b.classification,
                        "audio_url": f"/api/studio/recordings/{rec_meta_b.recording_id}/audio",
                    }

        # 3. Finalize pair statuses and integrity
        for pid, pair_info in pairs_dict.items():
            has_a = pair_info["utterance_a"] is not None
            has_b = pair_info["utterance_b"] is not None
            pair_info["is_complete_pair"] = (has_a and has_b)

            a_verified = pair_info["utterance_a"]["is_transcript_verified"] if has_a else False
            b_verified = pair_info["utterance_b"]["is_transcript_verified"] if has_b else False

            txt_a = (pair_info["utterance_a"]["verified_transcript"] or pair_info["utterance_a"]["reference_transcript"] if has_a else "").strip().lower()
            txt_b = (pair_info["utterance_b"]["verified_transcript"] or pair_info["utterance_b"]["reference_transcript"] if has_b else "").strip().lower()

            if has_a and has_b:
                if not a_verified or not b_verified:
                    status = "awaiting_verification"
                elif txt_a == txt_b or not txt_a or not txt_b:
                    status = "awaiting_verification"
                else:
                    if pair_info.get("is_evaluated"):
                        status = pair_info.get("status") or "completed"
                    else:
                        status = "ready_for_evaluation"
            else:
                status = "incomplete"

            pair_info["status"] = status
            pair_info["is_ready_for_evaluation"] = (status in ("ready_for_evaluation", "completed"))
            pair_info["is_ready_for_verification"] = (has_a and has_b and a_verified and b_verified)

            # Determine / backfill classification if absent
            pdata = pair_info.get("verification") or {}
            pair_class = pdata.get("classification") or pair_info.get("classification")
            if not pair_class:
                rec_a_id = pair_info["utterance_a"]["recording_id"] if has_a else None
                rec_b_id = pair_info["utterance_b"]["recording_id"] if has_b else None
                rec_a_meta = self.get_recording(rec_a_id) if rec_a_id else None
                rec_b_meta = self.get_recording(rec_b_id) if rec_b_id else None
                pair_class = determine_pair_classification(
                    pair_data=pair_info,
                    rec_a_classification=rec_a_meta.classification if rec_a_meta else None,
                    rec_b_classification=rec_b_meta.classification if rec_b_meta else None,
                    trusted_speakers=self._trusted_speakers,
                )
            pair_info["classification"] = pair_class

        results = list(pairs_dict.values())

        # Apply classification and cohort filtering
        if classification:
            results = [p for p in results if p.get("classification") == classification]
        if curated_only:
            results = [p for p in results if p.get("classification") == DatasetClassification.HUMAN_VERIFIED.value]
        if exclude_test_artifacts:
            results = [p for p in results if p.get("classification") != DatasetClassification.TEST_ARTIFACT.value]

        results.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return results

    def verify_contrast_pair(
        self,
        pair_id: str,
        confirmed_semantic_distinction: bool,
        distinction_field: Optional[str] = None,
        expected_distinction_desc: Optional[str] = None,
        operator_notes: Optional[str] = None,
        verified_by: Optional[str] = None,
    ) -> ContrastPairVerification:
        """Verifies and seals a candidate contrastive pair.

        Strict Dataset Integrity Rules:
        - Both Utterance A and Utterance B must exist.
        - Both utterances must have manually verified transcripts.
        - Operator must explicitly confirm genuine semantic distinction.
        - Utterance A and B transcripts cannot be identical.
        - Does NOT modify existing synthetic manifests or evaluation tables.
        """
        clean_pair_id = self.sanitize_id(pair_id, "pair_id")

        if not confirmed_semantic_distinction:
            raise StorageValidationError(
                "Cannot verify pair: Operator must explicitly confirm that Utterance A and Utterance B test a genuine semantic distinction."
            )

        # Retrieve recordings for this pair
        recordings = self.list_recordings(pair_id=clean_pair_id)
        utt_a = next((r for r in recordings if r.get("utterance_role") == "utterance_a"), None)
        utt_b = next((r for r in recordings if r.get("utterance_role") == "utterance_b"), None)

        if not utt_a:
            raise StorageValidationError(f"Cannot verify pair '{clean_pair_id}': Missing Utterance A recording.")
        if not utt_b:
            raise StorageValidationError(f"Cannot verify pair '{clean_pair_id}': Missing Utterance B recording.")

        # Full metadata for detailed checks
        meta_a = self.get_recording(utt_a["recording_id"])
        meta_b = self.get_recording(utt_b["recording_id"])

        if not meta_a or not meta_b:
            raise StorageValidationError("Could not load complete recording metadata for pair utterances.")

        if not meta_a.is_transcript_verified:
            raise StorageValidationError("Utterance A transcript must be manually verified before pair verification.")
        if not meta_b.is_transcript_verified:
            raise StorageValidationError("Utterance B transcript must be manually verified before pair verification.")

        # Explicit consent requirement
        if not meta_a.consent_given or not meta_b.consent_given:
            raise StorageValidationError("Explicit consent must be confirmed for both utterances before pair verification.")

        # Audio quality and diagnostics verification
        for label, meta in [("Utterance A", meta_a), ("Utterance B", meta_b)]:
            if meta.diagnostics:
                if not meta.diagnostics.decoding_successful:
                    raise StorageValidationError(f"Cannot verify pair: {label} failed audio decoding.")
                if meta.diagnostics.is_silent:
                    raise StorageValidationError(f"Cannot verify pair: {label} audio is silent.")
                if meta.diagnostics.is_clipped:
                    raise StorageValidationError(f"Cannot verify pair: {label} audio suffers from digital clipping.")

        # Trusted speaker check: fail closed if trust cannot be established
        if not self.is_speaker_trusted(meta_a.speaker_id) or not self.is_speaker_trusted(meta_b.speaker_id):
            raise StorageValidationError("Cannot verify pair: Both speakers must be registered in the trusted speaker registry.")

        transcript_a = (meta_a.verified_transcript or meta_a.reference_transcript).strip().lower()
        transcript_b = (meta_b.verified_transcript or meta_b.reference_transcript).strip().lower()

        if transcript_a == transcript_b:
            raise StorageValidationError(
                "Contrast pair rejected: Utterance A and Utterance B have identical transcripts. A contrastive pair must test differing semantic values."
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        rec_a_class = determine_recording_classification(meta_a, trusted_speakers=self._trusted_speakers)
        rec_b_class = determine_recording_classification(meta_b, trusted_speakers=self._trusted_speakers)
        pair_class = determine_pair_classification(
            pair_data={
                "pair_id": clean_pair_id,
                "speaker_id": meta_a.speaker_id,
                "is_verified_pair": True,
                "confirmed_semantic_distinction": True,
                "utterance_a_recording_id": meta_a.recording_id,
                "utterance_b_recording_id": meta_b.recording_id,
            },
            rec_a_classification=rec_a_class,
            rec_b_classification=rec_b_class,
            trusted_speakers=self._trusted_speakers,
        )

        pair_record = ContrastPairVerification(
            pair_id=clean_pair_id,
            category=meta_a.category,
            language=meta_a.language,
            utterance_a_recording_id=meta_a.recording_id,
            utterance_b_recording_id=meta_b.recording_id,
            utterance_a_transcript=meta_a.verified_transcript or meta_a.reference_transcript,
            utterance_b_transcript=meta_b.verified_transcript or meta_b.reference_transcript,
            distinction_field=distinction_field,
            expected_distinction_desc=expected_distinction_desc,
            is_verified_pair=True,
            confirmed_semantic_distinction=True,
            classification=pair_class,
            operator_notes=operator_notes.strip() if operator_notes else None,
            verified_by=verified_by.strip() if verified_by else "operator",
            verified_at=now_iso,
            created_at=now_iso,
        )

        pair_file = self.pairs_dir / f"{clean_pair_id}.json"
        with open(pair_file, "w", encoding="utf-8") as f:
            json.dump(pair_record.model_dump(), f, indent=2, ensure_ascii=False)

        # Also update global manifest with verified pair
        with self._lock:
            manifest_data = self._read_manifest_unsafe()
            pairs_list = manifest_data.get("pairs", [])
            existing_idx = next((i for i, p in enumerate(pairs_list) if p.get("pair_id") == clean_pair_id), -1)
            pair_summary = {
                "pair_id": clean_pair_id,
                "category": pair_record.category,
                "language": pair_record.language,
                "utterance_a": pair_record.utterance_a_recording_id,
                "utterance_b": pair_record.utterance_b_recording_id,
                "is_verified": True,
                "verified_at": now_iso,
            }
            if existing_idx >= 0:
                pairs_list[existing_idx] = pair_summary
            else:
                pairs_list.append(pair_summary)
            manifest_data["pairs"] = pairs_list
            manifest_data["updated_at"] = now_iso
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        return pair_record
