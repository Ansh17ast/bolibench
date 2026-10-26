"""Dataset and Audio Generator for BoliBench 600-Pair Expanded Benchmark (v2.0.0).

Generates 600 unique pairs across 6 categories:
- Quantities and numbers (100 pairs)
- Negation (100 pairs)
- Corrections and mind changes (100 pairs)
- Command intent (100 pairs)
- Named entities (100 pairs)
- Time extraction (100 pairs)

Strict Safety:
- Zero provider requests (0 Gnani credits used).
- Offline no-cost Edge-TTS + ffmpeg PCM conversion.
- Full offline AudioValidator checks (16kHz, mono, 16-bit linear PCM).
- Outputs to benchmark_manifest_v2.0.0.json and benchmark_audio_v2/.
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import wave

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import edge_tts

from app.evaluation.batch_manifest import (
    AudioSourceType,
    BatchManifest,
    BatchManifestItem,
    save_manifest,
    validate_manifest,
)
from app.services.audio_validator import AudioValidator
from scripts.dataset_v2_specifications import generate_all_600_specifications

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_v2_benchmark")

BENCHMARK_AUDIO_V2_DIR = WORKSPACE_DIR / "benchmark_audio_v2"
MANIFEST_V2_PATH = WORKSPACE_DIR / "benchmark_manifest_v2.0.0.json"
DRY_RUN_REPORT_V2_PATH = WORKSPACE_DIR / "benchmark_v2_dry_run_report.json"

SEM = asyncio.Semaphore(10)


async def generate_single_tts(text: str, voice: str, out_wav_path: Path) -> None:
    """Generates speech via Edge-TTS and converts to 16 kHz mono linear PCM WAV via ffmpeg."""
    async with SEM:
        temp_mp3 = out_wav_path.with_suffix(".temp.mp3")
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(temp_mp3))

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_mp3),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(out_wav_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        finally:
            if temp_mp3.exists():
                temp_mp3.unlink()


def apply_acoustic_transformation(in_wav: Path, out_wav: Path, filter_params: str) -> None:
    """Applies ffmpeg audio transformation filter."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_wav),
        "-filter:a",
        filter_params,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


async def build_v2_dataset() -> None:
    BENCHMARK_AUDIO_V2_DIR.mkdir(parents=True, exist_ok=True)
    validator = AudioValidator()

    specs = generate_all_600_specifications()
    logger.info(f"Loaded {len(specs)} pair specifications for BoliBench v2.0.0.")

    manifest_items: List[BatchManifestItem] = []
    tts_tasks = []

    # First pass: identify missing TTS audio files to generate
    for spec in specs:
        pair_id = spec["pair_id"]
        source_type = spec["source_type"]
        wav_a = BENCHMARK_AUDIO_V2_DIR / f"{pair_id}_a.wav"
        wav_b = BENCHMARK_AUDIO_V2_DIR / f"{pair_id}_b.wav"

        if source_type == AudioSourceType.TTS_SYNTHETIC:
            voice = spec["voice"]
            if not wav_a.exists():
                tts_tasks.append(generate_single_tts(spec["reference_transcript_a"], voice, wav_a))
            if not wav_b.exists():
                tts_tasks.append(generate_single_tts(spec["reference_transcript_b"], voice, wav_b))

    if tts_tasks:
        logger.info(f"Generating {len(tts_tasks)} TTS audio files with concurrency={SEM._value}...")
        t0 = time.time()
        await asyncio.gather(*tts_tasks)
        logger.info(f"Generated {len(tts_tasks)} audio files in {time.time() - t0:.2f}s.")
    else:
        logger.info("All TTS synthetic audio files already exist.")

    # Second pass: apply acoustic transformations (which depend on parents)
    transformed_count = 0
    for spec in specs:
        if spec["source_type"] == AudioSourceType.ACOUSTIC_TRANSFORMED:
            pair_id = spec["pair_id"]
            parent_id = spec["source_pair_id"]
            wav_a = BENCHMARK_AUDIO_V2_DIR / f"{pair_id}_a.wav"
            wav_b = BENCHMARK_AUDIO_V2_DIR / f"{pair_id}_b.wav"
            parent_a = BENCHMARK_AUDIO_V2_DIR / f"{parent_id}_a.wav"
            parent_b = BENCHMARK_AUDIO_V2_DIR / f"{parent_id}_b.wav"

            if not parent_a.exists() or not parent_b.exists():
                raise FileNotFoundError(f"Missing parent audio files for {pair_id}: {parent_a} / {parent_b}")

            params = spec["transformation_params"]
            if not wav_a.exists():
                apply_acoustic_transformation(parent_a, wav_a, params)
            if not wav_b.exists():
                apply_acoustic_transformation(parent_b, wav_b, params)
            transformed_count += 1

    logger.info(f"Verified {transformed_count} acoustically transformed pairs.")

    # Third pass: Validate all 1,200 WAV files & build manifest items
    logger.info("Validating all 1,200 WAV audio files with AudioValidator...")
    for idx, spec in enumerate(specs):
        pair_id = spec["pair_id"]
        source_type = spec["source_type"]
        wav_a = BENCHMARK_AUDIO_V2_DIR / f"{pair_id}_a.wav"
        wav_b = BENCHMARK_AUDIO_V2_DIR / f"{pair_id}_b.wav"

        bytes_a = wav_a.read_bytes()
        bytes_b = wav_b.read_bytes()

        # Offline AudioValidator check (format, rate, channels, bit-depth, duration)
        validator.validate_live_audio(wav_a.name, bytes_a)
        validator.validate_live_audio(wav_b.name, bytes_b)

        with wave.open(str(wav_a), "rb") as wf_a:
            dur_a = round(wf_a.getnframes() / float(wf_a.getframerate()), 2)
        with wave.open(str(wav_b), "rb") as wf_b:
            dur_b = round(wf_b.getnframes() / float(wf_b.getframerate()), 2)

        # Check distinctness
        hash_a = hashlib.sha256(bytes_a).hexdigest()
        hash_b = hashlib.sha256(bytes_b).hexdigest()
        if hash_a == hash_b:
            raise ValueError(f"Identical audio files detected in pair {pair_id}!")

        rel_path_a = str(wav_a.relative_to(WORKSPACE_DIR)).replace("\\", "/")
        rel_path_b = str(wav_b.relative_to(WORKSPACE_DIR)).replace("\\", "/")

        item = BatchManifestItem(
            pair_id=pair_id,
            category=spec["category"],
            language_code=spec["language_code"],
            audio_condition="clean_recorded" if source_type != AudioSourceType.ACOUSTIC_TRANSFORMED else "acoustic_transformed",
            source_type=source_type,
            audio_path_a=rel_path_a,
            audio_path_b=rel_path_b,
            reference_transcript_a=spec["reference_transcript_a"],
            reference_transcript_b=spec["reference_transcript_b"],
            expected_critical_fields_a=spec.get("expected_critical_fields_a", {}),
            expected_critical_fields_b=spec.get("expected_critical_fields_b", {}),
            expected_intent_a=spec.get("expected_intent_a"),
            expected_intent_b=spec.get("expected_intent_b"),
            expected_action_a=spec.get("expected_action_a"),
            expected_action_b=spec.get("expected_action_b"),
            distinction_field=spec.get("distinction_field"),
            description=spec.get("description"),
            speaker_metadata={
                "voice": spec.get("voice"),
                "duration_a": dur_a,
                "duration_b": dur_b,
                "sha256_a": hash_a,
                "sha256_b": hash_b,
            },
            tts_voice_or_source=spec.get("tts_voice_or_source"),
            transformation_params=spec.get("transformation_params"),
            dataset_version="2.0.0",
            notes=spec.get("notes"),
        )
        manifest_items.append(item)

    manifest = BatchManifest(
        manifest_version="2.0.0",
        name="BoliBench 600-Pair Contrast Evaluation Benchmark",
        description="Stage 2 expanded benchmark consisting of 600 unique paired contrast tests across 6 categories (100 per category)",
        created_at=datetime.now(timezone.utc).isoformat(),
        items=manifest_items,
    )

    save_manifest(manifest, MANIFEST_V2_PATH)
    logger.info(f"Manifest v2.0.0 saved successfully to: {MANIFEST_V2_PATH}")

    # Run full manifest pre-flight audit
    report = validate_manifest(manifest, base_dir=WORKSPACE_DIR, audio_validator=validator)
    logger.info(f"Pre-flight audit report: total={report.total_items}, ready={report.ready_count}, invalid={len(report.invalid_items)}")
    logger.info(f"Counts by category: {report.counts_by_category}")
    logger.info(f"Counts by source provenance: {report.counts_by_source}")

    dry_run_dict = {
        "manifest_version": manifest.manifest_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_items": report.total_items,
        "ready_items": report.ready_count,
        "invalid_items": report.invalid_items,
        "duplicated_items": report.duplicated_items,
        "missing_ground_truth_items": report.missing_ground_truth_items,
        "manual_review_items": report.manual_review_items,
        "counts_by_category": report.counts_by_category,
        "counts_by_source": report.counts_by_source,
    }

    with open(DRY_RUN_REPORT_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(dry_run_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Dry-run report saved to: {DRY_RUN_REPORT_V2_PATH}")


if __name__ == "__main__":
    asyncio.run(build_v2_dataset())
