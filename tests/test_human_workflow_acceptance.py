"""End-to-End Acceptance Tests for BoliBench Human Recording Studio Workflow.

Validates the 6 critical user acceptance tests:
1. Record A with custom business script -> save -> reload and confirm actual audio & transcript.
2. Record B with contrasting business script -> save -> reload and confirm audio & transcript.
3. Confirm both clips share the same pair ID and have correct distinct A/B roles.
4. Confirm the pair appears in the library and is selectable in Run Evaluation.
5. Trigger a mocked evaluation and verify both utterances and comparison results are persisted correctly in SQLite.
6. Verify that an actual 3-second audio file cannot be marked as a valid 50-second recording merely because client reports 50 seconds.
"""

import io
import wave
import uuid
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.recording_schemas import ValidationStatus
from app.services.recording_processor import HumanAudioProcessor
from app.services.recording_storage import RecordingStorage


def generate_pcm_wav(duration_sec: float, sample_rate: int = 16000, freq_hz: float = 440.0) -> bytes:
    """Generate in-memory 16kHz mono PCM WAV bytes with valid headers."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    samples = (0.5 * 32767.0 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.mark.asyncio
async def test_acceptance_criteria_1_to_5_end_to_end_pair_workflow():
    """Executes Acceptance Criteria 1 through 5:
    1. Record A -> save -> reload and confirm actual audio and transcript.
    2. Record B -> save -> reload and confirm audio and transcript.
    3. Confirm shared pair ID and correct distinct A/B roles.
    4. Confirm pair appears in library and is selectable in Run Evaluation.
    5. Trigger mocked evaluation and verify both utterances and comparison are persisted in SQLite.
    """
    uid = uuid.uuid4().hex[:6]
    pair_id = f"human_acc_pair_{uid}"
    speaker_id = f"spk_acc_{uid}"

    script_a = "Enterprise procurement order for 250 Dell PowerEdge rack servers with five year 24/7 support"
    verified_a = "Enterprise procurement order for 250 Dell PowerEdge rack servers with five year 24/7 support"

    script_b = "Enterprise procurement order for 750 Dell PowerEdge rack servers with three year standard support"
    verified_b = "Enterprise procurement order for 750 Dell PowerEdge rack servers with three year standard support"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Step 0: Create contrast pair specification
        create_pair_resp = await client.post("/api/studio/pairs/create", json={
            "pair_id": pair_id,
            "speaker_id": speaker_id,
            "language": "hinglish",
            "category": "quantities_and_numbers",
            "scenario_description": "Enterprise server procurement quantity contrast (250 vs 750 units)",
            "expected_distinction": "Quantity 250 units with 5yr SLA vs Quantity 750 units with 3yr SLA",
        })
        assert create_pair_resp.status_code == 200, f"Pair creation failed: {create_pair_resp.text}"

        # ---------------------------------------------------------------------
        # Acceptance Test 1: Record A -> Save -> Reload and confirm audio & transcript
        # ---------------------------------------------------------------------
        wav_bytes_a = generate_pcm_wav(duration_sec=50.0, freq_hz=440.0)
        rec_id_a = f"rec_acc_a_{uid}"

        upload_a_resp = await client.post("/api/studio/upload", data={
            "recording_id": rec_id_a,
            "pair_id": pair_id,
            "utterance_role": "utterance_a",
            "speaker_id": speaker_id,
            "language": "hinglish",
            "category": "quantities_and_numbers",
            "reference_transcript": script_a,
            "verified_transcript": verified_a,
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "I explicitly consent to record and use this audio for BoliBench evaluation.",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("recording_a.wav", wav_bytes_a, "audio/wav")})

        assert upload_a_resp.status_code == 200, f"Upload A failed: {upload_a_resp.text}"
        meta_a = upload_a_resp.json()
        assert meta_a["recording_id"] == rec_id_a
        assert meta_a["pair_id"] == pair_id
        assert meta_a["utterance_role"] == "utterance_a"
        assert meta_a["reference_transcript"] == script_a
        assert meta_a["verified_transcript"] == verified_a
        assert meta_a["is_transcript_verified"] is True
        assert meta_a["validation_status"] == "valid_50s"
        assert meta_a["diagnostics"]["duration_seconds"] == pytest.approx(50.0, abs=0.1)

        # Reload A metadata and actual audio stream from backend
        reload_a_resp = await client.get(f"/api/studio/recordings/{rec_id_a}")
        assert reload_a_resp.status_code == 200
        reloaded_a_data = reload_a_resp.json()
        assert reloaded_a_data["reference_transcript"] == script_a
        assert reloaded_a_data["verified_transcript"] == verified_a

        audio_a_stream = await client.get(f"/api/studio/recordings/{rec_id_a}/audio")
        assert audio_a_stream.status_code == 200
        assert len(audio_a_stream.content) > 1000
        # Validate that reloaded audio has standard 16kHz mono WAV header
        with wave.open(io.BytesIO(audio_a_stream.content), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert round(wf.getnframes() / 16000.0, 1) == 50.0

        # ---------------------------------------------------------------------
        # Acceptance Test 2: Record B -> Save -> Reload and confirm audio & transcript
        # ---------------------------------------------------------------------
        wav_bytes_b = generate_pcm_wav(duration_sec=50.0, freq_hz=880.0)
        rec_id_b = f"rec_acc_b_{uid}"

        upload_b_resp = await client.post("/api/studio/upload", data={
            "recording_id": rec_id_b,
            "pair_id": pair_id,
            "utterance_role": "utterance_b",
            "speaker_id": speaker_id,
            "language": "hinglish",
            "category": "quantities_and_numbers",
            "reference_transcript": script_b,
            "verified_transcript": verified_b,
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "I explicitly consent to record and use this audio for BoliBench evaluation.",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("recording_b.wav", wav_bytes_b, "audio/wav")})

        assert upload_b_resp.status_code == 200, f"Upload B failed: {upload_b_resp.text}"
        meta_b = upload_b_resp.json()
        assert meta_b["recording_id"] == rec_id_b
        assert meta_b["pair_id"] == pair_id
        assert meta_b["utterance_role"] == "utterance_b"
        assert meta_b["reference_transcript"] == script_b
        assert meta_b["verified_transcript"] == verified_b
        assert meta_b["is_transcript_verified"] is True
        assert meta_b["validation_status"] == "valid_50s"
        assert meta_b["diagnostics"]["duration_seconds"] == pytest.approx(50.0, abs=0.1)

        # Reload B metadata and actual audio stream
        reload_b_resp = await client.get(f"/api/studio/recordings/{rec_id_b}")
        assert reload_b_resp.status_code == 200
        reloaded_b_data = reload_b_resp.json()
        assert reloaded_b_data["reference_transcript"] == script_b
        assert reloaded_b_data["verified_transcript"] == verified_b

        audio_b_stream = await client.get(f"/api/studio/recordings/{rec_id_b}/audio")
        assert audio_b_stream.status_code == 200
        assert len(audio_b_stream.content) > 1000
        with wave.open(io.BytesIO(audio_b_stream.content), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert round(wf.getnframes() / 16000.0, 1) == 50.0

        # ---------------------------------------------------------------------
        # Acceptance Test 3: Confirm both clips share same pair ID & correct roles
        # ---------------------------------------------------------------------
        assert meta_a["pair_id"] == meta_b["pair_id"] == pair_id
        assert meta_a["recording_id"] != meta_b["recording_id"]
        assert meta_a["utterance_role"] == "utterance_a"
        assert meta_b["utterance_role"] == "utterance_b"

        # ---------------------------------------------------------------------
        # Acceptance Test 4: Confirm pair appears in library & is selectable in Run Evaluation
        # ---------------------------------------------------------------------
        # Check studio library candidate pairs endpoint
        studio_pairs_resp = await client.get("/api/studio/pairs")
        assert studio_pairs_resp.status_code == 200
        studio_pairs = studio_pairs_resp.json()
        target_studio_pair = next((p for p in studio_pairs if p["pair_id"] == pair_id), None)
        assert target_studio_pair is not None, f"Pair {pair_id} not found in /api/studio/pairs"
        assert target_studio_pair["utterance_a"]["recording_id"] == rec_id_a
        assert target_studio_pair["utterance_b"]["recording_id"] == rec_id_b

        # Check Run Evaluation query endpoint /api/eval/human-pairs
        eval_pairs_resp = await client.get("/api/eval/human-pairs")
        assert eval_pairs_resp.status_code == 200
        eval_pairs = eval_pairs_resp.json()
        target_eval_pair = next((p for p in eval_pairs if p["pair_id"] == pair_id), None)
        assert target_eval_pair is not None, f"Pair {pair_id} not returned in /api/eval/human-pairs"
        assert target_eval_pair["utterance_a"] is not None
        assert target_eval_pair["utterance_b"] is not None

        # Also verify filtered query by status="all" and category
        filtered_resp = await client.get(f"/api/eval/human-pairs?category=quantities_and_numbers&status=all")
        assert filtered_resp.status_code == 200
        assert any(p["pair_id"] == pair_id for p in filtered_resp.json())

        # ---------------------------------------------------------------------
        # Acceptance Test 5: Trigger mocked evaluation & verify persistence in SQLite
        # ---------------------------------------------------------------------
        mock_eval_resp = await client.post(f"/api/eval/run-human-pair-mock/{pair_id}")
        assert mock_eval_resp.status_code == 200, f"Mock eval failed: {mock_eval_resp.text}"
        eval_result = mock_eval_resp.json()

        assert eval_result["test_case_id"] == pair_id
        assert eval_result["human_pair_id"] == pair_id
        assert eval_result["dataset_source"] == "human_recording"
        assert eval_result["overall_status"] == "completed"
        assert eval_result["semantic_distinction_preserved"] is True
        assert eval_result["utterance_a"]["actual_transcript"] == verified_a
        assert eval_result["utterance_b"]["actual_transcript"] == verified_b
        assert eval_result["utterance_a"]["reference_transcript"] == verified_a
        assert eval_result["utterance_b"]["reference_transcript"] == verified_b
        run_id = eval_result["run_id"]

        # Confirm run is persisted in SQLite and retrieved in Results & Evidence history
        history_resp = await client.get("/api/eval/results?dataset_source=human_recording")
        assert history_resp.status_code == 200
        human_runs = history_resp.json()
        persisted_run = next((r for r in human_runs if r["run_id"] == run_id), None)
        assert persisted_run is not None, f"Evaluation run {run_id} not found in SQLite results history"
        assert persisted_run["human_pair_id"] == pair_id
        assert persisted_run["utterance_a"]["actual_transcript"] == verified_a
        assert persisted_run["utterance_b"]["actual_transcript"] == verified_b


@pytest.mark.asyncio
async def test_acceptance_criterion_6_short_audio_cannot_be_marked_50s_valid():
    """Acceptance Criterion 6:
    Verify that an actual 3-second audio file cannot be marked as a valid 50-second recording
    merely because the client reports 50 seconds (or countdown reaches zero).
    """
    uid = uuid.uuid4().hex[:6]
    rec_id = f"rec_short_spoof_{uid}"

    # Generate a true 3.0-second WAV file
    short_wav = generate_pcm_wav(duration_sec=3.0, sample_rate=16000)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Client falsely claims client_duration_seconds = 50.0
        resp = await client.post("/api/studio/upload", data={
            "recording_id": rec_id,
            "speaker_id": "spk_duration_test",
            "language": "hi-IN",
            "category": "quantities_and_numbers",
            "reference_transcript": "Short three second audio",
            "consent_given": "true",
            "consent_statement": "Consent confirmed.",
            "client_duration_seconds": "50.0",  # Claiming 50s
        }, files={"audio_file": ("spoof_50s.wav", short_wav, "audio/wav")})

        assert resp.status_code == 200
        meta = resp.json()

        # The backend must decode the true audio frames and refuse to classify as valid 50s!
        assert meta["validation_status"] == "incomplete_early_stop", (
            f"Expected 'incomplete_early_stop' but got {meta['validation_status']}"
        )
        assert meta["diagnostics"]["is_complete_50s"] is False
        assert meta["diagnostics"]["duration_seconds"] == pytest.approx(3.0, abs=0.1)


@pytest.mark.asyncio
async def test_distinct_transcripts_intended_vs_verified_preservation():
    """Confirms intended reference transcript and verified transcript remain strictly separate."""
    uid = uuid.uuid4().hex[:6]
    rec_id = f"rec_distinct_{uid}"
    intended = "Acme Corp quarterly procurement order for 200 server units"
    spoken_verified = "Acme Corp quarterly procurement order for 200 server units, with urgent shipping"

    wav_bytes = generate_pcm_wav(duration_sec=50.0)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/studio/upload", data={
            "recording_id": rec_id,
            "speaker_id": "spk_trans_sep",
            "language": "hinglish",
            "category": "quantities_and_numbers",
            "reference_transcript": intended,
            "verified_transcript": spoken_verified,
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "Consent confirmed.",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("test.wav", wav_bytes, "audio/wav")})

        assert resp.status_code == 200
        saved = resp.json()
        assert saved["reference_transcript"] == intended
        assert saved["verified_transcript"] == spoken_verified
        assert saved["reference_transcript"] != saved["verified_transcript"]

        # Confirm listing endpoint returns both fields
        recordings = (await client.get("/api/studio/recordings")).json()
        target = next((r for r in recordings if r["recording_id"] == rec_id), None)
        assert target is not None
        assert target["reference_transcript"] == intended
        assert target["verified_transcript"] == spoken_verified
