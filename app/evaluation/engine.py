"""EvaluationEngine orchestrating test-case retrieval, paired contrast scoring, and persistence."""

from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from app.evaluation.classifier import detect_semantic_contrast
from app.evaluation.dataset import get_test_case_by_id, load_seed_dataset
from app.evaluation.metrics import (
    calculate_cross_script_normalized_wer,
    evaluate_utterance,
    score_pair,
)
from app.evaluation.schemas import (
    AggregateMetrics,
    AudioCondition,
    DerivedAssessment,
    EvaluateTranscriptsRequest,
    ExecutionMode,
    FailureCategory,
    PairEvaluationResult,
    TestCasePair,
    UtteranceEvaluationResult,
)
from app.evaluation.storage import StorageManager
from app.services.gnani_client import GnaniClientError, GnaniPrismaClient

logger = logging.getLogger("bolibench.engine")


class EvaluationEngine:
    """Core evaluation engine for BoliBench paired-contrast speech AI crash testing."""

    def __init__(self, storage_manager: Optional[StorageManager] = None):
        self.storage = storage_manager or StorageManager()
        self._resumption_lock = threading.Lock()
        self._active_resumptions: Set[str] = set()

    def list_test_cases(self, category: Optional[str] = None) -> List[TestCasePair]:
        """Lists available test-case pairs."""
        all_cases = load_seed_dataset()
        if category:
            return [tc for tc in all_cases if tc.category.value == category]
        return all_cases

    def get_test_case(self, test_id: str) -> Optional[TestCasePair]:
        """Fetches a specific test-case pair by ID."""
        return get_test_case_by_id(test_id)

    def run_mock_evaluation(
        self,
        test_case_id: str,
        scenario: str = "ideal_preserved",
    ) -> PairEvaluationResult:
        """Executes a deterministic mock evaluation for a test case pair.

        Scenarios:
        - 'ideal_preserved': Both utterances transcribed perfectly (preserves semantic distinction).
        - 'crashed_distinction': Hand-authored simulated transcript corruption from curated seed dataset
          demonstrating downstream voice agent action crash (text-only simulation; not a measured model error).
        """
        test_case = self.get_test_case(test_case_id)
        if not test_case:
            raise ValueError(f"Test case with ID '{test_case_id}' not found.")

        run_id = f"eval_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        if scenario == "crashed_distinction":
            # Utterance A is transcribed accurately; Utterance B experiences simulated transcript corruption
            transcript_a = test_case.utterance_a.reference_transcript
            transcript_b = test_case.utterance_b.simulated_crashed_transcript or test_case.utterance_a.reference_transcript
        else:
            # Ideal preserved case
            transcript_a = test_case.utterance_a.reference_transcript
            transcript_b = test_case.utterance_b.reference_transcript

        audio_cond = test_case.audio_condition.value if hasattr(test_case.audio_condition, "value") else str(test_case.audio_condition)

        # Mock execution NEVER fabricates measured network latency values
        result = score_pair(
            run_id=run_id,
            test_case=test_case,
            transcript_a=transcript_a,
            transcript_b=transcript_b,
            execution_mode="mock",
            audio_condition=audio_cond,
            timestamp=timestamp,
            client_latency_ms=None,
            provider_latency_ms=None,
            simulated_latency_ms=None,
            latency_a_ms=None,
            latency_b_ms=None,
            request_id_a=None,
            request_id_b=None,
        )

        self.storage.save_run(result)
        return result

    def evaluate_transcripts(self, req: EvaluateTranscriptsRequest) -> PairEvaluationResult:
        """Evaluates arbitrary user or client provided transcripts against a test-case pair."""
        test_case = self.get_test_case(req.test_case_id)
        if not test_case:
            raise ValueError(f"Test case with ID '{req.test_case_id}' not found.")

        run_id = f"eval_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        is_live = (req.execution_mode == ExecutionMode.LIVE)
        exec_mode_str = req.execution_mode.value if hasattr(req.execution_mode, "value") else str(req.execution_mode)
        audio_cond_str = req.audio_condition.value if hasattr(req.audio_condition, "value") else str(req.audio_condition)

        # For mock runs, any provided latency is strictly categorized as simulated_latency_ms
        sim_latency = req.simulated_latency_ms or (req.client_latency_ms if not is_live else None)
        client_lat = req.client_latency_ms if is_live else None

        result = score_pair(
            run_id=run_id,
            test_case=test_case,
            transcript_a=req.transcript_a,
            transcript_b=req.transcript_b,
            execution_mode=exec_mode_str,
            audio_condition=audio_cond_str,
            timestamp=timestamp,
            client_latency_ms=client_lat,
            simulated_latency_ms=sim_latency,
            latency_a_ms=client_lat / 2.0 if client_lat else None,
            latency_b_ms=client_lat / 2.0 if client_lat else None,
        )

        self.storage.save_run(result)
        return result

    async def evaluate_live_audio(
        self,
        test_case_id: str,
        audio_a_bytes: bytes,
        filename_a: str,
        content_type_a: str,
        audio_b_bytes: bytes,
        filename_b: str,
        content_type_b: str,
        client: GnaniPrismaClient,
        audio_condition: str = "clean_recorded",
        resume_run_id: Optional[str] = None,
    ) -> PairEvaluationResult:
        """Evaluates a live audio pair using Gnani Prisma STT REST API.

        Guarantees:
        - Executes actual provider requests sequentially (Utterance A, then Utterance B).
        - Calculates genuine client-side elapsed time for each request.
        - Preserves provider request IDs and backend processing latency if reported.
        - If Utterance A fails, Utterance B is skipped (0 extra provider calls) and an audited
          failed run record (overall_status="failed") is persisted to SQLite before re-raising.
        - If Utterance A succeeds and Utterance B fails, an audited incomplete run record
          (overall_status="incomplete") is persisted to SQLite preserving Utterance A's real transcript
          and provider request ID, and noting that Utterance A consumed credits, before re-raising.
        - Failed and incomplete runs are never presented as successful paired evaluations and do
          not contaminate completed benchmark accuracy metrics.
        - Never marks a mock result as live.
        - Resumption: When resume_run_id is provided, safely reuses reliably recorded prior utterance
          results, retrying ONLY the incomplete/failed utterance to avoid duplicate billing.
        """
        test_case = self.get_test_case(test_case_id)
        if not test_case:
            raise ValueError(f"Test case with ID '{test_case_id}' not found.")

        # Check for resumption
        prior_run: Optional[PairEvaluationResult] = None
        if resume_run_id:
            with self._resumption_lock:
                if resume_run_id in self._active_resumptions:
                    raise ValueError(f"Run '{resume_run_id}' is currently being resumed by another concurrent request.")
                self._active_resumptions.add(resume_run_id)

            try:
                prior_run = self.storage.get_run(resume_run_id)
                if not prior_run:
                    raise ValueError(f"Run '{resume_run_id}' not found for evaluation resumption.")
                if prior_run.test_case_id != test_case_id:
                    raise ValueError(f"Run '{resume_run_id}' test case '{prior_run.test_case_id}' does not match '{test_case_id}'.")
                if prior_run.overall_status == "completed":
                    with self._resumption_lock:
                        self._active_resumptions.discard(resume_run_id)
                    return prior_run
                if prior_run.overall_status not in ("incomplete", "failed"):
                    raise ValueError(f"Run '{resume_run_id}' has status '{prior_run.overall_status}' and cannot be resumed.")
            except Exception:
                with self._resumption_lock:
                    self._active_resumptions.discard(resume_run_id)
                raise

        run_id = resume_run_id or f"eval_live_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Check if Utterance A can be reused from prior run
            can_reuse_a = (
                prior_run is not None
                and prior_run.utterance_a is not None
                and prior_run.utterance_a.status == "completed"
                and bool(prior_run.utterance_a.actual_transcript)
                and bool(prior_run.utterance_a.request_id)
            )

            if can_reuse_a:
                logger.info("Resuming %s: Reusing completed Utterance A (request_id=%s)", run_id, prior_run.utterance_a.request_id)
                resp_a = {
                    "transcript": prior_run.utterance_a.actual_transcript,
                    "request_id": prior_run.utterance_a.request_id,
                    "client_request_elapsed_ms": prior_run.utterance_a.client_request_elapsed_ms or prior_run.utterance_a.latency_ms or 0.0,
                    "provider_processing_time_ms": prior_run.utterance_a.provider_processing_time_ms,
                    "resumed": True,
                }
                elapsed_a_ms = resp_a["client_request_elapsed_ms"]
                prov_time_a = resp_a["provider_processing_time_ms"]
            else:
                # Step 1: Transcribe Utterance A
                start_time_a = time.perf_counter()
                try:
                    resp_a = await client.transcribe(
                        audio_bytes=audio_a_bytes,
                        filename=filename_a,
                        content_type=content_type_a,
                        language_code=test_case.language_code,
                        preferred_language=test_case.language_code,
                        format_mode="transcribe",
                    )
                except Exception as exc:
                    elapsed_a_ms = round((time.perf_counter() - start_time_a) * 1000.0, 2)
                    safe_error = getattr(exc, "message", str(exc))
                    failed_run = PairEvaluationResult(
                        run_id=run_id,
                        test_case_id=test_case_id,
                        category=test_case.category.value if hasattr(test_case.category, "value") else str(test_case.category),
                        language_code=test_case.language_code,
                        execution_mode=ExecutionMode.LIVE,
                        audio_condition=audio_condition,
                        timestamp=timestamp,
                        overall_status="failed",
                        utterance_a=UtteranceEvaluationResult(
                            utterance_id=test_case.utterance_a.id,
                            reference_transcript=test_case.utterance_a.reference_transcript,
                            actual_transcript="",
                            wer=1.0,
                            substitutions=0,
                            deletions=len(test_case.utterance_a.reference_transcript.split()),
                            insertions=0,
                            reference_length=len(test_case.utterance_a.reference_transcript.split()),
                            critical_fields_match=False,
                            detected_intent="UNKNOWN",
                            intent_match=False,
                            task_success=False,
                            status="failed",
                            error_message=safe_error,
                            latency_ms=elapsed_a_ms,
                            client_request_elapsed_ms=elapsed_a_ms,
                        ),
                        utterance_b=UtteranceEvaluationResult(
                            utterance_id=test_case.utterance_b.id,
                            reference_transcript=test_case.utterance_b.reference_transcript,
                            actual_transcript="",
                            wer=1.0,
                            substitutions=0,
                            deletions=len(test_case.utterance_b.reference_transcript.split()),
                            insertions=0,
                            reference_length=len(test_case.utterance_b.reference_transcript.split()),
                            critical_fields_match=False,
                            detected_intent="UNKNOWN",
                            intent_match=False,
                            task_success=False,
                            status="skipped",
                            error_message="Skipped because Utterance A failed.",
                        ),
                        wer_a=1.0,
                        wer_b=1.0,
                        average_wer=1.0,
                        critical_field_accuracy=0.0,
                        semantic_distinction_preserved=False,
                        task_success=False,
                        primary_failure_category=FailureCategory.PROVIDER_ERROR,
                        failure_explanation=f"Provider request for Utterance A failed: {safe_error}. Utterance B was not attempted.",
                        client_measured_latency_ms=elapsed_a_ms,
                        client_request_elapsed_ms=elapsed_a_ms,
                        provider_processing_time_ms=None,
                        simulated_latency_ms=None,
                    )
                    self.storage.save_run(failed_run)
                    raise

                elapsed_a_ms = resp_a.get("client_request_elapsed_ms")
                if elapsed_a_ms is None:
                    elapsed_a_ms = round((time.perf_counter() - start_time_a) * 1000.0, 2)
                prov_time_a = resp_a.get("provider_processing_time_ms")

            # Check if Utterance B can be reused from prior run
            can_reuse_b = (
                prior_run is not None
                and prior_run.utterance_b is not None
                and prior_run.utterance_b.status == "completed"
                and bool(prior_run.utterance_b.actual_transcript)
                and bool(prior_run.utterance_b.request_id)
            )

            if can_reuse_b:
                logger.info("Resuming %s: Reusing completed Utterance B (request_id=%s)", run_id, prior_run.utterance_b.request_id)
                resp_b = {
                    "transcript": prior_run.utterance_b.actual_transcript,
                    "request_id": prior_run.utterance_b.request_id,
                    "client_request_elapsed_ms": prior_run.utterance_b.client_request_elapsed_ms or prior_run.utterance_b.latency_ms or 0.0,
                    "provider_processing_time_ms": prior_run.utterance_b.provider_processing_time_ms,
                    "resumed": True,
                }
                elapsed_b_ms = resp_b["client_request_elapsed_ms"]
                prov_time_b = resp_b["provider_processing_time_ms"]
            else:
                # Step 2: Transcribe Utterance B
                start_time_b = time.perf_counter()
                try:
                    resp_b = await client.transcribe(
                        audio_bytes=audio_b_bytes,
                        filename=filename_b,
                        content_type=content_type_b,
                        language_code=test_case.language_code,
                        preferred_language=test_case.language_code,
                        format_mode="transcribe",
                    )
                except Exception as exc:
                    elapsed_b_ms = round((time.perf_counter() - start_time_b) * 1000.0, 2)
                    safe_error = getattr(exc, "message", str(exc))
                    total_client_latency_ms = round(elapsed_a_ms + elapsed_b_ms, 2)
                    req_id_a = resp_a.get("request_id")

                    # Score Utterance A so its extracted fields, WER, and request ID are faithfully preserved
                    res_a = evaluate_utterance(
                        utterance_spec=test_case.utterance_a,
                        actual_transcript=resp_a.get("transcript", ""),
                        category=test_case.category,
                        language_code=test_case.language_code,
                        latency_ms=elapsed_a_ms,
                        client_request_elapsed_ms=elapsed_a_ms,
                        provider_processing_time_ms=prov_time_a,
                        request_id=req_id_a,
                    )
                    res_a.status = "completed"

                    res_b = UtteranceEvaluationResult(
                        utterance_id=test_case.utterance_b.id,
                        reference_transcript=test_case.utterance_b.reference_transcript,
                        actual_transcript="",
                        wer=1.0,
                        substitutions=0,
                        deletions=len(test_case.utterance_b.reference_transcript.split()),
                        insertions=0,
                        reference_length=len(test_case.utterance_b.reference_transcript.split()),
                        critical_fields_match=False,
                        detected_intent="UNKNOWN",
                        intent_match=False,
                        task_success=False,
                        status="failed",
                        error_message=safe_error,
                        latency_ms=elapsed_b_ms,
                        client_request_elapsed_ms=elapsed_b_ms,
                    )

                    # Persist incomplete run record documenting that Utterance A succeeded
                    # and that provider credits may have been consumed for Utterance A.
                    incomplete_run = PairEvaluationResult(
                        run_id=run_id,
                        test_case_id=test_case_id,
                        category=test_case.category.value if hasattr(test_case.category, "value") else str(test_case.category),
                        language_code=test_case.language_code,
                        execution_mode=ExecutionMode.LIVE,
                        audio_condition=audio_condition,
                        timestamp=timestamp,
                        overall_status="incomplete",
                        utterance_a=res_a,
                        utterance_b=res_b,
                        wer_a=res_a.wer,
                        wer_b=1.0,
                        average_wer=round((res_a.wer + 1.0) / 2.0, 4),
                        critical_field_accuracy=round(float(res_a.critical_fields_match) * 0.5, 4),
                        semantic_distinction_preserved=False,
                        task_success=False,
                        primary_failure_category=FailureCategory.PROVIDER_ERROR,
                        failure_explanation=(
                            f"Partial live evaluation failure: Utterance A succeeded (request_id={req_id_a}), "
                            f"but Utterance B failed with error: {safe_error}. "
                            "Note: Provider credits may have been consumed for Utterance A."
                        ),
                        client_measured_latency_ms=total_client_latency_ms,
                        client_request_elapsed_ms=total_client_latency_ms,
                        provider_processing_time_ms=None,
                        simulated_latency_ms=None,
                    )
                    self.storage.save_run(incomplete_run)
                    raise

                elapsed_b_ms = resp_b.get("client_request_elapsed_ms")
                if elapsed_b_ms is None:
                    elapsed_b_ms = round((time.perf_counter() - start_time_b) * 1000.0, 2)
                prov_time_b = resp_b.get("provider_processing_time_ms")

            total_client_latency_ms = round(elapsed_a_ms + elapsed_b_ms, 2)

            # Provider latency is ONLY populated if genuinely reported by upstream provider
            total_provider_latency_ms = None
            if prov_time_a is not None and prov_time_b is not None:
                total_provider_latency_ms = round(prov_time_a + prov_time_b, 2)

            result = score_pair(
                run_id=run_id,
                test_case=test_case,
                transcript_a=resp_a["transcript"],
                transcript_b=resp_b["transcript"],
                execution_mode="live",
                audio_condition=audio_condition,
                timestamp=timestamp,
                client_latency_ms=total_client_latency_ms,
                client_request_elapsed_ms=total_client_latency_ms,
                provider_latency_ms=total_provider_latency_ms,
                simulated_latency_ms=None,
                latency_a_ms=elapsed_a_ms,
                latency_b_ms=elapsed_b_ms,
                client_elapsed_a_ms=elapsed_a_ms,
                client_elapsed_b_ms=elapsed_b_ms,
                provider_time_a_ms=prov_time_a,
                provider_time_b_ms=prov_time_b,
                request_id_a=resp_a.get("request_id"),
                request_id_b=resp_b.get("request_id"),
            )

            self.storage.save_run(result)
            return result
        finally:
            if resume_run_id:
                with self._resumption_lock:
                    self._active_resumptions.discard(resume_run_id)

    async def evaluate_auto_pair_live(
        self,
        audio_a_bytes: bytes,
        filename_a: str,
        content_type_a: str,
        audio_b_bytes: bytes,
        filename_b: str,
        content_type_b: str,
        client: GnaniPrismaClient,
        audio_condition: str = "clean_recorded",
        language_code: str = "hi-IN",
    ) -> PairEvaluationResult:
        """Evaluates an uploaded audio pair with automatic scenario detection using Gnani Prisma STT.

        Primary Product Flow:
        - Transcribes Utterance A once, measuring client-side elapsed time.
        - If Utterance A fails, Utterance B is skipped (0 extra provider calls) and an audited failed run is saved.
        - Transcribes Utterance B once, measuring client-side elapsed time.
        - If Utterance B fails, an audited incomplete run is saved (preserving Utterance A's transcript and request ID).
        - Infers semantic contrast automatically from recognized speech using detect_semantic_contrast().
        - If transcripts match a known benchmark seed case, scores against curated reference and links test case.
        - If unknown input, marks evaluation_mode='automatic_inference' and reference WER as unavailable (0.0 / reference_available=False).
        - Persists completed evaluation run to SQLite.
        """
        run_id = f"eval_live_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Step 1: Transcribe Utterance A
        start_time_a = time.perf_counter()
        try:
            resp_a = await client.transcribe(
                audio_bytes=audio_a_bytes,
                filename=filename_a,
                content_type=content_type_a,
                language_code=language_code,
                preferred_language=language_code,
                format_mode="transcribe",
            )
            if isinstance(resp_a, dict) and not resp_a.get("success", True):
                raise GnaniClientError(
                    message=resp_a.get("error", "Gnani Prisma STT provider error on Utterance A"),
                    status_code=502,
                    raw_response=resp_a,
                )
        except Exception as exc:
            elapsed_a_ms = round((time.perf_counter() - start_time_a) * 1000.0, 2)
            safe_error = getattr(exc, "message", str(exc))
            failed_run = PairEvaluationResult(
                run_id=run_id,
                test_case_id="auto_detected",
                category="unknown",
                language_code=language_code,
                execution_mode=ExecutionMode.LIVE,
                audio_condition=audio_condition,
                timestamp=timestamp,
                overall_status="failed",
                utterance_a=UtteranceEvaluationResult(
                    utterance_id="auto_utterance_a",
                    reference_transcript="N/A (live upload)",
                    actual_transcript="",
                    wer=1.0,
                    substitutions=0,
                    deletions=0,
                    insertions=0,
                    reference_length=0,
                    critical_fields_match=False,
                    detected_intent="UNKNOWN",
                    intent_match=False,
                    task_success=False,
                    status="failed",
                    error_message=safe_error,
                    latency_ms=elapsed_a_ms,
                    client_request_elapsed_ms=elapsed_a_ms,
                    reference_available=False,
                ),
                utterance_b=UtteranceEvaluationResult(
                    utterance_id="auto_utterance_b",
                    reference_transcript="N/A (live upload)",
                    actual_transcript="",
                    wer=1.0,
                    substitutions=0,
                    deletions=0,
                    insertions=0,
                    reference_length=0,
                    critical_fields_match=False,
                    detected_intent="UNKNOWN",
                    intent_match=False,
                    task_success=False,
                    status="skipped",
                    error_message="Skipped because Utterance A failed.",
                    reference_available=False,
                ),
                wer_a=1.0,
                wer_b=1.0,
                average_wer=1.0,
                critical_field_accuracy=0.0,
                semantic_distinction_preserved=False,
                task_success=False,
                primary_failure_category=FailureCategory.PROVIDER_ERROR,
                failure_explanation=f"Provider request for Utterance A failed: {safe_error}. Utterance B was not attempted.",
                client_measured_latency_ms=elapsed_a_ms,
                client_request_elapsed_ms=elapsed_a_ms,
                provider_processing_time_ms=None,
                simulated_latency_ms=None,
                evaluation_mode="automatic_inference",
                detection_confidence="ambiguous",
                distinction_field="unknown",
                supporting_evidence={"error": safe_error},
                is_curated_reference=False,
            )
            self.storage.save_run(failed_run)
            raise

        elapsed_a_ms = resp_a.get("client_request_elapsed_ms")
        if elapsed_a_ms is None:
            elapsed_a_ms = round((time.perf_counter() - start_time_a) * 1000.0, 2)
        prov_time_a = resp_a.get("provider_processing_time_ms")
        transcript_a = resp_a.get("transcript", "")

        # Step 2: Transcribe Utterance B
        start_time_b = time.perf_counter()
        try:
            resp_b = await client.transcribe(
                audio_bytes=audio_b_bytes,
                filename=filename_b,
                content_type=content_type_b,
                language_code=language_code,
                preferred_language=language_code,
                format_mode="transcribe",
            )
            if isinstance(resp_b, dict) and not resp_b.get("success", True):
                raise GnaniClientError(
                    message=resp_b.get("error", "Gnani Prisma STT provider error on Utterance B"),
                    status_code=502,
                    raw_response=resp_b,
                )
        except Exception as exc:
            elapsed_b_ms = round((time.perf_counter() - start_time_b) * 1000.0, 2)
            safe_error = getattr(exc, "message", str(exc))
            total_client_latency_ms = round(elapsed_a_ms + elapsed_b_ms, 2)
            req_id_a = resp_a.get("request_id")

            res_a = UtteranceEvaluationResult(
                utterance_id="auto_utterance_a",
                reference_transcript="N/A (live upload)",
                actual_transcript=transcript_a,
                wer=0.0,
                substitutions=0,
                deletions=0,
                insertions=0,
                reference_length=len(transcript_a.split()),
                critical_fields_match=False,
                detected_intent="UNKNOWN",
                intent_match=False,
                task_success=False,
                status="completed",
                latency_ms=elapsed_a_ms,
                client_request_elapsed_ms=elapsed_a_ms,
                provider_processing_time_ms=prov_time_a,
                request_id=req_id_a,
                reference_available=False,
            )
            res_b = UtteranceEvaluationResult(
                utterance_id="auto_utterance_b",
                reference_transcript="N/A (live upload)",
                actual_transcript="",
                wer=1.0,
                substitutions=0,
                deletions=0,
                insertions=0,
                reference_length=0,
                critical_fields_match=False,
                detected_intent="UNKNOWN",
                intent_match=False,
                task_success=False,
                status="failed",
                error_message=safe_error,
                latency_ms=elapsed_b_ms,
                client_request_elapsed_ms=elapsed_b_ms,
                reference_available=False,
            )

            incomplete_run = PairEvaluationResult(
                run_id=run_id,
                test_case_id="auto_detected",
                category="unknown",
                language_code=language_code,
                execution_mode=ExecutionMode.LIVE,
                audio_condition=audio_condition,
                timestamp=timestamp,
                overall_status="incomplete",
                utterance_a=res_a,
                utterance_b=res_b,
                wer_a=0.0,
                wer_b=1.0,
                average_wer=0.5,
                critical_field_accuracy=0.0,
                semantic_distinction_preserved=False,
                task_success=False,
                primary_failure_category=FailureCategory.PROVIDER_ERROR,
                failure_explanation=(
                    f"Partial live evaluation failure: Utterance A succeeded (request_id={req_id_a}), "
                    f"but Utterance B failed with error: {safe_error}. "
                    "Note: Provider credits were consumed for Utterance A."
                ),
                client_measured_latency_ms=total_client_latency_ms,
                client_request_elapsed_ms=total_client_latency_ms,
                provider_processing_time_ms=None,
                simulated_latency_ms=None,
                evaluation_mode="automatic_inference",
                detection_confidence="ambiguous",
                distinction_field="unknown",
                supporting_evidence={"transcript_a": transcript_a, "error_b": safe_error},
                is_curated_reference=False,
            )
            self.storage.save_run(incomplete_run)
            raise

        elapsed_b_ms = resp_b.get("client_request_elapsed_ms")
        if elapsed_b_ms is None:
            elapsed_b_ms = round((time.perf_counter() - start_time_b) * 1000.0, 2)
        prov_time_b = resp_b.get("provider_processing_time_ms")
        transcript_b = resp_b.get("transcript", "")
        req_id_a = resp_a.get("request_id")
        req_id_b = resp_b.get("request_id")

        total_client_latency_ms = round(elapsed_a_ms + elapsed_b_ms, 2)
        total_provider_latency_ms = None
        if prov_time_a is not None and prov_time_b is not None:
            total_provider_latency_ms = round(prov_time_a + prov_time_b, 2)

        # Step 3: Run automatic semantic contrast inference
        inference = detect_semantic_contrast(transcript_a, transcript_b, language_code=language_code)

        # Map to PairEvaluationResult
        result = self._build_inference_evaluation_result(
            run_id=run_id,
            timestamp=timestamp,
            transcript_a=transcript_a,
            transcript_b=transcript_b,
            inference=inference,
            execution_mode=ExecutionMode.LIVE,
            audio_condition=audio_condition,
            language_code=language_code,
            elapsed_a_ms=elapsed_a_ms,
            elapsed_b_ms=elapsed_b_ms,
            total_client_latency_ms=total_client_latency_ms,
            prov_time_a=prov_time_a,
            prov_time_b=prov_time_b,
            total_provider_latency_ms=total_provider_latency_ms,
            req_id_a=req_id_a,
            req_id_b=req_id_b,
            simulated_latency_ms=None,
        )

        self.storage.save_run(result)
        return result

    def evaluate_auto_pair_mock(
        self,
        transcript_a: str,
        transcript_b: str,
        audio_condition: str = "clean_recorded",
        language_code: str = "hi-IN",
        simulated_latency_ms: Optional[float] = 120.0,
    ) -> PairEvaluationResult:
        """Offline deterministic mock evaluation with automatic semantic contrast detection.

        Consumes 0 credits and makes 0 provider requests.
        """
        run_id = f"eval_mock_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        inference = detect_semantic_contrast(transcript_a, transcript_b, language_code=language_code)

        result = self._build_inference_evaluation_result(
            run_id=run_id,
            timestamp=timestamp,
            transcript_a=transcript_a,
            transcript_b=transcript_b,
            inference=inference,
            execution_mode=ExecutionMode.MOCK,
            audio_condition=audio_condition,
            language_code=language_code,
            elapsed_a_ms=None,
            elapsed_b_ms=None,
            total_client_latency_ms=None,
            prov_time_a=None,
            prov_time_b=None,
            total_provider_latency_ms=None,
            req_id_a=None,
            req_id_b=None,
            simulated_latency_ms=simulated_latency_ms,
        )

        self.storage.save_run(result)
        return result

    async def evaluate_human_pair_live(
        self,
        pair_id: str,
        client: GnaniPrismaClient,
        audio_condition: str = "clean_recorded",
        resume_run_id: Optional[str] = None,
        recording_storage: Optional[Any] = None,
    ) -> PairEvaluationResult:
        """Executes a live paired-contrast evaluation for a human-recorded contrast pair.

        Strict Safety & Credit Protection Guarantees:
        - Submits both Utterance A and Utterance B through Gnani Prisma STT REST API.
        - Continues with Utterance B even if Utterance A receives a provider error.
        - Records provider failures independently rather than discarding the pair.
        - Calculates Raw WER and Normalized WER against verified human ground truth.
        - Sets dataset_source='human_recording' to isolate from synthetic benchmark cohorts.
        - Updates pair status in human_recordings_v1/ and persists run to SQLite database.
        - Resumption: When resume_run_id is provided, safely reuses reliably recorded prior utterance
          results, retrying ONLY the incomplete/failed utterance to avoid duplicate billing.
        """
        if recording_storage is None:
            from app.services.recording_storage import RecordingStorage
            recording_storage = RecordingStorage()
        pair_info = recording_storage.get_pair(pair_id)
        if not pair_info:
            raise ValueError(f"Human contrast pair '{pair_id}' not found in human_recordings_v1.")

        utt_a_info = pair_info.get("utterance_a")
        utt_b_info = pair_info.get("utterance_b")
        if not utt_a_info or not utt_b_info or not utt_a_info.get("recording_id") or not utt_b_info.get("recording_id"):
            raise ValueError(f"Human contrast pair '{pair_id}' is incomplete (both Utterance A and B must be recorded).")

        # Check for resumption
        prior_run: Optional[PairEvaluationResult] = None
        if resume_run_id:
            with self._resumption_lock:
                if resume_run_id in self._active_resumptions:
                    raise ValueError(f"Run '{resume_run_id}' is currently being resumed by another concurrent request.")
                self._active_resumptions.add(resume_run_id)

            try:
                prior_run = self.storage.get_run(resume_run_id)
                if not prior_run:
                    raise ValueError(f"Run '{resume_run_id}' not found for evaluation resumption.")
                if prior_run.test_case_id != pair_id:
                    raise ValueError(f"Run '{resume_run_id}' test case '{prior_run.test_case_id}' does not match '{pair_id}'.")
                if prior_run.overall_status == "completed":
                    with self._resumption_lock:
                        self._active_resumptions.discard(resume_run_id)
                    return prior_run
                if prior_run.overall_status not in ("incomplete", "failed"):
                    raise ValueError(f"Run '{resume_run_id}' has status '{prior_run.overall_status}' and cannot be resumed.")
            except Exception:
                with self._resumption_lock:
                    self._active_resumptions.discard(resume_run_id)
                raise

        rec_id_a = utt_a_info["recording_id"]
        rec_id_b = utt_b_info["recording_id"]

        path_a = recording_storage.get_audio_file_path(rec_id_a)
        path_b = recording_storage.get_audio_file_path(rec_id_b)

        with open(path_a, "rb") as f:
            audio_a_bytes = f.read()
        with open(path_b, "rb") as f:
            audio_b_bytes = f.read()

        ref_a = (utt_a_info.get("verified_transcript") or utt_a_info.get("reference_transcript") or "").strip()
        ref_b = (utt_b_info.get("verified_transcript") or utt_b_info.get("reference_transcript") or "").strip()
        category = pair_info.get("category") or "quantities_and_numbers"
        language_code = pair_info.get("language") or "hi-IN"

        run_id = resume_run_id or f"eval_human_live_{pair_id}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Check if Utterance A can be reused from prior run
            can_reuse_a = (
                prior_run is not None
                and prior_run.utterance_a is not None
                and prior_run.utterance_a.status == "completed"
                and bool(prior_run.utterance_a.actual_transcript)
                and bool(prior_run.utterance_a.request_id)
            )

            if can_reuse_a:
                logger.info("Resuming %s: Reusing completed Utterance A (request_id=%s)", run_id, prior_run.utterance_a.request_id)
                resp_a = {
                    "transcript": prior_run.utterance_a.actual_transcript,
                    "request_id": prior_run.utterance_a.request_id,
                    "client_request_elapsed_ms": prior_run.utterance_a.client_request_elapsed_ms or prior_run.utterance_a.latency_ms or 0.0,
                    "provider_processing_time_ms": prior_run.utterance_a.provider_processing_time_ms,
                    "resumed": True,
                }
                error_a = None
                elapsed_a_ms = resp_a["client_request_elapsed_ms"]
            else:
                # Step 1: Transcribe Utterance A independently
                start_a = time.perf_counter()
                resp_a = None
                error_a = None
                try:
                    resp_a = await client.transcribe(
                        audio_bytes=audio_a_bytes,
                        filename=f"{rec_id_a}.wav",
                        content_type="audio/wav",
                        language_code=language_code,
                        preferred_language=language_code,
                        format_mode="transcribe",
                    )
                    if isinstance(resp_a, dict) and not resp_a.get("success", True):
                        error_a = resp_a.get("error", "Gnani Prisma STT provider error on Utterance A")
                except Exception as exc:
                    error_a = getattr(exc, "message", str(exc))
                elapsed_a_ms = round((time.perf_counter() - start_a) * 1000.0, 2)

            # Check if Utterance B can be reused from prior run
            can_reuse_b = (
                prior_run is not None
                and prior_run.utterance_b is not None
                and prior_run.utterance_b.status == "completed"
                and bool(prior_run.utterance_b.actual_transcript)
                and bool(prior_run.utterance_b.request_id)
            )

            if can_reuse_b:
                logger.info("Resuming %s: Reusing completed Utterance B (request_id=%s)", run_id, prior_run.utterance_b.request_id)
                resp_b = {
                    "transcript": prior_run.utterance_b.actual_transcript,
                    "request_id": prior_run.utterance_b.request_id,
                    "client_request_elapsed_ms": prior_run.utterance_b.client_request_elapsed_ms or prior_run.utterance_b.latency_ms or 0.0,
                    "provider_processing_time_ms": prior_run.utterance_b.provider_processing_time_ms,
                    "resumed": True,
                }
                error_b = None
                elapsed_b_ms = resp_b["client_request_elapsed_ms"]
            else:
                # Step 2: Transcribe Utterance B independently (ALWAYS continue even if A had a provider error)
                start_b = time.perf_counter()
                resp_b = None
                error_b = None
                try:
                    resp_b = await client.transcribe(
                        audio_bytes=audio_b_bytes,
                        filename=f"{rec_id_b}.wav",
                        content_type="audio/wav",
                        language_code=language_code,
                        preferred_language=language_code,
                        format_mode="transcribe",
                    )
                    if isinstance(resp_b, dict) and not resp_b.get("success", True):
                        error_b = resp_b.get("error", "Gnani Prisma STT provider error on Utterance B")
                except Exception as exc:
                    error_b = getattr(exc, "message", str(exc))
                elapsed_b_ms = round((time.perf_counter() - start_b) * 1000.0, 2)

            total_client_latency_ms = round(elapsed_a_ms + elapsed_b_ms, 2)

            transcript_a = resp_a.get("transcript", "") if (resp_a and not error_a) else ""
            transcript_b = resp_b.get("transcript", "") if (resp_b and not error_b) else ""
            req_id_a = resp_a.get("request_id") if resp_a else None
            req_id_b = resp_b.get("request_id") if resp_b else None
            prov_time_a = resp_a.get("provider_processing_time_ms") if resp_a else None
            prov_time_b = resp_b.get("provider_processing_time_ms") if resp_b else None
            total_prov_time = round(prov_time_a + prov_time_b, 2) if (prov_time_a is not None and prov_time_b is not None) else None

            # Build Utterance Evaluation Results
            if error_a:
                res_a = UtteranceEvaluationResult(
                    utterance_id=f"{pair_id}_a",
                    reference_transcript=ref_a,
                    actual_transcript="",
                    wer=1.0,
                    substitutions=0,
                    deletions=0,
                    insertions=0,
                    reference_length=len(ref_a.split()) if ref_a else 0,
                    critical_fields_match=False,
                    detected_intent="UNKNOWN",
                    intent_match=False,
                    task_success=False,
                    status="failed",
                    error_message=error_a,
                    latency_ms=elapsed_a_ms,
                    client_request_elapsed_ms=elapsed_a_ms,
                    reference_available=bool(ref_a),
                )
            else:
                from app.evaluation.metrics import calculate_wer
                wer_a_val, subs_a, dels_a, ins_a, r_len_a = calculate_wer(ref_a, transcript_a, language_code)
                norm_wer_a_val, _, _, _, _ = calculate_cross_script_normalized_wer(ref_a, transcript_a, language_code)
                res_a = UtteranceEvaluationResult(
                    utterance_id=f"{pair_id}_a",
                    reference_transcript=ref_a,
                    actual_transcript=transcript_a,
                    wer=wer_a_val,
                    substitutions=subs_a,
                    deletions=dels_a,
                    insertions=ins_a,
                    reference_length=r_len_a,
                    critical_fields_match=True,
                    detected_intent="UNKNOWN",
                    intent_match=True,
                    task_success=True,
                    status="completed",
                    latency_ms=elapsed_a_ms,
                    client_request_elapsed_ms=elapsed_a_ms,
                    provider_processing_time_ms=prov_time_a,
                    request_id=req_id_a,
                    normalized_wer=norm_wer_a_val,
                    reference_available=bool(ref_a),
                )

            if error_b:
                res_b = UtteranceEvaluationResult(
                    utterance_id=f"{pair_id}_b",
                    reference_transcript=ref_b,
                    actual_transcript="",
                    wer=1.0,
                    substitutions=0,
                    deletions=0,
                    insertions=0,
                    reference_length=len(ref_b.split()) if ref_b else 0,
                    critical_fields_match=False,
                    detected_intent="UNKNOWN",
                    intent_match=False,
                    task_success=False,
                    status="failed",
                    error_message=error_b,
                    latency_ms=elapsed_b_ms,
                    client_request_elapsed_ms=elapsed_b_ms,
                    reference_available=bool(ref_b),
                )
            else:
                from app.evaluation.metrics import calculate_wer
                wer_b_val, subs_b, dels_b, ins_b, r_len_b = calculate_wer(ref_b, transcript_b, language_code)
                norm_wer_b_val, _, _, _, _ = calculate_cross_script_normalized_wer(ref_b, transcript_b, language_code)
                res_b = UtteranceEvaluationResult(
                    utterance_id=f"{pair_id}_b",
                    reference_transcript=ref_b,
                    actual_transcript=transcript_b,
                    wer=wer_b_val,
                    substitutions=subs_b,
                    deletions=dels_b,
                    insertions=ins_b,
                    reference_length=r_len_b,
                    critical_fields_match=True,
                    detected_intent="UNKNOWN",
                    intent_match=True,
                    task_success=True,
                    status="completed",
                    latency_ms=elapsed_b_ms,
                    client_request_elapsed_ms=elapsed_b_ms,
                    provider_processing_time_ms=prov_time_b,
                    request_id=req_id_b,
                    normalized_wer=norm_wer_b_val,
                    reference_available=bool(ref_b),
                )

            # Step 3: Compute Pair-Level Contrast Evaluation
            if error_a and error_b:
                overall_status = "failed"
                primary_fail = FailureCategory.PROVIDER_ERROR
                fail_expl = f"Provider failure on both utterances for human pair '{pair_id}': Utterance A ({error_a}), Utterance B ({error_b})."
                distinction_preserved = False
                task_success = False
                field_acc = 0.0
                avg_wer = 1.0
                avg_norm_wer = 1.0
            elif error_a or error_b:
                overall_status = "incomplete"
                primary_fail = FailureCategory.PROVIDER_ERROR
                succeeded_utt = "B" if error_a else "A"
                failed_utt = "A" if error_a else "B"
                failed_err = error_a if error_a else error_b
                fail_expl = (
                    f"Partial live evaluation failure on human pair '{pair_id}': Utterance {succeeded_utt} succeeded, "
                    f"but Utterance {failed_utt} failed with error: {failed_err}. "
                    f"Note: Provider credits were consumed for Utterance {succeeded_utt}."
                )
                distinction_preserved = False
                task_success = False
                field_acc = 0.5
                succ_wer = res_b.wer if error_a else res_a.wer
                avg_wer = round((succ_wer + 1.0) / 2.0, 4)
                avg_norm_wer = None
            else:
                overall_status = "completed"
                avg_wer = round((res_a.wer + res_b.wer) / 2.0, 4)
                norm_a_calc = res_a.normalized_wer if res_a.normalized_wer is not None else res_a.wer
                norm_b_calc = res_b.normalized_wer if res_b.normalized_wer is not None else res_b.wer
                avg_norm_wer = round((norm_a_calc + norm_b_calc) / 2.0, 4)

                inference = detect_semantic_contrast(transcript_a, transcript_b, language_code=language_code)
                distinction_preserved = inference.semantic_distinction_preserved
                task_success = inference.task_success
                field_acc = 1.0 if distinction_preserved else 0.0

                if not distinction_preserved:
                    primary_fail = FailureCategory.EXTRACTION_FAILURE
                    for fc in FailureCategory:
                        if fc.value == inference.primary_failure_category:
                            primary_fail = fc
                            break
                    fail_expl = inference.failure_explanation or f"Semantic distinction collapsed between Utterance A and B under category {category}."
                else:
                    primary_fail = FailureCategory.NONE
                    fail_expl = f"Semantic contrast successfully preserved for human pair '{pair_id}' ({inference.distinction_field or category})."

            pair_result = PairEvaluationResult(
                run_id=run_id,
                test_case_id=pair_id,
                category=category,
                language_code=language_code,
                execution_mode=ExecutionMode.LIVE,
                audio_condition=audio_condition,
                timestamp=timestamp,
                overall_status=overall_status,
                utterance_a=res_a,
                utterance_b=res_b,
                wer_a=res_a.wer,
                wer_b=res_b.wer,
                average_wer=avg_wer,
                normalized_wer_a=res_a.normalized_wer,
                normalized_wer_b=res_b.normalized_wer,
                average_normalized_wer=avg_norm_wer,
                critical_field_accuracy=field_acc,
                semantic_distinction_preserved=distinction_preserved,
                task_success=task_success,
                primary_failure_category=primary_fail,
                failure_explanation=fail_expl,
                client_measured_latency_ms=total_client_latency_ms,
                client_request_elapsed_ms=total_client_latency_ms,
                provider_processing_time_ms=total_prov_time,
                simulated_latency_ms=None,
                evaluation_mode="human_recorded_pair",
                distinction_field=pair_info.get("distinction_field") or category,
                supporting_evidence={"human_pair_id": pair_id, "speaker_id": pair_info.get("speaker_id")},
                is_curated_reference=True,
                dataset_source="human_recording",
                dataset_classification=pair_info.get("classification") or "unverified",
                human_pair_id=pair_id,
            )

            self.storage.save_run(pair_result)
            recording_storage.update_pair_evaluation_status(pair_id, run_id, overall_status)
            return pair_result
        finally:
            if resume_run_id:
                with self._resumption_lock:
                    self._active_resumptions.discard(resume_run_id)

    def evaluate_human_pair_mock(
        self,
        pair_id: str,
        simulated_crashed: bool = False,
        audio_condition: str = "clean_recorded",
        recording_storage: Optional[Any] = None,
    ) -> PairEvaluationResult:
        """Deterministic mock evaluation for a human-recorded contrast pair (0 credits consumed)."""
        if recording_storage is None:
            from app.services.recording_storage import RecordingStorage
            recording_storage = RecordingStorage()
        pair_info = recording_storage.get_pair(pair_id)
        if not pair_info:
            raise ValueError(f"Human contrast pair '{pair_id}' not found in human_recordings_v1.")

        utt_a_info = pair_info.get("utterance_a")
        utt_b_info = pair_info.get("utterance_b")
        if not utt_a_info or not utt_b_info:
            raise ValueError(f"Human contrast pair '{pair_id}' is incomplete.")

        ref_a = (utt_a_info.get("verified_transcript") or utt_a_info.get("reference_transcript") or "").strip()
        ref_b = (utt_b_info.get("verified_transcript") or utt_b_info.get("reference_transcript") or "").strip()
        category = pair_info.get("category") or "quantities_and_numbers"
        language_code = pair_info.get("language") or "hi-IN"

        run_id = f"eval_human_mock_{pair_id}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # In mock mode, Utterance A uses reference text
        transcript_a = ref_a
        # In mock mode, Utterance B uses reference text (ideal) or simulated crash (collapse to A)
        transcript_b = ref_a if simulated_crashed else ref_b

        from app.evaluation.metrics import calculate_wer
        wer_a_val, s_a, d_a, i_a, l_a = calculate_wer(ref_a, transcript_a, language_code)
        wer_b_val, s_b, d_b, i_b, l_b = calculate_wer(ref_b, transcript_b, language_code)
        norm_a, _, _, _, _ = calculate_cross_script_normalized_wer(ref_a, transcript_a, language_code)
        norm_b, _, _, _, _ = calculate_cross_script_normalized_wer(ref_b, transcript_b, language_code)

        res_a = UtteranceEvaluationResult(
            utterance_id=f"{pair_id}_a",
            reference_transcript=ref_a,
            actual_transcript=transcript_a,
            wer=wer_a_val,
            substitutions=s_a,
            deletions=d_a,
            insertions=i_a,
            reference_length=l_a,
            critical_fields_match=True,
            detected_intent="INTENT_MOCK",
            intent_match=True,
            task_success=True,
            status="completed",
            simulated_latency_ms=60.0,
            normalized_wer=norm_a,
            reference_available=True,
        )
        res_b = UtteranceEvaluationResult(
            utterance_id=f"{pair_id}_b",
            reference_transcript=ref_b,
            actual_transcript=transcript_b,
            wer=wer_b_val,
            substitutions=s_b,
            deletions=d_b,
            insertions=i_b,
            reference_length=l_b,
            critical_fields_match=not simulated_crashed,
            detected_intent="INTENT_MOCK",
            intent_match=not simulated_crashed,
            task_success=not simulated_crashed,
            status="completed",
            simulated_latency_ms=60.0,
            normalized_wer=norm_b,
            reference_available=True,
        )

        distinction_preserved = not simulated_crashed
        pair_result = PairEvaluationResult(
            run_id=run_id,
            test_case_id=pair_id,
            category=category,
            language_code=language_code,
            execution_mode=ExecutionMode.MOCK,
            audio_condition=audio_condition,
            timestamp=timestamp,
            overall_status="completed",
            utterance_a=res_a,
            utterance_b=res_b,
            wer_a=res_a.wer,
            wer_b=res_b.wer,
            average_wer=round((res_a.wer + res_b.wer) / 2.0, 4),
            normalized_wer_a=norm_a,
            normalized_wer_b=norm_b,
            average_normalized_wer=round((norm_a + norm_b) / 2.0, 4),
            critical_field_accuracy=0.0 if simulated_crashed else 1.0,
            semantic_distinction_preserved=distinction_preserved,
            task_success=not simulated_crashed,
            primary_failure_category=FailureCategory.EXTRACTION_FAILURE if simulated_crashed else FailureCategory.NONE,
            failure_explanation="Simulated distinction collapse (mock)." if simulated_crashed else f"Deterministic mock evaluation on human pair '{pair_id}'. Distinction preserved.",
            client_measured_latency_ms=None,
            client_request_elapsed_ms=None,
            provider_processing_time_ms=None,
            simulated_latency_ms=120.0,
            evaluation_mode="human_recorded_pair",
            distinction_field=pair_info.get("distinction_field") or category,
            supporting_evidence={"human_pair_id": pair_id, "mock": True},
            is_curated_reference=True,
            dataset_source="human_recording",
            dataset_classification=pair_info.get("classification") or "unverified",
            human_pair_id=pair_id,
        )

        self.storage.save_run(pair_result)
        recording_storage.update_pair_evaluation_status(pair_id, run_id, "completed")
        return pair_result

    def _build_inference_evaluation_result(
        self,
        run_id: str,
        timestamp: str,
        transcript_a: str,
        transcript_b: str,
        inference: Any,
        execution_mode: ExecutionMode,
        audio_condition: str,
        language_code: str,
        elapsed_a_ms: Optional[float],
        elapsed_b_ms: Optional[float],
        total_client_latency_ms: Optional[float],
        prov_time_a: Optional[float],
        prov_time_b: Optional[float],
        total_provider_latency_ms: Optional[float],
        req_id_a: Optional[str],
        req_id_b: Optional[str],
        simulated_latency_ms: Optional[float],
    ) -> PairEvaluationResult:
        is_curated = bool(inference.is_curated_reference and inference.matched_test_case_id)
        test_case_id = inference.matched_test_case_id if is_curated else "auto_detected"
        eval_mode = "reference_benchmark" if is_curated else "automatic_inference"

        fail_cat = FailureCategory.NONE
        if inference.primary_failure_category:
            for fc in FailureCategory:
                if fc.value == inference.primary_failure_category:
                    fail_cat = fc
                    break

        ref_a = inference.reference_a if is_curated else "N/A (arbitrary upload; no benchmark reference)"
        ref_b = inference.reference_b if is_curated else "N/A (arbitrary upload; no benchmark reference)"
        wer_a_val = inference.wer_a if (is_curated and inference.wer_a is not None) else 0.0
        wer_b_val = inference.wer_b if (is_curated and inference.wer_b is not None) else 0.0
        avg_wer_val = inference.average_wer if (is_curated and inference.average_wer is not None) else 0.0

        res_a = UtteranceEvaluationResult(
            utterance_id=f"{test_case_id}_a",
            reference_transcript=ref_a,
            actual_transcript=transcript_a,
            wer=wer_a_val,
            substitutions=0,
            deletions=0,
            insertions=0,
            reference_length=len(ref_a.split()) if is_curated else len(transcript_a.split()),
            extracted_fields=inference.fields_a,
            critical_fields_match=inference.semantic_distinction_preserved,
            detected_intent=inference.intent_a or "UNKNOWN",
            intent_match=True if is_curated else False,
            task_success=inference.task_success,
            status="completed",
            latency_ms=elapsed_a_ms,
            client_request_elapsed_ms=elapsed_a_ms,
            provider_processing_time_ms=prov_time_a,
            simulated_latency_ms=simulated_latency_ms / 2.0 if simulated_latency_ms else None,
            request_id=req_id_a,
            normalized_wer=inference.normalized_wer_a if is_curated else None,
            reference_available=is_curated,
        )

        res_b = UtteranceEvaluationResult(
            utterance_id=f"{test_case_id}_b",
            reference_transcript=ref_b,
            actual_transcript=transcript_b,
            wer=wer_b_val,
            substitutions=0,
            deletions=0,
            insertions=0,
            reference_length=len(ref_b.split()) if is_curated else len(transcript_b.split()),
            extracted_fields=inference.fields_b,
            critical_fields_match=inference.semantic_distinction_preserved,
            detected_intent=inference.intent_b or "UNKNOWN",
            intent_match=True if is_curated else False,
            task_success=inference.task_success,
            status="completed",
            latency_ms=elapsed_b_ms,
            client_request_elapsed_ms=elapsed_b_ms,
            provider_processing_time_ms=prov_time_b,
            simulated_latency_ms=simulated_latency_ms / 2.0 if simulated_latency_ms else None,
            request_id=req_id_b,
            normalized_wer=inference.normalized_wer_b if is_curated else None,
            reference_available=is_curated,
        )

        return PairEvaluationResult(
            run_id=run_id,
            test_case_id=test_case_id,
            category=inference.category,
            language_code=language_code,
            execution_mode=execution_mode,
            audio_condition=audio_condition,
            timestamp=timestamp,
            overall_status="completed",
            utterance_a=res_a,
            utterance_b=res_b,
            wer_a=wer_a_val,
            wer_b=wer_b_val,
            average_wer=avg_wer_val,
            normalized_wer_a=inference.normalized_wer_a if is_curated else None,
            normalized_wer_b=inference.normalized_wer_b if is_curated else None,
            average_normalized_wer=inference.average_normalized_wer if is_curated else None,
            critical_field_accuracy=1.0 if inference.semantic_distinction_preserved else 0.0,
            semantic_distinction_preserved=inference.semantic_distinction_preserved,
            task_success=inference.task_success,
            primary_failure_category=fail_cat,
            failure_explanation=inference.failure_explanation or inference.explanation,
            client_measured_latency_ms=total_client_latency_ms,
            client_request_elapsed_ms=total_client_latency_ms,
            provider_processing_time_ms=total_provider_latency_ms,
            simulated_latency_ms=simulated_latency_ms,
            evaluation_mode=eval_mode,
            detection_confidence=inference.confidence.value if hasattr(inference.confidence, "value") else str(inference.confidence),
            distinction_field=inference.distinction_field,
            supporting_evidence=inference.supporting_evidence,
            is_curated_reference=is_curated,
        )

    def get_run(self, run_id: str) -> Optional[PairEvaluationResult]:
        """Retrieves a single evaluation run."""
        return self.storage.get_run(run_id)

    def list_runs(
        self,
        limit: int = 100,
        category: Optional[str] = None,
        status: Optional[str] = None,
        dataset_source: Optional[str] = None,
        dataset_classification: Optional[str] = None,
    ) -> List[PairEvaluationResult]:
        """Lists recent evaluation runs, optionally filtered by category, status, dataset_source, or dataset_classification."""
        return self.storage.list_runs(
            limit=limit,
            category=category,
            status=status,
            dataset_source=dataset_source,
            dataset_classification=dataset_classification,
        )

    def get_aggregate_metrics(self, dataset_classification: Optional[str] = None) -> AggregateMetrics:
        """Calculates aggregate metrics across evaluation runs, optionally filtered by dataset_classification."""
        return self.storage.get_aggregate_metrics(dataset_classification=dataset_classification)

    def export_csv(self) -> str:
        """Exports evaluation runs as CSV."""
        return self.storage.export_csv()

    def export_json(self) -> str:
        """Exports evaluation runs as JSON."""
        return self.storage.export_json()

    # -------------------------------------------------------------------------
    # Offline Derived Re-Scoring (Isolated from Live Runs)
    # -------------------------------------------------------------------------

    def generate_derived_rescore(self, source_run_id: str, persist: bool = True) -> DerivedAssessment:
        """Performs an offline deterministic derived re-scoring of a historical evaluation run.

        Guarantees:
        - Strictly 0 external provider calls and 0 credit usage.
        - Source run in evaluation_runs is completely untouched.
        - Preserves original raw WER, S/D/I counts, provider request IDs, and latencies.
        - Computes supplementary cross-script-normalized WER and corrected Hindi intent matching.
        """
        source_run = self.get_run(source_run_id)
        if not source_run:
            raise ValueError(f"Source evaluation run '{source_run_id}' not found.")

        test_case = self.get_test_case(source_run.test_case_id)
        if not test_case:
            raise ValueError(f"Test case '{source_run.test_case_id}' not found.")

        # Re-score deterministically using existing stored transcripts and references
        res = score_pair(
            run_id=f"rescore_{source_run_id}",
            test_case=test_case,
            transcript_a=source_run.utterance_a.actual_transcript,
            transcript_b=source_run.utterance_b.actual_transcript,
            execution_mode=source_run.execution_mode.value if hasattr(source_run.execution_mode, "value") else str(source_run.execution_mode),
            audio_condition=source_run.audio_condition,
            timestamp=datetime.now(timezone.utc).isoformat(),
            client_latency_ms=source_run.client_measured_latency_ms,
            client_request_elapsed_ms=source_run.client_request_elapsed_ms,
            provider_latency_ms=source_run.provider_processing_time_ms,
            simulated_latency_ms=source_run.simulated_latency_ms,
            latency_a_ms=source_run.utterance_a.latency_ms,
            latency_b_ms=source_run.utterance_b.latency_ms,
            client_elapsed_a_ms=source_run.utterance_a.client_request_elapsed_ms,
            client_elapsed_b_ms=source_run.utterance_b.client_request_elapsed_ms,
            provider_time_a_ms=source_run.utterance_a.provider_processing_time_ms,
            provider_time_b_ms=source_run.utterance_b.provider_processing_time_ms,
            request_id_a=source_run.utterance_a.request_id,
            request_id_b=source_run.utterance_b.request_id,
        )

        norm_wer_a, ns_a, nd_a, ni_a, _ = calculate_cross_script_normalized_wer(
            reference=test_case.utterance_a.reference_transcript,
            hypothesis=source_run.utterance_a.actual_transcript,
            language_code=test_case.language_code,
        )
        norm_wer_b, ns_b, nd_b, ni_b, _ = calculate_cross_script_normalized_wer(
            reference=test_case.utterance_b.reference_transcript,
            hypothesis=source_run.utterance_b.actual_transcript,
            language_code=test_case.language_code,
        )

        assessment = DerivedAssessment(
            derived_id=f"rescore_{source_run_id}",
            source_run_id=source_run_id,
            test_case_id=source_run.test_case_id,
            category=source_run.category,
            language_code=source_run.language_code,
            provenance="offline_deterministic_rescore",
            created_at=datetime.now(timezone.utc).isoformat(),
            provider_calls_made=0,
            credits_consumed=0,

            # Original historical evidence
            original_execution_mode=source_run.execution_mode.value if hasattr(source_run.execution_mode, "value") else str(source_run.execution_mode),
            original_overall_status=source_run.overall_status,
            original_raw_wer_a=source_run.wer_a,
            original_raw_wer_b=source_run.wer_b,
            original_average_raw_wer=source_run.average_wer,
            original_substitutions_a=source_run.utterance_a.substitutions,
            original_substitutions_b=source_run.utterance_b.substitutions,
            original_deletions_a=source_run.utterance_a.deletions,
            original_deletions_b=source_run.utterance_b.deletions,
            original_insertions_a=source_run.utterance_a.insertions,
            original_insertions_b=source_run.utterance_b.insertions,
            original_detected_intent_a=source_run.utterance_a.detected_intent,
            original_detected_intent_b=source_run.utterance_b.detected_intent,
            original_task_success=source_run.task_success,
            original_primary_failure_category=source_run.primary_failure_category.value if hasattr(source_run.primary_failure_category, "value") else str(source_run.primary_failure_category),
            original_failure_explanation=source_run.failure_explanation,
            request_id_a=source_run.utterance_a.request_id,
            request_id_b=source_run.utterance_b.request_id,
            client_request_elapsed_ms=source_run.client_request_elapsed_ms,
            latency_a_ms=source_run.utterance_a.latency_ms,
            latency_b_ms=source_run.utterance_b.latency_ms,

            # Text payloads
            reference_a=source_run.utterance_a.reference_transcript,
            reference_b=source_run.utterance_b.reference_transcript,
            actual_transcript_a=source_run.utterance_a.actual_transcript,
            actual_transcript_b=source_run.utterance_b.actual_transcript,

            # Derived Cross-Script Normalized Metrics
            normalized_wer_a=norm_wer_a,
            normalized_wer_b=norm_wer_b,
            average_normalized_wer=round((norm_wer_a + norm_wer_b) / 2.0, 4),
            normalized_substitutions_a=ns_a,
            normalized_substitutions_b=ns_b,
            normalized_deletions_a=nd_a,
            normalized_deletions_b=nd_b,
            normalized_insertions_a=ni_a,
            normalized_insertions_b=ni_b,

            # Corrected Deterministic Intents & Outcomes
            rescore_detected_intent_a=res.utterance_a.detected_intent,
            rescore_detected_intent_b=res.utterance_b.detected_intent,
            rescore_intent_match_a=res.utterance_a.intent_match,
            rescore_intent_match_b=res.utterance_b.intent_match,
            rescore_critical_fields_match_a=res.utterance_a.critical_fields_match,
            rescore_critical_fields_match_b=res.utterance_b.critical_fields_match,
            rescore_task_success_a=res.utterance_a.task_success,
            rescore_task_success_b=res.utterance_b.task_success,
            rescore_semantic_distinction_preserved=res.semantic_distinction_preserved,
            rescore_task_success=res.task_success,
            rescore_primary_failure_category=res.primary_failure_category.value if hasattr(res.primary_failure_category, "value") else str(res.primary_failure_category),
            rescore_failure_explanation=res.failure_explanation,
        )

        if persist:
            self.storage.save_derived_assessment(assessment)

        return assessment

    def get_derived_assessment(self, identifier: str) -> Optional[DerivedAssessment]:
        """Retrieves a derived assessment by derived_id or source_run_id."""
        return self.storage.get_derived_assessment(identifier)

    def list_derived_assessments(self, limit: int = 100) -> List[DerivedAssessment]:
        """Lists offline derived evaluation assessments."""
        return self.storage.list_derived_assessments(limit=limit)

    def export_derived_csv(self) -> str:
        """Exports offline derived assessments as CSV."""
        return self.storage.export_derived_assessments_csv()

    def export_derived_json(self) -> str:
        """Exports offline derived assessments as JSON."""
        return self.storage.export_derived_assessments_json()
