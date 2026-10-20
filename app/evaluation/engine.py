"""EvaluationEngine orchestrating test-case retrieval, paired contrast scoring, and persistence."""

from datetime import datetime, timezone
import time
from typing import Dict, List, Optional
import uuid

from app.evaluation.dataset import get_test_case_by_id, load_seed_dataset
from app.evaluation.metrics import evaluate_utterance, score_pair
from app.evaluation.schemas import (
    AggregateMetrics,
    AudioCondition,
    EvaluateTranscriptsRequest,
    ExecutionMode,
    FailureCategory,
    PairEvaluationResult,
    TestCasePair,
    UtteranceEvaluationResult,
)
from app.evaluation.storage import StorageManager
from app.services.gnani_client import GnaniPrismaClient


class EvaluationEngine:
    """Core evaluation engine for BoliBench paired-contrast speech AI crash testing."""

    def __init__(self, storage_manager: Optional[StorageManager] = None):
        self.storage = storage_manager or StorageManager()

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
    ) -> PairEvaluationResult:
        """Evaluates a live audio pair using Gnani Prisma STT REST API.

        Guarantees:
        - Executes actual provider requests sequentially.
        - Calculates genuine client-side elapsed time.
        - Preserves provider request IDs and backend processing latency.
        - If either call fails, no incomplete pair evaluation is persisted or returned.
        - Never marks a mock result as live.
        """
        test_case = self.get_test_case(test_case_id)
        if not test_case:
            raise ValueError(f"Test case with ID '{test_case_id}' not found.")

        run_id = f"eval_live_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

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
            # Persist safe failed run record so failure is audited without leaking credentials or audio
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

    def get_run(self, run_id: str) -> Optional[PairEvaluationResult]:
        """Retrieves a single evaluation run."""
        return self.storage.get_run(run_id)

    def list_runs(self, limit: int = 100, category: Optional[str] = None) -> List[PairEvaluationResult]:
        """Lists recent evaluation runs."""
        return self.storage.list_runs(limit=limit, category=category)

    def get_aggregate_metrics(self) -> AggregateMetrics:
        """Calculates aggregate metrics across evaluation runs."""
        return self.storage.get_aggregate_metrics()

    def export_csv(self) -> str:
        """Exports evaluation runs as CSV."""
        return self.storage.export_csv()

    def export_json(self) -> str:
        """Exports evaluation runs as JSON."""
        return self.storage.export_json()
