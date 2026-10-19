"""Gnani Prisma Speech-to-Text API client.

Implements the official v3 REST specification verified from Gnani AI documentation:
- Endpoint: POST https://api.vachana.ai/stt/v3
- Header: X-API-Key-ID: <api-key>
- Content-Type: multipart/form-data
"""

import json
import logging
import time
from typing import Any, Dict, Optional
import httpx

from app.config import Settings, get_settings

logger = logging.getLogger("bolibench.gnani_client")


class GnaniClientError(Exception):
    """Base exception for Gnani Prisma STT client operations."""

    def __init__(self, message: str, status_code: int = 500, error_type: str = "API_ERROR", details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.details = details or {}


class MissingCredentialsError(GnaniClientError):
    """Raised when GNANI_API_KEY is not configured."""

    def __init__(self, message: str = "Gnani API key is missing. Set GNANI_API_KEY in your .env file."):
        super().__init__(message=message, status_code=401, error_type="MISSING_CREDENTIALS")


class GnaniBadRequestError(GnaniClientError):
    """Raised on 400 Bad Request from Gnani STT API."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message=message, status_code=400, error_type="INVALID_REQUEST_ERROR", details=details)


class GnaniAuthenticationError(GnaniClientError):
    """Raised on 401 Unauthorized or 403 Forbidden from Gnani STT API."""

    def __init__(self, message: str, status_code: int = 401, details: Optional[dict] = None):
        error_type = "UNAUTHORIZED" if status_code == 401 else "FORBIDDEN"
        super().__init__(message=message, status_code=status_code, error_type=error_type, details=details)


class GnaniRateLimitError(GnaniClientError):
    """Raised on 429 Rate Limit from Gnani STT API."""

    def __init__(self, message: str = "Rate limit exceeded. Please wait before retrying.", details: Optional[dict] = None):
        super().__init__(message=message, status_code=429, error_type="RATE_LIMIT_ERROR", details=details)


class GnaniServerError(GnaniClientError):
    """Raised on 500 Internal Server Error or upstream response issues from Gnani STT API."""

    def __init__(self, message: str, status_code: int = 500, error_type: str = "API_ERROR", details: Optional[dict] = None):
        super().__init__(message=message, status_code=status_code, error_type=error_type, details=details)


class GnaniServiceUnavailableError(GnaniClientError):
    """Raised on 503 Service Unavailable from Gnani STT API."""

    def __init__(self, message: str = "Gnani STT service is temporarily unavailable.", details: Optional[dict] = None):
        super().__init__(message=message, status_code=503, error_type="SERVICE_UNAVAILABLE", details=details)


class GnaniTimeoutError(GnaniClientError):
    """Raised when an HTTP request times out."""

    def __init__(self, timeout_seconds: float):
        message = f"Gnani STT API request timed out after {timeout_seconds:.1f} seconds."
        super().__init__(message=message, status_code=504, error_type="GATEWAY_TIMEOUT")


class GnaniNetworkError(GnaniClientError):
    """Raised when network connection fails."""

    def __init__(self, message: str = "Failed to establish connection to Gnani STT API."):
        super().__init__(message=message, status_code=502, error_type="BAD_GATEWAY")


class GnaniPrismaClient:
    """Production client for Gnani Prisma v2.5 STT (REST v3)."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.settings = settings or get_settings()
        self._http_client = http_client

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
        language_code: str,
        format_mode: str = "verbatim",
        itn_native_numerals: bool = False,
        enable_substitution: bool = False,
        substitution_map: Optional[str] = None,
        bias_list: Optional[str] = None,
        bias_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Transcribes audio using Gnani Prisma v2.5 REST STT API.

        Measures and returns request latency in milliseconds.
        """
        if not self.settings.is_api_configured:
            raise MissingCredentialsError(
                "Gnani Prisma API key is not configured. Please add GNANI_API_KEY to your .env file."
            )

        endpoint = self.settings.gnani_base_url
        headers = {
            "X-API-Key-ID": self.settings.gnani_api_key.strip(),  # type: ignore[union-attr]
        }

        # Prepare multipart form data
        data_fields: Dict[str, str] = {
            "language_code": language_code,
            "format": format_mode,
        }

        if format_mode == "transcribe" and itn_native_numerals:
            data_fields["itn_native_numerals"] = "true"

        if enable_substitution and substitution_map:
            data_fields["enable_substitution"] = "true"
            data_fields["substitution_map"] = substitution_map

        if bias_list and bias_score and bias_score > 0.0:
            data_fields["bias_list"] = bias_list
            data_fields["bias_score"] = str(bias_score)

        files = {
            "audio_file": (filename, audio_bytes, content_type),
        }

        timeout = httpx.Timeout(self.settings.gnani_request_timeout_seconds)

        start_time = time.perf_counter()

        try:
            if self._http_client:
                response = await self._http_client.post(
                    endpoint,
                    headers=headers,
                    data=data_fields,
                    files=files,
                    timeout=timeout,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        data=data_fields,
                        files=files,
                        timeout=timeout,
                    )
        except httpx.TimeoutException:
            raise GnaniTimeoutError(self.settings.gnani_request_timeout_seconds)
        except httpx.ConnectError as exc:
            logger.error("Connection error while connecting to Gnani STT API: %s", exc)
            raise GnaniNetworkError(f"Cannot connect to Gnani STT endpoint: {endpoint}")
        except httpx.RequestError as exc:
            logger.error("Network request error: %s", exc)
            raise GnaniNetworkError(f"Network error during transcription request: {str(exc)}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return self._handle_response(response, elapsed_ms, language_code, format_mode)

    def _handle_response(
        self,
        response: httpx.Response,
        elapsed_ms: float,
        language_code: str,
        format_mode: str,
    ) -> Dict[str, Any]:
        """Parses and validates the HTTP response from Gnani API."""
        status_code = response.status_code

        try:
            resp_json = response.json()
        except Exception:
            resp_json = {}

        if not isinstance(resp_json, dict):
            resp_json = {}

        if status_code == 200:
            if "transcript" not in resp_json:
                raise GnaniServerError(
                    "Gnani STT returned 200 OK but response is missing the required 'transcript' field.",
                    status_code=502,
                    error_type="MALFORMED_RESPONSE",
                    details={"response_payload": resp_json},
                )

            transcript = resp_json.get("transcript", "")
            request_id = resp_json.get("request_id")
            # Never assume the API always returns a model name
            model = resp_json.get("model")
            timestamp = resp_json.get("timestamp")

            client_elapsed = round(elapsed_ms, 2)
            # Inspect official Gnani response payload for provider processing time
            # Note: The official Gnani STT v3 REST API does not report provider-side processing duration
            prov_processing_time = resp_json.get("provider_processing_time_ms")
            if prov_processing_time is None:
                prov_processing_time = resp_json.get("processing_time_ms")
            if prov_processing_time is not None:
                try:
                    prov_processing_time = round(float(prov_processing_time), 2)
                except (ValueError, TypeError):
                    prov_processing_time = None

            return {
                "success": True,
                "transcript": transcript,
                "request_id": request_id,
                "model": model,
                "server_timestamp": timestamp,
                "language_code": language_code,
                "format": format_mode,
                "client_request_elapsed_ms": client_elapsed,
                "client_measured_latency_ms": client_elapsed,
                "backend_latency_ms": client_elapsed,
                "provider_processing_time_ms": prov_processing_time,
                "is_mocked": False,
            }

        # Error handling mapping
        error_detail = resp_json.get("error", {}) if isinstance(resp_json.get("error"), dict) else {}
        error_type = error_detail.get("type", "UNKNOWN_ERROR")
        error_message = error_detail.get("message") or response.text or f"HTTP {status_code} error from Gnani STT"

        if status_code == 400:
            raise GnaniBadRequestError(error_message, details={"api_error_type": error_type})
        elif status_code == 401:
            raise GnaniAuthenticationError(
                f"Gnani authentication failed (401 Unauthorized): {error_message}",
                status_code=401,
                details={"api_error_type": error_type},
            )
        elif status_code == 403:
            raise GnaniAuthenticationError(
                f"Gnani authentication failed (403 Forbidden): {error_message}",
                status_code=403,
                details={"api_error_type": error_type},
            )
        elif status_code == 429:
            raise GnaniRateLimitError(
                f"Gnani STT rate limit exceeded: {error_message}",
                details={"api_error_type": error_type},
            )
        elif status_code == 503:
            raise GnaniServiceUnavailableError(
                f"Gnani STT service unavailable: {error_message}",
                details={"api_error_type": error_type},
            )
        else:
            raise GnaniServerError(
                f"Gnani STT server error ({status_code}): {error_message}",
                details={"api_error_type": error_type, "status_code": status_code},
            )
