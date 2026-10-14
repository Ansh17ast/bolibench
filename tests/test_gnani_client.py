"""Unit and integration tests for GnaniPrismaClient with deterministic mocks."""

import httpx
import pytest
from app.services.gnani_client import (
    GnaniAuthenticationError,
    GnaniBadRequestError,
    GnaniNetworkError,
    GnaniPrismaClient,
    GnaniRateLimitError,
    GnaniServerError,
    GnaniServiceUnavailableError,
    GnaniTimeoutError,
    MissingCredentialsError,
)


@pytest.mark.asyncio
async def test_gnani_client_missing_credentials(unconfigured_settings, valid_wav_bytes):
    """Client raises MissingCredentialsError when GNANI_API_KEY is not configured."""
    client = GnaniPrismaClient(settings=unconfigured_settings)
    with pytest.raises(MissingCredentialsError) as exc_info:
        await client.transcribe(
            audio_bytes=valid_wav_bytes,
            filename="test.wav",
            content_type="audio/wav",
            language_code="hi-IN",
        )
    assert "GNANI_API_KEY" in exc_info.value.message


@pytest.mark.asyncio
async def test_gnani_client_success_200(mock_settings, valid_wav_bytes):
    """Client successfully handles 200 OK response matching official Prisma v3 schema."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        # Verify official auth header
        assert request.headers.get("X-API-Key-ID") == "test_mock_api_key_12345"
        assert "multipart/form-data" in request.headers.get("content-type", "")

        return httpx.Response(
            status_code=200,
            json={
                "success": True,
                "request_id": "req_mock_12345678",
                "timestamp": "20261014_233000.123",
                "transcript": "नमस्ते, आप कैसे हैं?",
                "model": "gnani-prisma-v2.5",
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
        result = await client.transcribe(
            audio_bytes=valid_wav_bytes,
            filename="hindi_sample.wav",
            content_type="audio/wav",
            language_code="hi-IN",
            format_mode="transcribe",
            itn_native_numerals=True,
        )

        assert result["success"] is True
        assert result["transcript"] == "नमस्ते, आप कैसे हैं?"
        assert result["request_id"] == "req_mock_12345678"
        assert result["model"] == "gnani-prisma-v2.5"
        assert result["backend_latency_ms"] >= 0.0
        assert result["is_mocked"] is False


@pytest.mark.asyncio
async def test_gnani_client_bad_request_400(mock_settings, valid_wav_bytes):
    """Client translates 400 Bad Request to GnaniBadRequestError."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=400,
            json={
                "success": False,
                "error": {
                    "type": "INVALID_REQUEST_ERROR",
                    "message": "Audio duration exceeds maximum limit of 60 seconds.",
                },
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
        with pytest.raises(GnaniBadRequestError) as exc_info:
            await client.transcribe(
                audio_bytes=valid_wav_bytes,
                filename="test.wav",
                content_type="audio/wav",
                language_code="hi-IN",
            )
        assert exc_info.value.status_code == 400
        assert "exceeds maximum limit" in exc_info.value.message


@pytest.mark.asyncio
async def test_gnani_client_forbidden_403(mock_settings, valid_wav_bytes):
    """Client translates 403 Forbidden to GnaniAuthenticationError."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=403,
            json={
                "success": False,
                "error": {
                    "type": "FORBIDDEN",
                    "message": "You do not have permission to access this resource. Check API key or credits.",
                },
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
        with pytest.raises(GnaniAuthenticationError) as exc_info:
            await client.transcribe(
                audio_bytes=valid_wav_bytes,
                filename="test.wav",
                content_type="audio/wav",
                language_code="en-IN",
            )
        assert exc_info.value.status_code == 403
        assert "authentication failed" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_gnani_client_rate_limit_429(mock_settings, valid_wav_bytes):
    """Client translates 429 to GnaniRateLimitError."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            json={
                "success": False,
                "error": {
                    "type": "RATE_LIMIT_ERROR",
                    "message": "Rate limit exceeded. Please try again later.",
                },
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
        with pytest.raises(GnaniRateLimitError) as exc_info:
            await client.transcribe(
                audio_bytes=valid_wav_bytes,
                filename="test.wav",
                content_type="audio/wav",
                language_code="hi-IN",
            )
        assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_gnani_client_service_unavailable_503(mock_settings, valid_wav_bytes):
    """Client translates 503 to GnaniServiceUnavailableError."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=503,
            json={
                "success": False,
                "error": {
                    "type": "SERVICE_UNAVAILABLE",
                    "message": "Speech recognition service is temporarily unavailable.",
                },
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
        with pytest.raises(GnaniServiceUnavailableError) as exc_info:
            await client.transcribe(
                audio_bytes=valid_wav_bytes,
                filename="test.wav",
                content_type="audio/wav",
                language_code="hi-IN",
            )
        assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_gnani_client_server_error_500(mock_settings, valid_wav_bytes):
    """Client translates 500 to GnaniServerError."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=500,
            json={
                "success": False,
                "error": {
                    "type": "API_ERROR",
                    "message": "An unexpected error occurred while processing.",
                },
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
        with pytest.raises(GnaniServerError) as exc_info:
            await client.transcribe(
                audio_bytes=valid_wav_bytes,
                filename="test.wav",
                content_type="audio/wav",
                language_code="hi-IN",
            )
        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_gnani_client_timeout(mock_settings, valid_wav_bytes):
    """Client catches httpx.TimeoutException and raises GnaniTimeoutError."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Mocked request timeout")

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
        with pytest.raises(GnaniTimeoutError) as exc_info:
            await client.transcribe(
                audio_bytes=valid_wav_bytes,
                filename="test.wav",
                content_type="audio/wav",
                language_code="hi-IN",
            )
        assert exc_info.value.status_code == 504
        assert "timed out" in exc_info.value.message


@pytest.mark.asyncio
async def test_gnani_client_unauthorized_401(mock_settings, valid_wav_bytes):
    """Client translates 401 Unauthorized to GnaniAuthenticationError with 401 status."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=401,
            json={
                "success": False,
                "error": {
                    "type": "UNAUTHORIZED",
                    "message": "Invalid API Key provided.",
                },
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
        with pytest.raises(GnaniAuthenticationError) as exc_info:
            await client.transcribe(
                audio_bytes=valid_wav_bytes,
                filename="test.wav",
                content_type="audio/wav",
                language_code="hi-IN",
            )
        assert exc_info.value.status_code == 401
        assert exc_info.value.error_type == "UNAUTHORIZED"
        assert "authentication failed" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_gnani_client_success_200_without_model(mock_settings, valid_wav_bytes):
    """Client handles 200 OK without assuming a model name is returned."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "success": True,
                "request_id": "req_no_model_123",
                "timestamp": "20261015_000000.000",
                "transcript": "बिना मॉडल का जवाब",
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
        result = await client.transcribe(
            audio_bytes=valid_wav_bytes,
            filename="test.wav",
            content_type="audio/wav",
            language_code="hi-IN",
        )
        assert result["success"] is True
        assert result["transcript"] == "बिना मॉडल का जवाब"
        assert result["model"] is None


@pytest.mark.asyncio
async def test_gnani_client_malformed_response_missing_transcript(mock_settings, valid_wav_bytes):
    """Client raises GnaniServerError on 200 response missing the required transcript field."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "success": True,
                "request_id": "req_missing_transcript",
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
        with pytest.raises(GnaniServerError) as exc_info:
            await client.transcribe(
                audio_bytes=valid_wav_bytes,
                filename="test.wav",
                content_type="audio/wav",
                language_code="hi-IN",
            )
        assert exc_info.value.error_type == "MALFORMED_RESPONSE"
