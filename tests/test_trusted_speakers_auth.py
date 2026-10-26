"""Tests for Administrative Authorization Guard on Trusted Speaker Endpoints.

Verifies:
1. Server unconfigured (ADMIN_API_KEY is None or empty): Fails closed with 403 Forbidden.
2. Missing credentials (no Authorization or X-Admin-Token header): Rejects with 401 Unauthorized.
3. Empty credentials (empty Bearer or empty X-Admin-Token): Rejects with 401 Unauthorized.
4. Invalid credentials (incorrect key): Rejects with 403 Forbidden.
5. Valid Authorization: Bearer <key>: Authorizes registration (200 OK).
6. Valid X-Admin-Token: <key>: Authorizes registration (200 OK).
7. Test/mock prefixes are strictly rejected even with valid administrative credentials (400 Bad Request).
8. Absence of bypass: No alternate routes or methods allow unauthenticated speaker modification.
9. Credential privacy: Settings.__repr__ never leaks the administrative key.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.config import get_settings


TEST_ADMIN_SECRET = "adm_secret_key_verified_secure_7731"


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Ensure cached settings singleton is cleared before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_auth_server_unconfigured_fails_closed(monkeypatch, isolate_recording_storage):
    """When ADMIN_API_KEY is not configured on the server, endpoint must fail closed with 403."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    get_settings.cache_clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Request with no credentials
        res_no_cred = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_authorized_human_01"},
        )
        assert res_no_cred.status_code == 403
        assert "ADMIN_API_KEY is not configured" in res_no_cred.json()["detail"]

        # Request even with attempted token
        res_with_token = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_authorized_human_01"},
            headers={"Authorization": "Bearer any_token"},
        )
        assert res_with_token.status_code == 403
        assert "ADMIN_API_KEY is not configured" in res_with_token.json()["detail"]


@pytest.mark.asyncio
async def test_admin_auth_missing_credentials_rejected(monkeypatch, isolate_recording_storage):
    """When ADMIN_API_KEY is configured, unauthenticated requests must be rejected with 401."""
    monkeypatch.setenv("ADMIN_API_KEY", TEST_ADMIN_SECRET)
    get_settings.cache_clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_authorized_human_01"},
        )
        assert res.status_code == 401
        assert "Missing administrative credentials" in res.json()["detail"]
        assert "Bearer" in res.headers.get("WWW-Authenticate", "")


@pytest.mark.asyncio
async def test_admin_auth_empty_credentials_rejected(monkeypatch, isolate_recording_storage):
    """Empty or whitespace-only credentials must be rejected with 401."""
    monkeypatch.setenv("ADMIN_API_KEY", TEST_ADMIN_SECRET)
    get_settings.cache_clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Empty Bearer token
        res1 = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_authorized_human_01"},
            headers={"Authorization": "Bearer "},
        )
        assert res1.status_code == 401

        # Empty X-Admin-Token
        res2 = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_authorized_human_01"},
            headers={"X-Admin-Token": "   "},
        )
        assert res2.status_code == 401


@pytest.mark.asyncio
async def test_admin_auth_invalid_credentials_rejected(monkeypatch, isolate_recording_storage):
    """Invalid credentials must be rejected with 403 Forbidden."""
    monkeypatch.setenv("ADMIN_API_KEY", TEST_ADMIN_SECRET)
    get_settings.cache_clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Invalid Bearer token
        res_bearer = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_authorized_human_01"},
            headers={"Authorization": "Bearer wrong_secret_key"},
        )
        assert res_bearer.status_code == 403
        assert "Invalid administrative credentials" in res_bearer.json()["detail"]

        # Invalid X-Admin-Token
        res_custom = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_authorized_human_01"},
            headers={"X-Admin-Token": "wrong_secret_key"},
        )
        assert res_custom.status_code == 403
        assert "Invalid administrative credentials" in res_custom.json()["detail"]


@pytest.mark.asyncio
async def test_admin_auth_bearer_token_success(monkeypatch, isolate_recording_storage):
    """Valid Bearer token authorizes registration and updates registry."""
    monkeypatch.setenv("ADMIN_API_KEY", TEST_ADMIN_SECRET)
    get_settings.cache_clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_valid_bearer_01"},
            headers={"Authorization": f"Bearer {TEST_ADMIN_SECRET}"},
        )
        assert res.status_code == 200
        assert res.json() == {"status": "registered", "speaker_id": "spk_valid_bearer_01"}

        # Verify listed in public read endpoint
        list_res = await client.get("/api/studio/trusted-speakers")
        assert list_res.status_code == 200
        assert "spk_valid_bearer_01" in list_res.json()["trusted_speakers"]


@pytest.mark.asyncio
async def test_admin_auth_x_admin_token_success(monkeypatch, isolate_recording_storage):
    """Valid X-Admin-Token authorizes registration and updates registry."""
    monkeypatch.setenv("ADMIN_API_KEY", TEST_ADMIN_SECRET)
    get_settings.cache_clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_valid_custom_02"},
            headers={"X-Admin-Token": TEST_ADMIN_SECRET},
        )
        assert res.status_code == 200
        assert res.json() == {"status": "registered", "speaker_id": "spk_valid_custom_02"}

        # Verify listed in public read endpoint
        list_res = await client.get("/api/studio/trusted-speakers")
        assert list_res.status_code == 200
        assert "spk_valid_custom_02" in list_res.json()["trusted_speakers"]


@pytest.mark.asyncio
async def test_admin_auth_rejects_test_prefixes_even_when_authorized(monkeypatch, isolate_recording_storage):
    """Test and mock speaker prefixes must be rejected with 400 even if caller is an authorized admin."""
    monkeypatch.setenv("ADMIN_API_KEY", TEST_ADMIN_SECRET)
    get_settings.cache_clear()

    test_prefixes = ["spk_test_user", "spk_mock_01", "spk_fail_check", "spk_acc_verify"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for prefix in test_prefixes:
            res = await client.post(
                "/api/studio/trusted-speakers",
                json={"speaker_id": prefix},
                headers={"Authorization": f"Bearer {TEST_ADMIN_SECRET}"},
            )
            assert res.status_code == 400
            assert "Cannot register test/mock speaker" in res.json()["detail"]


@pytest.mark.asyncio
async def test_admin_auth_no_bypass_via_alternate_methods_or_routes(monkeypatch, isolate_recording_storage):
    """Confirm no alternate HTTP methods or unauthenticated routes can modify trusted speakers."""
    monkeypatch.setenv("ADMIN_API_KEY", TEST_ADMIN_SECRET)
    get_settings.cache_clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # PUT /api/studio/trusted-speakers -> 405 Method Not Allowed
        put_res = await client.put(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_bypass_attempt"},
        )
        assert put_res.status_code == 405

        # DELETE /api/studio/trusted-speakers -> 405 Method Not Allowed
        del_res = await client.delete("/api/studio/trusted-speakers")
        assert del_res.status_code == 405

        # Non-existent route /api/recording/trusted-speakers -> 404
        old_route_res = await client.post(
            "/api/recording/trusted-speakers",
            json={"speaker_id": "spk_bypass_attempt"},
        )
        assert old_route_res.status_code == 404


def test_settings_repr_never_leaks_admin_api_key(monkeypatch):
    """Proves Settings.__repr__ never reveals the admin secret key value."""
    secret = "very_secret_admin_token_never_log_this_xyz999"
    monkeypatch.setenv("ADMIN_API_KEY", secret)
    get_settings.cache_clear()

    settings = get_settings()
    repr_str = repr(settings)

    assert secret not in repr_str
    assert "admin_configured=True" in repr_str
