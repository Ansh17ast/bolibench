"""Server-side live evaluation credit guard and authorization manager.

Enforces:
1. Origin & Fetch Metadata Validation: Rejects cross-origin / cross-site requests.
2. Cryptographic Single-Use Authorization Tokens: Live evaluation requires an explicit,
   validated authorization handshake before reading audio uploads or invoking provider APIs.
3. Anti-CSRF Custom Headers: Requires a custom header that cannot be forged by standard HTML forms.
4. Privacy & Masking: Confirmation tokens and API credentials never appear in logs or responses.
"""

import logging
import secrets
import time
from typing import Dict, NamedTuple, Optional
from urllib.parse import urlparse
from fastapi import HTTPException, Request, status

logger = logging.getLogger("bolibench.live_guard")


class AuthTokenRecord(NamedTuple):
    test_id: str
    token: str
    created_at: float
    expires_at: float
    consumed: bool


class LiveEvaluationGuard:
    """Manages authorization tokens and validates same-origin requests for live evaluations.

    Token Architecture:
    - Cryptographically random 256-bit token (secrets.token_hex(32)).
    - Server-stored in an in-memory dictionary (_tokens) with TTL of 300 seconds (5 minutes).
    - Single-use: marked consumed immediately upon verification to prevent replay attacks.
    - Not a signed JWT; validation is stateful and bound to the issued test_case_id.
    """

    REQUIRED_CONFIRM_HEADER = "X-BoliBench-Confirm-Live"
    EXPECTED_HEADER_VALUE = "CONFIRM_LIVE_EVALUATION"
    TOKEN_TTL_SECONDS = 300.0  # 5 minutes validity

    # Explicit complete allowed origins (scheme + host + port)
    ALLOWED_SCHEMES = {"http", "https"}
    ALLOWED_ORIGINS = {
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://test",
        "http://testserver",
        "https://test",
        "https://testserver",
    }

    def __init__(self):
        # In-memory server token store: { token: AuthTokenRecord }
        self._tokens: Dict[str, AuthTokenRecord] = {}

    def issue_authorization_token(self, test_id: str, request: Request) -> str:
        """Issues a cryptographically random, server-stored, time-limited, single-use authorization token."""
        self.validate_same_origin(request)

        token = secrets.token_hex(32)
        now = time.time()
        record = AuthTokenRecord(
            test_id=test_id,
            token=token,
            created_at=now,
            expires_at=now + self.TOKEN_TTL_SECONDS,
            consumed=False,
        )
        self._tokens[token] = record
        self._cleanup_expired(now)

        # Security: Never log the confirmation token itself
        logger.info("Live evaluation authorization token issued for test_id=%s", test_id)
        return token

    def validate_same_origin(self, request: Request) -> None:
        """Rejects cross-origin, cross-site, or untrusted-origin requests to live evaluation routes.

        Origin Verification Policy:
        - In browser flows, browsers provide either 'Origin' (on POST) or 'Referer'.
        - Validates complete origins (scheme, host, and port) against explicit allowed local origins.
        - Requests missing both Origin and Referer are rejected on non-local hosts.
        - Documented Development / Test Exception:
          Automated test harnesses (e.g. starlette TestClient targeting 'testserver' or 'test')
          and local loopback development callers on 127.0.0.1 or localhost may omit Origin headers.
        """
        # 1. Sec-Fetch-Site (modern browser defense against CSRF)
        fetch_site = request.headers.get("sec-fetch-site")
        if fetch_site in ("cross-site",):
            logger.warning("Rejected cross-site request to live evaluation (sec-fetch-site=%s)", fetch_site)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Cross-origin requests to live evaluation are prohibited to protect programme credits.",
            )

        # 2. Origin & Referer Verification
        raw_origin = request.headers.get("origin") or request.headers.get("referer")
        req_hostname = (request.url.hostname or "").lower()

        if not raw_origin:
            # Documented development/test exception:
            # Test harnesses (host='test' or 'testserver' in pytest/httpx) and local loopback development callers
            # on 127.0.0.1 or localhost may omit Origin headers.
            # Non-local hosts (public deployments, remote domains) are strictly rejected.
            if req_hostname in ("test", "testserver", "localhost", "127.0.0.1"):
                logger.debug(
                    "Permitting missing Origin under documented local development/test exception for host=%s",
                    req_hostname,
                )
            else:
                logger.warning(
                    "Rejected live evaluation request: Missing Origin or Referer header on non-local host '%s'",
                    req_hostname,
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Missing Origin or Referer header. Live evaluation requires verified browser origin information.",
                )
        else:
            parsed = urlparse(raw_origin)
            origin_scheme = (parsed.scheme or "").lower()
            origin_netloc = (parsed.netloc or "").lower()

            if origin_scheme not in self.ALLOWED_SCHEMES or not origin_netloc:
                logger.warning("Rejected live evaluation with untrusted origin: %s", raw_origin)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Untrusted origin. Cross-origin access to live evaluation is prohibited.",
                )

            full_origin = f"{origin_scheme}://{origin_netloc}"
            req_origin = f"{(request.url.scheme or 'http').lower()}://{(request.url.netloc or '').lower()}"

            is_valid = full_origin in self.ALLOWED_ORIGINS
            if not is_valid and req_hostname in ("localhost", "127.0.0.1", "test", "testserver"):
                is_valid = (full_origin == req_origin)

            if not is_valid:
                logger.warning("Rejected live evaluation with untrusted origin: %s", raw_origin)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Untrusted origin. Cross-origin access to live evaluation is prohibited.",
                )

    def verify_authorization(
        self,
        test_id: str,
        confirmation_token: Optional[str],
        request: Request,
    ) -> None:
        """Verifies that the live evaluation request is authorized, same-origin, and confirmed.

        Must be executed BEFORE reading audio files or initiating any network calls.
        """
        # Step 1: Same-origin check
        self.validate_same_origin(request)

        # Step 2: Validate anti-CSRF custom header
        header_val = request.headers.get(self.REQUIRED_CONFIRM_HEADER)
        if not header_val or header_val.strip() != self.EXPECTED_HEADER_VALUE:
            logger.warning("Live evaluation rejected: missing or invalid %s header", self.REQUIRED_CONFIRM_HEADER)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Missing required authorization header '{self.REQUIRED_CONFIRM_HEADER}: {self.EXPECTED_HEADER_VALUE}'.",
            )

        # Step 3: Validate confirmation token
        if not confirmation_token or not confirmation_token.strip():
            logger.warning("Live evaluation rejected: missing confirmation token for test_id=%s", test_id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Missing live evaluation authorization token. Explicit authorization via /api/eval/authorize-live is required.",
            )

        token = confirmation_token.strip()
        record = self._tokens.get(token)

        if not record:
            logger.warning("Live evaluation rejected: token not found or invalid for test_id=%s", test_id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid authorization token. Please authorize again.",
            )

        now = time.time()
        if now > record.expires_at:
            logger.warning("Live evaluation rejected: token expired for test_id=%s", test_id)
            del self._tokens[token]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Authorization token has expired. Please authorize again.",
            )

        if record.consumed:
            logger.warning("Live evaluation rejected: token already consumed for test_id=%s", test_id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Authorization token has already been consumed. Each evaluation requires a fresh authorization.",
            )

        if record.test_id != test_id:
            logger.warning("Live evaluation rejected: token test_id mismatch (%s != %s)", record.test_id, test_id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Authorization token was issued for a different test case.",
            )

        # Mark token consumed (single-use)
        self._tokens[token] = AuthTokenRecord(
            test_id=record.test_id,
            token=record.token,
            created_at=record.created_at,
            expires_at=record.expires_at,
            consumed=True,
        )
        logger.info("Live evaluation authorized and confirmed for test_id=%s", test_id)

    def _cleanup_expired(self, now: float) -> None:
        """Removes expired tokens to prevent unbounded memory growth."""
        expired = [t for t, rec in self._tokens.items() if now > rec.expires_at]
        for t in expired:
            del self._tokens[t]


# Global singleton guard
live_guard = LiveEvaluationGuard()
