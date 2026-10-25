"""Server-side administrative authorization guard for sensitive BoliBench operations.

Enforces:
1. Environment-configured secret validation (ADMIN_API_KEY).
2. Constant-time comparison (secrets.compare_digest) to prevent timing attacks.
3. Fail-closed behavior: Rejects requests if ADMIN_API_KEY is missing/unconfigured on the server.
4. Flexible credentials support: Accepts 'Authorization: Bearer <token>' or 'X-Admin-Token: <token>'.
5. Security & Privacy: Never logs credential values or exposes secrets in error messages.
"""

import logging
import secrets
from typing import Optional
from fastapi import Header, HTTPException, Request, status

from app.config import get_settings

logger = logging.getLogger("bolibench.admin_guard")


async def verify_admin_authorization(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> str:
    """FastAPI dependency to verify administrative authorization.

    Raises:
        HTTPException(403): If the server has no ADMIN_API_KEY configured (fail-closed),
                            or if the provided credential does not match.
        HTTPException(401): If no credentials or empty credentials are provided.
    """
    settings = get_settings()

    # 1. Fail-closed server configuration check
    if not settings.is_admin_configured:
        logger.warning(
            "Administrative action rejected: ADMIN_API_KEY is not configured on the server."
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Administrative operations are disabled because ADMIN_API_KEY is not configured on the server.",
        )

    # 2. Extract credential from either X-Admin-Token or Authorization header
    token: Optional[str] = None
    if x_admin_token and x_admin_token.strip():
        token = x_admin_token.strip()
    elif authorization and authorization.strip():
        parts = authorization.strip().split(maxsplit=1)
        if parts[0].lower() == "bearer":
            token = parts[1].strip() if len(parts) > 1 else ""
        else:
            token = authorization.strip()

    # 3. Check for missing or empty credentials
    if not token or not token.strip():
        client_host = request.client.host if request.client else "unknown"
        logger.warning(
            "Administrative action rejected: missing credentials from client %s",
            client_host,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing administrative credentials. Provide via 'Authorization: Bearer <token>' or 'X-Admin-Token' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 4. Constant-time credential comparison
    expected_key = settings.admin_api_key.strip()  # type: ignore[union-attr]
    if not secrets.compare_digest(token, expected_key):
        client_host = request.client.host if request.client else "unknown"
        logger.warning(
            "Administrative action rejected: invalid credentials from client %s",
            client_host,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid administrative credentials.",
        )

    return "admin"
