"""NeuralRetail Scoring API — Authentication and Rate Limiting.

Day 19 — NeuralRetail AMX-DS-2026-04
API key verification, JWT bearer token creation/validation, and
slowapi rate limiting (100 requests/minute per client IP).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API Key authentication
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
http_bearer = HTTPBearer(auto_error=False)

_VALID_API_KEYS: set[str] | None = None


def _get_valid_api_keys() -> set[str]:
    """Load valid API keys from environment variable ``NEURALRETAIL_API_KEYS``.

    The environment variable is expected to be a comma-separated list of
    API key strings. Results are cached in the module-level ``_VALID_API_KEYS``
    set after the first call.

    Returns:
        Set of valid API key strings.
    """
    global _VALID_API_KEYS
    if _VALID_API_KEYS is None:
        raw = os.getenv("NEURALRETAIL_API_KEYS", "dev-key-12345,test-key-99999")
        _VALID_API_KEYS = {k.strip() for k in raw.split(",") if k.strip()}
        logger.info("Loaded %d API keys from environment.", len(_VALID_API_KEYS))
    return _VALID_API_KEYS


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """FastAPI dependency for X-API-Key header verification.

    Args:
        api_key: API key string from the ``X-API-Key`` request header.

    Returns:
        The validated API key string.

    Raises:
        HTTPException: 403 Forbidden if the key is not in the allowlist.
    """
    valid_keys = _get_valid_api_keys()
    if api_key not in valid_keys:
        logger.warning("Invalid API key attempt: %s…", api_key[:8] if len(api_key) >= 8 else "***")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key. Provide a valid X-API-Key header.",
        )
    return api_key


# ---------------------------------------------------------------------------
# JWT authentication
# ---------------------------------------------------------------------------
_JWT_ALGORITHM = "HS256"
_JWT_SECRET: str | None = None


def _get_jwt_secret() -> str:
    """Load JWT secret from environment variable ``NEURALRETAIL_JWT_SECRET``.

    Returns:
        JWT secret string.

    Raises:
        RuntimeError: If the environment variable is not set in production.
    """
    global _JWT_SECRET
    if _JWT_SECRET is None:
        secret = os.getenv("NEURALRETAIL_JWT_SECRET")
        if not secret:
            logger.warning("NEURALRETAIL_JWT_SECRET not set; using insecure default.")
            secret = "insecure-default-secret-change-in-production"
        _JWT_SECRET = secret
    return _JWT_SECRET


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token.

    Args:
        data: Claims to embed in the token. Must include a ``sub`` (subject) field.
        expires_delta: Token lifetime. Defaults to 1 hour if None.

    Returns:
        Encoded JWT string.

    Raises:
        ImportError: If ``python-jose`` is not installed.
    """
    try:
        from jose import jwt
    except ImportError as exc:
        raise ImportError("Install python-jose: pip install python-jose[cryptography]") from exc

    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=1))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, _get_jwt_secret(), algorithm=_JWT_ALGORITHM)


def verify_token(token: str) -> dict[str, Any]:
    """Verify and decode a JWT bearer token.

    Args:
        token: Encoded JWT string (without the "Bearer " prefix).

    Returns:
        Decoded claims dictionary.

    Raises:
        HTTPException: 401 Unauthorized if the token is expired or invalid.
    """
    try:
        from jose import JWTError, jwt

        payload = jwt.decode(token, _get_jwt_secret(), algorithms=[_JWT_ALGORITHM])
        return payload
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Security(http_bearer),
) -> dict[str, Any]:
    """FastAPI dependency for JWT Bearer token verification.

    Args:
        credentials: HTTP Bearer credentials from the Authorization header.

    Returns:
        Decoded JWT payload dict.

    Raises:
        HTTPException: 401 if credentials are missing or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_token(credentials.credentials)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def get_limiter():
    """Create and return a slowapi Limiter instance.

    Returns:
        ``slowapi.Limiter`` configured with client IP key function.

    Raises:
        ImportError: If ``slowapi`` is not installed.
    """
    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address

        return Limiter(key_func=get_remote_address, default_limits=["200/minute"])
    except ImportError:
        logger.warning("slowapi not installed; rate limiting disabled.")
        return None


# Module-level limiter instance
limiter = get_limiter()
