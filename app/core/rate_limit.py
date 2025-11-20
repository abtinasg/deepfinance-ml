"""
Rate limiting configuration using SlowAPI.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import get_settings


def _get_rate_limit_key(request: Request) -> str:
    """
    Get rate limit key based on API key or IP address.

    This allows rate limiting per API key when authenticated.
    """
    # Try to get API key from header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key[:8]}..."

    # Fall back to IP address
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=_get_rate_limit_key,
    default_limits=["100/minute"]
)


def get_rate_limit_string() -> str:
    """Get the rate limit string from settings."""
    settings = get_settings()
    return f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_WINDOW} seconds"
