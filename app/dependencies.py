"""
FastAPI dependency injection utilities.
"""

from typing import AsyncGenerator

import httpx

from app.config import get_settings
from app.core.auth import verify_api_key
from app.core.cache import CacheService, get_cache_service
from app.services.data_fetcher import DataFetcher
from app.services.prediction_service import PredictionService
from app.services.risk_service import RiskService


# Global HTTP client for connection pooling
_http_client: httpx.AsyncClient | None = None


async def get_http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Get HTTP client with connection pooling."""
    global _http_client
    settings = get_settings()

    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.FINNHUB_TIMEOUT),
            limits=httpx.Limits(
                max_keepalive_connections=settings.HTTP_POOL_SIZE,
                keepalive_expiry=settings.HTTP_POOL_KEEPALIVE
            )
        )

    yield _http_client


async def close_http_client() -> None:
    """Close the HTTP client on shutdown."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def get_data_fetcher() -> DataFetcher:
    """Get data fetcher service instance."""
    return DataFetcher()


def get_prediction_service() -> PredictionService:
    """Get prediction service instance."""
    return PredictionService()


def get_risk_service() -> RiskService:
    """Get risk service instance."""
    return RiskService()


# Re-export for convenience
__all__ = [
    "verify_api_key",
    "get_cache_service",
    "get_http_client",
    "close_http_client",
    "get_data_fetcher",
    "get_prediction_service",
    "get_risk_service",
]
