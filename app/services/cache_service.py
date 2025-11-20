"""
High-level cache management service.
"""

from typing import Any, Optional

from app.core.cache import CacheService, get_cache_service
from app.core.logging import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    High-level cache manager with domain-specific caching logic.
    """

    def __init__(self):
        self._cache: Optional[CacheService] = None

    async def _get_cache(self) -> CacheService:
        """Get cache service instance."""
        if self._cache is None:
            self._cache = await get_cache_service()
        return self._cache

    async def get_market_data(
        self,
        symbol: str,
        period_days: int
    ) -> Optional[dict]:
        """Get cached market data."""
        cache = await self._get_cache()
        return await cache.get_market_data(symbol, period_days)

    async def set_market_data(
        self,
        symbol: str,
        period_days: int,
        data: dict
    ) -> bool:
        """Cache market data."""
        cache = await self._get_cache()
        return await cache.set_market_data(symbol, period_days, data)

    async def get_prediction(
        self,
        symbol: str,
        horizon: int,
        models: tuple
    ) -> Optional[dict]:
        """Get cached prediction."""
        cache = await self._get_cache()
        return await cache.get_prediction(symbol, horizon, models)

    async def set_prediction(
        self,
        symbol: str,
        horizon: int,
        models: tuple,
        data: dict
    ) -> bool:
        """Cache prediction."""
        cache = await self._get_cache()
        return await cache.set_prediction(symbol, horizon, models, data)

    async def invalidate_symbol(self, symbol: str) -> int:
        """
        Invalidate all cached data for a symbol.

        Returns number of keys deleted.
        """
        cache = await self._get_cache()

        # Clear market data
        market_deleted = await cache.clear_pattern(f"market:{symbol.upper()}")

        # Clear predictions
        pred_deleted = await cache.clear_pattern(f"prediction:{symbol.upper()}")

        total = market_deleted + pred_deleted
        logger.info(
            f"Invalidated {total} cache entries for {symbol}",
            extra={"symbol": symbol}
        )

        return total

    async def invalidate_all_predictions(self) -> int:
        """Invalidate all cached predictions."""
        cache = await self._get_cache()
        deleted = await cache.clear_pattern("prediction")
        logger.info(f"Invalidated {deleted} prediction cache entries")
        return deleted

    async def invalidate_all_market_data(self) -> int:
        """Invalidate all cached market data."""
        cache = await self._get_cache()
        deleted = await cache.clear_pattern("market")
        logger.info(f"Invalidated {deleted} market data cache entries")
        return deleted

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        cache = await self._get_cache()

        if not cache.is_connected:
            return {
                "connected": False,
                "status": "Redis not connected"
            }

        return {
            "connected": True,
            "status": "operational"
        }
