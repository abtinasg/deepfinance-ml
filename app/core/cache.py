"""
Redis caching wrapper for DeepFinance ML Engine.
"""

import json
import hashlib
from typing import Any, Optional

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CacheService:
    """Redis-based caching service with connection pooling."""

    def __init__(self) -> None:
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        settings = get_settings()
        try:
            self._pool = ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=50,
                decode_responses=True
            )
            self._client = redis.Redis(connection_pool=self._pool)
            await self._client.ping()
            logger.info("Connected to Redis", extra={"url": settings.REDIS_URL})
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self._client = None

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        logger.info("Disconnected from Redis")

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._client is not None

    @staticmethod
    def _generate_key(prefix: str, *args: Any) -> str:
        """Generate a cache key from prefix and arguments."""
        key_data = json.dumps(args, sort_keys=True, default=str)
        hash_val = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"deepfinance:{prefix}:{hash_val}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        if not self._client:
            return None

        try:
            value = await self._client.get(key)
            if value:
                logger.debug(f"Cache hit: {key}")
                return json.loads(value)
            logger.debug(f"Cache miss: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}", extra={"key": key})
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time to live in seconds.

        Returns:
            True if successful, False otherwise.
        """
        if not self._client:
            return False

        try:
            settings = get_settings()
            ttl = ttl or settings.DATA_CACHE_TTL
            serialized = json.dumps(value, default=str)
            await self._client.setex(key, ttl, serialized)
            logger.debug(f"Cache set: {key}", extra={"ttl": ttl})
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}", extra={"key": key})
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        if not self._client:
            return False

        try:
            await self._client.delete(key)
            logger.debug(f"Cache delete: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}", extra={"key": key})
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        if not self._client:
            return 0

        try:
            keys = []
            async for key in self._client.scan_iter(f"deepfinance:{pattern}:*"):
                keys.append(key)

            if keys:
                deleted = await self._client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries", extra={"pattern": pattern})
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache clear error: {e}", extra={"pattern": pattern})
            return 0

    # Convenience methods for specific cache types
    async def get_market_data(self, symbol: str, period_days: int) -> Optional[dict]:
        """Get cached market data."""
        key = self._generate_key("market", symbol.upper(), period_days)
        return await self.get(key)

    async def set_market_data(
        self,
        symbol: str,
        period_days: int,
        data: dict
    ) -> bool:
        """Cache market data."""
        settings = get_settings()
        key = self._generate_key("market", symbol.upper(), period_days)
        return await self.set(key, data, settings.DATA_CACHE_TTL)

    async def get_prediction(
        self,
        symbol: str,
        horizon: int,
        models: tuple
    ) -> Optional[dict]:
        """Get cached prediction."""
        key = self._generate_key("prediction", symbol.upper(), horizon, models)
        return await self.get(key)

    async def set_prediction(
        self,
        symbol: str,
        horizon: int,
        models: tuple,
        data: dict
    ) -> bool:
        """Cache prediction."""
        settings = get_settings()
        key = self._generate_key("prediction", symbol.upper(), horizon, models)
        return await self.set(key, data, settings.PREDICTION_CACHE_TTL)


# Singleton instance
_cache_service: Optional[CacheService] = None


async def get_cache_service() -> CacheService:
    """Get the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
        await _cache_service.connect()
    return _cache_service


async def close_cache_service() -> None:
    """Close the global cache service."""
    global _cache_service
    if _cache_service is not None:
        await _cache_service.disconnect()
        _cache_service = None
