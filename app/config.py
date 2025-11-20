"""
Configuration management for DeepFinance ML Engine v2.
Uses pydantic-settings for type-safe environment variable handling.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "DeepFinance ML Engine"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security
    ML_SERVICE_API_KEY: str = ""
    FRONTEND_ORIGIN: str = "https://deepin.app"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    MODEL_CACHE_TTL: int = 3600  # 1 hour
    DATA_CACHE_TTL: int = 300    # 5 minutes
    PREDICTION_CACHE_TTL: int = 60  # 1 minute

    # Data providers
    FINNHUB_API_KEY: Optional[str] = None
    FINNHUB_TIMEOUT: int = 15

    # Model configuration
    PREDICTION_HORIZON: int = 5
    LOOKBACK_PERIOD: int = 60
    MAX_PREDICTION_HORIZON: int = 30

    # Risk metrics
    RISK_FREE_RATE: float = 0.05
    BENCHMARK_SYMBOL: str = "^GSPC"

    # Confidence scoring
    MIN_CONFIDENCE: float = 0.0
    MAX_CONFIDENCE: float = 1.0

    # Model weights for ensemble
    XGBOOST_WEIGHT: float = 0.35
    LSTM_WEIGHT: float = 0.35
    TRANSFORMER_WEIGHT: float = 0.30

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # Circuit breaker
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT: int = 60

    # Connection pooling
    HTTP_POOL_SIZE: int = 100
    HTTP_POOL_KEEPALIVE: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
