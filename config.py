"""
Configuration settings for DeepFinance ML Microservice
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Settings
    APP_NAME: str = "DeepFinance ML Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", 8000))

    # Data Provider Settings
    FINNHUB_API_KEY: Optional[str] = os.getenv("FINNHUB_API_KEY", "")
    YAHOO_TIMEOUT: int = 10
    FINNHUB_TIMEOUT: int = 10

    # Model Settings
    PREDICTION_HORIZON: int = 5  # Days to predict ahead
    LOOKBACK_PERIOD: int = 60  # Days of historical data

    # Risk Calculation Settings
    RISK_FREE_RATE: float = 0.05  # Annual risk-free rate (5%)
    BENCHMARK_SYMBOL: str = "^GSPC"  # S&P 500 as benchmark

    # Confidence Engine Settings
    MIN_CONFIDENCE: float = 0.0
    MAX_CONFIDENCE: float = 1.0

    # Model Weights for Ensemble
    XGBOOST_WEIGHT: float = 0.35
    LSTM_WEIGHT: float = 0.35
    TRANSFORMER_WEIGHT: float = 0.30

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
