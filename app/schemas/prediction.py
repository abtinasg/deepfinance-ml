"""
Prediction-related schemas.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request schema for price prediction."""

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol"
    )
    horizon: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Prediction horizon in days"
    )
    models: Optional[list[str]] = Field(
        default=None,
        description="Specific models to use (default: all)"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        v = v.upper().strip()
        if not v.replace("^", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError("Invalid symbol format")
        return v

    @field_validator("models")
    @classmethod
    def validate_models(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate model names."""
        if v is None:
            return None
        valid_models = {"xgboost", "lstm", "transformer"}
        v = [m.lower() for m in v]
        invalid = set(v) - valid_models
        if invalid:
            raise ValueError(f"Invalid models: {invalid}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "horizon": 5,
                "models": ["xgboost", "lstm", "transformer"]
            }
        }


class ModelPrediction(BaseModel):
    """Individual model prediction result."""

    model: str
    predictions: list[float]
    final_price: float
    change_percent: float
    trend: str
    confidence: float
    variance: Optional[float] = None
    feature_importance: Optional[dict[str, float]] = None
    attention_weights: Optional[dict[str, float]] = None


class EnsemblePrediction(BaseModel):
    """Ensemble prediction combining all models."""

    predictions: list[float]
    final_price: float
    change_percent: float
    confidence: float
    confidence_breakdown: dict[str, str]
    consensus: str
    consensus_strength: float
    prediction_range: dict[str, float]


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""

    rsi: float = Field(..., ge=0, le=100)
    macd: dict[str, float]
    bollinger_bands: dict[str, float]
    atr: float
    sma_20: float
    sma_50: float


class PredictResponse(BaseModel):
    """Response schema for price prediction."""

    symbol: str
    current_price: float
    prediction_horizon: int
    model_predictions: list[ModelPrediction]
    ensemble: EnsemblePrediction
    technical_indicators: TechnicalIndicators
    metadata: dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "current_price": 185.50,
                "prediction_horizon": 5,
                "model_predictions": [
                    {
                        "model": "XGBoost",
                        "predictions": [186.0, 186.5, 187.0, 187.5, 188.0],
                        "final_price": 188.0,
                        "change_percent": 1.35,
                        "trend": "bullish",
                        "confidence": 0.75,
                        "variance": 0.02
                    }
                ],
                "ensemble": {
                    "predictions": [186.1, 186.6, 187.1, 187.6, 188.1],
                    "final_price": 188.1,
                    "change_percent": 1.40,
                    "confidence": 0.78,
                    "confidence_breakdown": {
                        "level": "high",
                        "description": "Strong agreement"
                    },
                    "consensus": "bullish",
                    "consensus_strength": 0.85,
                    "prediction_range": {"min": 187.5, "max": 188.5}
                },
                "technical_indicators": {
                    "rsi": 55.5,
                    "macd": {"macd": 1.2, "signal": 1.0, "histogram": 0.2},
                    "bollinger_bands": {"upper": 190.0, "middle": 185.0, "lower": 180.0},
                    "atr": 2.5,
                    "sma_20": 184.0,
                    "sma_50": 182.0
                },
                "metadata": {
                    "data_source": "finnhub",
                    "lookback_period": "60 days"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class BatchPredictRequest(BaseModel):
    """Request for batch predictions."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of symbols to predict"
    )
    horizon: int = Field(default=5, ge=1, le=30)
    models: Optional[list[str]] = None

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        """Validate and normalize symbols."""
        validated = []
        for symbol in v:
            s = symbol.upper().strip()
            if not s.replace("^", "").replace("-", "").replace(".", "").isalnum():
                raise ValueError(f"Invalid symbol: {symbol}")
            validated.append(s)
        return validated


class BatchPredictResponse(BaseModel):
    """Response for batch predictions."""

    predictions: dict[str, PredictResponse]
    failed: list[str]
    total_processed: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PredictionHistoryItem(BaseModel):
    """Single prediction history entry."""

    timestamp: datetime
    predicted_price: float
    actual_price: Optional[float] = None
    confidence: float
    error_percent: Optional[float] = None


class PredictionHistoryResponse(BaseModel):
    """Response for prediction history."""

    symbol: str
    history: list[PredictionHistoryItem]
    accuracy_metrics: dict[str, float]
    total_predictions: int
