"""Pydantic schemas for request/response validation."""

from app.schemas.common import (
    APIResponse,
    ErrorResponse,
    HealthResponse,
    PaginatedResponse,
)
from app.schemas.prediction import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionHistoryResponse,
    ModelPrediction,
    EnsemblePrediction,
)
from app.schemas.risk import (
    RiskRequest,
    RiskResponse,
    RiskMetrics,
    VolatilityMetrics,
)

__all__ = [
    # Common
    "APIResponse",
    "ErrorResponse",
    "HealthResponse",
    "PaginatedResponse",
    # Prediction
    "PredictRequest",
    "PredictResponse",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "PredictionHistoryResponse",
    "ModelPrediction",
    "EnsemblePrediction",
    # Risk
    "RiskRequest",
    "RiskResponse",
    "RiskMetrics",
    "VolatilityMetrics",
]
