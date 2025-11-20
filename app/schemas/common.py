"""
Common schemas used across multiple endpoints.
"""

from datetime import datetime
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid symbol format",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class APIResponse(BaseModel, Generic[T]):
    """Generic API response wrapper."""

    success: bool = Field(default=True)
    data: Optional[T] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated API response."""

    items: list[T]
    total: int
    page: int
    page_size: int
    has_more: bool


class HealthResponse(BaseModel):
    """Health check response."""

    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    models: dict[str, Any] = Field(
        default_factory=dict,
        description="Model availability status"
    )
    redis: bool = Field(default=False, description="Redis connection status")
    uptime_seconds: float = Field(default=0, description="Service uptime")

    class Config:
        json_schema_extra = {
            "example": {
                "service": "DeepFinance ML Engine",
                "version": "2.0.0",
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "models": {
                    "xgboost": {"loaded": True, "version": "1.0"},
                    "lstm": {"loaded": True, "version": "1.0"},
                    "transformer": {"loaded": True, "version": "1.0"}
                },
                "redis": True,
                "uptime_seconds": 3600.5
            }
        }


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    type: str
    version: str
    loaded: bool
    weight: float
    last_trained: Optional[datetime] = None
    metrics: Optional[dict[str, float]] = None


class ModelsResponse(BaseModel):
    """Response for model listing endpoint."""

    models: list[ModelInfo]
    total: int
    ensemble_weights: dict[str, float]


class ModelStatusResponse(BaseModel):
    """Model status response."""

    model_name: str
    status: str
    loaded: bool
    memory_usage_mb: Optional[float] = None
    device: str = "cpu"
    version: str
    last_prediction: Optional[datetime] = None


class ReloadResponse(BaseModel):
    """Model reload response."""

    success: bool
    message: str
    models_reloaded: list[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SymbolInfo(BaseModel):
    """Symbol information response."""

    symbol: str
    current_price: float
    change_percent: float
    high: float
    low: float
    volume: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "current_price": 185.50,
                "change_percent": 1.25,
                "high": 186.20,
                "low": 183.75,
                "volume": 45678900,
                "timestamp": "2024-01-15T16:00:00Z"
            }
        }
