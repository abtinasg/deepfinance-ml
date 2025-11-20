"""API routes for DeepFinance ML Engine."""

from fastapi import APIRouter

from app.api.v1 import health, predictions, risk, symbols

# Create main API router
api_router = APIRouter()

# Include all v1 routes
api_router.include_router(health.router, tags=["health"])
api_router.include_router(predictions.router, tags=["predictions"])
api_router.include_router(risk.router, tags=["risk"])
api_router.include_router(symbols.router, tags=["symbols"])

__all__ = ["api_router"]
