"""API routes for DeepFinance ML Engine + DeepHealth Medical Analysis."""

from fastapi import APIRouter

from app.api.v1 import health, medical, predictions, risk, symbols

# Create main API router
api_router = APIRouter()

# Include all v1 routes
api_router.include_router(health.router, tags=["health"])
api_router.include_router(predictions.router, tags=["predictions"])
api_router.include_router(risk.router, tags=["risk"])
api_router.include_router(symbols.router, tags=["symbols"])

# Include DeepHealth medical analysis routes
api_router.include_router(medical.router, tags=["medical"])

__all__ = ["api_router"]
