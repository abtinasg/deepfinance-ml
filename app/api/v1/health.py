"""
Health check and monitoring endpoints.
"""

import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from starlette.responses import Response

from app.config import get_settings
from app.core.auth import verify_api_key
from app.core.cache import get_cache_service
from app.core.rate_limit import limiter
from app.models.registry import get_model_registry
from app.schemas.common import (
    HealthResponse,
    ModelsResponse,
    ModelStatusResponse,
    ReloadResponse,
)

router = APIRouter()

# Track startup time
_startup_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check service health and model availability.

    No authentication required for health checks.
    """
    settings = get_settings()
    registry = await get_model_registry()
    cache = await get_cache_service()

    return HealthResponse(
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        status="healthy",
        timestamp=datetime.utcnow(),
        models=registry.get_status()["models"],
        redis=cache.is_connected,
        uptime_seconds=time.time() - _startup_time
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("100/minute")
async def list_models(request: Any = None):
    """
    Get list of available ML models.

    Returns model names, types, and ensemble weights.
    """
    settings = get_settings()
    registry = await get_model_registry()

    models_info = registry.get_all_models_info()

    return ModelsResponse(
        models=[
            {
                "name": m["name"],
                "type": m["type"],
                "version": m["version"],
                "loaded": m["loaded"],
                "weight": m["weight"],
                "last_trained": None,
                "metrics": None
            }
            for m in models_info
        ],
        total=len(models_info),
        ensemble_weights={
            "xgboost": settings.XGBOOST_WEIGHT,
            "lstm": settings.LSTM_WEIGHT,
            "transformer": settings.TRANSFORMER_WEIGHT
        }
    )


@router.get(
    "/models/status",
    response_model=list[ModelStatusResponse],
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("100/minute")
async def get_models_status(request: Any = None):
    """
    Get detailed status of all models.

    Returns load status, memory usage, and device info.
    """
    import torch

    registry = await get_model_registry()
    status_list = []

    for name in registry.get_available_models():
        model = registry.get_model(name)
        if model:
            # Get device info for PyTorch models
            device = "cpu"
            memory_mb = None

            if hasattr(model, "device"):
                device = str(model.device)
                if torch.cuda.is_available() and "cuda" in device:
                    memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

            status_list.append(ModelStatusResponse(
                model_name=model.name,
                status="loaded" if model.is_trained else "not_loaded",
                loaded=model.is_trained,
                memory_usage_mb=memory_mb,
                device=device,
                version=model.version,
                last_prediction=None
            ))

    return status_list


@router.post(
    "/models/reload",
    response_model=ReloadResponse,
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("10/minute")
async def reload_models(
    request: Any = None,
    models: list[str] | None = None
):
    """
    Reload model weights from disk.

    Optionally specify which models to reload.
    """
    registry = await get_model_registry()

    results = await registry.reload_models(models)

    reloaded = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    if failed:
        message = f"Reloaded {len(reloaded)} models. Failed: {failed}"
    else:
        message = f"Successfully reloaded {len(reloaded)} models"

    return ReloadResponse(
        success=len(reloaded) > 0,
        message=message,
        models_reloaded=reloaded
    )


@router.get("/metrics")
async def prometheus_metrics():
    """
    Expose Prometheus metrics.

    No authentication for metrics endpoint.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
