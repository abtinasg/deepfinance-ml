"""
DeepFinance ML Engine v2 - Main Application Entry Point

A production-ready FastAPI application for financial price prediction
using ensemble ML models with security, caching, and monitoring.
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import api_router
from app.config import get_settings
from app.core.cache import close_cache_service, get_cache_service
from app.core.logging import get_logger, setup_logging
from app.core.rate_limit import limiter
from app.dependencies import close_http_client
from app.models.registry import close_model_registry, get_model_registry

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    settings = get_settings()

    # Startup
    logger.info(
        f"Starting {settings.APP_NAME} v{settings.APP_VERSION}",
        extra={"environment": "production" if not settings.DEBUG else "development"}
    )

    # Initialize Redis cache
    try:
        cache = await get_cache_service()
        logger.info(f"Redis connected: {cache.is_connected}")
    except Exception as e:
        logger.warning(f"Redis initialization failed: {e}")

    # Initialize model registry and load weights
    try:
        registry = await get_model_registry()
        loaded = sum(1 for m in registry.models.values() if m.is_trained)
        logger.info(f"Model registry initialized: {loaded}/{len(registry.models)} models loaded")
    except Exception as e:
        logger.error(f"Model registry initialization failed: {e}")

    logger.info(
        f"Application started on {settings.HOST}:{settings.PORT}",
        extra={"debug": settings.DEBUG}
    )

    yield

    # Shutdown
    logger.info("Shutting down application...")

    await close_cache_service()
    await close_model_registry()
    await close_http_client()

    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Production ML service for financial price prediction",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan
    )

    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_ORIGIN],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"]
    )

    # Add request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        Handle all unhandled exceptions.

        Returns sanitized error response without sensitive information.
        """
        logger.error(
            f"Unhandled exception: {type(exc).__name__}",
            extra={
                "path": request.url.path,
                "method": request.method
            },
            exc_info=exc
        )

        # Return sanitized error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with service information."""
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs" if settings.DEBUG else "Disabled in production",
            "health": "/api/v1/health"
        }

    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4,
        log_level="debug" if settings.DEBUG else "info"
    )
