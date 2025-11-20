"""
DeepFinance ML Engine - Main Application Entry Point

A FastAPI-based microservice for financial price prediction and risk analysis
using ensemble machine learning models (XGBoost, LSTM, Transformer).
"""
import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from api import router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Running on {settings.HOST}:{settings.PORT}")

    # Log configuration
    logger.info(f"Prediction horizon: {settings.PREDICTION_HORIZON} days")
    logger.info(f"Lookback period: {settings.LOOKBACK_PERIOD} days")

    if settings.FINNHUB_API_KEY:
        logger.info("Finnhub API key configured (fallback enabled)")
    else:
        logger.warning("Finnhub API key not configured (fallback disabled)")

    yield

    # Shutdown
    logger.info("Shutting down DeepFinance ML Engine")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""
## DeepFinance ML Engine

AI-powered financial analysis microservice providing:

### Features
- **Price Prediction**: Ensemble predictions using XGBoost, LSTM, and Transformer models
- **Risk Analysis**: Comprehensive risk metrics including Beta, Volatility, Sharpe Ratio
- **Confidence Scoring**: Model confidence and consensus analysis
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and more

### Data Sources
- Provider: Finnhub API

### Models
- **XGBoost**: Gradient boosting for feature-based prediction
- **LSTM**: Recurrent neural network for sequence patterns
- **Transformer**: Attention-based deep learning model
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else None
        }
    )


# Include API routes
app.include_router(router, prefix="/api/v1", tags=["DeepFinance ML"])

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )
