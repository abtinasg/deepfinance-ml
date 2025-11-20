"""
FastAPI routes for DeepFinance ML API
"""
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import settings
from services import DataFetcher, PredictionService, RiskService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
prediction_service = PredictionService()
risk_service = RiskService()
data_fetcher = DataFetcher()


# Request/Response Models
class PredictRequest(BaseModel):
    """Request model for price prediction"""
    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    horizon: Optional[int] = Field(
        default=5,
        ge=1,
        le=30,
        description="Number of days to predict"
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="List of models to use (xgboost, lstm, transformer)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "horizon": 5,
                "models": ["xgboost", "lstm", "transformer"]
            }
        }


class RiskRequest(BaseModel):
    """Request model for risk calculation"""
    symbol: str = Field(..., description="Stock ticker symbol")
    period_days: Optional[int] = Field(
        default=60,
        ge=10,
        le=365,
        description="Number of days for risk calculation"
    )
    include_benchmark: Optional[bool] = Field(
        default=True,
        description="Include benchmark comparison for beta calculation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "period_days": 60,
                "include_benchmark": True
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    service: str
    version: str
    timestamp: str
    models_available: List[str]


class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check service health and status

    Returns service information and available models
    """
    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow().isoformat(),
        models_available=["XGBoost", "LSTM", "Transformer"]
    )


@router.post("/predict")
async def predict_price(request: PredictRequest):
    """
    Generate price predictions for a stock symbol

    Uses ensemble of ML models (XGBoost, LSTM, Transformer) to predict
    future prices with confidence scoring.

    Returns:
    - Individual model predictions
    - Weighted ensemble prediction
    - Confidence scores
    - Technical indicators
    """
    try:
        result = await prediction_service.predict(
            symbol=request.symbol.upper(),
            horizon=request.horizon,
            models=request.models
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("message")
                or result.get("error")
                or "Prediction failed",
            )

        return {
            "success": True,
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/risk")
async def calculate_risk(request: RiskRequest):
    """
    Calculate risk metrics for a stock symbol

    Returns comprehensive risk metrics including:
    - Volatility (daily, weekly, annualized)
    - Beta (relative to S&P 500)
    - Sharpe Ratio
    - Sortino Ratio
    - Value at Risk (VaR)
    - Maximum Drawdown
    """
    try:
        # Fetch asset data
        asset_data = await data_fetcher.fetch_historical_data(
            request.symbol.upper(),
            request.period_days
        )

        if asset_data is None or asset_data.empty:
            raise HTTPException(
                status_code=400,
                detail="insufficient_data",
            )

        # Fetch benchmark data if requested
        benchmark_data = None
        if request.include_benchmark:
            benchmark_data = await data_fetcher.fetch_benchmark_data(
                request.period_days
            )

        # Calculate risk metrics
        risk_metrics = await risk_service.calculate_risk_metrics(
            asset_data,
            benchmark_data
        )

        if risk_metrics.get("success") is False:
            raise HTTPException(status_code=400, detail="insufficient_data")

        # Get current price info
        current_price = asset_data['close'].iloc[-1]
        price_change = (
            asset_data['close'].iloc[-1] - asset_data['close'].iloc[0]
        ) / asset_data['close'].iloc[0] * 100

        return {
            "success": True,
            "data": {
                "symbol": request.symbol.upper(),
                "current_price": float(current_price),
                "period_return_percent": float(price_change),
                "risk_metrics": risk_metrics,
                "benchmark": settings.BENCHMARK_SYMBOL if request.include_benchmark else None,
                "period_days": request.period_days
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/models")
async def get_models():
    """
    Get information about available prediction models
    """
    return {
        "success": True,
        "data": {
            "models": prediction_service.get_available_models(),
            "weights": {
                "xgboost": settings.XGBOOST_WEIGHT,
                "lstm": settings.LSTM_WEIGHT,
                "transformer": settings.TRANSFORMER_WEIGHT
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/symbols/{symbol}")
async def get_symbol_info(symbol: str):
    """
    Get current information for a symbol
    """
    try:
        data = await data_fetcher.fetch_historical_data(symbol.upper(), 5)

        if data is None or data.empty:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch data for {symbol}"
            )

        current_price = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2] if len(data) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100

        return {
            "success": True,
            "data": {
                "symbol": symbol.upper(),
                "current_price": float(current_price),
                "previous_close": float(prev_close),
                "change": float(change),
                "change_percent": float(change_pct),
                "high": float(data['high'].iloc[-1]),
                "low": float(data['low'].iloc[-1]),
                "volume": int(data['volume'].iloc[-1])
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Symbol info error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
