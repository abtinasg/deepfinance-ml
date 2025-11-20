"""
Risk analysis endpoints.
"""

from typing import Any

from fastapi import APIRouter, Depends
from starlette.requests import Request

from app.core.auth import verify_api_key
from app.core.rate_limit import limiter
from app.schemas.risk import RiskRequest, RiskResponse
from app.services.risk_service import RiskService

router = APIRouter()


@router.post(
    "/risk",
    response_model=RiskResponse,
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("30/minute")
async def calculate_risk(
    request: Request,
    body: RiskRequest
):
    """
    Calculate risk metrics for a symbol.

    Returns volatility, Sharpe ratio, VaR, max drawdown, and more.
    """
    service = RiskService()

    try:
        result = await service.calculate_risk_metrics(
            symbol=body.symbol,
            period_days=body.period_days,
            include_benchmark=body.include_benchmark
        )

        return RiskResponse(
            symbol=result["symbol"],
            period_days=result["period_days"],
            current_price=result["current_price"],
            metrics=result["metrics"],
            benchmark_symbol=result.get("benchmark_symbol"),
            benchmark_comparison=result.get("benchmark_comparison"),
            recommendations=result["recommendations"]
        )
    finally:
        await service.close()
