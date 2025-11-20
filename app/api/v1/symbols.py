"""
Symbol information endpoints.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from starlette.requests import Request

from app.core.auth import verify_api_key
from app.core.rate_limit import limiter
from app.schemas.common import SymbolInfo
from app.services.data_fetcher import DataFetcher

router = APIRouter()


@router.get(
    "/symbols/{symbol}",
    response_model=SymbolInfo,
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("100/minute")
async def get_symbol_info(
    request: Request,
    symbol: str
):
    """
    Get current price and market data for a symbol.

    Returns current price, change %, high, low, and volume.
    """
    fetcher = DataFetcher()

    try:
        data = await fetcher.fetch_historical_data(symbol.upper(), period_days=5)

        if data is None or data.empty:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=404,
                detail=f"Symbol not found: {symbol}"
            )

        # Get latest data
        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else latest

        current_price = float(latest["close"])
        previous_close = float(previous["close"])
        change_percent = ((current_price - previous_close) / previous_close) * 100

        return SymbolInfo(
            symbol=symbol.upper(),
            current_price=current_price,
            change_percent=change_percent,
            high=float(latest["high"]),
            low=float(latest["low"]),
            volume=int(latest["volume"]),
            timestamp=datetime.utcnow()
        )
    finally:
        await fetcher.close()
