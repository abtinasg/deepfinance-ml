"""
Prediction endpoints.
"""

import asyncio
import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from starlette.requests import Request

from app.core.auth import verify_api_key
from app.core.logging import get_logger
from app.core.rate_limit import limiter
from app.schemas.prediction import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    PredictionHistoryResponse,
)
from app.services.prediction_service import PredictionService

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("30/minute")
async def predict(
    request: Request,
    body: PredictRequest
):
    """
    Generate price predictions for a symbol.

    Uses ensemble of ML models (XGBoost, LSTM, Transformer).
    """
    service = PredictionService()

    try:
        result = await service.predict(
            symbol=body.symbol,
            horizon=body.horizon,
            models=body.models
        )

        # Convert to response model
        return PredictResponse(
            symbol=result["symbol"],
            current_price=result["current_price"],
            prediction_horizon=result["prediction_horizon"],
            model_predictions=result["model_predictions"],
            ensemble=result["ensemble"],
            technical_indicators=result["technical_indicators"],
            metadata=result["metadata"]
        )
    finally:
        await service.close()


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("10/minute")
async def batch_predict(
    request: Request,
    body: BatchPredictRequest
):
    """
    Generate predictions for multiple symbols.

    Limited to 50 symbols per request.
    """
    service = PredictionService()

    try:
        result = await service.get_batch_predictions(
            symbols=body.symbols,
            horizon=body.horizon,
            models=body.models
        )

        return BatchPredictResponse(
            predictions=result["predictions"],
            failed=result["failed"],
            total_processed=result["total_processed"]
        )
    finally:
        await service.close()


@router.get(
    "/predict/{symbol}/history",
    response_model=PredictionHistoryResponse,
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("60/minute")
async def get_prediction_history(
    request: Request,
    symbol: str,
    limit: int = 100
):
    """
    Get historical predictions for a symbol.

    Returns past predictions with accuracy metrics.

    Note: This endpoint requires a database for persistence.
    Currently returns placeholder data.
    """
    # TODO: Implement with database persistence
    return PredictionHistoryResponse(
        symbol=symbol.upper(),
        history=[],
        accuracy_metrics={
            "mean_error_percent": 0.0,
            "hit_rate": 0.0,
            "directional_accuracy": 0.0
        },
        total_predictions=0
    )


@router.websocket("/stream")
async def websocket_predictions(websocket: WebSocket):
    """
    WebSocket endpoint for real-time predictions.

    Clients can subscribe to symbols for continuous updates.
    """
    await websocket.accept()

    # Verify API key from query params
    api_key = websocket.query_params.get("api_key")
    if not api_key:
        await websocket.send_json({
            "error": "API key required",
            "code": "AUTH_REQUIRED"
        })
        await websocket.close()
        return

    from app.config import get_settings
    settings = get_settings()

    if api_key != settings.ML_SERVICE_API_KEY:
        await websocket.send_json({
            "error": "Invalid API key",
            "code": "AUTH_FAILED"
        })
        await websocket.close()
        return

    subscriptions = set()
    service = PredictionService()

    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connected successfully"
        })

        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "error": "Invalid JSON",
                    "code": "PARSE_ERROR"
                })
                continue

            action = message.get("action")

            if action == "subscribe":
                symbol = message.get("symbol", "").upper()
                if symbol:
                    subscriptions.add(symbol)
                    await websocket.send_json({
                        "type": "subscribed",
                        "symbol": symbol
                    })

                    # Send initial prediction
                    try:
                        result = await service.predict(symbol, horizon=5)
                        await websocket.send_json({
                            "type": "prediction",
                            "symbol": symbol,
                            "data": {
                                "current_price": result["current_price"],
                                "final_price": result["ensemble"]["final_price"],
                                "change_percent": result["ensemble"]["change_percent"],
                                "confidence": result["ensemble"]["confidence"]
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "symbol": symbol,
                            "message": str(e)
                        })

            elif action == "unsubscribe":
                symbol = message.get("symbol", "").upper()
                subscriptions.discard(symbol)
                await websocket.send_json({
                    "type": "unsubscribed",
                    "symbol": symbol
                })

            elif action == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })

            else:
                await websocket.send_json({
                    "error": f"Unknown action: {action}",
                    "code": "UNKNOWN_ACTION"
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
    finally:
        await service.close()
