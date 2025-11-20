"""
Prediction orchestration service.
"""

import asyncio
from datetime import datetime
from typing import Any, Optional

from app.config import get_settings
from app.core.cache import get_cache_service
from app.core.logging import get_logger
from app.models.registry import get_model_registry
from app.services.confidence_engine import ConfidenceEngine
from app.services.data_fetcher import DataFetcher

logger = get_logger(__name__)


class PredictionService:
    """
    Service for orchestrating price predictions across all models.
    """

    def __init__(self):
        self._data_fetcher = DataFetcher()
        self._confidence_engine = ConfidenceEngine()
        self._prediction_lock = asyncio.Lock()

    async def predict(
        self,
        symbol: str,
        horizon: int = 5,
        models: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """
        Generate predictions using ensemble of models.

        Args:
            symbol: Stock ticker symbol.
            horizon: Prediction horizon in days.
            models: Specific models to use (None = all).

        Returns:
            Complete prediction response.
        """
        settings = get_settings()

        # Check cache
        cache = await get_cache_service()
        models_tuple = tuple(sorted(models)) if models else ("all",)
        cached = await cache.get_prediction(symbol, horizon, models_tuple)
        if cached:
            logger.debug(f"Cache hit for prediction {symbol}")
            return cached

        # Fetch historical data
        data = await self._data_fetcher.fetch_historical_data(
            symbol,
            settings.LOOKBACK_PERIOD
        )

        if data is None or data.empty:
            raise ValueError(
                f"Failed to fetch data for {symbol} using Finnhub"
            )

        current_price = float(data["close"].iloc[-1])

        # Get technical indicators
        indicators = self._data_fetcher.calculate_technical_indicators(data)

        # Get model registry
        registry = await get_model_registry()

        # Determine which models to use
        if models:
            model_names = [m.lower() for m in models]
        else:
            model_names = registry.get_available_models()

        # Run predictions concurrently
        async with self._prediction_lock:
            tasks = []
            for name in model_names:
                model = registry.get_model(name)
                if model:
                    tasks.append(self._run_model_prediction(model, data, horizon))

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        model_predictions = []
        successful_results = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Model prediction failed: {result}")
                continue
            if result:
                model_predictions.append(result)
                successful_results.append(result)

        if not successful_results:
            raise ValueError("All model predictions failed")

        # Create ensemble prediction
        historical_prices = data["close"].values
        ensemble = self._confidence_engine.create_ensemble_prediction(
            successful_results,
            historical_prices
        )

        # Add confidence to each model prediction
        for pred in model_predictions:
            confidence = self._confidence_engine.calculate_model_confidence(
                pred["predictions"],
                historical_prices
            )
            pred["confidence"] = confidence

        # Build response
        response = {
            "symbol": symbol,
            "current_price": current_price,
            "prediction_horizon": horizon,
            "model_predictions": model_predictions,
            "ensemble": ensemble,
            "technical_indicators": indicators,
            "metadata": {
                "data_source": "finnhub",
                "lookback_period": f"{settings.LOOKBACK_PERIOD} days",
                "generated_at": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        # Cache the result
        await cache.set_prediction(symbol, horizon, models_tuple, response)

        return response

    async def _run_model_prediction(
        self,
        model,
        data,
        horizon: int
    ) -> Optional[dict[str, Any]]:
        """Run prediction for a single model with error handling."""
        try:
            return await model.predict(data, horizon)
        except Exception as e:
            logger.error(
                f"Model {model.name} failed: {e}",
                extra={"model": model.name}
            )
            return None

    async def get_quick_prediction(
        self,
        symbol: str,
        horizon: int = 5
    ) -> dict[str, Any]:
        """
        Get a quick prediction using only XGBoost (fastest model).

        Args:
            symbol: Stock ticker symbol.
            horizon: Prediction horizon.

        Returns:
            Quick prediction result.
        """
        return await self.predict(symbol, horizon, models=["xgboost"])

    async def get_batch_predictions(
        self,
        symbols: list[str],
        horizon: int = 5,
        models: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """
        Get predictions for multiple symbols.

        Args:
            symbols: List of stock symbols.
            horizon: Prediction horizon.
            models: Models to use.

        Returns:
            Batch prediction results.
        """
        predictions = {}
        failed = []

        # Run predictions concurrently with semaphore to limit parallelism
        semaphore = asyncio.Semaphore(5)

        async def predict_with_semaphore(symbol: str):
            async with semaphore:
                try:
                    return symbol, await self.predict(symbol, horizon, models)
                except Exception as e:
                    logger.error(f"Batch prediction failed for {symbol}: {e}")
                    return symbol, None

        tasks = [predict_with_semaphore(s) for s in symbols]
        results = await asyncio.gather(*tasks)

        for symbol, result in results:
            if result:
                predictions[symbol] = result
            else:
                failed.append(symbol)

        return {
            "predictions": predictions,
            "failed": failed,
            "total_processed": len(symbols),
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_available_models(self) -> list[dict[str, Any]]:
        """Get information about available models."""
        settings = get_settings()
        return [
            {
                "name": "XGBoost",
                "type": "Gradient Boosting",
                "weight": settings.XGBOOST_WEIGHT,
                "description": "Fast, accurate for short-term predictions"
            },
            {
                "name": "LSTM",
                "type": "Recurrent Neural Network",
                "weight": settings.LSTM_WEIGHT,
                "description": "Good at capturing sequential patterns"
            },
            {
                "name": "Transformer",
                "type": "Attention-based Neural Network",
                "weight": settings.TRANSFORMER_WEIGHT,
                "description": "Captures complex temporal dependencies"
            }
        ]

    async def close(self) -> None:
        """Cleanup resources."""
        await self._data_fetcher.close()
