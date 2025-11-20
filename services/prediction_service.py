"""
Prediction service that orchestrates all ML models
"""
import logging
import asyncio
from typing import Dict, Any, Optional

import numpy as np

from config import settings
from models import XGBoostPredictor, LSTMPredictor, TransformerPredictor
from .data_fetcher import DataFetcher
from .confidence_engine import ConfidenceEngine

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for orchestrating price predictions across all models"""

    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.confidence_engine = ConfidenceEngine()

        # Initialize models
        self.models = {
            "xgboost": XGBoostPredictor(),
            "lstm": LSTMPredictor(),
            "transformer": TransformerPredictor()
        }

        self.lookback_period = settings.LOOKBACK_PERIOD
        self.prediction_horizon = settings.PREDICTION_HORIZON

    async def predict(
        self,
        symbol: str,
        horizon: Optional[int] = None,
        models: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate price predictions for a symbol

        Args:
            symbol: Stock ticker symbol
            horizon: Number of days to predict (default from config)
            models: List of models to use (default: all)

        Returns:
            Complete prediction results with ensemble
        """
        try:
            horizon = horizon or self.prediction_horizon

            # Fetch historical data
            data = await self.data_fetcher.fetch_historical_data(
                symbol,
                self.lookback_period
            )

            if data is None or data.empty:
                return {
                    "success": False,
                    "error": f"Failed to fetch data for {symbol}",
                    "symbol": symbol
                }

            # Get current price
            current_price = data['close'].iloc[-1]

            # Calculate technical indicators
            indicators = self.data_fetcher.calculate_technical_indicators(data)

            # Determine which models to use
            if models:
                selected_models = {
                    k: v for k, v in self.models.items()
                    if k.lower() in [m.lower() for m in models]
                }
            else:
                selected_models = self.models

            if not selected_models:
                return {
                    "success": False,
                    "error": "No valid models selected",
                    "symbol": symbol
                }

            # Run predictions concurrently
            model_results = await self._run_predictions(
                selected_models,
                data,
                horizon
            )

            if not model_results:
                return {
                    "success": False,
                    "error": "All model predictions failed",
                    "symbol": symbol
                }

            # Create ensemble prediction
            historical_prices = data['close'].values
            ensemble = self.confidence_engine.create_ensemble_prediction(
                model_results,
                historical_prices
            )

            # Get confidence breakdown
            confidence_breakdown = self.confidence_engine.get_confidence_breakdown(
                ensemble["ensemble_confidence"]
            )

            # Calculate price change metrics
            ensemble_final = ensemble["final_price"]
            price_change = ensemble_final - current_price
            price_change_pct = (price_change / current_price) * 100

            return {
                "success": True,
                "symbol": symbol.upper(),
                "current_price": float(current_price),
                "predictions": {
                    "xgboost": next(
                        (r for r in model_results if r["model"] == "XGBoost"),
                        None
                    ),
                    "lstm": next(
                        (r for r in model_results if r["model"] == "LSTM"),
                        None
                    ),
                    "transformer": next(
                        (r for r in model_results if r["model"] == "Transformer"),
                        None
                    )
                },
                "ensemble": {
                    "predictions": ensemble["ensemble_predictions"],
                    "final_price": ensemble["final_price"],
                    "price_change": float(price_change),
                    "price_change_percent": float(price_change_pct),
                    "model_weights": ensemble["adjusted_weights"],
                    "consensus": ensemble["consensus"],
                    "prediction_range": ensemble["prediction_range"]
                },
                "confidence": {
                    "overall": ensemble["ensemble_confidence"],
                    "breakdown": confidence_breakdown,
                    "by_model": ensemble["model_confidences"]
                },
                "indicators": {
                    "rsi": indicators.get("rsi"),
                    "macd": indicators.get("macd"),
                    "macd_signal": indicators.get("macd_signal"),
                    "sma_20": indicators.get("sma_20"),
                    "sma_50": indicators.get("sma_50"),
                    "bollinger_upper": indicators.get("bollinger_upper"),
                    "bollinger_lower": indicators.get("bollinger_lower"),
                    "atr": indicators.get("atr")
                },
                "metadata": {
                    "horizon_days": horizon,
                    "lookback_days": self.lookback_period,
                    "data_points": len(data),
                    "models_used": [r["model"] for r in model_results]
                }
            }

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    async def _run_predictions(
        self,
        models: dict,
        data,
        horizon: int
    ) -> list:
        """Run predictions for all selected models concurrently"""
        tasks = []

        for name, model in models.items():
            tasks.append(self._safe_predict(model, data, horizon))

        results = await asyncio.gather(*tasks)

        # Filter out failed predictions
        return [r for r in results if r is not None]

    async def _safe_predict(self, model, data, horizon) -> Optional[Dict]:
        """Safely run a model prediction with error handling"""
        try:
            return await model.predict(data, horizon)
        except Exception as e:
            logger.error(f"Error in {model.name} prediction: {str(e)}")
            return None

    async def get_quick_prediction(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Get a quick prediction using only XGBoost (fastest model)
        """
        return await self.predict(
            symbol,
            horizon=3,
            models=["xgboost"]
        )

    def get_available_models(self) -> list:
        """Get list of available prediction models"""
        return [
            {
                "name": model.name,
                "type": name,
                "is_trained": model.is_trained
            }
            for name, model in self.models.items()
        ]
