"""
XGBoost-based price predictor.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.core.logging import get_logger
from app.models.base_model import BasePredictor

logger = get_logger(__name__)


class XGBoostPredictor(BasePredictor):
    """XGBoost model for price prediction."""

    def __init__(self):
        super().__init__(name="XGBoost", version="2.0")
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self._feature_importance: dict[str, float] = {}

    def load_weights(self, path: str) -> bool:
        """Load pre-trained XGBoost model."""
        try:
            filepath = Path(path)
            if not filepath.exists():
                logger.warning(f"Weights file not found: {path}")
                return False

            with open(filepath, "rb") as f:
                saved_data = pickle.load(f)

            self.model = saved_data["model"]
            self._feature_importance = saved_data.get("feature_importance", {})
            self.is_trained = True
            self._weights_path = path

            logger.info(f"Loaded XGBoost weights from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load XGBoost weights: {e}")
            return False

    def save_weights(self, path: str) -> bool:
        """Save trained XGBoost model."""
        try:
            filepath = Path(path)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            saved_data = {
                "model": self.model,
                "feature_importance": self._feature_importance,
                "version": self.version
            }

            with open(filepath, "wb") as f:
                pickle.dump(saved_data, f)

            self._weights_path = path
            logger.info(f"Saved XGBoost weights to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save XGBoost weights: {e}")
            return False

    def train(self, data: pd.DataFrame) -> bool:
        """
        Train the XGBoost model on historical data.

        Args:
            data: Historical OHLCV data.

        Returns:
            True if training successful.
        """
        try:
            features = self.prepare_features(data)
            if len(features) < 20:
                logger.warning("Insufficient data for XGBoost training")
                return False

            # Target: next day's return
            returns = data["close"].pct_change().dropna().values
            X = features[:-1]
            y = returns[len(returns) - len(X):]

            self.model.fit(X, y)

            # Store feature importance
            importance = self.model.feature_importances_
            feature_names = [
                "returns", "volatility", "momentum",
                "volume_change", "daily_range", "daily_return"
            ]
            self._feature_importance = dict(zip(feature_names, importance.tolist()))

            self.is_trained = True
            logger.info("XGBoost model trained successfully")
            return True

        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return False

    async def predict(
        self,
        data: pd.DataFrame,
        horizon: int = 5
    ) -> dict[str, Any]:
        """
        Generate price predictions using XGBoost.

        Args:
            data: Historical OHLCV data.
            horizon: Number of days to predict.

        Returns:
            Prediction results with confidence and features.
        """
        try:
            if not self.is_trained:
                # Train if no pre-trained weights
                if not self.train(data):
                    raise ValueError("Model training failed")

            features = self.prepare_features(data)
            current_price = float(data["close"].iloc[-1])
            predictions = []
            current_features = features[-1:].copy()

            for _ in range(horizon):
                # Predict next return
                predicted_return = self.model.predict(current_features)[0]

                # Calculate next price
                next_price = current_price * (1 + predicted_return)
                predictions.append(float(next_price))

                # Update features for next iteration
                current_features[0][0] = predicted_return  # returns
                current_features[0][2] = predicted_return  # momentum (simplified)
                current_price = next_price

            final_price = predictions[-1]
            price_change = (final_price - float(data["close"].iloc[-1])) / float(data["close"].iloc[-1])

            # Calculate variance from predictions
            variance = float(np.var(predictions) / (np.mean(predictions) ** 2))

            return {
                "model": self.name,
                "predictions": predictions,
                "final_price": final_price,
                "change_percent": price_change * 100,
                "variance": variance,
                "trend": "bullish" if price_change > 0 else "bearish",
                "feature_importance": self._feature_importance
            }

        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            raise
