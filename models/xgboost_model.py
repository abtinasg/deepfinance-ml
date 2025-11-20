"""
XGBoost model for price prediction
"""
import logging
from typing import Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .base_model import BasePredictor

logger = logging.getLogger(__name__)


class XGBoostPredictor(BasePredictor):
    """XGBoost-based price predictor"""

    def __init__(self):
        super().__init__("XGBoost")
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.feature_importance = None

    def train(self, data: pd.DataFrame) -> None:
        """Train XGBoost model on historical data"""
        try:
            features = self.prepare_features(data)
            close_prices = data['close'].values

            # Create target (next day's return)
            returns = np.diff(close_prices) / close_prices[:-1]

            # Align features and targets
            X = features[:-1]
            y = returns

            # Train model
            self.model.fit(X, y)
            self.feature_importance = self.model.feature_importances_
            self.is_trained = True
            self.last_train_date = datetime.now()

            logger.info(f"{self.name} model trained successfully")

        except Exception as e:
            logger.error(f"Error training {self.name}: {str(e)}")
            raise

    async def predict(
        self,
        data: pd.DataFrame,
        horizon: int = 5
    ) -> Dict[str, Any]:
        """Make price predictions using XGBoost"""
        try:
            # Train on provided data
            self.train(data)

            features = self.prepare_features(data)
            close_prices = data['close'].values
            current_price = close_prices[-1]

            predictions = []
            prediction_features = features[-1].reshape(1, -1)

            # Predict iteratively for each day in horizon
            for _ in range(horizon):
                predicted_return = self.model.predict(prediction_features)[0]
                predicted_price = current_price * (1 + predicted_return)
                predictions.append(predicted_price)
                current_price = predicted_price

                # Update features for next prediction
                prediction_features = self._update_features(
                    prediction_features,
                    predicted_return
                )

            # Calculate prediction metrics
            variance = np.var(predictions)
            trend = "bullish" if predictions[-1] > predictions[0] else "bearish"

            return {
                "model": self.name,
                "predictions": [float(p) for p in predictions],
                "final_price": float(predictions[-1]),
                "variance": float(variance),
                "trend": trend,
                "feature_importance": {
                    "returns": float(self.feature_importance[0]),
                    "volatility": float(self.feature_importance[1]),
                    "momentum": float(self.feature_importance[2]),
                    "volume_change": float(self.feature_importance[3]),
                    "daily_range": float(self.feature_importance[4]),
                    "daily_return": float(self.feature_importance[5])
                }
            }

        except Exception as e:
            logger.error(f"Error in {self.name} prediction: {str(e)}")
            raise

    def _update_features(
        self,
        features: np.ndarray,
        predicted_return: float
    ) -> np.ndarray:
        """Update feature vector for next prediction"""
        new_features = features.copy()
        new_features[0, 0] = predicted_return  # Update return
        new_features[0, 2] = predicted_return  # Update momentum
        return new_features
