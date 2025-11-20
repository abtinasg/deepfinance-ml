"""
Base predictor class for all ML models.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)


class BasePredictor(ABC):
    """Abstract base class for all prediction models."""

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.is_trained = False
        self._weights_path: str | None = None

    @abstractmethod
    async def predict(
        self,
        data: pd.DataFrame,
        horizon: int = 5
    ) -> dict[str, Any]:
        """
        Generate price predictions.

        Args:
            data: Historical OHLCV data.
            horizon: Number of days to predict.

        Returns:
            Dictionary with predictions and metadata.
        """
        pass

    @abstractmethod
    def load_weights(self, path: str) -> bool:
        """
        Load pre-trained weights.

        Args:
            path: Path to weights file.

        Returns:
            True if successful.
        """
        pass

    @abstractmethod
    def save_weights(self, path: str) -> bool:
        """
        Save trained weights.

        Args:
            path: Path to save weights.

        Returns:
            True if successful.
        """
        pass

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix from OHLCV data.

        Creates 6 features:
        - Daily returns
        - Rolling volatility (5-day)
        - Momentum (5-day)
        - Volume change
        - Daily range
        - Daily return (close - open)

        Args:
            data: DataFrame with OHLCV columns.

        Returns:
            Feature matrix as numpy array.
        """
        df = data.copy()

        # Calculate features
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=5).std()
        df["momentum"] = df["close"].pct_change(periods=5)
        df["volume_change"] = df["volume"].pct_change()
        df["daily_range"] = (df["high"] - df["low"]) / df["close"]
        df["daily_return"] = (df["close"] - df["open"]) / df["open"]

        # Drop NaN values
        df = df.dropna()

        features = df[
            ["returns", "volatility", "momentum",
             "volume_change", "daily_range", "daily_return"]
        ].values

        return features

    def get_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "name": self.name,
            "version": self.version,
            "is_trained": self.is_trained,
            "weights_path": self._weights_path
        }
