"""
Base class for all prediction models
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Abstract base class for price prediction models"""

    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.last_train_date = None

    @abstractmethod
    async def predict(
        self,
        data: pd.DataFrame,
        horizon: int = 5
    ) -> Dict[str, Any]:
        """
        Make price predictions

        Args:
            data: Historical OHLCV data
            horizon: Number of days to predict

        Returns:
            Dict containing predictions and metadata
        """
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """Train the model on historical data"""
        pass

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from OHLCV data"""
        features = []

        # Price features
        close = df['close'].values
        open_prices = df['open'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[0], returns])

        # Volatility (rolling std of returns)
        vol = pd.Series(returns).rolling(window=5, min_periods=1).std().values

        # Price momentum
        momentum = close / np.roll(close, 5) - 1
        momentum[:5] = 0

        # Volume change
        vol_change = np.diff(volume) / (volume[:-1] + 1e-8)
        vol_change = np.concatenate([[0], vol_change])

        # Combine features
        for i in range(len(df)):
            feature_row = [
                returns[i],
                vol[i],
                momentum[i],
                vol_change[i],
                (high[i] - low[i]) / close[i],  # Daily range
                (close[i] - open_prices[i]) / open_prices[i],  # Daily return
            ]
            features.append(feature_row)

        return np.array(features)

    def normalize_data(self, data: np.ndarray) -> tuple:
        """Normalize data with min-max scaling"""
        min_val = np.min(data)
        max_val = np.max(data)

        if max_val - min_val == 0:
            return data, min_val, max_val

        normalized = (data - min_val) / (max_val - min_val)
        return normalized, min_val, max_val

    def denormalize_data(
        self,
        data: np.ndarray,
        min_val: float,
        max_val: float
    ) -> np.ndarray:
        """Denormalize data back to original scale"""
        return data * (max_val - min_val) + min_val
