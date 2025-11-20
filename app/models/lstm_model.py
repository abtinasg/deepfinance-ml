"""
LSTM neural network for price prediction.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.core.logging import get_logger
from app.models.base_model import BasePredictor

logger = get_logger(__name__)


class LSTMNetwork(nn.Module):
    """LSTM neural network architecture."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # Use the last timestep output
        out = self.fc(lstm_out[:, -1, :])
        return out


class LSTMPredictor(BasePredictor):
    """LSTM model for price prediction."""

    def __init__(self):
        super().__init__(name="LSTM", version="2.0")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = LSTMNetwork().to(self.device)
        self.sequence_length = 10
        self._scaler_params: dict[str, tuple] = {}

    def load_weights(self, path: str) -> bool:
        """Load pre-trained LSTM weights."""
        try:
            filepath = Path(path)
            if not filepath.exists():
                logger.warning(f"Weights file not found: {path}")
                return False

            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self._scaler_params = checkpoint.get("scaler_params", {})
            self.is_trained = True
            self._weights_path = path

            logger.info(f"Loaded LSTM weights from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load LSTM weights: {e}")
            return False

    def save_weights(self, path: str) -> bool:
        """Save trained LSTM weights."""
        try:
            filepath = Path(path)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "scaler_params": self._scaler_params,
                "version": self.version
            }

            torch.save(checkpoint, filepath)
            self._weights_path = path
            logger.info(f"Saved LSTM weights to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save LSTM weights: {e}")
            return False

    def _normalize(
        self,
        data: np.ndarray,
        name: str
    ) -> np.ndarray:
        """Normalize data and store parameters."""
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        self._scaler_params[name] = (mean.tolist(), std.tolist())
        return (data - mean) / std

    def _denormalize(
        self,
        data: np.ndarray,
        name: str
    ) -> np.ndarray:
        """Denormalize data using stored parameters."""
        if name not in self._scaler_params:
            return data
        mean, std = self._scaler_params[name]
        mean = np.array(mean)
        std = np.array(std)
        return data * std + mean

    def _create_sequences(
        self,
        features: np.ndarray,
        prices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(prices[i + self.sequence_length])
        return np.array(X), np.array(y)

    def train(
        self,
        data: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 16
    ) -> bool:
        """
        Train the LSTM model.

        Args:
            data: Historical OHLCV data.
            epochs: Number of training epochs.
            batch_size: Training batch size.

        Returns:
            True if training successful.
        """
        try:
            features = self.prepare_features(data)
            prices = data["close"].iloc[len(data) - len(features):].values

            if len(features) < self.sequence_length + 10:
                logger.warning("Insufficient data for LSTM training")
                return False

            # Normalize
            features_norm = self._normalize(features, "features")
            prices_norm = self._normalize(prices.reshape(-1, 1), "prices").flatten()

            # Create sequences
            X, y = self._create_sequences(features_norm, prices_norm)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    output = self.model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    logger.debug(f"LSTM Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

            # Cleanup
            del X_tensor, y_tensor, dataset, dataloader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_trained = True
            logger.info("LSTM model trained successfully")
            return True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA out of memory during LSTM training")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return False

    async def predict(
        self,
        data: pd.DataFrame,
        horizon: int = 5
    ) -> dict[str, Any]:
        """
        Generate price predictions using LSTM.

        Args:
            data: Historical OHLCV data.
            horizon: Number of days to predict.

        Returns:
            Prediction results.
        """
        try:
            if not self.is_trained:
                if not self.train(data):
                    raise ValueError("Model training failed")

            features = self.prepare_features(data)
            prices = data["close"].iloc[len(data) - len(features):].values
            current_price = float(prices[-1])

            # Normalize using stored parameters
            if "features" in self._scaler_params:
                mean, std = self._scaler_params["features"]
                features_norm = (features - np.array(mean)) / np.array(std)
            else:
                features_norm = self._normalize(features, "features")

            # Get last sequence
            sequence = features_norm[-self.sequence_length:]

            self.model.eval()
            predictions = []

            with torch.no_grad():
                for _ in range(horizon):
                    # Predict
                    input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                    output = self.model(input_tensor)

                    # Denormalize prediction
                    pred_norm = output.cpu().numpy()[0, 0]
                    if "prices" in self._scaler_params:
                        mean, std = self._scaler_params["prices"]
                        pred_price = pred_norm * std[0] + mean[0]
                    else:
                        pred_price = pred_norm

                    predictions.append(float(pred_price))

                    # Update sequence for next prediction
                    # Create new feature row (simplified - use last features with updated return)
                    new_feature = sequence[-1].copy()
                    new_feature[0] = (pred_price - current_price) / current_price  # return
                    sequence = np.vstack([sequence[1:], new_feature])
                    current_price = pred_price

                    # Cleanup
                    del input_tensor

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            final_price = predictions[-1]
            original_price = float(data["close"].iloc[-1])
            price_change = (final_price - original_price) / original_price

            return {
                "model": self.name,
                "predictions": predictions,
                "final_price": final_price,
                "change_percent": price_change * 100,
                "trend": "bullish" if price_change > 0 else "bearish",
                "device": str(self.device)
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA out of memory during LSTM prediction")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            raise
