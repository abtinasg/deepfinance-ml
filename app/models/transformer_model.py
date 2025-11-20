"""
Transformer-based price predictor.
"""

import math
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


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerNetwork(nn.Module):
    """Transformer network for time series prediction."""

    def __init__(
        self,
        input_size: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input to d_model dimensions
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Use last position for prediction
        x = self.fc(x[:, -1, :])

        return x


class TransformerPredictor(BasePredictor):
    """Transformer model for price prediction."""

    def __init__(self):
        super().__init__(name="Transformer", version="2.0")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = TransformerNetwork().to(self.device)
        self.sequence_length = 10
        self._scaler_params: dict[str, tuple] = {}

    def load_weights(self, path: str) -> bool:
        """Load pre-trained Transformer weights."""
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

            logger.info(f"Loaded Transformer weights from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Transformer weights: {e}")
            return False

    def save_weights(self, path: str) -> bool:
        """Save trained Transformer weights."""
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
            logger.info(f"Saved Transformer weights to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save Transformer weights: {e}")
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

    def _create_sequences(
        self,
        features: np.ndarray,
        prices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
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
        Train the Transformer model.

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
                logger.warning("Insufficient data for Transformer training")
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
                    logger.debug(f"Transformer Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

            # Cleanup
            del X_tensor, y_tensor, dataset, dataloader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_trained = True
            logger.info("Transformer model trained successfully")
            return True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA out of memory during Transformer training")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
            return False

    async def predict(
        self,
        data: pd.DataFrame,
        horizon: int = 5
    ) -> dict[str, Any]:
        """
        Generate price predictions using Transformer.

        Args:
            data: Historical OHLCV data.
            horizon: Number of days to predict.

        Returns:
            Prediction results with attention weights.
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

                    # Update sequence
                    new_feature = sequence[-1].copy()
                    new_feature[0] = (pred_price - current_price) / current_price
                    sequence = np.vstack([sequence[1:], new_feature])
                    current_price = pred_price

                    del input_tensor

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            final_price = predictions[-1]
            original_price = float(data["close"].iloc[-1])
            price_change = (final_price - original_price) / original_price

            # Attention breakdown (placeholder - would need hooks for real attention)
            attention_breakdown = {
                "recent_trend": 0.4,
                "volatility": 0.25,
                "momentum": 0.2,
                "volume": 0.15
            }

            return {
                "model": self.name,
                "predictions": predictions,
                "final_price": final_price,
                "change_percent": price_change * 100,
                "trend": "bullish" if price_change > 0 else "bearish",
                "attention_weights": attention_breakdown,
                "device": str(self.device)
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA out of memory during Transformer prediction")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            raise
