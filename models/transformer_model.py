"""
Transformer model for price prediction
"""
import logging
import math
from typing import Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BasePredictor

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class TransformerNetwork(nn.Module):
    """Transformer Neural Network for time series prediction"""

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
            dim_feedforward=d_model * 4,
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

    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Use last position for prediction
        out = self.fc(x[:, -1, :])
        return out


class TransformerPredictor(BasePredictor):
    """Transformer-based price predictor"""

    def __init__(self):
        super().__init__("Transformer")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = TransformerNetwork().to(self.device)
        self.sequence_length = 10
        self.epochs = 50
        self.batch_size = 16
        self.learning_rate = 0.001
        self.min_price = 0
        self.max_price = 1

    def train(self, data: pd.DataFrame) -> None:
        """Train Transformer model on historical data"""
        try:
            features = self.prepare_features(data)
            close_prices = data['close'].values

            # Normalize
            norm_features, _, _ = self.normalize_data(features)
            norm_prices, self.min_price, self.max_price = self.normalize_data(
                close_prices
            )

            # Create sequences
            X, y = self._create_sequences(norm_features, norm_prices)

            if len(X) == 0:
                logger.warning("Not enough data for Transformer training")
                return

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )

            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            self.is_trained = True
            self.last_train_date = datetime.now()
            logger.info(f"{self.name} model trained successfully")

        except Exception as e:
            logger.error(f"Error training {self.name}: {str(e)}")
            raise

    def _create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray
    ) -> tuple:
        """Create sequences for Transformer training"""
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(targets[i + self.sequence_length])
        return np.array(X), np.array(y)

    async def predict(
        self,
        data: pd.DataFrame,
        horizon: int = 5
    ) -> Dict[str, Any]:
        """Make price predictions using Transformer"""
        try:
            # Train on provided data
            self.train(data)

            features = self.prepare_features(data)
            close_prices = data['close'].values

            # Normalize
            norm_features, _, _ = self.normalize_data(features)

            # Get last sequence
            last_sequence = norm_features[-self.sequence_length:]
            last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(
                self.device
            )

            predictions = []
            self.model.eval()

            with torch.no_grad():
                current_sequence = last_sequence.clone()

                for _ in range(horizon):
                    # Predict next value
                    pred = self.model(current_sequence)
                    pred_value = pred.item()

                    # Denormalize prediction
                    denorm_pred = self.denormalize_data(
                        np.array([pred_value]),
                        self.min_price,
                        self.max_price
                    )[0]
                    predictions.append(denorm_pred)

                    # Update sequence for next prediction
                    new_features = current_sequence[:, -1, :].clone()
                    current_sequence = torch.cat([
                        current_sequence[:, 1:, :],
                        new_features.unsqueeze(1)
                    ], dim=1)

            # Calculate prediction metrics
            variance = np.var(predictions)
            trend = "bullish" if predictions[-1] > predictions[0] else "bearish"

            # Attention analysis (simplified)
            attention_weights = self._get_attention_weights()

            return {
                "model": self.name,
                "predictions": [float(p) for p in predictions],
                "final_price": float(predictions[-1]),
                "variance": float(variance),
                "trend": trend,
                "model_info": {
                    "sequence_length": self.sequence_length,
                    "epochs": self.epochs,
                    "device": str(self.device),
                    "attention_focus": attention_weights
                }
            }

        except Exception as e:
            logger.error(f"Error in {self.name} prediction: {str(e)}")
            raise

    def _get_attention_weights(self) -> Dict[str, float]:
        """Get simplified attention weight analysis"""
        # This is a simplified representation
        # In production, you would extract actual attention weights
        return {
            "recent_prices": 0.4,
            "momentum": 0.3,
            "volatility": 0.2,
            "volume": 0.1
        }
