"""
Confidence scoring and ensemble prediction engine.
"""

from typing import Any

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ConfidenceEngine:
    """
    Engine for calculating prediction confidence and creating ensemble predictions.
    """

    def calculate_model_confidence(
        self,
        predictions: list[float],
        historical_prices: np.ndarray
    ) -> float:
        """
        Calculate confidence score for a model's predictions.

        Factors:
        - Prediction consistency (30%)
        - Trend alignment (40%)
        - Magnitude reasonableness (30%)

        Args:
            predictions: Model's price predictions.
            historical_prices: Historical price data.

        Returns:
            Confidence score between 0 and 1.
        """
        if not predictions or len(historical_prices) < 5:
            return 0.5

        predictions = np.array(predictions)
        current_price = float(historical_prices[-1])

        # 1. Prediction Consistency (30%)
        # Lower variance = higher confidence
        pred_returns = np.diff(predictions) / predictions[:-1]
        consistency = 1 - min(np.std(pred_returns) * 10, 1)
        consistency_score = max(0, consistency) * 0.3

        # 2. Trend Alignment (40%)
        # Check if prediction aligns with recent trend
        recent_returns = np.diff(historical_prices[-10:]) / historical_prices[-10:-1]
        recent_trend = np.mean(recent_returns)
        predicted_return = (predictions[-1] - current_price) / current_price

        # Same direction = higher confidence
        if recent_trend * predicted_return > 0:
            trend_score = 0.4
        elif abs(predicted_return) < 0.01:
            trend_score = 0.2  # Neutral prediction
        else:
            trend_score = 0.1  # Opposing trend

        # 3. Magnitude Reasonableness (30%)
        # Predictions within reasonable bounds
        abs_return = abs(predicted_return)

        if abs_return < 0.02:
            magnitude_score = 0.3
        elif abs_return < 0.05:
            magnitude_score = 0.25
        elif abs_return < 0.10:
            magnitude_score = 0.15
        else:
            magnitude_score = 0.05  # Extreme predictions

        total_confidence = consistency_score + trend_score + magnitude_score

        return float(min(max(total_confidence, 0), 1))

    def create_ensemble_prediction(
        self,
        model_results: list[dict[str, Any]],
        historical_prices: np.ndarray
    ) -> dict[str, Any]:
        """
        Create weighted ensemble prediction from multiple models.

        Args:
            model_results: List of model prediction results.
            historical_prices: Historical price data.

        Returns:
            Ensemble prediction with confidence and consensus.
        """
        if not model_results:
            raise ValueError("No model results provided")

        settings = get_settings()
        current_price = float(historical_prices[-1])

        # Get base weights from settings
        weight_map = {
            "XGBoost": settings.XGBOOST_WEIGHT,
            "LSTM": settings.LSTM_WEIGHT,
            "Transformer": settings.TRANSFORMER_WEIGHT
        }

        # Calculate confidence-adjusted weights
        weighted_predictions = []
        total_weight = 0

        for result in model_results:
            model_name = result["model"]
            base_weight = weight_map.get(model_name, 0.33)

            # Calculate confidence
            confidence = self.calculate_model_confidence(
                result["predictions"],
                historical_prices
            )

            # Adjust weight by confidence
            adjusted_weight = base_weight * (0.5 + confidence * 0.5)
            total_weight += adjusted_weight

            weighted_predictions.append({
                "predictions": np.array(result["predictions"]),
                "weight": adjusted_weight,
                "confidence": confidence,
                "trend": result.get("trend", "neutral")
            })

        # Normalize weights
        for wp in weighted_predictions:
            wp["weight"] /= total_weight

        # Create ensemble predictions
        horizon = len(weighted_predictions[0]["predictions"])
        ensemble_preds = np.zeros(horizon)

        for wp in weighted_predictions:
            ensemble_preds += wp["predictions"] * wp["weight"]

        # Calculate ensemble statistics
        all_predictions = np.array([wp["predictions"] for wp in weighted_predictions])
        pred_min = np.min(all_predictions[:, -1])
        pred_max = np.max(all_predictions[:, -1])

        # Calculate confidence
        # Higher agreement between models = higher confidence
        pred_std = np.std(all_predictions[:, -1])
        agreement = 1 - (pred_std / (current_price * 0.1))
        base_confidence = np.mean([wp["confidence"] for wp in weighted_predictions])
        ensemble_confidence = (base_confidence * 0.7 + max(0, agreement) * 0.3)

        # Calculate consensus
        trends = [wp["trend"] for wp in weighted_predictions]
        bullish_count = sum(1 for t in trends if t == "bullish")
        bearish_count = sum(1 for t in trends if t == "bearish")

        if bullish_count > bearish_count:
            consensus = "bullish"
            consensus_strength = bullish_count / len(trends)
        elif bearish_count > bullish_count:
            consensus = "bearish"
            consensus_strength = bearish_count / len(trends)
        else:
            consensus = "neutral"
            consensus_strength = 0.5

        final_price = float(ensemble_preds[-1])
        change_percent = ((final_price - current_price) / current_price) * 100

        # Get confidence breakdown
        confidence_breakdown = self.get_confidence_breakdown(ensemble_confidence)

        return {
            "predictions": ensemble_preds.tolist(),
            "final_price": final_price,
            "change_percent": change_percent,
            "confidence": float(ensemble_confidence),
            "confidence_breakdown": confidence_breakdown,
            "consensus": consensus,
            "consensus_strength": float(consensus_strength),
            "prediction_range": {
                "min": float(pred_min),
                "max": float(pred_max)
            }
        }

    def get_confidence_breakdown(
        self,
        confidence: float
    ) -> dict[str, str]:
        """
        Get human-readable confidence breakdown.

        Args:
            confidence: Confidence score (0-1).

        Returns:
            Dictionary with level and description.
        """
        if confidence >= 0.8:
            return {
                "level": "very_high",
                "description": "Strong model agreement with clear trend signals"
            }
        elif confidence >= 0.65:
            return {
                "level": "high",
                "description": "Good model agreement with consistent predictions"
            }
        elif confidence >= 0.5:
            return {
                "level": "moderate",
                "description": "Mixed signals - predictions vary between models"
            }
        elif confidence >= 0.35:
            return {
                "level": "low",
                "description": "High uncertainty - models disagree significantly"
            }
        else:
            return {
                "level": "very_low",
                "description": "Very high uncertainty - consider with caution"
            }
