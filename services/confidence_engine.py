"""
Confidence Engine for model ensemble and confidence scoring
"""
import logging
from typing import Dict, Any, List

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class ConfidenceEngine:
    """Engine for calculating confidence scores and ensemble predictions"""

    def __init__(self):
        self.weights = {
            "XGBoost": settings.XGBOOST_WEIGHT,
            "LSTM": settings.LSTM_WEIGHT,
            "Transformer": settings.TRANSFORMER_WEIGHT
        }
        self.min_confidence = settings.MIN_CONFIDENCE
        self.max_confidence = settings.MAX_CONFIDENCE

    def calculate_model_confidence(
        self,
        predictions: List[float],
        historical_prices: np.ndarray
    ) -> float:
        """
        Calculate confidence score for a single model's predictions

        Factors considered:
        - Prediction variance (lower is better)
        - Alignment with recent trend
        - Prediction magnitude relative to historical volatility
        """
        if not predictions or len(historical_prices) < 2:
            return 0.5

        # Factor 1: Prediction consistency (inverse of variance)
        pred_variance = np.var(predictions)
        price_variance = np.var(historical_prices)

        if price_variance > 0:
            variance_ratio = pred_variance / price_variance
            consistency_score = 1 / (1 + variance_ratio)
        else:
            consistency_score = 0.5

        # Factor 2: Trend alignment
        recent_trend = (
            historical_prices[-1] - historical_prices[-5]
        ) / historical_prices[-5] if len(historical_prices) >= 5 else 0

        predicted_change = (
            predictions[-1] - historical_prices[-1]
        ) / historical_prices[-1]

        # Check if prediction aligns with recent trend
        if (recent_trend > 0 and predicted_change > 0) or \
           (recent_trend < 0 and predicted_change < 0):
            trend_score = 0.7
        elif abs(recent_trend) < 0.01:  # Sideways market
            trend_score = 0.6
        else:
            trend_score = 0.3

        # Factor 3: Magnitude reasonableness
        daily_volatility = np.std(
            np.diff(historical_prices) / historical_prices[:-1]
        )
        expected_max_change = daily_volatility * len(predictions) * 2

        total_predicted_change = abs(
            (predictions[-1] - historical_prices[-1]) / historical_prices[-1]
        )

        if total_predicted_change <= expected_max_change:
            magnitude_score = 0.8
        elif total_predicted_change <= expected_max_change * 2:
            magnitude_score = 0.5
        else:
            magnitude_score = 0.2

        # Weighted combination
        confidence = (
            consistency_score * 0.3 +
            trend_score * 0.4 +
            magnitude_score * 0.3
        )

        # Clamp to valid range
        confidence = max(
            self.min_confidence,
            min(self.max_confidence, confidence)
        )

        return float(confidence)

    def create_ensemble_prediction(
        self,
        model_results: List[Dict[str, Any]],
        historical_prices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create weighted ensemble prediction from multiple models

        Args:
            model_results: List of model prediction results
            historical_prices: Historical close prices

        Returns:
            Ensemble prediction with confidence scores
        """
        if not model_results:
            return {"error": "No model results provided"}

        # Calculate confidence for each model
        model_confidences = {}
        model_predictions = {}

        for result in model_results:
            model_name = result["model"]
            predictions = result["predictions"]

            confidence = self.calculate_model_confidence(
                predictions,
                historical_prices
            )

            model_confidences[model_name] = confidence
            model_predictions[model_name] = predictions

        # Adjust weights by confidence
        adjusted_weights = {}
        total_weight = 0

        for model_name, base_weight in self.weights.items():
            if model_name in model_confidences:
                # Multiply base weight by confidence
                adjusted = base_weight * model_confidences[model_name]
                adjusted_weights[model_name] = adjusted
                total_weight += adjusted

        # Normalize weights
        if total_weight > 0:
            for model_name in adjusted_weights:
                adjusted_weights[model_name] /= total_weight

        # Calculate ensemble predictions
        horizon = len(model_results[0]["predictions"])
        ensemble_predictions = []

        for i in range(horizon):
            weighted_sum = 0
            for model_name, weight in adjusted_weights.items():
                if model_name in model_predictions:
                    weighted_sum += (
                        model_predictions[model_name][i] * weight
                    )
            ensemble_predictions.append(weighted_sum)

        # Calculate overall ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(
            model_results,
            model_confidences,
            adjusted_weights
        )

        # Determine consensus
        trends = [r["trend"] for r in model_results]
        bullish_count = sum(1 for t in trends if t == "bullish")
        consensus = "bullish" if bullish_count > len(trends) / 2 else "bearish"
        consensus_strength = max(
            bullish_count,
            len(trends) - bullish_count
        ) / len(trends)

        return {
            "ensemble_predictions": [float(p) for p in ensemble_predictions],
            "final_price": float(ensemble_predictions[-1]),
            "ensemble_confidence": float(ensemble_confidence),
            "model_confidences": {
                k: float(v) for k, v in model_confidences.items()
            },
            "adjusted_weights": {
                k: float(v) for k, v in adjusted_weights.items()
            },
            "consensus": {
                "direction": consensus,
                "strength": float(consensus_strength),
                "model_agreement": f"{max(bullish_count, len(trends) - bullish_count)}/{len(trends)}"
            },
            "prediction_range": {
                "min": float(min(ensemble_predictions)),
                "max": float(max(ensemble_predictions)),
                "volatility": float(np.std(ensemble_predictions))
            }
        }

    def _calculate_ensemble_confidence(
        self,
        model_results: List[Dict[str, Any]],
        model_confidences: Dict[str, float],
        adjusted_weights: Dict[str, float]
    ) -> float:
        """Calculate overall ensemble confidence"""
        # Factor 1: Weighted average of individual confidences
        weighted_conf = sum(
            model_confidences.get(name, 0) * weight
            for name, weight in adjusted_weights.items()
        )

        # Factor 2: Model agreement (variance of final predictions)
        final_prices = [r["final_price"] for r in model_results]
        mean_price = np.mean(final_prices)

        if mean_price > 0:
            # Coefficient of variation
            cv = np.std(final_prices) / mean_price
            agreement_score = 1 / (1 + cv * 10)  # Scale CV impact
        else:
            agreement_score = 0.5

        # Factor 3: Trend consensus
        trends = [r["trend"] for r in model_results]
        consensus_ratio = max(
            sum(1 for t in trends if t == "bullish"),
            sum(1 for t in trends if t == "bearish")
        ) / len(trends)

        # Combine factors
        ensemble_confidence = (
            weighted_conf * 0.4 +
            agreement_score * 0.35 +
            consensus_ratio * 0.25
        )

        return max(
            self.min_confidence,
            min(self.max_confidence, ensemble_confidence)
        )

    def get_confidence_breakdown(
        self,
        confidence: float
    ) -> Dict[str, Any]:
        """Get human-readable confidence breakdown"""
        if confidence >= 0.8:
            level = "very_high"
            description = "Strong conviction in prediction accuracy"
        elif confidence >= 0.6:
            level = "high"
            description = "Good confidence with moderate uncertainty"
        elif confidence >= 0.4:
            level = "moderate"
            description = "Balanced confidence with notable uncertainty"
        elif confidence >= 0.2:
            level = "low"
            description = "Limited confidence - use with caution"
        else:
            level = "very_low"
            description = "Minimal confidence - prediction unreliable"

        return {
            "level": level,
            "score": float(confidence),
            "description": description
        }
