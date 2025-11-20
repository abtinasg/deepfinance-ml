"""
Tests for confidence engine.
"""

import pytest
import numpy as np

from app.services.confidence_engine import ConfidenceEngine


@pytest.fixture
def confidence_engine():
    """Create confidence engine instance."""
    return ConfidenceEngine()


def test_calculate_confidence_consistent_predictions(confidence_engine: ConfidenceEngine):
    """Test confidence for consistent predictions."""
    # Predictions that follow recent trend
    historical = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    predictions = [111, 112, 113, 114, 115]

    confidence = confidence_engine.calculate_model_confidence(predictions, historical)

    assert 0 <= confidence <= 1
    assert confidence > 0.5  # Should be relatively high


def test_calculate_confidence_extreme_predictions(confidence_engine: ConfidenceEngine):
    """Test confidence for extreme predictions."""
    historical = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    # Extreme prediction
    predictions = [150, 200, 250, 300, 350]

    confidence = confidence_engine.calculate_model_confidence(predictions, historical)

    assert confidence < 0.5  # Should be low


def test_create_ensemble_prediction(confidence_engine: ConfidenceEngine):
    """Test ensemble prediction creation."""
    model_results = [
        {
            "model": "XGBoost",
            "predictions": [111, 112, 113, 114, 115],
            "trend": "bullish"
        },
        {
            "model": "LSTM",
            "predictions": [112, 113, 114, 115, 116],
            "trend": "bullish"
        },
        {
            "model": "Transformer",
            "predictions": [110, 111, 112, 113, 114],
            "trend": "bullish"
        }
    ]
    historical = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

    ensemble = confidence_engine.create_ensemble_prediction(model_results, historical)

    assert "predictions" in ensemble
    assert "final_price" in ensemble
    assert "confidence" in ensemble
    assert "consensus" in ensemble
    assert ensemble["consensus"] == "bullish"


def test_confidence_breakdown(confidence_engine: ConfidenceEngine):
    """Test confidence level description."""
    breakdown = confidence_engine.get_confidence_breakdown(0.85)
    assert breakdown["level"] == "very_high"

    breakdown = confidence_engine.get_confidence_breakdown(0.5)
    assert breakdown["level"] == "moderate"

    breakdown = confidence_engine.get_confidence_breakdown(0.2)
    assert breakdown["level"] == "very_low"
