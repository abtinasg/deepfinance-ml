"""
Tests for XGBoost model.
"""

import pytest
import pandas as pd
import numpy as np

from app.models.xgboost_model import XGBoostPredictor


@pytest.fixture
def xgboost_model():
    """Create XGBoost model instance."""
    return XGBoostPredictor()


def test_model_initialization(xgboost_model: XGBoostPredictor):
    """Test model initializes correctly."""
    assert xgboost_model.name == "XGBoost"
    assert xgboost_model.version == "2.0"
    assert xgboost_model.is_trained is False


def test_prepare_features(xgboost_model: XGBoostPredictor, sample_ohlcv_data: pd.DataFrame):
    """Test feature preparation."""
    features = xgboost_model.prepare_features(sample_ohlcv_data)

    assert isinstance(features, np.ndarray)
    assert features.shape[1] == 6  # 6 features


def test_train_model(xgboost_model: XGBoostPredictor, sample_ohlcv_data: pd.DataFrame):
    """Test model training."""
    result = xgboost_model.train(sample_ohlcv_data)

    assert result is True
    assert xgboost_model.is_trained is True


@pytest.mark.asyncio
async def test_predict(xgboost_model: XGBoostPredictor, sample_ohlcv_data: pd.DataFrame):
    """Test prediction generation."""
    xgboost_model.train(sample_ohlcv_data)

    result = await xgboost_model.predict(sample_ohlcv_data, horizon=5)

    assert "predictions" in result
    assert "final_price" in result
    assert "trend" in result
    assert len(result["predictions"]) == 5


def test_insufficient_data(xgboost_model: XGBoostPredictor):
    """Test model handles insufficient data."""
    small_data = pd.DataFrame({
        "open": [100, 101],
        "high": [102, 103],
        "low": [99, 100],
        "close": [101, 102],
        "volume": [1000, 1100]
    })

    result = xgboost_model.train(small_data)
    assert result is False
