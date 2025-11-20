"""
Tests for risk service calculations.
"""

import pytest
import numpy as np

from app.services.risk_service import RiskService


@pytest.fixture
def risk_service():
    """Create risk service instance."""
    return RiskService()


def test_calculate_volatility(risk_service: RiskService, sample_returns: np.ndarray):
    """Test volatility calculation."""
    vol = risk_service.calculate_volatility(sample_returns)

    assert "daily" in vol
    assert "weekly" in vol
    assert "annualized" in vol
    assert vol["daily"] > 0
    assert vol["annualized"] > vol["daily"]


def test_calculate_sharpe_ratio(risk_service: RiskService, sample_returns: np.ndarray):
    """Test Sharpe ratio calculation."""
    sharpe = risk_service.calculate_sharpe_ratio(sample_returns)

    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)


def test_calculate_sortino_ratio(risk_service: RiskService, sample_returns: np.ndarray):
    """Test Sortino ratio calculation."""
    sortino = risk_service.calculate_sortino_ratio(sample_returns)

    assert isinstance(sortino, float)


def test_calculate_var(risk_service: RiskService, sample_returns: np.ndarray):
    """Test VaR calculation."""
    var_95 = risk_service.calculate_var(sample_returns, 0.95)
    var_99 = risk_service.calculate_var(sample_returns, 0.99)

    assert var_95 < 0  # VaR is typically negative
    assert var_99 < var_95  # 99% VaR should be more extreme


def test_calculate_max_drawdown(risk_service: RiskService, sample_ohlcv_data):
    """Test max drawdown calculation."""
    prices = sample_ohlcv_data["close"].values
    max_dd = risk_service.calculate_max_drawdown(prices)

    assert max_dd <= 0  # Drawdown is negative
    assert max_dd >= -1  # Cannot be worse than -100%


def test_calculate_beta(risk_service: RiskService):
    """Test beta calculation."""
    np.random.seed(42)
    market_returns = np.random.normal(0.001, 0.01, 100)
    # Asset with beta ~1.5
    asset_returns = 1.5 * market_returns + np.random.normal(0, 0.005, 100)

    beta = risk_service.calculate_beta(asset_returns, market_returns)

    assert 1.0 < beta < 2.0  # Should be close to 1.5
