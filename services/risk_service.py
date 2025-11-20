"""
Risk calculation service for portfolio risk metrics
"""
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class RiskService:
    """Service for calculating risk metrics"""

    def __init__(self):
        self.risk_free_rate = settings.RISK_FREE_RATE
        self.trading_days = 252  # Annual trading days

    async def calculate_risk_metrics(
        self,
        asset_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics

        Args:
            asset_data: DataFrame with asset OHLCV data
            benchmark_data: DataFrame with benchmark OHLCV data

        Returns:
            Dict containing all risk metrics
        """
        try:
            close_prices = asset_data['close'].values

            # Calculate returns
            returns = np.diff(close_prices) / close_prices[:-1]

            # Core metrics
            volatility = self.calculate_volatility(returns)
            sharpe_ratio = self.calculate_sharpe_ratio(returns)

            # Calculate beta if benchmark data provided
            beta = None
            if benchmark_data is not None and not benchmark_data.empty:
                benchmark_returns = np.diff(
                    benchmark_data['close'].values
                ) / benchmark_data['close'].values[:-1]
                beta = self.calculate_beta(returns, benchmark_returns)

            # Additional risk metrics
            var_95 = self.calculate_var(returns, 0.95)
            var_99 = self.calculate_var(returns, 0.99)
            max_drawdown = self.calculate_max_drawdown(close_prices)
            sortino_ratio = self.calculate_sortino_ratio(returns)

            return {
                "volatility": {
                    "daily": float(volatility),
                    "annualized": float(volatility * np.sqrt(self.trading_days)),
                    "weekly": float(volatility * np.sqrt(5))
                },
                "beta": float(beta) if beta is not None else None,
                "sharpe_ratio": float(sharpe_ratio),
                "sortino_ratio": float(sortino_ratio),
                "value_at_risk": {
                    "var_95": float(var_95),
                    "var_99": float(var_99)
                },
                "max_drawdown": float(max_drawdown),
                "risk_assessment": self._assess_risk_level(
                    volatility, beta, max_drawdown
                )
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            raise

    def calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate historical volatility (standard deviation of returns)"""
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns, ddof=1))

    def calculate_beta(
        self,
        asset_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate beta coefficient

        Beta = Covariance(Asset, Benchmark) / Variance(Benchmark)
        """
        if len(asset_returns) != len(benchmark_returns):
            # Align lengths
            min_len = min(len(asset_returns), len(benchmark_returns))
            asset_returns = asset_returns[-min_len:]
            benchmark_returns = benchmark_returns[-min_len:]

        if len(asset_returns) < 2:
            return 1.0

        covariance = np.cov(asset_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)

        if benchmark_variance == 0:
            return 1.0

        return float(covariance / benchmark_variance)

    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe Ratio

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev
        """
        if len(returns) < 2:
            return 0.0

        # Convert annual risk-free rate to daily
        daily_rf = self.risk_free_rate / self.trading_days

        excess_returns = returns - daily_rf
        mean_excess = np.mean(excess_returns)
        std_dev = np.std(returns, ddof=1)

        if std_dev == 0:
            return 0.0

        # Annualize
        sharpe = (mean_excess / std_dev) * np.sqrt(self.trading_days)
        return float(sharpe)

    def calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation)

        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
        """
        if len(returns) < 2:
            return 0.0

        # Convert annual risk-free rate to daily
        daily_rf = self.risk_free_rate / self.trading_days

        excess_returns = returns - daily_rf
        mean_excess = np.mean(excess_returns)

        # Calculate downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if mean_excess > 0 else 0.0

        downside_dev = np.std(negative_returns, ddof=1)

        if downside_dev == 0:
            return 0.0

        # Annualize
        sortino = (mean_excess / downside_dev) * np.sqrt(self.trading_days)
        return float(sortino)

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical method

        Returns the maximum expected loss at given confidence level
        """
        if len(returns) < 2:
            return 0.0

        # Historical VaR
        var = np.percentile(returns, (1 - confidence) * 100)
        return float(abs(var))

    def calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """
        Calculate Maximum Drawdown

        Max Drawdown = (Peak - Trough) / Peak
        """
        if len(prices) < 2:
            return 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)

        # Calculate drawdowns
        drawdowns = (running_max - prices) / running_max

        return float(np.max(drawdowns))

    def _assess_risk_level(
        self,
        volatility: float,
        beta: Optional[float],
        max_drawdown: float
    ) -> Dict[str, Any]:
        """Assess overall risk level based on metrics"""
        # Annualized volatility
        ann_vol = volatility * np.sqrt(self.trading_days)

        # Determine volatility risk level
        if ann_vol < 0.15:
            vol_level = "low"
        elif ann_vol < 0.25:
            vol_level = "moderate"
        elif ann_vol < 0.40:
            vol_level = "high"
        else:
            vol_level = "very_high"

        # Determine beta risk level
        if beta is None:
            beta_level = "unknown"
        elif abs(beta) < 0.8:
            beta_level = "defensive"
        elif abs(beta) < 1.2:
            beta_level = "neutral"
        else:
            beta_level = "aggressive"

        # Determine drawdown risk
        if max_drawdown < 0.10:
            dd_level = "low"
        elif max_drawdown < 0.20:
            dd_level = "moderate"
        elif max_drawdown < 0.30:
            dd_level = "high"
        else:
            dd_level = "severe"

        # Overall assessment
        risk_scores = {
            "low": 1, "moderate": 2, "high": 3,
            "very_high": 4, "severe": 4,
            "defensive": 1, "neutral": 2, "aggressive": 3,
            "unknown": 2
        }

        avg_score = (
            risk_scores.get(vol_level, 2) +
            risk_scores.get(beta_level, 2) +
            risk_scores.get(dd_level, 2)
        ) / 3

        if avg_score < 1.5:
            overall = "low"
        elif avg_score < 2.5:
            overall = "moderate"
        elif avg_score < 3.5:
            overall = "high"
        else:
            overall = "very_high"

        return {
            "overall": overall,
            "volatility_level": vol_level,
            "beta_level": beta_level,
            "drawdown_level": dd_level,
            "recommendation": self._get_recommendation(overall)
        }

    def _get_recommendation(self, risk_level: str) -> str:
        """Get investment recommendation based on risk level"""
        recommendations = {
            "low": "Suitable for conservative investors seeking stable returns",
            "moderate": "Balanced risk-reward profile for moderate investors",
            "high": "Higher potential returns with significant volatility",
            "very_high": "Speculative asset - suitable only for risk-tolerant investors"
        }
        return recommendations.get(risk_level, "Risk assessment unavailable")
