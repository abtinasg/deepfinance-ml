"""
Risk analysis and metrics calculation service.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from app.config import get_settings
from app.core.logging import get_logger
from app.services.data_fetcher import DataFetcher

logger = get_logger(__name__)


class RiskService:
    """
    Service for calculating financial risk metrics.
    """

    def __init__(self):
        self._data_fetcher = DataFetcher()

    async def calculate_risk_metrics(
        self,
        symbol: str,
        period_days: int = 30,
        include_benchmark: bool = True
    ) -> dict[str, Any]:
        """
        Calculate comprehensive risk metrics for a symbol.

        Args:
            symbol: Stock ticker symbol.
            period_days: Analysis period.
            include_benchmark: Include S&P 500 comparison.

        Returns:
            Complete risk analysis.
        """
        settings = get_settings()

        # Fetch asset data
        asset_data = await self._data_fetcher.fetch_historical_data(
            symbol,
            period_days
        )

        if asset_data is None or asset_data.empty:
            raise ValueError(f"Failed to fetch data for {symbol}")

        # Calculate asset returns
        asset_returns = asset_data["close"].pct_change().dropna().values
        asset_prices = asset_data["close"].values

        # Fetch benchmark if requested
        benchmark_returns = None
        benchmark_comparison = None

        if include_benchmark:
            benchmark_data = await self._data_fetcher.fetch_benchmark_data(
                period_days
            )
            if benchmark_data is not None and not benchmark_data.empty:
                benchmark_returns = benchmark_data["close"].pct_change().dropna().values
                # Align lengths
                min_len = min(len(asset_returns), len(benchmark_returns))
                asset_returns_aligned = asset_returns[-min_len:]
                benchmark_returns_aligned = benchmark_returns[-min_len:]
            else:
                asset_returns_aligned = asset_returns
                benchmark_returns_aligned = None
        else:
            asset_returns_aligned = asset_returns
            benchmark_returns_aligned = None

        # Calculate metrics
        volatility = self.calculate_volatility(asset_returns)
        sharpe = self.calculate_sharpe_ratio(asset_returns)
        sortino = self.calculate_sortino_ratio(asset_returns)
        var_95 = self.calculate_var(asset_returns, 0.95)
        var_99 = self.calculate_var(asset_returns, 0.99)
        max_dd = self.calculate_max_drawdown(asset_prices)

        # Calculate beta if benchmark available
        beta = None
        if benchmark_returns_aligned is not None:
            beta = self.calculate_beta(
                asset_returns_aligned,
                benchmark_returns_aligned
            )
            benchmark_comparison = {
                "correlation": float(np.corrcoef(
                    asset_returns_aligned,
                    benchmark_returns_aligned
                )[0, 1]),
                "relative_volatility": float(
                    np.std(asset_returns_aligned) /
                    (np.std(benchmark_returns_aligned) + 1e-10)
                )
            }

        # Determine risk level
        risk_level, risk_description = self._assess_risk_level(
            volatility["annualized"],
            sharpe,
            max_dd
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            volatility["annualized"],
            sharpe,
            max_dd,
            beta
        )

        return {
            "symbol": symbol,
            "period_days": period_days,
            "current_price": float(asset_prices[-1]),
            "metrics": {
                "volatility": volatility,
                "beta": beta,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "value_at_risk": {
                    "var_95": var_95,
                    "var_99": var_99,
                    "expected_shortfall": self.calculate_expected_shortfall(
                        asset_returns,
                        0.95
                    )
                },
                "max_drawdown": max_dd,
                "risk_level": risk_level,
                "risk_description": risk_description
            },
            "benchmark_symbol": settings.BENCHMARK_SYMBOL if include_benchmark else None,
            "benchmark_comparison": benchmark_comparison,
            "recommendations": recommendations
        }

    def calculate_volatility(self, returns: np.ndarray) -> dict[str, float]:
        """
        Calculate volatility metrics.

        Returns daily, weekly, and annualized volatility.
        """
        daily_vol = float(np.std(returns))
        weekly_vol = daily_vol * np.sqrt(5)
        annual_vol = daily_vol * np.sqrt(252)

        return {
            "daily": daily_vol,
            "weekly": weekly_vol,
            "annualized": annual_vol
        }

    def calculate_beta(
        self,
        asset_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate beta relative to benchmark.

        Beta = Cov(asset, benchmark) / Var(benchmark)
        """
        if len(asset_returns) != len(benchmark_returns):
            min_len = min(len(asset_returns), len(benchmark_returns))
            asset_returns = asset_returns[-min_len:]
            benchmark_returns = benchmark_returns[-min_len:]

        covariance = np.cov(asset_returns, benchmark_returns)[0, 1]
        variance = np.var(benchmark_returns)

        if variance == 0:
            return 1.0

        return float(covariance / variance)

    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe = (mean_return - risk_free) / std_dev
        """
        settings = get_settings()

        # Convert annual risk-free rate to daily
        daily_rf = settings.RISK_FREE_RATE / 252

        mean_return = np.mean(returns)
        std_dev = np.std(returns)

        if std_dev == 0:
            return 0.0

        # Annualize
        sharpe = (mean_return - daily_rf) / std_dev * np.sqrt(252)

        return float(sharpe)

    def calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sortino ratio (penalizes only downside volatility).
        """
        settings = get_settings()
        daily_rf = settings.RISK_FREE_RATE / 252

        mean_return = np.mean(returns)

        # Downside returns only
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return float("inf") if mean_return > daily_rf else 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        sortino = (mean_return - daily_rf) / downside_std * np.sqrt(252)

        return float(sortino)

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk using historical method.
        """
        percentile = (1 - confidence) * 100
        var = float(np.percentile(returns, percentile))
        return var

    def calculate_expected_shortfall(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR).

        Average of losses exceeding VaR.
        """
        var = self.calculate_var(returns, confidence)
        tail_losses = returns[returns <= var]

        if len(tail_losses) == 0:
            return var

        return float(np.mean(tail_losses))

    def calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Max percentage drop from peak to trough.
        """
        peak = prices[0]
        max_dd = 0

        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if drawdown > max_dd:
                max_dd = drawdown

        return float(-max_dd)  # Return as negative

    def _assess_risk_level(
        self,
        annual_volatility: float,
        sharpe: float,
        max_drawdown: float
    ) -> tuple[str, str]:
        """
        Assess overall risk level based on metrics.

        Returns:
            Tuple of (risk_level, description).
        """
        # Score based on volatility
        vol_score = 0
        if annual_volatility < 0.15:
            vol_score = 1
        elif annual_volatility < 0.25:
            vol_score = 2
        elif annual_volatility < 0.40:
            vol_score = 3
        else:
            vol_score = 4

        # Adjust based on Sharpe
        if sharpe > 1.5:
            vol_score = max(1, vol_score - 1)
        elif sharpe < 0.5:
            vol_score = min(4, vol_score + 1)

        # Adjust based on drawdown
        if max_drawdown < -0.20:
            vol_score = min(4, vol_score + 1)

        levels = {
            1: ("low", "Low risk with stable returns"),
            2: ("moderate", "Moderate risk with reasonable volatility"),
            3: ("high", "High risk with significant volatility"),
            4: ("very_high", "Very high risk - suitable for risk-tolerant investors")
        }

        return levels.get(vol_score, ("unknown", "Unable to assess risk"))

    def _generate_recommendations(
        self,
        annual_volatility: float,
        sharpe: float,
        max_drawdown: float,
        beta: Optional[float]
    ) -> list[str]:
        """Generate risk-based recommendations."""
        recommendations = []

        # Volatility recommendations
        if annual_volatility > 0.30:
            recommendations.append(
                "Consider smaller position sizes due to high volatility"
            )
        elif annual_volatility < 0.15:
            recommendations.append(
                "Low volatility suggests potential for larger positions"
            )

        # Sharpe recommendations
        if sharpe < 0.5:
            recommendations.append(
                "Risk-adjusted returns are low - consider alternatives"
            )
        elif sharpe > 2.0:
            recommendations.append(
                "Excellent risk-adjusted returns"
            )

        # Drawdown recommendations
        if max_drawdown < -0.15:
            recommendations.append(
                "Historical drawdowns suggest using stop-loss orders"
            )

        # Beta recommendations
        if beta is not None:
            if beta > 1.5:
                recommendations.append(
                    "High beta - more volatile than market"
                )
            elif beta < 0.5:
                recommendations.append(
                    "Low beta - good for portfolio diversification"
                )

        if not recommendations:
            recommendations.append(
                "Risk profile appears balanced"
            )

        return recommendations

    async def close(self) -> None:
        """Cleanup resources."""
        await self._data_fetcher.close()
