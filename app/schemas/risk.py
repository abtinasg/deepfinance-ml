"""
Risk analysis schemas.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RiskRequest(BaseModel):
    """Request schema for risk analysis."""

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol"
    )
    period_days: int = Field(
        default=30,
        ge=10,
        le=365,
        description="Analysis period in days"
    )
    include_benchmark: bool = Field(
        default=True,
        description="Include S&P 500 benchmark comparison"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        v = v.upper().strip()
        if not v.replace("^", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError("Invalid symbol format")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "period_days": 30,
                "include_benchmark": True
            }
        }


class VolatilityMetrics(BaseModel):
    """Volatility breakdown."""

    daily: float = Field(..., description="Daily volatility")
    weekly: float = Field(..., description="Weekly volatility")
    annualized: float = Field(..., description="Annualized volatility")


class ValueAtRisk(BaseModel):
    """Value at Risk metrics."""

    var_95: float = Field(..., description="95% VaR")
    var_99: float = Field(..., description="99% VaR")
    expected_shortfall: Optional[float] = Field(
        None,
        description="Expected shortfall (CVaR)"
    )


class RiskMetrics(BaseModel):
    """Complete risk metrics."""

    volatility: VolatilityMetrics
    beta: Optional[float] = Field(None, description="Beta vs benchmark")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    value_at_risk: ValueAtRisk
    max_drawdown: float = Field(..., description="Maximum drawdown")
    risk_level: str = Field(..., description="Risk assessment level")
    risk_description: str = Field(..., description="Risk assessment description")


class RiskResponse(BaseModel):
    """Response schema for risk analysis."""

    symbol: str
    period_days: int
    current_price: float
    metrics: RiskMetrics
    benchmark_symbol: Optional[str] = None
    benchmark_comparison: Optional[dict[str, float]] = None
    recommendations: list[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "period_days": 30,
                "current_price": 185.50,
                "metrics": {
                    "volatility": {
                        "daily": 0.015,
                        "weekly": 0.033,
                        "annualized": 0.238
                    },
                    "beta": 1.15,
                    "sharpe_ratio": 1.85,
                    "sortino_ratio": 2.10,
                    "value_at_risk": {
                        "var_95": -0.025,
                        "var_99": -0.038,
                        "expected_shortfall": -0.042
                    },
                    "max_drawdown": -0.085,
                    "risk_level": "moderate",
                    "risk_description": "Moderate risk with reasonable volatility"
                },
                "benchmark_symbol": "^GSPC",
                "benchmark_comparison": {
                    "correlation": 0.85,
                    "relative_volatility": 1.2
                },
                "recommendations": [
                    "Consider position sizing",
                    "Monitor key support levels"
                ],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
