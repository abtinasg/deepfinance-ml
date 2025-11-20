"""
Data fetching service with circuit breaker and caching.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

from app.config import get_settings
from app.core.cache import get_cache_service
from app.core.logging import get_logger

logger = get_logger(__name__)


class CircuitBreaker:
    """Circuit breaker for external API calls."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: int = 60
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failure_count = 0
        self._last_failure: Optional[datetime] = None
        self._state = "closed"  # closed, open, half-open

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if self._state == "open":
            # Check if we should try again
            if self._last_failure:
                elapsed = (datetime.utcnow() - self._last_failure).total_seconds()
                if elapsed >= self.reset_timeout:
                    self._state = "half-open"
                    return False
            return True
        return False

    def record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        self._state = "closed"

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure = datetime.utcnow()

        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning(
                f"Circuit breaker {self.name} opened",
                extra={"failures": self._failure_count}
            )


class DataFetcher:
    """
    Service for fetching market data from multiple sources.

    Uses circuit breaker pattern and caching for reliability.
    """

    def __init__(self):
        self._yahoo_breaker = CircuitBreaker("yahoo_finance")
        self._finnhub_breaker = CircuitBreaker("finnhub")
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            settings = get_settings()
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.YAHOO_TIMEOUT),
                limits=httpx.Limits(
                    max_keepalive_connections=50,
                    keepalive_expiry=30
                )
            )
        return self._http_client

    async def fetch_historical_data(
        self,
        symbol: str,
        period_days: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data with caching.

        Args:
            symbol: Stock ticker symbol.
            period_days: Number of days of history.

        Returns:
            DataFrame with OHLCV data or None if failed.
        """
        # Check cache first
        cache = await get_cache_service()
        cached = await cache.get_market_data(symbol, period_days)
        if cached:
            logger.debug(f"Cache hit for {symbol}")
            return pd.DataFrame(cached)

        # Try Yahoo Finance first
        df = await self._fetch_from_yahoo(symbol, period_days)

        if df is None:
            # Fallback to Finnhub
            df = await self._fetch_from_finnhub(symbol, period_days)

        if df is not None and not df.empty:
            # Cache the result
            await cache.set_market_data(
                symbol,
                period_days,
                df.to_dict(orient="list")
            )
            return df

        logger.error(f"Failed to fetch data for {symbol}")
        return None

    async def _fetch_from_yahoo(
        self,
        symbol: str,
        period_days: int
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        if self._yahoo_breaker.is_open:
            logger.warning("Yahoo Finance circuit breaker is open")
            return None

        try:
            # Use thread pool for blocking yfinance call
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None,
                self._fetch_yahoo_sync,
                symbol,
                period_days
            )

            if df is not None and not df.empty:
                self._yahoo_breaker.record_success()
                logger.debug(f"Fetched {len(df)} rows from Yahoo for {symbol}")
                return df

            self._yahoo_breaker.record_failure()
            return None

        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}", extra={"symbol": symbol})
            self._yahoo_breaker.record_failure()
            return None

    def _fetch_yahoo_sync(
        self,
        symbol: str,
        period_days: int
    ) -> Optional[pd.DataFrame]:
        """Synchronous Yahoo Finance fetch."""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days + 10)

            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                return None

            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            df = df[["open", "high", "low", "close", "volume"]]
            df = df.reset_index(drop=True)

            return df.tail(period_days)

        except Exception as e:
            logger.error(f"Yahoo sync error: {e}")
            return None

    async def _fetch_from_finnhub(
        self,
        symbol: str,
        period_days: int
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Finnhub API."""
        settings = get_settings()

        if not settings.FINNHUB_API_KEY:
            logger.warning("Finnhub API key not configured")
            return None

        if self._finnhub_breaker.is_open:
            logger.warning("Finnhub circuit breaker is open")
            return None

        try:
            client = await self._get_client()

            end_ts = int(datetime.now().timestamp())
            start_ts = int((datetime.now() - timedelta(days=period_days)).timestamp())

            url = (
                f"https://finnhub.io/api/v1/stock/candle"
                f"?symbol={symbol}&resolution=D"
                f"&from={start_ts}&to={end_ts}"
                f"&token={settings.FINNHUB_API_KEY}"
            )

            response = await client.get(
                url,
                timeout=settings.FINNHUB_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()

            if data.get("s") != "ok":
                self._finnhub_breaker.record_failure()
                return None

            df = pd.DataFrame({
                "open": data["o"],
                "high": data["h"],
                "low": data["l"],
                "close": data["c"],
                "volume": data["v"]
            })

            self._finnhub_breaker.record_success()
            logger.debug(f"Fetched {len(df)} rows from Finnhub for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Finnhub error: {e}", extra={"symbol": symbol})
            self._finnhub_breaker.record_failure()
            return None

    async def fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price for a symbol."""
        df = await self.fetch_historical_data(symbol, period_days=5)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None

    async def fetch_benchmark_data(
        self,
        period_days: int = 60
    ) -> Optional[pd.DataFrame]:
        """Fetch S&P 500 benchmark data."""
        settings = get_settings()
        return await self.fetch_historical_data(
            settings.BENCHMARK_SYMBOL,
            period_days
        )

    def calculate_technical_indicators(
        self,
        df: pd.DataFrame
    ) -> dict:
        """
        Calculate technical indicators for analysis.

        Returns:
            Dictionary with RSI, MACD, Bollinger Bands, ATR, SMAs.
        """
        if len(df) < 20:
            return {}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI (14-period)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        # Bollinger Bands
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)

        # ATR (14-period)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        # SMAs
        sma50 = close.rolling(window=50).mean() if len(close) >= 50 else close.rolling(window=len(close)).mean()

        return {
            "rsi": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0,
            "macd": {
                "macd": float(macd_line.iloc[-1]) if not np.isnan(macd_line.iloc[-1]) else 0.0,
                "signal": float(signal_line.iloc[-1]) if not np.isnan(signal_line.iloc[-1]) else 0.0,
                "histogram": float(histogram.iloc[-1]) if not np.isnan(histogram.iloc[-1]) else 0.0
            },
            "bollinger_bands": {
                "upper": float(upper_band.iloc[-1]) if not np.isnan(upper_band.iloc[-1]) else float(close.iloc[-1]),
                "middle": float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else float(close.iloc[-1]),
                "lower": float(lower_band.iloc[-1]) if not np.isnan(lower_band.iloc[-1]) else float(close.iloc[-1])
            },
            "atr": float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0,
            "sma_20": float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else float(close.iloc[-1]),
            "sma_50": float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else float(close.iloc[-1])
        }

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
