"""
Data fetching service using Finnhub as the sole provider.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import finnhub
import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches market data exclusively from Finnhub."""

    def __init__(self):
        self.finnhub_timeout = settings.FINNHUB_TIMEOUT
        self.finnhub_api_key = settings.FINNHUB_API_KEY
        self._client: Optional[finnhub.Client] = None

    def _get_client(self) -> Optional[finnhub.Client]:
        if not self.finnhub_api_key:
            logger.error("Finnhub API key not configured")
            return None

        if self._client is None:
            self._client = finnhub.Client(api_key=self.finnhub_api_key)

        return self._client

    async def fetch_price_history(
        self, symbol: str, period_days: int = 60
    ) -> Dict[str, Any]:
        """Fetch OHLCV candles from Finnhub.

        Returns a dictionary with success flag and data/ reason.
        """

        client = self._get_client()
        if client is None:
            return {"success": False, "reason": "finnhub_failed"}

        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=period_days)

            candles = await asyncio.to_thread(
                client.stock_candles,
                symbol.upper(),
                "D",
                int(start_time.timestamp()),
                int(end_time.timestamp()),
            )

            if not candles or candles.get("s") != "ok":
                logger.error("Finnhub returned invalid status for %s", symbol)
                return {"success": False, "reason": "finnhub_failed"}

            timestamps = candles.get("t") or []
            if not timestamps:
                logger.error("Finnhub returned no timestamps for %s", symbol)
                return {"success": False, "reason": "finnhub_failed"}

            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(timestamps, unit="s"),
                    "open": candles.get("o", []),
                    "high": candles.get("h", []),
                    "low": candles.get("l", []),
                    "close": candles.get("c", []),
                    "volume": candles.get("v", []),
                }
            )

            if df.empty:
                logger.error("Finnhub returned empty candles for %s", symbol)
                return {"success": False, "reason": "finnhub_failed"}

            logger.debug({"datasource": "finnhub", "symbol": symbol, "status": "success"})
            return {"success": True, "data": df}

        except Exception as exc:
            logger.error("Finnhub error for %s: %s", symbol, exc)
            return {"success": False, "reason": "finnhub_failed"}

    async def fetch_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch symbol profile details from Finnhub."""

        client = self._get_client()
        if client is None:
            return {"success": False, "reason": "finnhub_failed"}

        try:
            profile = await asyncio.to_thread(
                client.company_profile2, {"symbol": symbol.upper()}
            )

            if not profile:
                logger.error("Finnhub returned no profile for %s", symbol)
                return {"success": False, "reason": "finnhub_failed"}

            logger.debug({"datasource": "finnhub", "symbol": symbol, "status": "success"})
            return {"success": True, "data": profile}

        except Exception as exc:
            logger.error("Finnhub profile error for %s: %s", symbol, exc)
            return {"success": False, "reason": "finnhub_failed"}

    async def fetch_historical_data(
        self,
        symbol: str,
        period_days: int = 60,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data for a symbol."""

        result = await self.fetch_price_history(symbol, period_days)
        data = result.get("data") if result.get("success") else None

        if data is None or data.empty:
            logger.error("Failed to fetch data for %s using Finnhub", symbol)
            return None

        return data

    async def fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price for a symbol using Finnhub."""
        client = self._get_client()
        if client is None:
            return None

        try:
            quote = await asyncio.to_thread(client.quote, symbol.upper())
            current_price = quote.get("c") if isinstance(quote, dict) else None

            if current_price is None:
                logger.error(
                    "Failed to fetch current price for %s using Finnhub", symbol
                )
                return None

            logger.debug({"datasource": "finnhub", "symbol": symbol, "status": "success"})
            return float(current_price)

        except Exception as exc:
            logger.error("Error fetching current price for %s: %s", symbol, exc)
            return None

    async def fetch_benchmark_data(
        self,
        period_days: int = 60
    ) -> Optional[pd.DataFrame]:
        """Fetch benchmark (S&P 500) data for risk calculations"""
        return await self.fetch_historical_data(
            settings.BENCHMARK_SYMBOL,
            period_days
        )

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators from OHLCV data"""
        if df is None or df.empty:
            return {}

        close = df['close'].values

        indicators = {}

        # RSI (14-day)
        indicators['rsi'] = self._calculate_rsi(close, 14)

        # MACD
        macd, signal, hist = self._calculate_macd(close)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_histogram'] = hist

        # Moving Averages
        indicators['sma_20'] = np.mean(close[-20:]) if len(close) >= 20 else np.mean(close)
        indicators['sma_50'] = np.mean(close[-50:]) if len(close) >= 50 else np.mean(close)

        # Bollinger Bands
        sma_20 = indicators['sma_20']
        std_20 = np.std(close[-20:]) if len(close) >= 20 else np.std(close)
        indicators['bollinger_upper'] = sma_20 + (2 * std_20)
        indicators['bollinger_lower'] = sma_20 - (2 * std_20)

        # ATR (Average True Range)
        indicators['atr'] = self._calculate_atr(df)

        return indicators

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0

        # EMA calculation
        def ema(data, period):
            alpha = 2 / (period + 1)
            ema_values = [data[0]]
            for price in data[1:]:
                ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
            return np.array(ema_values)

        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line

        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period:
            return 0.0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        atr = np.mean(tr[-period:])
        return float(atr)
