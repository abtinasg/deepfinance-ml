"""
Data fetching service using Finnhub as the sole provider.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import httpx
import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches market data exclusively from Finnhub."""

    def __init__(self):
        self.finnhub_timeout = settings.FINNHUB_TIMEOUT
        self.finnhub_api_key = settings.FINNHUB_API_KEY

    async def fetch_historical_data(
        self,
        symbol: str,
        period_days: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol
            period_days: Number of days of historical data

        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        data = await self._fetch_from_finnhub(symbol, period_days)

        if data is None or data.empty:
            logger.error(f"Failed to fetch data for {symbol} using Finnhub")
            return None

        return data

    async def _fetch_from_finnhub(
        self,
        symbol: str,
        period_days: int
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Finnhub."""
        if not self.finnhub_api_key:
            logger.warning("Finnhub API key not configured")
            return None

        try:
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=period_days + 30)).timestamp())

            url = "https://finnhub.io/api/v1/stock/candle"
            params = {
                "symbol": symbol,
                "resolution": "D",
                "from": start_time,
                "to": end_time,
                "token": self.finnhub_api_key
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    params=params,
                    timeout=self.finnhub_timeout
                )

                if response.status_code != 200:
                    logger.error(f"Finnhub returned status {response.status_code}")
                    return None

                data = response.json()

                if data.get("s") != "ok":
                    logger.error(f"Finnhub returned error: {data.get('s')}")
                    return None

                required_fields = {"o", "h", "l", "c", "v", "t"}
                if not required_fields.issubset(data):
                    logger.error(
                        f"Finnhub response missing fields for {symbol}",
                        extra={"missing": required_fields - set(data.keys())}
                    )
                    return None

                if not data.get("t"):
                    logger.error(f"Finnhub returned no timestamps for {symbol}")
                    return None

                df = pd.DataFrame({
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                })

                if df.empty:
                    logger.error(f"Finnhub returned empty data for {symbol}")
                    return None

                df.index = pd.to_datetime(data['t'], unit='s')

                return df.tail(period_days)

        except Exception as e:
            logger.error(f"Finnhub error for {symbol}: {str(e)}")
            return None

    async def fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price for a symbol using Finnhub."""
        if not self.finnhub_api_key:
            logger.warning("Finnhub API key not configured")
            return None

        try:
            url = (
                f"https://finnhub.io/api/v1/quote"
                f"?symbol={symbol}&token={self.finnhub_api_key}"
            )

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    timeout=self.finnhub_timeout
                )

                if response.status_code != 200:
                    logger.error(f"Finnhub quote status {response.status_code}")
                    return None

                data = response.json()
                current_price = data.get('c')

                if current_price is None:
                    logger.error(
                        f"Failed to fetch current price for {symbol} using Finnhub"
                    )
                    return None

                return float(current_price)

        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {str(e)}")
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
