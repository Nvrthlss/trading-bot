"""
Alpaca stocks exchange client
Implements the ExchangeClient interface using the alpaca-py SDK
"""

from datetime import datetime, timedelta, timezone
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from src.exchanges.base import ExchangeClient
from src.config import ALPACA_API_KEY_ID, ALPACA_API_SECRET, ensure_alpaca_credentials


# Mapping: standard interval string -> (number, TimeFrameUnit)
INTERVAL_MAP = {
    "1m":  (1,  TimeFrameUnit.Minute),
    "5m":  (5,  TimeFrameUnit.Minute),
    "15m": (15, TimeFrameUnit.Minute),
    "30m": (30, TimeFrameUnit.Minute),
    "1h":  (1,  TimeFrameUnit.Hour),
    "4h":  (4,  TimeFrameUnit.Hour),
    "1d":  (1,  TimeFrameUnit.Day),
}


class AlpacaClient(ExchangeClient):
    def __init__(self):
        ensure_alpaca_credentials()
        self._client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY_ID,
            secret_key=ALPACA_API_SECRET,
        )

    @property
    def name(self) -> str:
        return "alpaca"

    def _to_timeframe(self, interval: str) -> TimeFrame:
        if interval not in INTERVAL_MAP:
            raise ValueError(
                f"Unsupported interval '{interval}' for Alpaca. "
                f"Supported: {list(INTERVAL_MAP.keys())}"
            )
        amount, unit = INTERVAL_MAP[interval]
        return TimeFrame(amount, unit)

    def _estimate_lookback(self, interval: str, limit: int) -> timedelta:
        interval_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 24 * 60,
        }[interval]

        # Trading minutes per calendar day
        if interval == "1d":
            minutes_per_day = 1 * 24 * 60  # daily bars count differently
            calendar_days = int(limit * 1.5) + 10
        else:
            # 6.5 trading hours per day × 5 days / 7 calendar days ≈ 2.3h/day avg
            # Divide bars needed by (bars-per-trading-day × 5/7) to get calendar days
            bars_per_trading_day = (6.5 * 60) / interval_minutes
            trading_days_needed = limit / bars_per_trading_day
            calendar_days = int(trading_days_needed * (7 / 5)) + 10  # weekend buffer

        return timedelta(days=calendar_days)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        timeframe = self._to_timeframe(interval)
        lookback = self._estimate_lookback(interval, limit)

        end = datetime.now(timezone.utc)
        start = end - lookback

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = self._client.get_stock_bars(request)

        # bars.df is a multi-index DataFrame (symbol, timestamp). Flatten to our format.
        df = bars.df

        if df.empty:
            raise RuntimeError(f"No data returned for {symbol} {interval}")

        # Drop the symbol level — we requested one symbol so it's redundant
        df = df.reset_index(level="symbol", drop=True)

        # Keep only the columns we need, in our standard order
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "timestamp"

        # Ensure timestamp is UTC, naive (matches our Binance format)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        # Keep only the most recent `limit` bars
        df = df.tail(limit)

        return df