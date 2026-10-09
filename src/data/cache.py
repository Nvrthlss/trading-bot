"""
Data cache for OHLCV market data.
Stores candles as CSV files on disk, re-downloads when stale.
"""

from pathlib import Path
import pandas as pd

from src.exchanges.binance_client import BinanceClient

class DataCache:
    """
    Caches OHLCV data as CSV files.
    Wraps a BinanceClient and only hits the AI when the cache is missing or stale.
    """

    def __init__(self, client: BinanceClient, cache_dir: str = "data"):
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_path(self, symbol: str, interval: str) -> Path:
        """Return the cache file path for a given symbol/interval combination."""
        return self.cache_dir / f"{symbol}_{interval}.csv"

    def _is_fresh(self, df: pd.DataFrame, interval: str) -> bool:
        if df.empty:
            return False

        interval_minutes = self._interval_to_minutes(interval)

        newest_candle = df.index[-1]
        age = pd.Timestamp.utcnow().tz_localize(None) - newest_candle
        max_age = pd.Timedelta(minutes=interval_minutes)

        return age < max_age

    def _interval_to_minutes(self, interval: str) -> int:
        unit = interval[-1]
        number = int(interval[:-1])

        if unit == "m":
            return number
        elif unit == "h":
            return number * 60
        elif unit == "d":
            return number * 60 * 24
        elif unit == "w":
            return number * 60 * 24 * 7
        else:
            raise ValueError(f"Unknown interval unit: {unit}")

    def _load_from_disk(self, symbol: str, interval: str) -> pd.DataFrame:
        path = self._cache_path(symbol, interval)
        df = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
        return df

    def _save_to_disk(self, df: pd.DataFrame, symbol: str, interval: str):
        path = self._cache_path(symbol, interval)
        df.to_csv(path)

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        Get OHLCV candles with caching.

        - If cache file is missing: download, save, return.
        - If cache is fresh and has enough candles: return from cache.
        - Otherwise: re-download fully, save, return.
        """
        path = self._cache_path(symbol, interval)

        if path.exists():
            cached_df = self._load_from_disk(symbol, interval)

            if self._is_fresh(cached_df, interval) and len(cached_df) >= limit:
                print(f"[CACHE HIT] {symbol} {interval} ({len(cached_df)} candles)")
                return cached_df.tail(limit)

            print(f"[CACHE STALE] {symbol} {interval} — re-downloading")
        else:
            print(f"[CACHE MISS] {symbol} {interval} — downloading")

        # Download fresh data
        df = self.client.get_klines(symbol, interval, limit)
        self._save_to_disk(df, symbol, interval)
        return df

    