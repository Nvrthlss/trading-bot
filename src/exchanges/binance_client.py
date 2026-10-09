"""
Binance API client for fetching OHLCV market data.
"""
import requests
import pandas as pd

class BinanceClient:
    """Client for the Binance public API (no auth needed for market data)."""
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session = requests.Session()

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        all_data = []
        remaining = limit
        end_time = None

        while remaining > 0:
            batch_size = min(remaining, 1000)

            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": batch_size,
            }
            if end_time is not None:
                params["endTime"] = end_time
        
            url = self.base_url + "/api/v3/klines"
            response = self.session.get(url, params=params)
            response.raise_for_status()
            batch = response.json()

            if not batch:
                break

            all_data = batch + all_data
            remaining -= len(batch)

            if len(batch) < batch_size:
                break

            end_time = batch[0][0] - 1
        
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]

        return df