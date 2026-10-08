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
        url = self.base_url + "/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
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