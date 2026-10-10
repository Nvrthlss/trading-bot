"""
SMA Crossover strategy.
Buys when short SMA crosses above long SMA (Golden Cross)
Sells when short SMA crosses below long SMA (Death Cross)
"""

import pandas as pd

from src.indicators.trend import sma

BUY = 1
SELL = -1
HOLD = 0

class SMACrossoverStrategy:
    """
    Simple moving average crossover strategy
    """
    def __init__(self, short_window: int = 50, long_window: int = 200):
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[f"sma_{self.short_window}"] = sma(df, self.short_window)
        df[f"sma_{self.long_window}"] = sma(df, self.long_window)
        return df

    def signal(self, df: pd.DataFrame, i: int) -> int:
        if i < 1:
            return HOLD     # Not enough history for a crossover check

        short_col = f"sma_{self.short_window}"
        long_col = f"sma_{self.long_window}"

        short_now = df[short_col].iloc[i]
        long_now = df[long_col].iloc[i]
        short_prev = df[short_col].iloc[i - 1]
        long_prev = df[long_col].iloc[i - 1]

        # Any NaN means we don't have enough data yet
        if pd.isna(short_now) or pd.isna(long_now) or pd.isna(short_prev) or pd.isna(long_prev):
            return HOLD

        # Golden Cross: short crosses from below to above long
        if short_prev <= long_prev and short_now > long_now:
            return BUY

        # Death Cross: short crosses from above to below long
        if short_prev >= long_prev and short_now < long_now:
            return SELL

        return HOLD