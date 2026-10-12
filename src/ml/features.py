"""
Feature engineering for ML models
Transforms raw OHLCV data into normalized, model-friendly features
"""

import pandas as pd
import numpy as np

from src.indicators.trend import sma, ema
from src.indicators.momentum import rsi


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ML-ready features to a DataFrame
    Features are chosen to be scale-independent (work across any coin price range)

    Returns the DataFrame with new columns added (does not modify input)
    """
    df = df.copy()

    # --- Price-based features (relative to moving averages) ---
    sma_20 = sma(df, 20)
    sma_50 = sma(df, 50)
    ema_20 = ema(df, 20)

    df["price_vs_sma20"] = df["close"] / sma_20 - 1
    df["price_vs_sma50"] = df["close"] / sma_50 - 1
    df["price_vs_ema20"] = df["close"] / ema_20 - 1
    df["sma20_vs_sma50"] = sma_20 / sma_50 - 1

    # --- Momentum ---
    df["rsi_14"] = rsi(df, 14)
    df["rsi_7"] = rsi(df, 7)

    # --- Returns at different horizons ---
    df["return_1"] = df["close"].pct_change(1) * 100
    df["return_3"] = df["close"].pct_change(3) * 100
    df["return_6"] = df["close"].pct_change(6) * 100
    df["return_12"] = df["close"].pct_change(12) * 100

    # --- Volatility (rolling standard deviation of returns) ---
    df["volatility_12"] = df["return_1"].rolling(12).std()
    df["volatility_24"] = df["return_1"].rolling(24).std()

    # --- Volume features ---
    volume_sma_20 = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / volume_sma_20

    # --- Candle shape (relative to range) ---
    candle_range = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    # Avoid division by zero
    safe_range = candle_range.replace(0, np.nan)
    df["body_pct"] = body / safe_range
    df["upper_wick_pct"] = upper_wick / safe_range
    df["lower_wick_pct"] = lower_wick / safe_range

    # --- Candle direction (1 = green, -1 = red, 0 = doji) ---
    df["candle_direction"] = np.sign(df["close"] - df["open"])

    return df


FEATURE_COLUMNS = [
    "price_vs_sma20", "price_vs_sma50", "price_vs_ema20", "sma20_vs_sma50",
    "rsi_14", "rsi_7",
    "return_1", "return_3", "return_6", "return_12",
    "volatility_12", "volatility_24",
    "volume_ratio",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "candle_direction",
]