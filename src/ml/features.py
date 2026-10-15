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

def _compute_tf_features(df: pd.DataFrame, tf_suffix: str, feature_set: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    rsi_14 = rsi(df, 14)
    out[f"rsi_14{tf_suffix}"] = rsi_14
    out[f"return_1{tf_suffix}"] = df["close"].pct_change(1) * 100

    if feature_set in ("medium", "full"):
        sma_20 = sma(df, 20)
        out[f"price_vs_sma20{tf_suffix}"] = df["close"] / sma_20 - 1

    if feature_set == "full":
        sma_50 = sma(df, 50)
        out[f"sma20_vs_sma50{tf_suffix}"] = sma_20 / sma_50 - 1

    return out

def add_multi_tf_features(df_trade: pd.DataFrame, multi_tf_dfs: dict) -> pd.DataFrame:
    df_trade = df_trade.copy()

    tf_config = {
        "5m": ("minimal", "_5m"),
        "15m": ("medium", "_15m"),
        "1h": ("full", "_1h"),
        "4h": ("full", "_4h"),
        "1d": ("full", "_1d")
    }

    df_trade = df_trade.sort_index()
    # Thanks pandas, for giving me some debugging to do.
    # From now on, I'll remember that you don't know how many microseconds is a millisecond
    df_trade["_trade_ts"] = pd.to_datetime(df_trade.index).astype("datetime64[ns]")

    for tf, df_tf in multi_tf_dfs.items():
        if tf not in tf_config:
            continue
        feature_set, suffix = tf_config[tf]

        tf_features = _compute_tf_features(df_tf, suffix, feature_set)
        tf_features = tf_features.sort_index()
        tf_features["_tf_ts"] = pd.to_datetime(tf_features.index).astype("datetime64[ns]")

        merged = pd.merge_asof(
            df_trade[["_trade_ts"]],
            tf_features,
            left_on="_trade_ts",
            right_on="_tf_ts",
            direction="backward",
        )
        merged.index = df_trade.index

        for col in tf_features.columns:
            if col == "_tf_ts":
                continue
            df_trade[col] = merged[col].values

    df_trade = df_trade.drop(columns=["_trade_ts"])
    return df_trade

def multi_tf_feature_columns(intervals: list) -> list:
    tf_config = {
        "5m": ("minimal", "_5m"),
        "15m": ("medium", "_15m"),
        "1h": ("full", "_1h"),
        "4h": ("full", "_4h"),
        "1d": ("full", "_1d"),
    }

    columns = []
    for tf in intervals:
        if tf not in tf_config:
            continue
        feature_set, suffix = tf_config[tf]
        columns.append(f"rsi_14{suffix}")
        columns.append(f"return_1{suffix}")
        if feature_set in ("medium", "full"):
            columns.append(f"price_vs_sma20{suffix}")
        if feature_set == "full":
            columns.append(f"sma20_vs_sma50{suffix}")
    return columns

FEATURE_COLUMNS = [
    "price_vs_sma20", "price_vs_sma50", "price_vs_ema20", "sma20_vs_sma50",
    "rsi_14", "rsi_7",
    "return_1", "return_3", "return_6", "return_12",
    "volatility_12", "volatility_24",
    "volume_ratio",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "candle_direction",
]