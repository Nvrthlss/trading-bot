import pandas as pd

def sma(df: pd.DataFrame, window: int, column: str = "close") -> pd.Series:
    """
    Simple Moving Average
    """
    return df[column].rolling(window=window).mean()

def ema(df: pd.DataFrame, window: int, column: str = "close") -> pd.Series:
    """
    Exponential Moving Average
    Reacts faster to price changes than SMA by weighting recent values more
    """
    return df[column].ewm(span=window, adjust=False).mean()