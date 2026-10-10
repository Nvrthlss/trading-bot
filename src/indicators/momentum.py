import pandas as pd

def rsi(df: pd.DataFrame, window: int = 14, column: str = "close") -> pd.Series:
    """
    Relative Strength Index (RSI)
    """

    delta = df[column].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values