"""
Label generation for supervised learning
Based on future price movement
"""

import pandas as pd
import numpy as np

LABEL_BUY = 1
LABEL_HOLD = 0
LABEL_SELL = -1

def add_labels(
    df: pd.DataFrame,
    forward_periods: int = 8,
    threshold_pct: float = 1.0,
) -> pd.DataFrame:
    df = df.copy()

    future_price = df["close"].shift(-forward_periods)
    df["future_return"] = (future_price - df["close"]) / df["close"] * 100

    def classify(ret):
        if pd.isna(ret):
            return np.nan
        if ret >= threshold_pct:
            return LABEL_BUY
        if ret <= -threshold_pct:
            return LABEL_SELL
        return LABEL_HOLD

    df["label"] = df["future_return"].apply(classify)
    return df