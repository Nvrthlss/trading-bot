"""
Build a model-ready dataset from OHLCV data
Combines feature engineering + labeling + cleanup
"""

import pandas as pd

from src.ml.features import add_features, FEATURE_COLUMNS
from src.ml.labels import add_labels

def build_dataset(
    df: pd.DataFrame,
    forward_periods: int = 8,
    threshold_pct: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    df = add_features(df)
    df = add_labels(df, forward_periods=forward_periods, threshold_pct=threshold_pct)

    cols = FEATURE_COLUMNS + ["label"]
    df = df[cols].dropna()

    X = df[FEATURE_COLUMNS]
    y = df["label"].astype(int)

    return X, y

def label_distribution(y: pd.Series) -> dict:
    counts = y.value_counts().to_dict()
    total = len(y)
    return {
        label: {"count": count, "pct": count / total * 100} for label, count in counts.items()
    }