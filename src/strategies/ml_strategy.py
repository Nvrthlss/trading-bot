"""
ML-based trading strategy
Trains a Random Forest on historical data and uses its predictions as signals
Only emits a BUY/SELL if the model's confidence is high enough
"""

import pandas as pd
import numpy as np

from src.strategies.sma_crossover import BUY, SELL, HOLD
from src.ml.features import add_features, FEATURE_COLUMNS
from src.ml.labels import add_labels
from src.ml.model import TradingModel


class MLStrategy:

    def __init__(
        self,
        forward_periods: int = 8,
        threshold_pct: float = 1.0,
        train_ratio: float = 0.7,
        min_confidence: float = 0.5,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_leaf: int = 20,
    ):
        self.forward_periods = forward_periods
        self.threshold_pct = threshold_pct
        self.train_ratio = train_ratio
        self.min_confidence = min_confidence

        self.model = TradingModel(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )

        self._first_predict_idx: int = 0
        self._signals_cache: np.ndarray | None = None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_features(df)

        labeled_df = add_labels(
            df, forward_periods=self.forward_periods,
            threshold_pct=self.threshold_pct,
        )

        full = labeled_df[FEATURE_COLUMNS + ["label"]].dropna()

        split_idx = int(len(full) * self.train_ratio)
        train_df = full.iloc[:split_idx]

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["label"].astype(int)

        self.model.train(X_train, y_train)

        last_train_timestamp = train_df.index[-1]
        self._first_predict_idx = df.index.get_loc(last_train_timestamp) + 1

        self._signals_cache = self._precompute_signals(df)

        return df

    def prepare_live(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model.is_trained:
            raise RuntimeError("prepare_live called before model was trained. Call prepare() first.")
        df = add_features(df)
        self._signals_cache = self._precompute_signals(df)
        return df

    def _precompute_signals(self, df: pd.DataFrame) -> np.ndarray:
        signals = np.full(len(df), HOLD, dtype=int)

        features_df = df[FEATURE_COLUMNS]
        valid_mask = ~features_df.isna().any(axis=1)
        valid_df = features_df[valid_mask]

        if len(valid_df) == 0:
            return signals

        probs = self.model.predict_proba(valid_df)

        max_probs = probs.max(axis=1).values
        max_classes = probs.idxmax(axis=1).values

        signal_values = np.full(len(valid_df), HOLD, dtype=int)
        confident = max_probs >= self.min_confidence
        for cls, sig in [("BUY", BUY), ("SELL", SELL)]:
            mask = confident & (max_classes == cls)
            signal_values[mask] = sig

        valid_positions = np.where(valid_mask.values)[0]
        signals[valid_positions] = signal_values

        return signals

    def signal(self, df: pd.DataFrame, i: int) -> int:
        if i < self._first_predict_idx:
            return HOLD
        if self._signals_cache is None:
            return HOLD
        return int(self._signals_cache[i])