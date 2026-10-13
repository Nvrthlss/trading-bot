"""
ML model training and prediction
Wraps scikit-learn models with a trading-specific interface
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class TradingModel:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.feature_names: list = []
        self.is_trained: bool = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        probs = self.model.predict_proba(X)
        class_names = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        cols = [class_names[c] for c in self.model.classes_]
        return pd.DataFrame(probs, columns=cols, index=X.index)

    def feature_importance(self) -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        )
        return importances.sort_values(ascending=False)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        preds = self.predict(X)
        return {
            "accuracy": (preds == y).mean(),
            "confusion_matrix": confusion_matrix(y, preds, labels=[-1, 0, 1]),
            "report": classification_report(y, preds, labels=[-1, 0, 1], target_names=["SELL", "HOLD", "BUy"], zero_division=0)
        }

def time_based_split(
        X: pd.DataFrame,
        y: pd.Series,
        train_ratio: float = 0.8,
) -> tuple:
    split_idx = int(len(X) * train_ratio)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test