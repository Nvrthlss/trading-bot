"""
Advanced backtesting: Walk-Forward Optimization + Monte Carlo Simulation
Validates that a strategy is robust across different market regimes
"""

from dataclasses import dataclass, field
from typing import List, Callable
import pandas as pd
import numpy as np

from src.backtest.engine import BacktestEngine, BacktestResult

@dataclass
class WindowResult:
    window_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    return_pct: float
    buy_hold_pct: float
    num_trades: int
    win_rate: float

@dataclass
class WalkForwardResult:
    windows: List[WindowResult] = field(default_factory=list)

    @property
    def avg_return(self) -> float:
        if not self.windows:
            return 0.0
        return float(np.mean([w.return_pct for w in self.windows]))

    @property
    def consistency(self) -> float:
        if not self.windows:
            return 0.0
        profitable = sum(1 for w in self.windows if w.return_pct > 0)
        return (profitable / len(self.windows)) * 100

    @property
    def avg_win_rate(self) -> float:
        if not self.windows:
            return 0.0
        return float(np.mean([w.win_rate for w in self.windows]))

    @property
    def total_trades(self) -> int:
        return sum(w.num_trades for w in self.windows)

def walk_forward(
    df: pd.DataFrame,
    strategy_factory: Callable,
    engine: BacktestEngine,
    train_size: int = 1000,
    test_size: int = 250,
    step: int = 250,
) -> WalkForwardResult:
    result = WalkForwardResult()
    total_len = len(df)
    window_idx = 0
    start = 0

    while start + train_size + test_size <= total_len:
        train_end_idx = start + train_size
        test_end_idx = train_end_idx + test_size

        window_df = df.iloc[start:test_end_idx].copy()

        strategy = strategy_factory()

        intended_train_ratio = train_size / (train_size + test_size)
        if hasattr(strategy, "train_ratio"):
            strategy.train_ratio = intended_train_ratio

        backtest_result = engine.run(window_df, strategy)

        train_start_ts = window_df.index[0]
        train_end_ts = window_df.index[train_size - 1]
        test_start_ts = window_df.index[train_size]
        test_end_ts = window_df.index[-1]

        window_idx += 1
        print(f"  Window {window_idx}: "
              f"test {test_start_ts.date()} -> {test_end_ts.date()}  "
              f"return={backtest_result.total_return_pct:+.2f}%  "
              f"trades={backtest_result.num_trades}  "
              f"win_rate={backtest_result.win_rate:.0f}%")

        result.windows.append(WindowResult(
            window_idx=window_idx,
            train_start=train_start_ts,
            train_end=train_end_ts,
            test_start=test_start_ts,
            test_end=test_end_ts,
            return_pct=backtest_result.total_return_pct,
            buy_hold_pct=backtest_result.buy_hold_return_pct,
            num_trades=backtest_result.num_trades,
            win_rate=backtest_result.win_rate,
        ))

        start += step
    
    return result

@dataclass
class MonteCarloResult:
    n_simulations: int
    final_balances: List[float] = field(default_factory=list)
    max_drawdowns: List[float] = field(default_factory=list)

    @property
    def profit_probability(self) -> float:
        if not self.final_balances:
            return 0.0
        initial = self.final_balances[0] / (1 + 0)
        return 0.0

    @property
    def mean_final(self) -> float:
        if not self.final_balances:
            return 0.0
        return float(np.mean(self.final_balances))

    @property
    def median_final(self) -> float:
        if not self.final_balances:
            return 0.0
        return float(np.median(self.final_balances))

    @property
    def mean_max_drawdown(self) -> float:
        if not self.max_drawdowns:
            return 0.0
        return float(np.mean(self.max_drawdowns))


def monte_carlo(
    backtest_result: BacktestResult,
    n_simulations: int = 500,
    seed: int = 42,
) -> dict:
    if not backtest_result.trades:
        return {
            "n_simulations": 0,
            "final_balance": backtest_result.initial_balance,
            "total_return_pct": 0.0,
            "mean_max_drawdown": 0.0,
            "median_max_drawdown": 0.0,
            "worst_max_drawdown": 0.0,
            "best_max_drawdown": 0.0,
            "percentile_95_drawdown": 0.0,
            "pct_under_5pct_dd": 100.0,
            "pct_under_10pct_dd": 100.0,
        }

    rng = np.random.default_rng(seed)
    initial_balance = backtest_result.initial_balance
    pnls = np.array([t.pnl for t in backtest_result.trades])
    final_balance = initial_balance + pnls.sum()
    total_return_pct = (final_balance - initial_balance) / initial_balance * 100

    max_drawdowns = []

    for _ in range(n_simulations):
        shuffled = rng.permutation(pnls)

        equity = initial_balance + np.cumsum(shuffled)
        equity = np.concatenate([[initial_balance], equity])

        running_peak = np.maximum.accumulate(equity)
        drawdowns = (running_peak - equity) / running_peak * 100
        max_drawdowns.append(drawdowns.max())

    max_drawdowns = np.array(max_drawdowns)

    return {
        "n_simulations": n_simulations,
        "final_balance": float(final_balance),
        "total_return_pct": float(total_return_pct),
        "mean_max_drawdown": float(np.mean(max_drawdowns)),
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "worst_max_drawdown": float(np.max(max_drawdowns)),
        "best_max_drawdown": float(np.min(max_drawdowns)),
        "percentile_95_drawdown": float(np.percentile(max_drawdowns, 95)),
        "pct_under_5pct_dd": float((max_drawdowns < 5).sum() / len(max_drawdowns) * 100),
        "pct_under_10pct_dd": float((max_drawdowns < 10).sum() / len(max_drawdowns) * 100),
    }