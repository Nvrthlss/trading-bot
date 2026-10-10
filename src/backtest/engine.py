"""
Simple backtest engine
Simulates trading a strategy on historical OHLCV data
"""

from dataclasses import dataclass, field
from typing import List
import pandas as pd

from src.strategies.sma_crossover import BUY, SELL

@dataclass
class Trade:
    """A single completed round-trip trade (buy + sell)"""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    quantity: float
    fee_paid: float
    pnl: float          # Net profit/loss after fees (in USDT)
    return_pct: float   # Return percentage

@dataclass
class BacktestResult:
    """Summary of a backtest run"""
    initial_balance: float
    final_balance: float
    total_return_pct: float
    buy_hold_return_pct: float
    num_trades: float
    num_wins: float
    num_losses: float
    win_rate: float
    total_fees: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = None      #Balance over time, for plotting

class BacktestEngine:
    """
    Runs a strategy on historical data and reports performance
    """

    def __init__(self, initial_balance: float = 10_000.0, trade_size_pct: float = 5.0, fee_rate: float = 0.001):
        self.initial_balance = initial_balance
        self.trade_size_pct = trade_size_pct
        self.fee_rate = fee_rate

    def run(self, df: pd.DataFrame, strategy) -> BacktestResult:
        """
        Run a strategy on the given data
        """
        df = strategy.prepare(df)

        balance = self.initial_balance
        position_qty = 0.0
        entry_price = 0.0
        entry_time = None
        total_fees = 0.0
        trades: List[Trade] = []
        equity_curve = []

        for i in range(len(df)):
            price = df["close"].iloc[i]
            timestamp = df.index[i]

            sig = strategy.signal(df, i)

            if sig == BUY and position_qty == 0:
                trade_amount = balance * (self.trade_size_pct / 100)
                fee = trade_amount * self.fee_rate
                qty = (trade_amount - fee) / price

                balance -= trade_amount
                position_qty = qty
                entry_price = price
                entry_time = timestamp
                total_fees += fee

            elif sig == SELL and position_qty > 0:
                gross_proceeds = position_qty * price
                fee = gross_proceeds * self.fee_rate
                net_proceeds = gross_proceeds - fee

                balance += net_proceeds
                total_fees += fee

                entry_cost = position_qty * entry_price * (1 + self.fee_rate)
                pnl = net_proceeds - entry_cost
                return_pct = (pnl / entry_cost) * 100

                trades.append(Trade(
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=timestamp,
                    exit_price=price,
                    quantity=position_qty,
                    fee_paid=fee + (entry_cost - position_qty * entry_price),
                    pnl=pnl,
                    return_pct=return_pct,
                ))

                position_qty = 0.0
                entry_price = 0.0
                entry_time = None

            equity = balance + position_qty * price
            equity_curve.append(equity)

        if position_qty > 0:
            last_price = df["close"].iloc[-1]
            last_time = df.index[-1]
            gross_proceeds = position_qty * last_price
            fee = gross_proceeds * self.fee_rate
            net_proceeds = gross_proceeds - fee

            balance += net_proceeds
            total_fees += fee

            entry_cost = position_qty * entry_price * (1 + self.fee_rate)
            pnl = net_proceeds - entry_cost
            return_pct = (pnl / entry_cost) * 100

            trades.append(Trade(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=last_time,
                exit_price=last_price,
                quantity=position_qty,
                fee_paid=fee + (entry_cost - position_qty * entry_price),
                pnl=pnl,
                return_pct=return_pct,
            ))

            position_qty = 0.0

        final_balance = balance
        total_return_pct = (final_balance - self.initial_balance) / self.initial_balance * 100

        first_price = df["close"].iloc[0]
        last_price = df["close"].iloc[-1]
        buy_hold_return_pct = (last_price - first_price) / first_price * 100

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = (len(wins) / len(trades) * 100) if trades else 0

        return BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_return_pct=total_return_pct,
            buy_hold_return_pct=buy_hold_return_pct,
            num_trades=len(trades),
            num_wins=len(wins),
            num_losses=len(losses),
            win_rate=win_rate,
            total_fees=total_fees,
            trades=trades,
            equity_curve=pd.Series(equity_curve, index=df.index),
        )