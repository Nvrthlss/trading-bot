"""
Multi-coin backtest engine
Runs multiple strategies on multiple symbols simultaneously,
sharing a common balance with portfolio-level exposure limits
Mirrors the PaperTrader logic for historical data
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable
import pandas as pd

from src.strategies.sma_crossover import BUY, SELL
from src.risk.manager import RiskManager, EXIT_SIGNAL


@dataclass
class MultiCoinTrade:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    exit_reason: str


@dataclass
class MultiCoinResult:
    initial_balance: float
    final_balance: float
    total_return_pct: float
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    total_fees: float
    exit_reasons: dict = field(default_factory=dict)
    trades: List[MultiCoinTrade] = field(default_factory=list)
    equity_curve: pd.Series = None
    per_symbol_stats: Dict[str, dict] = field(default_factory=dict)


class MultiCoinBacktestEngine:
    def __init__(
        self,
        initial_balance: float = 10_000.0,
        trade_size_pct: float = 5.0,
        fee_rate: float = 0.001,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        max_total_exposure_pct: float = 60.0,
    ):
        self.initial_balance = initial_balance
        self.trade_size_pct = trade_size_pct
        self.fee_rate = fee_rate
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_total_exposure_pct = max_total_exposure_pct

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        strategies: Dict[str, object],
    ) -> MultiCoinResult:
        prepared_data = {}
        for sym, df in data.items():
            prepared_data[sym] = strategies[sym].prepare(df)

        all_timestamps = sorted(set().union(*(df.index for df in prepared_data.values())))

        balance = self.initial_balance
        open_positions: Dict[str, dict] = {}
        risk_managers = {
            sym: RiskManager(
                stop_loss_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
            )
            for sym in data.keys()
        }
        total_fees = 0.0
        trades: List[MultiCoinTrade] = []
        exit_reasons: List[str] = []
        equity_curve = []

        for ts in all_timestamps:
            prices: Dict[str, float] = {}
            signals: Dict[str, int] = {}
            for sym, df in prepared_data.items():
                if ts not in df.index:
                    continue
                i = df.index.get_loc(ts)
                prices[sym] = df["close"].iloc[i]
                signals[sym] = strategies[sym].signal(df, i)

            for sym in list(open_positions.keys()):
                if sym not in prices:
                    continue
                price = prices[sym]
                sig = signals.get(sym, 0)
                rm = risk_managers[sym]

                exit_decision = rm.check_exit(price)
                exit_reason = None
                if exit_decision.should_exit:
                    exit_reason = exit_decision.reason
                elif sig == SELL:
                    exit_reason = EXIT_SIGNAL

                if exit_reason is not None:
                    pos = open_positions[sym]
                    gross = pos["qty"] * price
                    fee = gross * self.fee_rate
                    net = gross - fee
                    balance += net
                    total_fees += fee

                    entry_cost = pos["qty"] * pos["entry_price"] * (1 + self.fee_rate)
                    pnl = net - entry_cost
                    return_pct = (pnl / entry_cost) * 100

                    trades.append(MultiCoinTrade(
                        symbol=sym,
                        entry_time=pos["entry_time"],
                        entry_price=pos["entry_price"],
                        exit_time=ts,
                        exit_price=price,
                        quantity=pos["qty"],
                        pnl=pnl,
                        return_pct=return_pct,
                        exit_reason=exit_reason,
                    ))
                    exit_reasons.append(exit_reason)
                    del open_positions[sym]
                    rm.close_position()

            for sym in prepared_data.keys():
                if sym in open_positions:
                    continue
                if sym not in prices:
                    continue
                if signals.get(sym) != BUY:
                    continue

                price = prices[sym]
                proposed_amount = balance * (self.trade_size_pct / 100)
                if proposed_amount <= 0:
                    continue

                current_pos_value = sum(
                    pos["qty"] * prices.get(s, pos["entry_price"])
                    for s, pos in open_positions.items()
                )
                portfolio = balance + current_pos_value
                future_pos_value = current_pos_value + proposed_amount
                if portfolio <= 0:
                    continue
                future_exposure = (future_pos_value / portfolio) * 100
                if future_exposure > self.max_total_exposure_pct:
                    continue

                fee = proposed_amount * self.fee_rate
                qty = (proposed_amount - fee) / price
                balance -= proposed_amount
                total_fees += fee

                open_positions[sym] = {
                    "entry_price": price,
                    "entry_time": ts,
                    "qty": qty,
                }
                risk_managers[sym].open_position(entry_price=price)

            pos_value = sum(
                pos["qty"] * prices.get(s, pos["entry_price"])
                for s, pos in open_positions.items()
            )
            equity_curve.append(balance + pos_value)

        last_ts = all_timestamps[-1]
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            df = prepared_data[sym]
            if last_ts not in df.index:
                continue
            price = df["close"].iloc[-1]
            gross = pos["qty"] * price
            fee = gross * self.fee_rate
            net = gross - fee
            balance += net
            total_fees += fee

            entry_cost = pos["qty"] * pos["entry_price"] * (1 + self.fee_rate)
            pnl = net - entry_cost
            return_pct = (pnl / entry_cost) * 100

            trades.append(MultiCoinTrade(
                symbol=sym,
                entry_time=pos["entry_time"],
                entry_price=pos["entry_price"],
                exit_time=last_ts,
                exit_price=price,
                quantity=pos["qty"],
                pnl=pnl,
                return_pct=return_pct,
                exit_reason="END_OF_DATA",
            ))
            del open_positions[sym]

        final_balance = balance
        total_return_pct = (final_balance - self.initial_balance) / self.initial_balance * 100
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = (len(wins) / len(trades) * 100) if trades else 0.0

        exit_counts = {}
        for r in exit_reasons:
            exit_counts[r] = exit_counts.get(r, 0) + 1

        per_symbol: Dict[str, dict] = {}
        for sym in data.keys():
            sym_trades = [t for t in trades if t.symbol == sym]
            sym_wins = [t for t in sym_trades if t.pnl > 0]
            per_symbol[sym] = {
                "num_trades": len(sym_trades),
                "num_wins": len(sym_wins),
                "win_rate": (len(sym_wins) / len(sym_trades) * 100) if sym_trades else 0.0,
                "total_pnl": sum(t.pnl for t in sym_trades),
            }

        return MultiCoinResult(
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_return_pct=total_return_pct,
            num_trades=len(trades),
            num_wins=len(wins),
            num_losses=len(losses),
            win_rate=win_rate,
            total_fees=total_fees,
            exit_reasons=exit_counts,
            trades=trades,
            equity_curve=pd.Series(equity_curve, index=all_timestamps),
            per_symbol_stats=per_symbol,
        )