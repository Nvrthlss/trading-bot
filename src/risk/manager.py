"""
Risk Management
Handles stop-loss and take-profit for open positions
"""

from dataclasses import dataclass
from typing import Optional

# Exit reason constants
EXIT_STOP_LOSS = "STOP_LOSS"
EXIT_TAKE_PROFIT = "TAKE_PROFIT"
EXIT_SIGNAL = "SIGNAL"

@dataclass
class ExitDecision:
    should_exit: bool
    reason: Optional[str] = None

class RiskManager:
    def __init__(self, stop_loss_pct: float = 2.0, take_profit_pct: float = 4.0):
        if stop_loss_pct <= 0 or take_profit_pct <= 0:
            raise ValueError("stop_loss_pct and take_profit_pct must be positive")

        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.entry_price: Optional[float] = None
        self.stop_loss_price: Optional[float] = None
        self.take_profit_price: Optional[float] = None

    def open_position(self, entry_price: float):
        self.entry_price = entry_price
        self.stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100)
        self.take_profit_price = entry_price * (1 + self.take_profit_pct / 100)

    def close_position(self):
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None

    def check_exit(self, current_price: float) -> ExitDecision:
        if self.entry_price is None:
            return ExitDecision(should_exit=False)

        if current_price <= self.stop_loss_price:
            return ExitDecision(should_exit=True, reason=EXIT_STOP_LOSS)

        if current_price >= self.take_profit_price:
            return ExitDecision(should_exit=True, reason=EXIT_TAKE_PROFIT)

        return ExitDecision(should_exit=False)

    def has_open_position(self) -> bool:
        return self.entry_price is not None