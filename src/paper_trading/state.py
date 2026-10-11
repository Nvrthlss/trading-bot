"""
Paper trading state persistence
Saves and loads balance, open position, and trade log to/from JSON files
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List
from datetime import datetime

@dataclass
class OpenPosition:
    symbol: str
    entry_time: str
    entry_price: float
    quantity: float
    stop_loss_price: float
    take_profit_price: float

@dataclass
class PaperState:
    balance: float
    initial_balance: float
    total_fees_paid: float = 0.0
    cycle_count: int = 0
    started_at: Optional[str] = None
    open_positions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "total_fees_paid": self.total_fees_paid,
            "cycle_count": self.cycle_count,
            "started_at": self.started_at,
            "open_positions": {
                sym: asdict(pos) for sym, pos in self.open_positions.items()
            },
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PaperState":
        positions_data = d.pop("open_position", {})
        state = cls(**d)
        state.open_position = {
            sym: OpenPosition(**pos_dict) for sym, pos_dict in positions_data.items()
        }
        return state

class StateStore:
    def __init__(self, path: str = "data/paper_state.json"):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> PaperState:
        with open(self.path, "r") as f:
            data = json.load(f)
        return PaperState.from_dict(data)

    def save(self, state: PaperState):
        with open(self.path, "w") as f:
            json.dump(state.to_dict(), f, indent=2, default=str)

class TradeLog:
    def __init__(self, path: str = "data/paper_trades.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True)

    def append(self, trade: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(trade, default=str) + "\n")

    def read_all(self) -> List[dict]:
        if not self.path.exists():
            return []
        trades = []
        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))
        return trades