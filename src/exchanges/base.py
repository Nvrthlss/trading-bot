"""
Abstract exchange client interface
All exchange implementations (Binance, Alpaca, OANDA, ...) must conform to this
"""

from abc import ABC, abstractmethod
import pandas as pd


class ExchangeClient(ABC):
    @abstractmethod
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...