"""SMA trend rule — simple-moving-average crossover, vol-normalized."""

from __future__ import annotations

import pandas as pd

from .base import Strategy
from .registry import register
from ..core.forecasts import normalize_by_vol
from ..core.volatility import ew_vol


@register
class SMACrossover(Strategy):
    key = "sma"
    label = "SMA Trend"
    params = {
        "fast": (20, 2, 150, 1),
        "slow": (100, 5, 300, 1),
    }
    spectrum_param = "fast"

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        fast = int(self.values["fast"])
        slow = int(self.values["slow"])
        diff = close.rolling(fast).mean() - close.rolling(slow).mean()
        vol = ew_vol(close, annualize=False) * close
        return normalize_by_vol(diff, vol)
