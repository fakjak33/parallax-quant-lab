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
        "fast": (20, 2, 1000, 1),
        "slow": (100, 5, 2000, 1),
        "smooth": (1, 1, 50, 1),
    }
    spectrum_param = "fast"

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        fast = int(self.values["fast"])
        slow = int(self.values["slow"])
        diff = close.rolling(fast).mean() - close.rolling(slow).mean()
        vol = ew_vol(close, annualize=False) * close
        fc = normalize_by_vol(diff, vol)
        sm = int(self.values.get("smooth", 1))
        return fc.rolling(sm).mean().fillna(fc) if sm > 1 else fc

    def indicator_lines(self, ohlcv):
        close = ohlcv["close"]
        return {
            f"SMA fast ({int(self.values['fast'])})": close.rolling(int(self.values["fast"])).mean(),
            f"SMA slow ({int(self.values['slow'])})": close.rolling(int(self.values["slow"])).mean(),
        }
