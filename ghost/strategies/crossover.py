"""Generic MA crossover with a binary-ish, smoothed forecast.

Distinct from the EWMAC trend rule: this measures the *percentage* gap
between two SMAs (a classic golden/death-cross proxy) and smooths it, giving
a forecast that saturates rather than scaling with raw price difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register


@register
class Crossover(Strategy):
    key = "crossover"
    label = "MA Crossover (%gap)"
    params = {
        "fast": (50, 5, 150, 1),
        "slow": (200, 20, 400, 1),
        "smooth": (5, 1, 30, 1),
    }
    spectrum_param = "fast"

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        fast = close.rolling(int(self.values["fast"])).mean()
        slow = close.rolling(int(self.values["slow"])).mean()
        pct_gap = (fast - slow) / slow.replace(0.0, np.nan)
        return pct_gap.rolling(int(self.values["smooth"])).mean().fillna(0.0)

    def indicator_lines(self, ohlcv):
        close = ohlcv["close"]
        return {
            f"SMA fast ({int(self.values['fast'])})": close.rolling(int(self.values["fast"])).mean(),
            f"SMA slow ({int(self.values['slow'])})": close.rolling(int(self.values["slow"])).mean(),
        }
