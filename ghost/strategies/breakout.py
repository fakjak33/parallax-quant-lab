"""Donchian channel breakout.

Forecast = position of price within its N-day high/low range, centered and
scaled so a fresh N-day high reads near +20 and a fresh low near -20.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register


@register
class Breakout(Strategy):
    key = "breakout"
    label = "Donchian Breakout"
    params = {
        "lookback": (40, 5, 1000, 1),
        "smooth": (10, 1, 200, 1),
    }
    spectrum_param = "lookback"

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        roll_max = close.rolling(lb).max()
        roll_min = close.rolling(lb).min()
        rng = (roll_max - roll_min).replace(0.0, np.nan)
        # position in range: 0 at low, 1 at high -> center to [-0.5, 0.5]
        pos = (close - roll_min) / rng - 0.5
        smoothed = pos.rolling(int(self.values["smooth"])).mean()
        return smoothed.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def indicator_lines(self, ohlcv):
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        return {
            f"Donchian high ({lb})": close.rolling(lb).max(),
            f"Donchian low ({lb})": close.rolling(lb).min(),
        }
