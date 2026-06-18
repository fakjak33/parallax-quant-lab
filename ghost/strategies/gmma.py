"""Guppy Multiple Moving Average (GMMA).

Two EMA ribbons: a short-term group (traders) and a long-term group
(investors). Forecast = separation between the two ribbon means, normalized
by price vol. Wide positive separation => strong, agreed-upon uptrend.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy
from .registry import register
from ..core.forecasts import normalize_by_vol
from ..core.volatility import ew_vol

_SHORT = (3, 5, 8, 10, 12, 15)
_LONG = (30, 35, 40, 45, 50, 60)


@register
class GMMA(Strategy):
    key = "gmma"
    label = "Guppy MMA"
    params = {
        # a single 'speed' multiplier scales both ribbons for the spectrum
        "speed": (1.0, 0.5, 3.0, 0.1),
    }
    spectrum_param = "speed"

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        speed = float(self.values["speed"])
        short = pd.concat(
            [close.ewm(span=max(2, int(s * speed))).mean() for s in _SHORT], axis=1
        ).mean(axis=1)
        long = pd.concat(
            [close.ewm(span=max(3, int(s * speed))).mean() for s in _LONG], axis=1
        ).mean(axis=1)
        vol = ew_vol(close, annualize=False) * close
        return normalize_by_vol(short - long, vol)
