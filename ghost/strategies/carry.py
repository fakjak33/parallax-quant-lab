"""Carry proxy for ETFs.

True carry needs a term structure / yield. For equity & commodity ETFs we
approximate the carry premium with a long-horizon risk-adjusted drift (the
asset's own tendency to earn a positive return per unit risk), smoothed.
This is a deliberately simple stand-in so the 'carry' style is represented;
swap in real yield/roll data later for futures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..core.volatility import ew_vol
from ..config import TRADING_DAYS


@register
class CarryProxy(Strategy):
    key = "carry"
    label = "Carry (drift proxy)"
    params = {
        "lookback": (252, 60, 2000, 5),
    }
    spectrum_param = "lookback"

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        ann_ret = (close / close.shift(lb)) ** (TRADING_DAYS / lb) - 1.0
        vol_annual = ew_vol(close, annualize=True)
        carry = ann_ret / vol_annual.replace(0.0, np.nan)
        return carry.rolling(20).mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
