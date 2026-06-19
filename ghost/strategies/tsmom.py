"""Time-series momentum (Moskowitz, Ooi & Pedersen 2012).

Forecast = past N-day return scaled by volatility. Positive past return =>
long. This is the academically canonical 'trend' factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..core.volatility import ew_vol
from ..config import TRADING_DAYS


@register
class TimeSeriesMomentum(Strategy):
    key = "tsmom"
    label = "Time-Series Momentum"
    params = {
        "lookback": (90, 10, 2000, 5),
        "smooth": (1, 1, 50, 1),
    }
    spectrum_param = "lookback"

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        past_ret = close / close.shift(lb) - 1.0
        # scale by annualized vol so the signal is risk-adjusted
        vol_annual = ew_vol(close, annualize=True)
        sharpe_like = past_ret / (vol_annual * np.sqrt(lb / TRADING_DAYS))
        fc = sharpe_like.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sm = int(self.values.get("smooth", 1))
        return fc.rolling(sm).mean().fillna(fc) if sm > 1 else fc
