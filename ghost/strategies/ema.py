"""EMA trend rule.

Carver's EWMAC: difference between a fast and slow EWMA, normalized by price
volatility so the forecast scale is stationary. Positive => uptrend.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy
from .registry import register
from ..core.forecasts import normalize_by_vol
from ..core.volatility import ew_vol


@register
class EMACrossover(Strategy):
    key = "ema"
    label = "EMA Trend (EWMAC)"
    params = {
        "fast": (16, 2, 1000, 1),
        "slow": (64, 4, 2000, 1),
        "smooth": (1, 1, 50, 1),
    }
    spectrum_param = "fast"

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        fast = int(self.values["fast"])
        slow = int(self.values["slow"])
        ewmac = close.ewm(span=fast).mean() - close.ewm(span=slow).mean()
        # normalize by price * daily vol => unit-free trend strength
        vol = ew_vol(close, annualize=False) * close
        fc = normalize_by_vol(ewmac, vol)
        sm = int(self.values.get("smooth", 1))
        return fc.rolling(sm).mean().fillna(fc) if sm > 1 else fc

    def indicator_lines(self, ohlcv):
        close = ohlcv["close"]
        return {
            f"EMA fast ({int(self.values['fast'])})": close.ewm(span=int(self.values["fast"])).mean(),
            f"EMA slow ({int(self.values['slow'])})": close.ewm(span=int(self.values["slow"])).mean(),
        }
