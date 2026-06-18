"""Cross-sectional momentum (Jegadeesh & Titman 1993).

Ranks assets by past return across the universe and goes long winners /
short losers. This rule is inherently multi-asset, so it overrides
``forecast_panel`` rather than ``raw_forecast``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..core.forecasts import scale_forecast


@register
class CrossSectionalMomentum(Strategy):
    key = "xsmom"
    label = "Cross-Sectional Momentum"
    cross_sectional = True
    params = {
        "lookback": (126, 20, 378, 5),
        "skip": (21, 0, 42, 1),   # skip most-recent days (reversal control)
    }
    spectrum_param = "lookback"

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:  # pragma: no cover
        raise NotImplementedError("xsmom is cross-sectional; use forecast_panel.")

    def forecast_panel(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        lb = int(self.values["lookback"])
        skip = int(self.values["skip"])
        closes = pd.DataFrame({t: df["close"] for t, df in panel.items()}).sort_index()
        past_ret = closes.shift(skip) / closes.shift(lb) - 1.0

        # cross-sectional z-score each day => demeaned long/short forecast
        mean = past_ret.mean(axis=1)
        std = past_ret.std(axis=1).replace(0.0, np.nan)
        z = past_ret.sub(mean, axis=0).div(std, axis=0)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # scale each column to the standard forecast magnitude
        return pd.DataFrame({c: scale_forecast(z[c]) for c in z.columns})
