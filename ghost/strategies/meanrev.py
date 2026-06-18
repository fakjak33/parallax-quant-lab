"""Mean reversion — the academically supported counterpoint to trend.

Forecast = negative z-score of price vs its moving average (buy dips, fade
rallies). Included so the lab can demonstrate regime dependence: this should
WIN on mean-reverting synthetic data and LOSE on trending data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register


@register
class MeanReversion(Strategy):
    key = "meanrev"
    label = "Mean Reversion (z-score)"
    params = {
        "lookback": (20, 3, 120, 1),
    }
    spectrum_param = "lookback"

    def raw_forecast(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        lb = int(self.values["lookback"])
        ma = close.rolling(lb).mean()
        sd = close.rolling(lb).std().replace(0.0, np.nan)
        z = (close - ma) / sd
        # negative => fade the move
        return (-z).replace([np.inf, -np.inf], np.nan).fillna(0.0)
