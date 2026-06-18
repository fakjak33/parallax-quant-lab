"""Accumulation buy/sell rules (registry-driven, like the trading strategies).

Each rule exposes ``params`` (name -> (default, lo, hi, step)) for the UI and a
``signal(close, ctx)`` returning a boolean Series (fire on True). Buy rules also
provide ``weight(close, ctx)`` in [0,1] to scale how aggressively to deploy
(e.g. deeper drawdowns deploy more). ``ctx`` carries extras like a VIX series.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import signals as sig
from .regression import regression_channel

BUY_RULES: dict[str, type] = {}
SELL_RULES: dict[str, type] = {}


def _reg(store):
    def deco(cls):
        store[cls.key] = cls
        return cls
    return deco


@dataclass
class Rule:
    key: str = ""
    label: str = ""
    params: dict = field(default_factory=dict)
    desc: str = ""

    def __init__(self, **kw):
        self.values = {n: spec[0] for n, spec in self.params.items()}
        for k, v in kw.items():
            if k in self.values:
                self.values[k] = v

    def signal(self, close: pd.Series, ctx: dict) -> pd.Series:
        raise NotImplementedError

    def weight(self, close: pd.Series, ctx: dict) -> pd.Series:
        return pd.Series(1.0, index=close.index)


# ----------------------------- BUY RULES -----------------------------------
@_reg(BUY_RULES)
class FixedDCA(Rule):
    key, label = "dca", "Fixed DCA (every period)"
    desc = "Invest the periodic contribution every cadence period (the baseline)."
    params = {}

    def signal(self, close, ctx):
        return pd.Series(True, index=close.index)


@_reg(BUY_RULES)
class DrawdownBuy(Rule):
    key, label = "drawdown", "Buy on drawdown from high"
    desc = "Deploy when price is X% below its running peak; deeper dips deploy more."
    params = {"threshold_pct": (10.0, 1.0, 80.0, 1.0), "scale_by_depth": (1, 0, 1, 1)}

    def signal(self, close, ctx):
        dd = sig.drawdown_from_high(close) * -100.0
        return dd >= float(self.values["threshold_pct"])

    def weight(self, close, ctx):
        if not int(self.values["scale_by_depth"]):
            return pd.Series(1.0, index=close.index)
        dd = (sig.drawdown_from_high(close) * -100.0)
        thr = float(self.values["threshold_pct"])
        return (dd / (thr * 2)).clip(0.25, 1.0)


@_reg(BUY_RULES)
class VIXBuy(Rule):
    key, label = "vix", "Buy when VIX elevated"
    desc = "Deploy when VIX ≥ level (fear). Needs the VIX series (auto-fetched)."
    params = {"level": (25.0, 10.0, 80.0, 1.0)}

    def signal(self, close, ctx):
        vix = ctx.get("vix")
        if vix is None:
            return pd.Series(False, index=close.index)
        v = vix.reindex(close.index).ffill()
        return v >= float(self.values["level"])


@_reg(BUY_RULES)
class MATouchBuy(Rule):
    key, label = "ma_touch", "Buy near moving average"
    desc = "Deploy when price is within tolerance of (or below) MA(n)."
    params = {"n": (200, 0, 2000, 1), "tol_pct": (1.0, 0.0, 20.0, 0.5)}

    def signal(self, close, ctx):
        n = int(self.values["n"])
        if n <= 1:
            return pd.Series(True, index=close.index)
        ma = sig.moving_average(close, n)
        return close <= ma * (1 + float(self.values["tol_pct"]) / 100.0)


@_reg(BUY_RULES)
class RSIBuy(Rule):
    key, label = "rsi", "Buy when RSI low"
    desc = "Deploy when RSI(period) ≤ level (oversold)."
    params = {"period": (14, 2, 100, 1), "level": (35.0, 5.0, 60.0, 1.0)}

    def signal(self, close, ctx):
        return sig.rsi(close, int(self.values["period"])) <= float(self.values["level"])


@_reg(BUY_RULES)
class SlopeBuy(Rule):
    key, label = "ma_slope", "Buy when MA slope ≤ level"
    desc = "Deploy when the slope of MA(n) falls below a threshold (falling trend)."
    params = {"n": (100, 5, 2000, 1), "level_bps": (0.0, -50.0, 50.0, 1.0)}

    def signal(self, close, ctx):
        slope = sig.ma_slope(close, int(self.values["n"])) * 10000  # bps/bar
        return slope <= float(self.values["level_bps"])


@_reg(BUY_RULES)
class RegressionBuy(Rule):
    key, label = "regression", "Buy below regression band"
    desc = "Deploy when price ≤ lin/log regression fit − k·σ over the lookback."
    params = {"lookback": (504, 60, 2000, 10), "k": (2.0, 0.5, 4.0, 0.25), "log": (1, 0, 1, 1)}

    def signal(self, close, ctx):
        ch = regression_channel(close, int(self.values["lookback"]),
                                float(self.values["k"]), bool(int(self.values["log"])))
        return close <= ch["lower"]

    def weight(self, close, ctx):
        ch = regression_channel(close, int(self.values["lookback"]),
                                float(self.values["k"]), bool(int(self.values["log"])))
        return (-ch["z"] / 3.0).clip(0.25, 1.0).fillna(0.5)


# ----------------------------- SELL RULES ----------------------------------
@_reg(SELL_RULES)
class MayerSell(Rule):
    key, label = "mayer", "Sell when far above MA (Mayer)"
    desc = "Trim when price ≥ MA(n)·(1+x%) — the Mayer-multiple overextension."
    params = {"n": (200, 5, 2000, 1), "pct": (40.0, 5.0, 200.0, 5.0)}

    def signal(self, close, ctx):
        return sig.pct_from_ma(close, int(self.values["n"])) * 100 >= float(self.values["pct"])


@_reg(SELL_RULES)
class RSISell(Rule):
    key, label = "rsi_sell", "Sell when RSI high"
    desc = "Trim when RSI(period) ≥ level (overbought)."
    params = {"period": (14, 2, 100, 1), "level": (75.0, 50.0, 95.0, 1.0)}

    def signal(self, close, ctx):
        return sig.rsi(close, int(self.values["period"])) >= float(self.values["level"])


@_reg(SELL_RULES)
class SlopeSell(Rule):
    key, label = "slope_sell", "Sell when MA slope ≥ level"
    desc = "Trim when the slope of MA(n) exceeds a threshold (overheated trend)."
    params = {"n": (100, 5, 2000, 1), "level_bps": (20.0, -50.0, 100.0, 1.0)}

    def signal(self, close, ctx):
        slope = sig.ma_slope(close, int(self.values["n"])) * 10000
        return slope >= float(self.values["level_bps"])


@_reg(SELL_RULES)
class RegressionSell(Rule):
    key, label = "regression_sell", "Sell above regression band"
    desc = "Trim when price ≥ lin/log regression fit + k·σ over the lookback."
    params = {"lookback": (504, 60, 2000, 10), "k": (2.0, 0.5, 4.0, 0.25), "log": (1, 0, 1, 1)}

    def signal(self, close, ctx):
        ch = regression_channel(close, int(self.values["lookback"]),
                                float(self.values["k"]), bool(int(self.values["log"])))
        return close >= ch["upper"]
