"""Volatility estimation and volatility-targeted position sizing.

Carver sizes positions so each instrument contributes a target risk. We use
an exponentially-weighted daily return vol, annualize it, and derive the
number of units (or the capital weight) implied by a continuous forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS, VOL_LOOKBACK, FORECAST_SCALAR_TARGET


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily percentage returns."""
    return prices.pct_change()


def ew_vol(prices: pd.Series, span: int = VOL_LOOKBACK, annualize: bool = False) -> pd.Series:
    """Exponentially-weighted daily return volatility.

    A small floor avoids divide-by-zero in flat periods. Set ``annualize`` to
    scale by sqrt(TRADING_DAYS).
    """
    rets = daily_returns(prices)
    vol = rets.ewm(span=span, min_periods=max(2, span // 2)).std()
    vol = vol.bfill().clip(lower=1e-6)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def position_from_forecast(
    forecast: pd.Series,
    prices: pd.Series,
    capital: float,
    target_vol: float,
    vol_span: int = VOL_LOOKBACK,
    use_vol_target: bool = True,
    sizing_mode: str | None = None,
    fixed_pct: float = 1.0,
    fixed_dollar: float | None = None,
) -> pd.Series:
    """Convert a capped forecast into a position in *units* (shares).

    ``sizing_mode`` selects how a forecast maps to size (falls back to
    ``use_vol_target`` for back-compat):

    - ``"vol_target"`` — Carver volatility targeting::
          units = (forecast/10) * (capital * target_vol)
                  / (price_vol_annual * price)
      +10 (average forecast) targets the full per-instrument risk budget.
    - ``"fixed_pct"`` — deploy ``fixed_pct`` of capital at full forecast,
      scaled linearly by forecast/10. No vol scaling.
    - ``"fixed_dollar"`` — deploy ``fixed_dollar`` notional at full forecast.
    """
    if sizing_mode is None:
        sizing_mode = "vol_target" if use_vol_target else "fixed_pct"

    if sizing_mode == "vol_target":
        price_vol_annual = ew_vol(prices, span=vol_span, annualize=True)
        risk_budget = capital * target_vol
        notional_vol_per_unit = price_vol_annual * prices
        units = (forecast / FORECAST_SCALAR_TARGET) * risk_budget / notional_vol_per_unit
    elif sizing_mode == "fixed_dollar":
        dollar = capital if fixed_dollar is None else fixed_dollar
        notional = (forecast / FORECAST_SCALAR_TARGET) * dollar
        units = notional / prices
    else:  # fixed_pct
        notional = (forecast / FORECAST_SCALAR_TARGET) * fixed_pct * capital
        units = notional / prices
    return units.replace([np.inf, -np.inf], np.nan).fillna(0.0)
