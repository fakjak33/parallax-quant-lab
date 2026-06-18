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
) -> pd.Series:
    """Convert a capped forecast into a position in *units* (shares).

    With ``use_vol_target`` (Carver default), volatility targeting:

        units = (forecast / 10) * (capital * target_vol)
                / (price_vol_annual * price)

    A forecast of +10 (the average) targets the full per-instrument risk
    budget; +20 doubles it, -10 is a full short.

    With ``use_vol_target=False``, fixed-notional sizing: the position scales
    linearly with the forecast (full forecast = full capital deployed), with
    NO volatility scaling — so high-vol periods carry proportionally more risk.
    """
    if use_vol_target:
        price_vol_annual = ew_vol(prices, span=vol_span, annualize=True)
        risk_budget = capital * target_vol
        notional_vol_per_unit = price_vol_annual * prices
        units = (forecast / FORECAST_SCALAR_TARGET) * risk_budget / notional_vol_per_unit
    else:
        # forecast/10 fraction of capital as notional, converted to units
        notional = (forecast / FORECAST_SCALAR_TARGET) * capital
        units = notional / prices
    return units.replace([np.inf, -np.inf], np.nan).fillna(0.0)
