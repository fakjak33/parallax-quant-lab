"""Accumulation backtester: trigger-based buying/selling vs DCA & buy-and-hold.

Model: you start with ``initial_cash`` and add ``contribution`` every cadence
period into a cash reserve. A *buy rule* deploys a fraction of available cash
when it fires; an optional *sell rule* moves a fraction of holdings back to cash.
Idle cash optionally earns ``cash_yield_annual``. Equity = shares·price + cash,
so every strategy is compared on the *same total money in* (dry powder counts).

Benchmarks (same contribution schedule):
- **DCA**: invest each contribution immediately every period.
- **Buy & hold (lump)**: invest the entire planned capital at t0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


@dataclass
class AccumConfig:
    initial_cash: float = 10_000.0
    contribution: float = 1_000.0
    cadence: str = "Weekly"            # Daily | Weekly | Monthly
    deploy_fraction: float = 1.0       # fraction of cash deployed per buy signal
    sell_fraction: float = 0.25        # fraction of shares sold per sell signal
    cash_yield_annual: float = 0.0     # optional yield on idle cash


@dataclass
class AccumResult:
    equity: pd.Series
    cash: pd.Series
    shares: pd.Series
    invested: pd.Series                # cumulative money contributed
    stats: dict


def _contribution_mask(index: pd.DatetimeIndex, cadence: str) -> np.ndarray:
    """Boolean mask: True on the first bar of each contribution period."""
    if cadence == "Daily":
        return np.ones(len(index), dtype=bool)
    freq = "W" if cadence == "Weekly" else "M"
    periods = pd.PeriodIndex(index, freq=freq)
    return np.r_[True, periods[1:] != periods[:-1]]


def run_accumulation(close: pd.Series, buy_rule, sell_rule, cfg: AccumConfig,
                     ctx: dict | None = None) -> AccumResult:
    ctx = ctx or {}
    close = close.dropna()
    idx = close.index
    px = close.to_numpy(dtype=float)
    contrib_mask = _contribution_mask(idx, cfg.cadence)

    buy_sig = buy_rule.signal(close, ctx).reindex(idx).fillna(False).to_numpy()
    buy_w = buy_rule.weight(close, ctx).reindex(idx).fillna(1.0).clip(0, 1).to_numpy()
    sell_sig = (sell_rule.signal(close, ctx).reindex(idx).fillna(False).to_numpy()
                if sell_rule is not None else np.zeros(len(idx), bool))

    daily_yield = (1 + cfg.cash_yield_annual) ** (1 / TRADING_DAYS) - 1
    cash = cfg.initial_cash
    shares = 0.0
    invested = cfg.initial_cash
    eq, csh, shs, inv = [], [], [], []

    for i in range(len(idx)):
        if i > 0 and daily_yield:
            cash *= (1 + daily_yield)
        if contrib_mask[i] and i > 0:
            cash += cfg.contribution
            invested += cfg.contribution
        # sell first (free up nothing needed, but realize before buy)
        if sell_sig[i] and shares > 0:
            sell_sh = shares * cfg.sell_fraction
            cash += sell_sh * px[i]
            shares -= sell_sh
        if buy_sig[i] and cash > 0:
            spend = cash * cfg.deploy_fraction * float(buy_w[i])
            shares += spend / px[i]
            cash -= spend
        eq.append(shares * px[i] + cash)
        csh.append(cash); shs.append(shares); inv.append(invested)

    equity = pd.Series(eq, index=idx)
    invested_s = pd.Series(inv, index=idx)
    stats = _accum_stats(equity, invested_s, close)
    return AccumResult(equity, pd.Series(csh, index=idx), pd.Series(shs, index=idx),
                       invested_s, stats)


def benchmarks(close: pd.Series, cfg: AccumConfig) -> dict[str, pd.Series]:
    """DCA and lump buy-and-hold equity curves on the same contribution plan."""
    close = close.dropna()
    idx = close.index
    px = close.to_numpy(dtype=float)
    mask = _contribution_mask(idx, cfg.cadence)

    # DCA: invest each contribution immediately
    shares, eq = 0.0, []
    shares += cfg.initial_cash / px[0]
    for i in range(len(idx)):
        if mask[i] and i > 0:
            shares += cfg.contribution / px[i]
        eq.append(shares * px[i])
    dca = pd.Series(eq, index=idx)

    # Lump: deploy the entire planned capital at t0
    total = cfg.initial_cash + cfg.contribution * (mask.sum() - 1)
    lump = (close / px[0]) * total

    return {"DCA": dca, "Buy & Hold (lump)": lump}


def _accum_stats(equity: pd.Series, invested: pd.Series, close: pd.Series) -> dict:
    from ..backtest import metrics
    ret = equity.pct_change().fillna(0.0)
    mkt = close.pct_change().reindex(ret.index).fillna(0.0)
    var = mkt.var()
    beta = float(ret.cov(mkt) / var) if var > 1e-18 else 0.0
    alpha_daily = ret.mean() - beta * mkt.mean()
    final_inv = float(invested.iloc[-1])
    final_eq = float(equity.iloc[-1])
    return {
        "FinalEquity": final_eq,
        "Invested": final_inv,
        "Profit": final_eq - final_inv,
        "ReturnOnInvested%": (final_eq / final_inv - 1) * 100 if final_inv else 0.0,
        "AnnVol%": metrics.annual_vol(ret) * 100,
        "MaxDD%": metrics.max_drawdown(ret) * 100,
        "Beta": beta,
        "Alpha(ann)%": alpha_daily * TRADING_DAYS * 100,
        "Corr": float(ret.corr(mkt)) if var > 1e-18 else 0.0,
    }
