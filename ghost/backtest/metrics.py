"""Performance metrics, including overfitting-aware Sharpe variants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import TRADING_DAYS


def _clean(returns: pd.Series) -> pd.Series:
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def sharpe(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.std(ddof=1) < 1e-12 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def sortino(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    downside = r[r < 0]
    dd = downside.std(ddof=1)
    if dd < 1e-12:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods))


def cagr(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.empty:
        return 0.0
    total = (1.0 + r).prod()
    years = len(r) / periods
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + _clean(returns)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())


def calmar(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(returns))
    return float(cagr(returns, periods) / mdd) if mdd > 1e-9 else 0.0


def hit_rate(returns: pd.Series) -> float:
    r = _clean(returns)
    nonzero = r[r != 0]
    return float((nonzero > 0).mean()) if len(nonzero) else 0.0


def turnover(position: pd.Series) -> float:
    """Average daily absolute change in position (proxy for trading activity)."""
    return float(position.diff().abs().mean())


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0,
                               periods: int = TRADING_DAYS) -> float:
    """PSR: probability the true Sharpe exceeds ``benchmark_sr`` (Bailey & Lopez de Prado).

    Accounts for track-record length, skew, and kurtosis.
    """
    r = _clean(returns)
    n = len(r)
    if n < 8 or r.std(ddof=1) < 1e-12:
        return 0.0
    sr = sharpe(r, periods) / np.sqrt(periods)          # per-period SR
    sr_bench = benchmark_sr / np.sqrt(periods)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    if denom < 1e-12:
        return 0.0
    z = (sr - sr_bench) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          trial_sr_std: float = 1.0, periods: int = TRADING_DAYS) -> float:
    """DSR: PSR against a benchmark inflated for ``n_trials`` independent tries.

    This is the headline overfitting metric for spectrum testing: testing many
    variants raises the bar a strategy must clear to be considered 'real'.
    """
    if n_trials < 1:
        n_trials = 1
    euler = 0.5772156649
    # expected max Sharpe of n_trials random strategies (annualized)
    e_max = trial_sr_std * (
        (1 - euler) * stats.norm.ppf(1 - 1.0 / n_trials)
        + euler * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    return probabilistic_sharpe_ratio(returns, benchmark_sr=e_max, periods=periods)


def summary(returns: pd.Series, position: pd.Series | None = None,
            n_trials: int = 1) -> dict[str, float]:
    """One-stop metrics dict for the UI table."""
    out = {
        "CAGR": cagr(returns),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "MaxDD": max_drawdown(returns),
        "Calmar": calmar(returns),
        "HitRate": hit_rate(returns),
        "PSR": probabilistic_sharpe_ratio(returns),
        "DSR": deflated_sharpe_ratio(returns, n_trials=n_trials),
    }
    if position is not None:
        out["Turnover"] = turnover(position)
    return out
