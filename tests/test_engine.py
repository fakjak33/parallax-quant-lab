import numpy as np
import pandas as pd

from ghost.data.synthetic import generate
from ghost.strategies import REGISTRY
from ghost.backtest.engine import run_strategy
from ghost.backtest.spectrum import run_spectrum
from ghost.backtest import metrics


def test_backtest_runs_and_has_metrics():
    df = generate("trending", n_days=600, seed=5)
    res = run_strategy(REGISTRY["ema"](fast=16, slow=64), df)
    assert len(res.equity) == len(df)
    assert set(["Sharpe", "CAGR", "MaxDD", "DSR"]).issubset(res.stats)
    assert np.isfinite(res.returns).all()


def test_trend_follower_beats_meanrev_on_trending_data():
    """Sanity: a trend rule should out-Sharpe mean reversion on trending data."""
    df = generate("trending", n_days=1200, seed=11)
    trend = run_strategy(REGISTRY["tsmom"](lookback=90), df)
    mr = run_strategy(REGISTRY["meanrev"](lookback=20), df)
    assert trend.stats["Sharpe"] > mr.stats["Sharpe"]


def test_meanrev_beats_trend_on_mean_reverting_data():
    df = generate("mean_reverting", n_days=1200, seed=12)
    trend = run_strategy(REGISTRY["tsmom"](lookback=90), df)
    mr = run_strategy(REGISTRY["meanrev"](lookback=15), df)
    assert mr.stats["Sharpe"] > trend.stats["Sharpe"]


def test_spectrum_produces_table():
    df = generate("trending", n_days=800, seed=7)
    results, table = run_spectrum(REGISTRY["ema"], df, param="fast",
                                  values=[2, 4, 8, 16, 32])
    assert len(results) == len(table) == 5
    assert "Sharpe" in table.columns
    # deflated Sharpe should be present and <= PSR (stricter bar)
    assert (table["DSR"] <= table["PSR"] + 1e-9).all()


def test_manual_sharpe_matches():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 1000))
    manual = r.mean() / r.std(ddof=1) * np.sqrt(256)
    assert abs(metrics.sharpe(r) - manual) < 1e-9
