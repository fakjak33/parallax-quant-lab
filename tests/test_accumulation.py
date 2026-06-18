import numpy as np
import pandas as pd

from ghost.accumulation.engine import AccumConfig, run_accumulation, benchmarks
from ghost.accumulation.strategies import BUY_RULES, SELL_RULES
from ghost.accumulation.regression import regression_channel
from ghost.accumulation import signals as sig
from ghost.data.universe import is_leveraged, nonleveraged_master_tickers
from ghost.data.stocks500 import STOCKS_500


def _ramp(n=500, start="2018-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(np.linspace(100, 200, n), index=idx)


def test_dca_invests_everything():
    close = _ramp()
    cfg = AccumConfig(initial_cash=1000, contribution=100, cadence="Weekly")
    bench = benchmarks(close, cfg)
    assert "DCA" in bench and "Buy & Hold (lump)" in bench
    # both benchmarks end positive and lump >= dca on a monotonic uptrend
    assert bench["Buy & Hold (lump)"].iloc[-1] >= bench["DCA"].iloc[-1] > 0


def test_accumulation_runs_and_holds_cash_until_dip():
    close = _ramp()
    cfg = AccumConfig(initial_cash=10000, contribution=0, cadence="Weekly")
    # drawdown rule never fires on a pure uptrend -> stays in cash
    res = run_accumulation(close, BUY_RULES["drawdown"](threshold_pct=10), None, cfg, {})
    assert res.shares.iloc[-1] == 0.0
    assert abs(res.equity.iloc[-1] - 10000) < 1e-6


def test_dca_rule_deploys():
    close = _ramp()
    cfg = AccumConfig(initial_cash=0, contribution=100, cadence="Weekly")
    res = run_accumulation(close, BUY_RULES["dca"](), None, cfg, {})
    assert res.shares.iloc[-1] > 0
    assert "Beta" in res.stats and "Alpha(ann)%" in res.stats


def test_regression_channel_bands_order():
    close = _ramp()
    ch = regression_channel(close, lookback=100, k=2.0, log=True)
    valid = ch.dropna()
    assert (valid["upper"] >= valid["fit"]).all()
    assert (valid["fit"] >= valid["lower"]).all()


def test_rsi_bounds():
    close = _ramp()
    r = sig.rsi(close, 14)
    assert (r >= 0).all() and (r <= 100).all()


def test_leverage_filter():
    assert is_leveraged("TQQQ")
    assert is_leveraged("SOXL")
    assert is_leveraged("X", "Direxion Daily 3x Bull")
    assert not is_leveraged("SPY")
    assert not is_leveraged("XLF")
    nonlev = nonleveraged_master_tickers()
    assert "TQQQ" not in nonlev and "SOXL" not in nonlev


def test_stock_universe_size():
    assert len(STOCKS_500) >= 400  # curated large/mid-cap set
    assert len(STOCKS_500) == len(set(STOCKS_500))  # de-duped
