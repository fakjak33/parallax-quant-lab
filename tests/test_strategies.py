import numpy as np
import pandas as pd
import pytest

from ghost import strategies
from ghost.strategies import REGISTRY
from ghost.data.synthetic import generate
from ghost.config import FORECAST_CAP


EXPECTED = {"ema", "sma", "gmma", "crossover", "tsmom", "xsmom", "breakout", "meanrev", "carry"}


def test_all_strategies_registered():
    assert EXPECTED.issubset(set(REGISTRY))


@pytest.mark.parametrize("key", sorted(EXPECTED - {"xsmom"}))
def test_single_instrument_forecast_shape_and_cap(key):
    df = generate("gbm", n_days=400, seed=1)
    strat = REGISTRY[key]()
    fc = strat.forecast(df)
    assert len(fc) == len(df)
    assert np.isfinite(fc).all()
    assert fc.abs().max() <= FORECAST_CAP + 1e-9


def test_xsmom_panel():
    from ghost.data.synthetic import generate_panel
    panel = generate_panel(n_assets=5, kind="trending", n_days=400, seed=3)
    strat = REGISTRY["xsmom"]()
    fcs = strat.forecast_panel(panel)
    assert fcs.shape[1] == 5
    assert np.isfinite(fcs.fillna(0)).all().all()
