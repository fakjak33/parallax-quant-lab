# GHOST — Systematic Trading Strategy R&D Lab

> *Section-9 quant division.* A plug-and-play backtesting lab for developing
> systematic strategies the **Robert Carver** way: continuous volatility-scaled
> forecasts, vol-targeted sizing, forecast combination, and testing a whole
> **spectrum** of variants at once to avoid overfitting. Styled after *Ghost in
> the Shell*.

## Features

- **Real + synthetic data** — yfinance (free, cached to parquet) and synthetic
  generators (GBM, trending, mean-reverting, regime, fat-tailed) with known
  ground truth for validating strategies.
- **Carver forecast pipeline** — raw rule → vol-normalized → scaled to avg |10|
  → capped ±20 → vol-targeted position sizing → forecast combination (FDM).
- **Strategy library** (plug-and-play, auto-registered):
  EMA (EWMAC), SMA, Guppy MMA, MA crossover, time-series momentum,
  cross-sectional momentum, Donchian breakout, mean reversion, carry proxy.
- **Spectrum testing** — sweep a parameter across a family; spot robust
  *plateaus* vs overfit *spikes*; deflated-Sharpe penalty over the trial count.
- **Overfitting defenses** — return correlation matrix, walk-forward IS/OOS,
  Monte Carlo block-bootstrap distributions, Probabilistic & Deflated Sharpe.
- **ATR risk overlays** — optional trailing stop-loss and take-profit (k·ATR),
  with an exportable trade blotter.
- **Realism** — transaction costs, slippage, next-day execution (no look-ahead).

## Quick start

```bash
# from the ghost/ folder
.venv/Scripts/python -m streamlit run app.py
```

Then open http://localhost:8501.

(Windows note: if `python` isn't on PATH, the project venv interpreter is at
`.venv/Scripts/python.exe`.)

## Adding a strategy

Drop a file in `ghost/strategies/`, subclass `Strategy`, decorate with
`@register`, and define `params` + `spectrum_param`. It appears in the UI
automatically — no UI edits needed.

```python
@register
class MyRule(Strategy):
    key = "myrule"
    label = "My Rule"
    params = {"lookback": (50, 5, 200, 1)}
    spectrum_param = "lookback"

    def raw_forecast(self, ohlcv):
        ...  # return an unscaled pandas Series
```

## Tests

```bash
.venv/Scripts/python -m pytest tests/ -q
```

The suite includes **regime sanity checks**: trend rules must beat mean
reversion on trending synthetic data, and vice versa — proving the engine is
directionally correct.

## Layout

```
app.py              Streamlit dashboard (GITS theme)
ghost/
  config.py         defaults, theme
  data/             providers (yfinance+cache), synthetic, universe
  core/             forecasts, volatility, combination
  strategies/       rule library + registry
  risk/             ATR + stop/TP overlays
  backtest/         engine, metrics, spectrum, diagnostics, montecarlo
tests/              pytest suite
```

## Disclaimer

For research and education only. Not investment advice. Backtested results do
not guarantee future performance.
