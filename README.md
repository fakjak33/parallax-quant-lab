# PARALLAX — Systematic Trading Strategy R&D Lab

> A plug-and-play backtesting lab for developing systematic strategies the
> **Robert Carver** way: continuous volatility-scaled forecasts, vol-targeted
> sizing, forecast combination, and testing a whole **spectrum** of variants at
> once to avoid overfitting. Modernist, minimalist UI — black canvas with a
> retro pantone palette.

## Features

- **Real + synthetic data** — yfinance (free, cached to parquet) and synthetic
  generators (GBM, trending, mean-reverting, regime, fat-tailed) with known
  ground truth for validating strategies. Adjustable **timeframe** (daily /
  weekly / monthly) and date window.
- **2,200+ ETFs** — bundled categorized ETF master list plus curated universes;
  type any yfinance ticker too.
- **Carver forecast pipeline** — raw rule → vol-normalized → scaled to avg |10|
  → capped ±20 → position sizing → forecast combination (FDM). Volatility
  targeting can be **toggled off** for fixed-notional sizing.
- **Full control per strategy** — edit *every* parameter of each rule, choose
  **which** parameter the spectrum sweeps, and set direction
  (**both / long-only / short-only**).
- **Strategy library** (plug-and-play, auto-registered):
  EMA (EWMAC), SMA, Guppy MMA, MA crossover, time-series momentum,
  cross-sectional momentum, Donchian breakout, mean reversion, carry proxy.
- **Trade visualization** — long/short entry markers on the price chart.
- **Spectrum testing** — sweep any parameter across a family; spot robust
  *plateaus* vs overfit *spikes*; deflated-Sharpe penalty over the trial count.
- **Diagnostics** — multi-select return-correlation matrix, **beta &
  correlation to the underlying ETF**, walk-forward IS/OOS.
- **Overfitting defenses** — Monte Carlo block-bootstrap distributions,
  Probabilistic & Deflated Sharpe.
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

## Modes
A sidebar **Mode** toggle switches between:
- **Strategy R&D** — the Carver-style systematic trading backtester (spectrum,
  drawdown, Monte Carlo, Kelly, diagnostics).
- **Accumulation Lab** — long-term DCA / dip-buying strategies (buy on drawdown,
  VIX, RSI, MA touch, MA slope, regression bands; optional sell rules like the
  Mayer multiple) compared against fixed DCA and lump-sum buy & hold, with
  beta/alpha/correlation/drawdown stats and linear/log regression channels.
- **ETF Lab** — design and backtest your own fund (`ghost/etf/`). Pick an
  explicit basket or **screen & rank** a universe by a point-in-time price
  factor (momentum / trailing or prior-calendar-year return / low-vol / beta /
  drawdown / dividend growth) **or a fundamental factor** (low P/E, FCF/share,
  profit margin, price/book, dividend yield, revenue), add **fundamental
  filters** (e.g. FCF/share ≤ 3, sector/industry, margin thresholds), choose a
  **weighting** scheme (equal, inverse-vol, market-cap / FCF / revenue snapshot,
  manual), a **rebalance** cadence (weekly→annual), and **long / short /
  long-short** with a separate short-leg ranking and a borrow-cost knob. Tabs:
  DESIGN (holdings & weights + optional fundamental snapshot table), BACKTEST
  (equity vs benchmarks, weights-over-time, drawdown, costs & turnover), COMPARE
  (risk/capture stats + correlation heatmap), OVERLAP (weighted holdings overlap
  between funds), and **FUND ECONOMICS** (issuer P&L: AUM, fee revenue,
  startup/maintenance/licensing costs, net profit, breakeven AUM, investor fee
  drag).
  - **Free-data honesty:** price/return/vol/beta factors are genuinely
    point-in-time. Fundamental screens & weighting (P/E, FCF/share,
    sector/industry, margin) use a **current snapshot held constant across
    history** — so any design using them carries look-ahead/survivorship bias.
    The Lab flags this loudly (a red banner via `spec_is_lookahead`) and the
    selection universe is current regardless (survivorship caveat).

Universe: ~500 curated liquid large/mid-cap stocks + ETFs (leveraged/inverse
ETFs excluded by default).

## Deploying / remote access

See [DEPLOY.md](DEPLOY.md) for a step-by-step guide to running Parallax on a
private, **password-protected** URL (Streamlit Community Cloud) so you can open
it from a phone or another computer.

Security measures baked in: a password gate (`ghost/auth.py`, configured via
Streamlit secrets), ticker-input sanitization with path-traversal protection
(`ghost.data.providers.clean_ticker`), and resource caps on tickers / sweep
grid sizes. Run `pip-audit` to check dependencies for known CVEs.

## Tests

```bash
.venv/Scripts/python -m pytest tests/ -q
```

The suite includes **regime sanity checks**: trend rules must beat mean
reversion on trending synthetic data, and vice versa — proving the engine is
directionally correct.

## Layout

```
app.py              Streamlit dashboard (Parallax theme)
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
