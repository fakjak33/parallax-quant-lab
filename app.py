"""PARALLAX — Streamlit dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ghost import strategies  # noqa: F401  (registers strategies)
from ghost.strategies import REGISTRY
from ghost.config import (
    BacktestConfig, RiskConfig, THEME, TIMEFRAME_DAYS, TIME_SCALED_PARAMS,
)
from ghost.data import providers, synthetic
from ghost.data.providers import TIMEFRAMES, resample_ohlcv, clean_ticker
from ghost.data.universe import UNIVERSES, master_categories, master_tickers, is_leveraged
from ghost.accumulation import strategies as accum_strats
from ghost.accumulation.engine import AccumConfig, run_accumulation, benchmarks
from ghost.accumulation.regression import regression_channel
from ghost.backtest.engine import run_single
from ghost.backtest.spectrum import run_spectrum, run_spectrum_2d, make_spectrum
from ghost.backtest.diagnostics import return_correlation, walk_forward, beta_and_correlation
from ghost.backtest.trades import extract_trades
from ghost.backtest import montecarlo, metrics, kelly
from ghost.etf import portfolio as etf_pf, screens as etf_screens, presets as etf_presets
from ghost.etf import factors as etf_factors, weighting as etf_weighting
from ghost.etf import economics as etf_econ, overlap as etf_overlap
from ghost.auth import require_password
from ghost.ui_theme import CSS, BANNER, section, style_fig

st.set_page_config(page_title="PARALLAX // quant lab", layout="wide", page_icon="◧")
st.markdown(CSS, unsafe_allow_html=True)
require_password()
st.markdown(BANNER, unsafe_allow_html=True)

MAX_TICKERS = 12

HELP = {
    "mode": "Real pulls actual price history from Yahoo Finance. Synthetic generates fake data with known behavior to validate strategies.",
    "capital": "Starting account equity. All P/L and position sizes scale from this.",
    "sizing": "How a forecast becomes a position size. Vol target = constant risk (Carver). % of capital / $ notional = fixed exposure regardless of volatility.",
    "vol_target": "Volatility targeting scales positions so the strategy aims for a constant annual risk level.",
    "target_vol": "Annualized volatility the strategy targets, e.g. 0.20 = 20%/yr.",
    "direction": "Restrict trades: 'both' allows long & short, 'long' only buys, 'short' only sells.",
    "stop": "Stop-loss: exit a losing trade. ATR adapts to volatility (k×ATR); Percent uses a fixed % move. Trailing ratchets the stop in your favor.",
    "tp": "Take-profit: exit a winning trade at a target. ATR uses k×ATR; Percent uses a fixed % move.",
    "timeframe": "Bar frequency. Day-based parameters (fast/slow/lookback) are auto-scaled to the chosen candle so a '60' means ~60 trading days at any frequency.",
    "cost": "Transaction cost on traded notional, in basis points (1 bp = 0.01%).",
    "slippage": "Estimated execution slippage (half-spread) in basis points.",
    "sweep": "Run the strategy across a range of one (or two) parameters at once. Look for a broad plateau (robust) rather than a single lucky spike (overfit).",
    "seed": "Random seed for synthetic data — change it for a different fake price path.",
    "corr": "How correlated two strategies' daily returns are. Near 0 = diversifying; near 1 = redundant. Carver combines low-correlation rules.",
    "beta": "Beta = sensitivity of the strategy's returns to the underlying ETF's returns. Correlation = how tightly they move together (-1 to 1). Low beta/correlation means the strategy is market-neutral-ish.",
    "walkforward": "Splits history into segments and compares in-sample (fitted period) vs out-of-sample (later, unseen period) Sharpe. A big drop OOS warns of overfitting.",
    "montecarlo": "Resamples the daily returns in blocks many times to build a distribution of possible outcomes, instead of trusting the single historical path.",
    "kelly": "The Kelly criterion gives the growth-optimal bet size. Full Kelly maximizes long-run growth but is very swingy; most use half- or quarter-Kelly.",
    "drawdown": "Drawdown = % below the prior equity peak. Shows pain/risk over time vs just buying and holding the underlying.",
    "capture": "Upside capture = how much of the underlying's up-day return the strategy captures (>100% = amplifies gains). Downside capture = same for down days (<100% = cushions losses).",
    "delay": "Wait a fixed number of bars after a signal fires before acting (e.g. confirm an EMA cross for 5 candles before entering). Entry delay applies to opening/adding; exit delay to closing/trimming.",
    "logscale": "Log scale spaces equal % moves equally — better for long histories and comparing assets at different price levels. Linear shows absolute dollar moves.",
    "mc_pct": "Distribution across all bootstrap simulations: p5 = pessimistic (5th percentile), p50 = median, p95 = optimistic (95th). A wide p5→p95 gap means the result is luck-sensitive.",
    "kelly_full": "Full Kelly: the growth-optimal leverage = mean/variance of returns. Maximizes long-run compounding but is very volatile and unforgiving of estimation error.",
    "kelly_half": "Half Kelly: half the full-Kelly leverage. ~75% of the growth with far less volatility — the common practical choice.",
    "kelly_quarter": "Quarter Kelly: a conservative quarter of full Kelly, for when return estimates are uncertain.",
    "wf_color": "IS = in-sample (earlier, 'fitted' period); OOS = out-of-sample (later, unseen). Green IS = healthy Sharpe; green OOS = the edge held up out-of-sample; red OOS = it decayed (overfitting warning).",
    "corr_under": "Add underlyings (SPY, TLT, GLD…) to see how your strategies' returns correlate with real assets — not just with each other.",
}

PARAM_HELP = {
    "fast": "Fast (short) moving-average span in trading days — reacts quickly.",
    "slow": "Slow (long) moving-average span in trading days — the trend baseline.",
    "lookback": "Number of trading days the rule looks back over.",
    "skip": "Most-recent trading days to skip (controls short-term reversal).",
    "smooth": "Smoothing window (trading days) applied to the raw signal.",
    "speed": "Scales all Guppy ribbon spans up/down together.",
}


def label_to_key(label):
    for k, cls in REGISTRY.items():
        if cls.label == label:
            return k
    raise KeyError(label)


def render_param_controls(cls, prefix):
    values = {}
    for name, (default, lo, hi, step) in cls.params.items():
        is_int = isinstance(default, int) and isinstance(step, int)
        key = f"{prefix}_{cls.key}_{name}"
        help_txt = PARAM_HELP.get(name, f"{name} parameter")
        if is_int:
            values[name] = int(st.number_input(name, int(lo), int(hi), int(default),
                                                int(step), key=key, help=help_txt))
        else:
            values[name] = float(st.number_input(name, float(lo), float(hi),
                                                 float(default), float(step),
                                                 key=key, help=help_txt))
    return values


def scale_params_for_tf(values, tf_label):
    """Convert day-based params into bars at the chosen candle frequency."""
    dpb = TIMEFRAME_DAYS.get(tf_label, 1)
    if dpb <= 1:
        return dict(values)
    out = {}
    for k, v in values.items():
        if k in TIME_SCALED_PARAMS and isinstance(v, (int, float)):
            out[k] = max(2, int(round(v / dpb))) if k != "skip" else max(0, int(round(v / dpb)))
        else:
            out[k] = v
    return out


INSTRUMENT_HOLDER = {"name": None}


def make_strategy(cls, raw_values, tf_label):
    return cls(**scale_params_for_tf(raw_values, tf_label))


def forecast_for(cls, strat, ohlcv, panel):
    if cls.cross_sectional:
        return strat.forecast_panel(panel)[INSTRUMENT_HOLDER["name"]]
    return strat.forecast(ohlcv)


def sb(label, idx):
    st.sidebar.markdown(section(label, idx), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def sidebar():
    cfg = {}
    sb("Capital", 0)
    capital = st.sidebar.number_input("Starting capital ($)", 1_000.0, 1e9,
                                      1_000_000.0, 10_000.0, help=HELP["capital"])

    sb("Data source", 1)
    cfg["mode"] = st.sidebar.radio("Mode", ["Real (yfinance)", "Synthetic"], index=0,
                                   help=HELP["mode"])
    if cfg["mode"].startswith("Real"):
        src = st.sidebar.radio("Ticker source", ["Curated", "ETF master list"], index=0)
        if src == "Curated":
            uni = st.sidebar.selectbox("Universe", list(UNIVERSES), index=0)
            options = UNIVERSES[uni]
        else:
            cat = st.sidebar.selectbox("Category", master_categories())
            options = master_tickers(cat)
            if st.sidebar.checkbox("Exclude leveraged/inverse ETFs", True):
                options = [t for t in options if not is_leveraged(t)]
        cfg["tickers"] = st.sidebar.multiselect("Tickers", options,
                                                default=options[:1] if options else [],
                                                help="Defaults to a single ticker.")
        custom = st.sidebar.text_input("Add tickers (comma-sep)", "")
        if custom.strip():
            for raw in custom.split(","):
                if raw.strip():
                    try:
                        cfg["tickers"].append(clean_ticker(raw))
                    except ValueError:
                        st.sidebar.warning(f"Ignored invalid ticker: {raw.strip()!r}")
        cfg["tickers"] = list(dict.fromkeys(cfg["tickers"]))[:MAX_TICKERS]
    else:
        cfg["kind"] = st.sidebar.selectbox("Synthetic kind",
            ["trending", "mean_reverting", "gbm", "regime", "fat_tailed"])
        cfg["n_days"] = st.sidebar.number_input("Days", 300, 6000, 1500, 100)
        cfg["seed"] = int(st.sidebar.number_input("Seed", value=42, step=1, help=HELP["seed"]))
        cfg["n_assets"] = st.sidebar.number_input("Synthetic assets", 1, 12, 4)

    sb("Timeframe", 2)
    c1, c2 = st.sidebar.columns(2)
    cfg["start"] = c1.text_input("Start", "2015-01-01")
    cfg["end"] = c2.text_input("End", "")
    cfg["tf"] = st.sidebar.selectbox("Bar frequency", list(TIMEFRAMES), index=0,
                                     help=HELP["timeframe"])

    sb("Strategies", 3)
    cfg["selected"] = st.sidebar.multiselect("Active rules",
        [REGISTRY[k].label for k in sorted(REGISTRY)], default=[REGISTRY["ema"].label])
    cfg["params"] = {}
    for lbl in cfg["selected"]:
        k = label_to_key(lbl)
        with st.sidebar.expander(f"{lbl} parameters", expanded=False):
            cfg["params"][k] = render_param_controls(REGISTRY[k], prefix="bt")

    sb("Position sizing", 4)
    bt = BacktestConfig(capital=capital)
    mode = st.sidebar.radio("Sizing method",
        ["Volatility target", "% of capital", "$ notional"], help=HELP["sizing"])
    if mode == "Volatility target":
        bt.sizing_mode = "vol_target"
        bt.target_vol = st.sidebar.slider("Target vol", 0.05, 0.50, 0.20, 0.01,
                                          help=HELP["target_vol"])
    elif mode == "% of capital":
        bt.sizing_mode = "fixed_pct"
        bt.fixed_pct = st.sidebar.slider("% of capital at full forecast", 0.1, 3.0, 1.0, 0.1)
    else:
        bt.sizing_mode = "fixed_dollar"
        bt.fixed_dollar = st.sidebar.number_input("$ notional at full forecast",
                                                  1_000.0, 1e9, capital, 10_000.0)
    bt.direction = st.sidebar.radio("Direction", ["both", "long", "short"],
                                    horizontal=True, help=HELP["direction"])

    sb("Execution delay", 5)
    bt.use_delay = st.sidebar.checkbox("Delay entries/exits", False, help=HELP["delay"])
    if bt.use_delay:
        bt.entry_delay = int(st.sidebar.number_input("Entry delay (bars)", 0, 100, 0, 1,
                             help="Bars to wait after a signal before opening/adding."))
        bt.exit_delay = int(st.sidebar.number_input("Exit delay (bars)", 0, 100, 0, 1,
                            help="Bars to wait after a signal before closing/trimming."))

    sb("Stops & take-profit", 6)
    risk = RiskConfig()
    stop_kind = st.sidebar.selectbox("Stop-loss", ["None", "ATR", "Percent"], help=HELP["stop"])
    if stop_kind == "ATR":
        risk.use_atr_stop = True
        risk.atr_stop_mult = st.sidebar.number_input("Stop k·ATR", 0.5, 15.0, 3.0, 0.5)
    elif stop_kind == "Percent":
        risk.use_pct_stop = True
        risk.pct_stop = st.sidebar.number_input("Stop %", 0.5, 50.0, 10.0, 0.5) / 100.0
    if stop_kind != "None":
        risk.trailing_stop = st.sidebar.checkbox("Trailing stop", True)
    tp_kind = st.sidebar.selectbox("Take-profit", ["None", "ATR", "Percent"], help=HELP["tp"])
    if tp_kind == "ATR":
        risk.use_atr_tp = True
        risk.atr_tp_mult = st.sidebar.number_input("TP k·ATR", 0.5, 25.0, 6.0, 0.5)
    elif tp_kind == "Percent":
        risk.use_pct_tp = True
        risk.pct_tp = st.sidebar.number_input("TP %", 0.5, 100.0, 20.0, 0.5) / 100.0
    if risk.use_atr_stop or risk.use_atr_tp:
        risk.atr_period = st.sidebar.number_input("ATR period", 5, 60, 14)

    sb("Costs", 0)
    bt.cost_bps = st.sidebar.number_input("Cost (bps)", 0.0, 20.0, 1.0, 0.5, help=HELP["cost"])
    bt.slippage_bps = st.sidebar.number_input("Slippage (bps)", 0.0, 20.0, 0.5, 0.5,
                                              help=HELP["slippage"])
    cfg["bt"], cfg["risk"], cfg["capital"] = bt, risk, capital
    return cfg


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_real(tickers, start, end):
    return {t: providers.get_prices(t, start=start or None, end=end or None) for t in tickers}


def load_data(cfg):
    if cfg["mode"].startswith("Real"):
        if not cfg["tickers"]:
            return {}
        data = load_real(tuple(cfg["tickers"]), cfg["start"], cfg["end"])
    else:
        data = synthetic.generate_panel(n_assets=int(cfg["n_assets"]), kind=cfg["kind"],
            n_days=int(cfg["n_days"]), seed=cfg["seed"], start=cfg["start"] or "2015-01-01")
    rule = TIMEFRAMES[cfg["tf"]]
    out = {}
    for t, df in data.items():
        d = resample_ohlcv(df, rule)
        if cfg["start"]:
            d = d[d.index >= pd.Timestamp(cfg["start"])]
        if cfg["end"]:
            d = d[d.index <= pd.Timestamp(cfg["end"])]
        out[t] = d
    return out


def add_trade_markers(fig, price, ledger):
    """Plot entry AND exit markers from the trade ledger so they match the table.

    Long entry ▲ / long exit ✕ (teal); short entry ▼ / short exit ✕ (coral).
    """
    if ledger is None or ledger.empty:
        return
    price = price.copy()
    price.index = pd.to_datetime(price.index)

    def _price_at(dates):
        idx = pd.to_datetime(pd.Series(list(dates)))
        return [float(price.reindex([d]).ffill().iloc[-1]) if d in price.index
                else float(price.asof(d)) for d in idx]

    longs = ledger[ledger["side"] == "LONG"]
    shorts = ledger[ledger["side"] == "SHORT"]
    # vertical offset so entry/exit markers at the same price don't overlap
    span = float(price.max() - price.min()) or 1.0
    off = span * 0.02
    specs = [
        # (dates, symbol, color, name, y-offset direction)
        (longs["entry_date"], "triangle-up", THEME.long_color, "Long entry", -off),
        (longs["exit_date"], "x-thin", THEME.long_color, "Long exit", +off),
        (shorts["entry_date"], "triangle-down", THEME.short_color, "Short entry", +off),
        (shorts["exit_date"], "x-thin", THEME.short_color, "Short exit", -off),
    ]
    for dates, sym, color, name, dy in specs:
        if len(dates) == 0:
            continue
        xs = pd.to_datetime(list(dates))
        ys = [y + dy for y in _price_at(dates)]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=name,
            marker=dict(symbol=sym, size=11, color=color,
                        line=dict(width=1.5, color="#fff"))))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    mode = st.sidebar.radio("◆ MODE", ["Strategy R&D", "Accumulation Lab", "ETF Lab"],
                            index=0,
                            help="Strategy R&D = systematic trading backtester. "
                                 "Accumulation Lab = long-term DCA / dip-buying. "
                                 "ETF Lab = design & backtest your own fund.")
    st.sidebar.markdown("<hr style='border-color:#2c2c2c'>", unsafe_allow_html=True)
    if mode == "Accumulation Lab":
        run_accum_lab()
    elif mode == "ETF Lab":
        run_etf_lab()
    else:
        run_rnd()


def run_rnd():
    cfg = sidebar()
    data = load_data(cfg)
    if not data:
        st.info("Select at least one ticker (or synthetic mode) to begin.")
        return
    if not cfg["selected"]:
        st.info("Activate at least one strategy in the sidebar.")
        return

    keys = [label_to_key(lbl) for lbl in cfg["selected"]]
    instrument = st.selectbox("Instrument", list(data))
    INSTRUMENT_HOLDER["name"] = instrument
    ohlcv = data[instrument]
    bt, risk, tf = cfg["bt"], cfg["risk"], cfg["tf"]

    def run_one(k):
        cls = REGISTRY[k]
        strat = make_strategy(cls, cfg["params"].get(k, {}), tf)
        fc = forecast_for(cls, strat, ohlcv, data)
        return run_single(fc, ohlcv, bt=bt, risk=risk, label=strat.describe()), strat

    bench_ret = ohlcv["close"].pct_change().fillna(0.0)

    tabs = st.tabs(["BACKTEST", "SPECTRUM", "DRAWDOWN", "MONTE CARLO", "KELLY", "DIAGNOSTICS"])

    # --- BACKTEST ----------------------------------------------------------
    with tabs[0]:
        results = {REGISTRY[k].label: run_one(k) for k in keys}
        log_eq = st.checkbox("Log scale", False, key="log_eq", help=HELP["logscale"])
        fig = go.Figure()
        for label, (res, _) in results.items():
            fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity, mode="lines", name=label))
        bench_eq = (1 + bench_ret).cumprod() * bt.capital
        fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, mode="lines",
            name=f"Buy&Hold {instrument}", line=dict(color=THEME.muted, dash="dot")))
        fig.update_layout(title=f"EQUITY CURVE — {instrument}")
        st.plotly_chart(style_fig(fig, log_y=log_eq), use_container_width=True)

        # --- Return summary (ROI / total profit / final equity) -------------
        st.markdown(section("Return summary", 4), unsafe_allow_html=True)
        bh_final = float(bench_eq.iloc[-1])
        summ_rows = []
        for label, (res, _) in results.items():
            fe = float(res.equity.iloc[-1])
            summ_rows.append({
                "Strategy": label,
                "Final equity": fe,
                "Total return %": (fe / bt.capital - 1.0) * 100,
                "Profit $": fe - bt.capital,
                "CAGR %": res.stats.get("CAGR", float("nan")) * 100,
                "Sharpe": res.stats.get("Sharpe", float("nan")),
                "Max DD %": res.stats.get("MaxDD", float("nan")) * 100,
                "vs Buy&Hold %": (fe / bh_final - 1.0) * 100 if bh_final else float("nan"),
            })
        summ = pd.DataFrame(summ_rows).set_index("Strategy")
        st.dataframe(summ.style.format({
            "Final equity": "${:,.0f}", "Total return %": "{:.1f}%", "Profit $": "${:,.0f}",
            "CAGR %": "{:.1f}%", "Sharpe": "{:.2f}", "Max DD %": "{:.1f}%",
            "vs Buy&Hold %": "{:+.1f}%"}), use_container_width=True)
        # Headline cards for the best strategy by final equity
        best_lbl = summ["Final equity"].idxmax()
        best = summ.loc[best_lbl]
        cc = st.columns(4)
        cc[0].metric("Final equity (best)", f"${best['Final equity']:,.0f}", best_lbl)
        cc[1].metric("Total return", f"{best['Total return %']:.1f}%")
        cc[2].metric("Profit", f"${best['Profit $']:,.0f}")
        cc[3].metric("vs Buy & Hold", f"{best['vs Buy&Hold %']:+.1f}%")

        st.markdown(section("Signals & indicators", 1), unsafe_allow_html=True)
        sig_label = st.selectbox("Strategy to inspect", list(results))
        sig_res, sig_strat = results[sig_label]
        ledger = extract_trades(sig_res.position, ohlcv["close"])
        log_px = st.checkbox("Log scale", False, key="log_px", help=HELP["logscale"])
        pfig = go.Figure()
        pfig.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv["close"], mode="lines",
            name=instrument, line=dict(color="#ffffff", width=1.5)))
        for i, (name, s) in enumerate(sig_strat.indicator_lines(ohlcv).items()):
            pfig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=name,
                line=dict(width=1, color=THEME.series[(i + 1) % len(THEME.series)]), opacity=0.85))
        add_trade_markers(pfig, ohlcv["close"], ledger)
        pfig.update_layout(title=f"{instrument} — price, indicators & entries/exits ({sig_label})")
        st.plotly_chart(style_fig(pfig, height=480, log_y=log_px), use_container_width=True)

        st.markdown(section("Trade details", 2), unsafe_allow_html=True)
        if ledger.empty:
            st.info("No trades for this strategy/period.")
        else:
            st.caption("Each row is one round-trip; markers on the chart above match these rows.")
            st.dataframe(ledger, use_container_width=True, height=260)
            st.download_button("Export trades CSV", ledger.to_csv(index=False),
                               file_name=f"trades_{instrument}_{sig_label}.csv")

        st.markdown(section("Metrics", 3), unsafe_allow_html=True)
        table = pd.DataFrame({lbl: r.stats for lbl, (r, _) in results.items()}).T
        table["FinalEquity"] = [r.equity.iloc[-1] for (r, _) in results.values()]
        st.dataframe(table.style.format("{:.3f}"), use_container_width=True)

    # --- SPECTRUM (+ beta/correlation) -------------------------------------
    with tabs[1]:
        st.markdown(section("Parameter sweep", 0), unsafe_allow_html=True)
        st.caption(HELP["sweep"])
        spec_key = st.selectbox("Strategy", keys, format_func=lambda k: REGISTRY[k].label)
        cls = REGISTRY[spec_key]
        if cls.cross_sectional:
            st.warning("Spectrum view supports single-instrument rules for now.")
        else:
            base = scale_params_for_tf(cfg["params"].get(spec_key, {}), tf)
            sweep_params = st.multiselect("Parameter(s) to sweep (pick 1 or 2)",
                list(cls.params), default=[cls.spectrum_param],
                help="One = bar chart; two = Sharpe heatmap.")
            if len(sweep_params) == 1:
                p = sweep_params[0]
                d, lo, hi, _ = cls.params[p]
                c1, c2, c3 = st.columns(3)
                v_lo = c1.number_input(f"{p} min", value=float(lo))
                v_hi = c2.number_input(f"{p} max", value=float(hi))
                n = c3.number_input("Variants", 4, 30, 12)
                vals = make_spectrum(v_lo, v_hi, int(n), integer=isinstance(d, int))
                fixed = {k: v for k, v in base.items() if k != p}
                res_map, tbl = run_spectrum(cls, ohlcv, param=p, values=vals, bt=bt,
                                            risk=risk, fixed_params=fixed)
                f = go.Figure()
                for lbl, res in res_map.items():
                    f.add_trace(go.Scatter(x=res.equity.index, y=res.equity, mode="lines",
                                           name=lbl, opacity=0.85))
                f.update_layout(title=f"SPECTRUM EQUITY — {cls.label} (sweep {p})")
                st.plotly_chart(style_fig(f), use_container_width=True)
                bar = go.Figure(go.Bar(x=tbl.index.astype(str), y=tbl["Sharpe"], marker_color=THEME.teal))
                bar.update_layout(title=f"SHARPE vs {p}")
                st.plotly_chart(style_fig(bar, height=320), use_container_width=True)
                st.dataframe(tbl.style.format("{:.3f}"), use_container_width=True)
            elif len(sweep_params) == 2:
                px, py = sweep_params[:2]
                dx, lox, hix, _ = cls.params[px]
                dy, loy, hiy, _ = cls.params[py]
                n = st.number_input("Grid size (per axis)", 4, 16, 8)
                vx = make_spectrum(float(lox), float(hix), int(n), integer=isinstance(dx, int))
                vy = make_spectrum(float(loy), float(hiy), int(n), integer=isinstance(dy, int))
                fixed = {k: v for k, v in base.items() if k not in (px, py)}
                grid = run_spectrum_2d(cls, ohlcv, px, vx, py, vy, bt=bt, risk=risk, fixed_params=fixed)
                hm = go.Figure(go.Heatmap(z=grid.values.astype(float),
                    x=[str(c) for c in grid.columns], y=[str(i) for i in grid.index],
                    colorscale="Viridis", colorbar=dict(title="Sharpe")))
                hm.update_layout(title=f"SHARPE HEATMAP — {px} (x) vs {py} (y)", xaxis_title=px, yaxis_title=py)
                st.plotly_chart(style_fig(hm, height=480), use_container_width=True)
            else:
                st.info("Select one or two parameters to sweep.")

        st.markdown(section("Beta & correlation to underlying", 2), unsafe_allow_html=True)
        st.caption(HELP["beta"])
        rows = []
        for k in keys:
            if REGISTRY[k].cross_sectional:
                continue
            r = run_one(k)[0]
            bc = beta_and_correlation(r.returns, ohlcv["close"])
            rows.append({"Strategy": REGISTRY[k].label, "Beta": bc["beta"], "Correlation": bc["correlation"]})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Strategy").style.format("{:.3f}"),
                         use_container_width=True)

    # --- DRAWDOWN ----------------------------------------------------------
    with tabs[2]:
        st.markdown(section("Drawdown history", 0), unsafe_allow_html=True)
        st.caption(HELP["drawdown"])
        ddfig = go.Figure()
        for k in keys:
            res = run_one(k)[0]
            dd = metrics.drawdown_series(res.returns)
            ddfig.add_trace(go.Scatter(x=dd.index, y=dd * 100, mode="lines", name=REGISTRY[k].label))
        bench_dd = metrics.drawdown_series(bench_ret)
        ddfig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd * 100, mode="lines",
            name=f"Buy&Hold {instrument}", line=dict(color=THEME.muted, dash="dot")))
        ddfig.update_layout(title="DRAWDOWN (%) vs UNDERLYING", yaxis_title="Drawdown %")
        st.plotly_chart(style_fig(ddfig), use_container_width=True)

        st.markdown(section("Risk & capture stats", 1), unsafe_allow_html=True)
        st.caption(HELP["capture"])
        bvol = metrics.annual_vol(bench_ret)
        rows = []
        for k in keys:
            res = run_one(k)[0]
            v = metrics.annual_vol(res.returns)
            rows.append({
                "Strategy": REGISTRY[k].label,
                "AnnVol %": v * 100,
                "Vol vs underlying": (v / bvol) if bvol > 1e-9 else 0.0,
                "MaxDD %": metrics.max_drawdown(res.returns) * 100,
                "Upside capture %": metrics.upside_capture(res.returns, bench_ret),
                "Downside capture %": metrics.downside_capture(res.returns, bench_ret),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Strategy").style.format("{:.2f}"),
                     use_container_width=True)

        st.markdown(section("Spectrum drawdown", 2), unsafe_allow_html=True)
        sk = st.selectbox("Strategy", [k for k in keys if not REGISTRY[k].cross_sectional] or keys,
                          format_func=lambda k: REGISTRY[k].label, key="ddspec")
        if not REGISTRY[sk].cross_sectional:
            cls = REGISTRY[sk]
            p = cls.spectrum_param
            d, lo, hi, _ = cls.params[p]
            vals = make_spectrum(float(lo), float(hi), 12, integer=isinstance(d, int))
            base = {kk: vv for kk, vv in scale_params_for_tf(cfg["params"].get(sk, {}), tf).items() if kk != p}
            _, tbl = run_spectrum(cls, ohlcv, param=p, values=vals, bt=bt, risk=risk, fixed_params=base)
            bar = go.Figure(go.Bar(x=tbl.index.astype(str), y=tbl["MaxDD"] * 100, marker_color=THEME.coral))
            bar.update_layout(title=f"MAX DRAWDOWN (%) vs {p}", yaxis_title="MaxDD %")
            st.plotly_chart(style_fig(bar, height=340), use_container_width=True)

    # --- MONTE CARLO -------------------------------------------------------
    with tabs[3]:
        st.markdown(section("Monte Carlo simulation", 0), unsafe_allow_html=True)
        st.caption(HELP["montecarlo"])
        with st.expander("How does this work? (methodology)"):
            st.markdown(
                "Monte Carlo here is a **block bootstrap of the strategy's realized "
                "daily returns** — not a re-run of the price model:\n\n"
                "1. Take the selected strategy's actual daily return series (real OR "
                "synthetic — same procedure either way).\n"
                "2. Resample it in contiguous **blocks** (default 20 bars) to preserve "
                "short-term autocorrelation, building one alternative ordering of the "
                "same returns.\n"
                "3. Repeat N times → a distribution of equity paths, Sharpe, drawdown, CAGR.\n\n"
                "It does **not** reseed synthetic data or invent new prices — it reshuffles "
                "the outcomes you actually got, answering: *how much of my result was "
                "ordering/luck?* A wide spread = luck-sensitive; a tight one = robust.")
        mk = st.selectbox("Strategy", [k for k in keys if not REGISTRY[k].cross_sectional] or keys,
                          format_func=lambda k: REGISTRY[k].label, key="mc")
        c1, c2, c3 = st.columns(3)
        n_sims = int(c1.number_input("Simulations", 50, 3000, 400, 50))
        block = int(c2.number_input("Block size (bars)", 1, 120, 20, 1,
                    help="Length of contiguous return blocks resampled (preserves short-term autocorrelation)."))
        show_paths = c3.number_input("Paths to draw", 0, 500, 120, 10,
                    help="How many simulated equity curves to overlay (visual only).")
        if not REGISTRY[mk].cross_sectional:
            res = run_one(mk)[0]
            paths = montecarlo.bootstrap_paths(res.returns, n_sims=n_sims, block=block,
                                               capital=bt.capital)
            if not paths.empty:
                pf = go.Figure()
                for col in list(paths.columns)[:int(show_paths)]:
                    pf.add_trace(go.Scatter(y=paths[col], mode="lines",
                        line=dict(width=0.6, color=THEME.teal), opacity=0.22, showlegend=False))
                pf.add_trace(go.Scatter(y=paths.median(axis=1), mode="lines",
                    name="median", line=dict(width=2.5, color=THEME.mustard)))
                pf.add_trace(go.Scatter(y=res.equity.reset_index(drop=True), mode="lines",
                    name="actual", line=dict(width=2.5, color=THEME.coral)))
                pf.update_layout(title=f"MONTE CARLO EQUITY PATHS ({n_sims} sims)")
                st.plotly_chart(style_fig(pf, height=460), use_container_width=True)

                boot = montecarlo.bootstrap(res.returns, n_sims=n_sims, block=block)
                hist = go.Figure(go.Histogram(x=boot["Sharpe"], nbinsx=40, marker_color=THEME.teal))
                hist.add_vline(x=res.stats["Sharpe"], line_color=THEME.coral, annotation_text="actual")
                hist.update_layout(title="BOOTSTRAP SHARPE DISTRIBUTION")
                st.plotly_chart(style_fig(hist, height=320), use_container_width=True)
                summ = montecarlo.summarize(boot)
                st.caption("Each metric shows **p5 / p50 / p95** across all simulations "
                           "— pessimistic / median / optimistic.")
                cols = st.columns(3)
                for i, (m, q) in enumerate(summ.items()):
                    cols[i % 3].metric(f"{m}  (p5 / p50 / p95)",
                                       f"{q['p5']:.2f} / {q['p50']:.2f} / {q['p95']:.2f}",
                                       help=HELP["mc_pct"])
            else:
                st.info("Not enough data for Monte Carlo on this selection.")

    # --- KELLY -------------------------------------------------------------
    with tabs[4]:
        st.markdown(section("Kelly position sizer", 0), unsafe_allow_html=True)
        st.caption(HELP["kelly"])
        kk = st.selectbox("Strategy", [k for k in keys if not REGISTRY[k].cross_sectional] or keys,
                          format_func=lambda k: REGISTRY[k].label, key="kelly")
        if not REGISTRY[kk].cross_sectional:
            res = run_one(kk)[0]
            kc = kelly.kelly_continuous(res.returns)
            c1, c2, c3 = st.columns(3)
            c1.metric("Full Kelly leverage", f"{kc['full']:.2f}×", help=HELP["kelly_full"])
            c2.metric("Half Kelly", f"{kc['half']:.2f}×", help=HELP["kelly_half"])
            c3.metric("Quarter Kelly", f"{kc['quarter']:.2f}×", help=HELP["kelly_quarter"])
            half_dollars = max(0.0, kc["half"]) * bt.capital
            st.write(f"Half-Kelly on ${bt.capital:,.0f} capital ≈ **${half_dollars:,.0f}** notional exposure.")
            st.caption("Continuous Kelly = mean/variance of returns. Full Kelly is "
                       "growth-optimal but very volatile; half/quarter Kelly is the practical choice.")

        st.markdown(section("Manual Kelly calculator", 1), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        wp = c1.slider("Win probability", 0.0, 1.0, 0.55, 0.01)
        wl = c2.number_input("Win/loss payoff ratio", 0.1, 10.0, 1.5, 0.1,
                             help="Average win size ÷ average loss size.")
        f = kelly.kelly_discrete(wp, wl)
        st.metric("Kelly fraction", f"{f*100:.1f}% of capital",
                  help="f* = p − (1−p)/b. Negative edges return 0%.")

    # --- DIAGNOSTICS -------------------------------------------------------
    with tabs[5]:
        st.markdown(section("Return correlation", 0), unsafe_allow_html=True)
        st.caption(HELP["corr"])
        chosen = st.multiselect("Strategies to compare", [REGISTRY[k].label for k in keys],
                                default=[REGISTRY[k].label for k in keys])
        underlyings = st.multiselect("Add underlyings (real tickers)",
                                     ["SPY", "QQQ", "IWM", "TLT", "GLD", "HYG", "UUP", "USO"],
                                     default=[], help=HELP["corr_under"])
        ck = [label_to_key(l) for l in chosen if not REGISTRY[label_to_key(l)].cross_sectional]
        series = {REGISTRY[k].label: run_one(k)[0].returns for k in ck}
        # Always include the current instrument as a baseline so a correlation
        # renders even with a single active strategy (strategy-vs-market is the
        # most useful comparison anyway).
        series[f"{instrument} (underlying)"] = bench_ret
        for u in underlyings:
            if u == instrument:
                continue
            try:
                up = providers.get_prices(u, start=cfg["start"] or None, end=cfg["end"] or None)
                # resample to the SELECTED timeframe so the series aligns with the
                # strategy/instrument returns (else weekly/monthly vs daily mismatch)
                up = resample_ohlcv(up, TIMEFRAMES[tf])
                if cfg["start"]:
                    up = up[up.index >= pd.Timestamp(cfg["start"])]
                if cfg["end"]:
                    up = up[up.index <= pd.Timestamp(cfg["end"])]
                series[u] = up["close"].pct_change().reindex(bench_ret.index)
            except Exception as e:
                st.warning(f"{u}: {e}")
        corr = pd.DataFrame(series).dropna(how="all").corr()
        if corr.shape[0] >= 2 and not corr.isna().all().all():
            labels = [str(c) for c in corr.columns]
            hm = go.Figure(go.Heatmap(z=corr.values, x=labels, y=labels,
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=np.round(corr.values, 2), texttemplate="%{text}",
                colorbar=dict(title="r")))
            hm.update_layout(title="RETURN CORRELATION")
            st.plotly_chart(style_fig(hm), use_container_width=True)
        else:
            st.info("Activate at least one strategy to see its correlation to the underlying.")

        st.markdown(section("Walk-forward (IS vs OOS)", 1), unsafe_allow_html=True)
        st.caption(HELP["walkforward"])
        st.caption(HELP["wf_color"])
        wf_keys = [k for k in keys if not REGISTRY[k].cross_sectional]
        if wf_keys:
            cwf1, cwf2 = st.columns(2)
            wf_key = cwf1.selectbox("Rule", wf_keys, format_func=lambda k: REGISTRY[k].label, key="wf")
            n_splits = int(cwf2.number_input("OOS windows (splits)", 2, 10, 4, 1,
                           help="How many sequential out-of-sample windows to test."))
            strat = make_strategy(REGISTRY[wf_key], cfg["params"].get(wf_key, {}), tf)
            wf = walk_forward(strat, ohlcv, n_splits=n_splits, bt=bt, risk=risk)
            if not wf.empty:
                def _is_color(v):
                    c = THEME.long_color if v > 0.5 else (THEME.mustard if v > 0 else THEME.short_color)
                    return f"color:{c}; font-weight:700"

                def _oos_row(row):
                    # OOS holds up if it keeps >=70% of a positive IS Sharpe
                    holds = row["OOS_Sharpe"] >= 0.7 * row["IS_Sharpe"] and row["IS_Sharpe"] > 0
                    col = THEME.long_color if holds else THEME.short_color
                    return ["", "", "", _is_color(row["IS_Sharpe"]), f"color:{col}; font-weight:700"]

                styler = (wf.style
                          .format({"IS_Sharpe": "{:.2f}", "OOS_Sharpe": "{:.2f}"})
                          .apply(_oos_row, axis=1))
                st.dataframe(styler, use_container_width=True)


ACCUM_HELP = {
    "lab": "Test long-term accumulation: a buy rule deploys cash on triggers (dips, VIX, RSI, MA touch, regression bands…) and is compared to fixed DCA and lump-sum buy & hold on the SAME contributions.",
    "initial": "Cash available at the start (your dry powder).",
    "contribution": "New cash added every cadence period (your ongoing savings).",
    "cadence": "How often you add the contribution and (for DCA) invest it.",
    "deploy": "Fraction of available cash deployed each time the buy rule fires (1.0 = go all-in on a signal).",
    "sell": "Fraction of holdings sold each time the (optional) sell rule fires.",
    "stats": "Beta/alpha/correlation are vs the underlying's own returns. Alpha = annualized excess return after removing beta·market.",
}


def _render_accum_params(rule_cls, prefix):
    vals = {}
    for name, (d, lo, hi, step) in rule_cls.params.items():
        is_int = isinstance(d, int) and isinstance(step, int)
        key = f"{prefix}_{rule_cls.key}_{name}"
        if is_int:
            vals[name] = int(st.number_input(name, int(lo), int(hi), int(d), int(step), key=key))
        else:
            vals[name] = float(st.number_input(name, float(lo), float(hi), float(d), float(step), key=key))
    return vals


def run_accum_lab():
    st.sidebar.markdown(section("Data source", 0), unsafe_allow_html=True)
    src_mode = st.sidebar.radio("Mode", ["Real (yfinance)", "Synthetic"], index=0,
                                key="acc_mode", help=HELP["mode"])
    cfg = {"mode": src_mode}
    if src_mode.startswith("Real"):
        grp = st.sidebar.selectbox("Universe", list(UNIVERSES), index=0, key="acc_uni")
        ticker = st.sidebar.selectbox("Ticker", UNIVERSES[grp], key="acc_tkr")
        custom = st.sidebar.text_input("Or type a ticker", "", key="acc_custom")
        if custom.strip():
            ticker = clean_ticker(custom)
        cfg["tickers"] = [ticker]
    else:
        cfg["kind"] = st.sidebar.selectbox("Synthetic kind",
            ["trending", "mean_reverting", "gbm", "regime", "fat_tailed"], key="acc_kind")
        cfg["n_days"] = st.sidebar.number_input("Days", 300, 6000, 2000, 100, key="acc_days")
        cfg["seed"] = int(st.sidebar.number_input("Seed", value=42, step=1, key="acc_seed"))
        cfg["n_assets"] = 1
        ticker = "SYNTH_1"

    st.sidebar.markdown(section("Timeframe", 1), unsafe_allow_html=True)
    c1, c2 = st.sidebar.columns(2)
    cfg["start"] = c1.text_input("Start", "2010-01-01", key="acc_start")
    cfg["end"] = c2.text_input("End", "", key="acc_end")
    cfg["tf"] = st.sidebar.selectbox("Candles", list(TIMEFRAMES), index=1, key="acc_tf")

    st.sidebar.markdown(section("Contributions", 2), unsafe_allow_html=True)
    acfg = AccumConfig()
    acfg.initial_cash = st.sidebar.number_input("Initial cash ($)", 0.0, 1e9, 10_000.0,
                                                1_000.0, help=ACCUM_HELP["initial"])
    acfg.contribution = st.sidebar.number_input("Contribution ($)", 0.0, 1e8, 1_000.0,
                                                100.0, help=ACCUM_HELP["contribution"])
    acfg.cadence = st.sidebar.selectbox("Cadence", ["Daily", "Weekly", "Monthly"], index=1,
                                        help=ACCUM_HELP["cadence"])
    dmode = st.sidebar.radio("Deploy per signal", ["% of cash", "Fixed $"],
                             help=ACCUM_HELP["deploy"])
    if dmode == "% of cash":
        acfg.deploy_mode = "pct_cash"
        acfg.deploy_fraction = st.sidebar.slider("% of available cash", 0.05, 1.0, 0.5, 0.05,
                                                 help="Hold the rest as dry powder for future signals.")
    else:
        acfg.deploy_mode = "fixed_dollar"
        acfg.deploy_dollar = st.sidebar.number_input("$ deployed per signal", 100.0, 1e8,
                                                     5_000.0, 500.0,
                                                     help="Spend a fixed dollar amount each signal; "
                                                          "the rest stays in cash.")
    acfg.min_signal_gap = int(st.sidebar.number_input("Min bars between buys", 0, 250, 0, 1,
                              help="Avoid buying on consecutive bars; 0 = no limit."))
    acfg.cash_yield_annual = st.sidebar.number_input("Idle cash yield (annual)", 0.0, 0.10,
                                                     0.0, 0.005,
                                                     help="Interest earned on un-deployed cash.")

    st.sidebar.markdown(section("Buy rule", 3), unsafe_allow_html=True)
    buy_label = st.sidebar.selectbox("Accumulation trigger",
        [c.label for c in accum_strats.BUY_RULES.values()],
        help="When to deploy cash. 'Fixed DCA' = always (baseline).")
    buy_cls = next(c for c in accum_strats.BUY_RULES.values() if c.label == buy_label)
    st.sidebar.caption(buy_cls.desc)
    with st.sidebar.expander("Buy parameters", expanded=True):
        buy_vals = _render_accum_params(buy_cls, "buy")

    st.sidebar.markdown(section("Sell rule (optional)", 4), unsafe_allow_html=True)
    sell_names = ["(none)"] + [c.label for c in accum_strats.SELL_RULES.values()]
    sell_label = st.sidebar.selectbox("Trim trigger", sell_names)
    sell_rule = None
    if sell_label != "(none)":
        sell_cls = next(c for c in accum_strats.SELL_RULES.values() if c.label == sell_label)
        st.sidebar.caption(sell_cls.desc)
        acfg.sell_fraction = st.sidebar.slider("Sell fraction per signal", 0.05, 1.0, 0.25,
                                               0.05, help=ACCUM_HELP["sell"])
        with st.sidebar.expander("Sell parameters", expanded=True):
            sell_vals = _render_accum_params(sell_cls, "sell")
        sell_rule = sell_cls(**sell_vals)

    # --- load data + VIX context ------------------------------------------
    try:
        data = load_data(cfg)
    except Exception as e:
        st.error(f"Data load failed: {e}")
        return
    if not data or ticker not in data:
        st.info("Select a valid ticker to begin.")
        return
    close = data[ticker]["close"].dropna()
    ctx = {}
    if buy_cls.key == "vix":
        try:
            ctx["vix"] = providers.get_prices("^VIX", start=cfg["start"] or None,
                                              end=cfg["end"] or None)["close"]
        except Exception:
            st.warning("Couldn't fetch ^VIX; VIX rule will not fire.")

    buy_rule = buy_cls(**buy_vals)
    res = run_accumulation(close, buy_rule, sell_rule, acfg, ctx)
    bench = benchmarks(close, acfg)

    st.markdown(section(f"Accumulation — {ticker}", 0), unsafe_allow_html=True)
    st.caption(ACCUM_HELP["lab"])

    t_eq, t_sig, t_dd, t_stats, t_reg = st.tabs(
        ["EQUITY", "SIGNALS", "DRAWDOWN", "STATS", "REGRESSION"])

    with t_eq:
        log_y = st.checkbox("Log scale", True, key="acc_log", help=HELP["logscale"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity, mode="lines",
                                 name=f"{buy_label}", line=dict(color=THEME.teal, width=2)))
        for name, eq in bench.items():
            fig.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name=name,
                                     line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=res.invested.index, y=res.invested, mode="lines",
                                 name="Contributed", line=dict(color=THEME.muted, dash="dash")))
        fig.add_trace(go.Scatter(x=res.deployed.index, y=res.deployed, mode="lines",
                                 name="Deployed (cost basis)", line=dict(color=THEME.mauve, dash="dot")))
        # the underlying asset's own price, on a secondary axis (right) so you can
        # see where the strategy bought relative to price moves
        fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines",
                                 name=f"{ticker} price (RHS)", yaxis="y2",
                                 line=dict(color=THEME.orange, width=1.2, dash="dot")))
        fig.update_layout(
            title=f"{ticker} — strategy vs DCA vs buy&hold",
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        title=f"{ticker} price ($)",
                        type="log" if log_y else "linear"))
        st.plotly_chart(style_fig(fig, log_y=log_y), use_container_width=True)

        # Dedicated P/L curve: profit = equity − money contributed. Flat at $0
        # until the first buy deploys capital, so the dry-powder phase is obvious.
        st.markdown(section("Profit / loss (vs money contributed)", 1), unsafe_allow_html=True)
        st.caption("P/L = equity − cash contributed. Stays flat at $0 until the buy "
                   "rule first deploys capital — then it reflects only real gains/losses.")
        plfig = go.Figure()
        plfig.add_trace(go.Scatter(x=res.profit.index, y=res.profit, mode="lines",
                                   name=f"{buy_label} P/L", line=dict(color=THEME.teal, width=2)))
        for name, eq in bench.items():
            plfig.add_trace(go.Scatter(x=eq.index, y=eq - res.invested.reindex(eq.index),
                                       mode="lines", name=f"{name} P/L", line=dict(dash="dot")))
        plfig.add_hline(y=0, line=dict(color=THEME.muted, width=1))
        plfig.update_layout(title=f"{ticker} — profit / loss ($)", yaxis_title="P/L $")
        st.plotly_chart(style_fig(plfig, height=320), use_container_width=True)

        c = st.columns(3)
        c[0].metric("Final equity", f"${res.stats['FinalEquity']:,.0f}")
        c[1].metric("Contributed", f"${res.stats['Contributed']:,.0f}")
        c[2].metric("Deployed (cost basis)", f"${res.stats['Deployed']:,.0f}")
        c2 = st.columns(3)
        c2[0].metric("Profit (P/L)", f"${res.stats['Profit']:,.0f}")
        c2[1].metric("Return on contributed", f"{res.stats['ReturnOnContributed%']:.1f}%",
                     help="Profit ÷ every dollar you saved (includes dry-powder drag).")
        c2[2].metric("Return on deployed", f"{res.stats['ReturnOnDeployed%']:.1f}%",
                     help="Return on capital actually put to work (excludes idle cash).")

    with t_sig:
        st.caption("Price with buy/sell signals, plus the indicator driving the "
                   "selected rule (e.g. RSI, VIX, drawdown) and its threshold.")
        log_s = st.checkbox("Log scale", True, key="acc_sig_log", help=HELP["logscale"])
        # Boolean masks aligned to the price index (numpy for safe positional use).
        buy_sig = (buy_rule.signal(close, ctx).reindex(close.index)
                   .fillna(False).astype(bool).to_numpy())
        sell_sig = (sell_rule.signal(close, ctx).reindex(close.index)
                    .fillna(False).astype(bool).to_numpy()
                    if sell_rule is not None else np.zeros(len(close), dtype=bool))
        n_buy, n_sell = int(buy_sig.sum()), int(sell_sig.sum())
        # Make "why are there no markers" obvious instead of silently blank.
        if buy_cls.key == "vix" and "vix" not in ctx:
            st.warning("VIX series unavailable (couldn't fetch ^VIX), so the VIX buy "
                       "rule can't fire. Try again or pick another rule.")
        st.caption(f"**{n_buy}** buy signal(s)"
                   + (f" · **{n_sell}** sell signal(s)" if sell_rule is not None else "")
                   + " over the selected window.")
        sp = float(close.max() - close.min()) or 1.0
        pricefig = go.Figure()
        pricefig.add_trace(go.Scatter(x=close.index, y=close, name=ticker,
                                      line=dict(color="#fff", width=1.4)))
        if n_buy:
            pricefig.add_trace(go.Scatter(x=close.index[buy_sig], y=close.to_numpy()[buy_sig] - sp * 0.02,
                mode="markers", name=f"Buy ({n_buy})", marker=dict(symbol="triangle-up", size=9,
                color=THEME.teal, line=dict(width=1, color="#fff"))))
        if n_sell:
            pricefig.add_trace(go.Scatter(x=close.index[sell_sig], y=close.to_numpy()[sell_sig] + sp * 0.02,
                mode="markers", name=f"Sell ({n_sell})", marker=dict(symbol="triangle-down", size=9,
                color=THEME.coral, line=dict(width=1, color="#fff"))))
        pricefig.update_layout(title=f"{ticker} — buy/sell signals")
        st.plotly_chart(style_fig(pricefig, height=380, log_y=log_s), use_container_width=True)
        # indicator sub-panel (RSI/VIX/drawdown/slope/...) for the active buy rule.
        # Guarded: a single misbehaving rule must never blank the whole tab.
        try:
            panel = buy_rule.panel_indicator(close, ctx)
        except Exception as e:  # noqa: BLE001
            panel = None
            st.warning(f"Indicator panel unavailable for this rule: {e}")
        if panel is not None:
            ser, ilabel, levels = panel
            ifig = go.Figure()
            ifig.add_trace(go.Scatter(x=ser.index, y=ser, name=ilabel, line=dict(color=THEME.mustard)))
            for val, lab in levels:
                ifig.add_hline(y=val, line=dict(color=THEME.coral, dash="dot"),
                               annotation_text=lab, annotation_position="right")
            ifig.update_layout(title=f"Indicator — {ilabel}")
            st.plotly_chart(style_fig(ifig, height=260), use_container_width=True)
        else:
            st.info("This buy rule is price-based (e.g. MA touch / regression) — see "
                    "the price chart above and the REGRESSION tab.")

    with t_dd:
        from ghost.backtest import metrics as M
        ddfig = go.Figure()
        ddfig.add_trace(go.Scatter(x=res.equity.index,
                                   y=M.drawdown_series(res.equity.pct_change().fillna(0)) * 100,
                                   mode="lines", name=buy_label, line=dict(color=THEME.teal)))
        for name, eq in bench.items():
            ddfig.add_trace(go.Scatter(x=eq.index,
                                       y=M.drawdown_series(eq.pct_change().fillna(0)) * 100,
                                       mode="lines", name=name, line=dict(dash="dot")))
        ddfig.update_layout(title="DRAWDOWN (%)", yaxis_title="Drawdown %")
        st.plotly_chart(style_fig(ddfig), use_container_width=True)

    with t_stats:
        st.caption(ACCUM_HELP["stats"])
        rows = {buy_label: res.stats}
        from ghost.accumulation.engine import _accum_stats
        for name, eq in bench.items():
            inv = res.invested  # same contribution schedule
            rows[name] = _accum_stats(eq, inv, close)  # benchmarks deploy everything
        tbl = pd.DataFrame(rows).T[["FinalEquity", "Contributed", "Deployed", "Profit",
                                    "ReturnOnContributed%", "ReturnOnDeployed%",
                                    "AnnVol%", "MaxDD%", "Beta", "Alpha(ann)%", "Corr"]]
        st.dataframe(tbl.style.format("{:.2f}"), use_container_width=True)

    with t_reg:
        st.caption("Linear/log regression channel with ±k·σ bands — buy near the "
                   "lower band, sell near the upper band.")
        c1, c2, c3 = st.columns(3)
        lb = int(c1.number_input("Lookback (bars)", 60, 2000, 504, 10, key="reg_lb"))
        k = float(c2.number_input("Std bands (k)", 0.5, 4.0, 2.0, 0.25, key="reg_k"))
        logfit = c3.checkbox("Log regression", True, key="reg_log")
        ch = regression_channel(close, lb, k, logfit)
        rfig = go.Figure()
        rfig.add_trace(go.Scatter(x=close.index, y=close, name=ticker,
                                  line=dict(color="#fff", width=1.5)))
        rfig.add_trace(go.Scatter(x=ch.index, y=ch["fit"], name="fit", line=dict(color=THEME.mustard)))
        rfig.add_trace(go.Scatter(x=ch.index, y=ch["upper"], name="+kσ (sell)", line=dict(color=THEME.coral, dash="dot")))
        rfig.add_trace(go.Scatter(x=ch.index, y=ch["lower"], name="−kσ (buy)", line=dict(color=THEME.teal, dash="dot")))
        rfig.update_layout(title=f"{ticker} — {'log' if logfit else 'linear'} regression channel")
        st.plotly_chart(style_fig(rfig, log_y=logfit), use_container_width=True)


# ----------------------------------------------------------------------------
# ETF Lab
# ----------------------------------------------------------------------------
ETF_UNIVERSES = ["Sector ETFs", "Broad ETFs", "Industry ETFs", "Futures-like ETFs",
                 "IWB (large-cap proxy)", "Stocks (500)"]
ETF_RANK_FACTORS = {
    "Momentum (12-1)": "momentum",
    "Trailing return": "trailing_return",
    "Prior calendar-year return": "calendar_return",
    "Low volatility": "low_volatility",
    "High volatility": "volatility",
    "High beta": "beta",
    "Low beta": "low_beta",
    "Shallowest drawdown": "max_drawdown",
}
ETF_WEIGHTS = {"Equal weight": "equal", "Inverse volatility": "inverse_vol",
               "Market cap (snapshot)": "market_cap", "Manual": "manual"}
ETF_BENCH = ["SPY", "QQQ", "IWM", "EFA", "AGG", "TLT", "GLD"]


@st.cache_data(show_spinner="Loading prices…")
def _etf_panel(tickers, start, end):
    from ghost.data import providers
    return providers.get_panel(tuple(sorted(set(tickers))), start=start or None,
                               end=end or None, field="close")


def _etf_sidebar():
    st.sidebar.markdown(section("Fund design", 0), unsafe_allow_html=True)
    use_preset = st.sidebar.radio("Start from", ["Preset", "Build your own"],
                                  horizontal=True, key="etf_src")
    spec = None
    if use_preset == "Preset":
        names = [p.name for p in etf_presets.PRESETS if p.enabled]
        choice = st.sidebar.selectbox("Example fund", names, key="etf_preset")
        spec = etf_presets.by_name(choice)
        st.sidebar.caption(spec.notes)
        disabled = [p.name for p in etf_presets.PRESETS if not p.enabled]
        if disabled:
            st.sidebar.caption("⏳ Needs Phase-2 fundamentals: " + "; ".join(disabled))
    else:
        sel_mode = st.sidebar.radio("Holdings", ["Explicit basket", "Screen & rank"],
                                    key="etf_selmode")
        weighting = ETF_WEIGHTS[st.sidebar.selectbox("Weighting", list(ETF_WEIGHTS),
                                                     key="etf_w")]
        max_w = st.sidebar.slider("Max weight per name", 0.05, 1.0, 1.0, 0.05, key="etf_maxw")
        rebal = st.sidebar.selectbox("Rebalance", ["Weekly", "Monthly", "Quarterly",
                                     "Biannual", "Annual"], index=1, key="etf_rebal")
        manual = None
        if sel_mode == "Explicit basket":
            grp = st.sidebar.selectbox("Pick from", ETF_UNIVERSES, key="etf_eu")
            opts = etf_screens.resolve_universe(grp)
            picks = st.sidebar.multiselect("Holdings", opts, default=opts[:4], key="etf_picks")
            custom = st.sidebar.text_input("Add tickers (comma-sep)", "", key="etf_custom")
            for raw in custom.split(","):
                if raw.strip():
                    try:
                        picks.append(clean_ticker(raw))
                    except ValueError:
                        pass
            picks = list(dict.fromkeys(picks))
            if weighting == "manual" and picks:
                manual = {}
                st.sidebar.caption("Relative manual weights:")
                for t in picks:
                    manual[t] = st.sidebar.number_input(t, 0.0, 100.0, 1.0, 0.5, key=f"etf_mw_{t}")
            sel = etf_screens.SelectionSpec(explicit=picks)
            direction = "long"
        else:
            uni = st.sidebar.selectbox("Universe", ETF_UNIVERSES, key="etf_uni")
            if uni in ("IWB (large-cap proxy)", "Stocks (500)"):
                st.sidebar.caption("⚠ First load fetches the whole universe — can take "
                                   "a minute (then cached).")
            rf = ETF_RANK_FACTORS[st.sidebar.selectbox("Rank by", list(ETF_RANK_FACTORS),
                                                       key="etf_rf")]
            lookback = int(st.sidebar.number_input("Lookback (trading days)", 21, 1000,
                                                   252, 21, key="etf_lb"))
            direction = st.sidebar.radio("Direction", ["long", "short", "long_short"],
                                         horizontal=True, key="etf_dir")
            top_n = int(st.sidebar.number_input("# long holdings", 1, 100, 10, 1, key="etf_topn"))
            bottom_n = None
            if direction in ("short", "long_short"):
                bottom_n = int(st.sidebar.number_input("# short holdings", 1, 100, 10, 1,
                                                       key="etf_botn"))
            sel = etf_screens.SelectionSpec(universe=uni, rank_factor=rf,
                                            rank_lookback=lookback, top_n=top_n,
                                            bottom_n=bottom_n)
        spec = etf_screens.ETFSpec(name="Custom fund", selection=sel, weighting=weighting,
                                   rebalance=rebal, direction=direction,
                                   manual_weights=manual, max_weight=max_w)

    st.sidebar.markdown(section("Costs & fees", 1), unsafe_allow_html=True)
    cap = st.sidebar.number_input("Backtest capital ($)", 10_000.0, 1e10, 1_000_000.0,
                                  10_000.0, key="etf_cap")
    cost = st.sidebar.number_input("Commission (bps)", 0.0, 50.0, 1.0, 0.5, key="etf_cost")
    slip = st.sidebar.number_input("Slippage (bps)", 0.0, 50.0, 0.5, 0.5, key="etf_slip")
    er = st.sidebar.number_input("Expense ratio (%/yr)", 0.0, 3.0, 0.40, 0.05,
                                 key="etf_er") / 100.0
    borrow = st.sidebar.number_input("Short borrow (bps/yr)", 0.0, 1000.0, 50.0, 10.0,
                                     key="etf_borrow") if spec.direction != "long" else 0.0
    cfg = etf_pf.PortfolioConfig(capital=cap, rebalance=spec.rebalance, cost_bps=cost,
                                 slippage_bps=slip, expense_ratio_annual=er,
                                 borrow_bps=borrow, max_weight=spec.max_weight,
                                 direction=spec.direction)

    st.sidebar.markdown(section("Timeframe & benchmark", 2), unsafe_allow_html=True)
    c1, c2 = st.sidebar.columns(2)
    start = c1.text_input("Start", "2015-01-01", key="etf_start")
    end = c2.text_input("End", "", key="etf_end")
    bench = st.sidebar.multiselect("Benchmarks", ETF_BENCH, default=["SPY"], key="etf_bench")
    return spec, cfg, start, end, bench


def run_etf_lab():
    spec, cfg, start, end, bench = _etf_sidebar()
    st.markdown(section(f"ETF Lab — {spec.name}", 0), unsafe_allow_html=True)
    st.caption("Design a fund (pick holdings or screen & rank a universe), backtest it "
               "with realistic costs & expense ratio, and compare it to benchmarks.")
    st.warning("⚠ Free-data limits: the selection universe is **current** (survivorship "
               "bias on historical backtests). Price/return/vol/beta factors are valid; "
               "market-cap weighting uses a **current snapshot**.")

    # assemble the ticker universe to load
    need = set(etf_screens.resolve_universe(spec.selection.universe))
    if spec.selection.explicit:
        need = set(spec.selection.explicit)
    need |= set(bench) | {"SPY"}
    try:
        panel = _etf_panel(tuple(need), start, end)
    except Exception as e:
        st.error(f"Price load failed: {e}")
        return
    if panel.empty:
        st.info("No price data for this selection.")
        return
    market_close = panel["SPY"] if "SPY" in panel else panel.iloc[:, 0]

    rebal_dts = etf_pf.rebalance_dates(panel.index, spec.rebalance)
    ws = etf_screens.build_weight_schedule(spec, panel, rebal_dts, market_close)
    if (ws.abs().sum(axis=1) == 0).all():
        st.warning("This design selects no holdings over the chosen window — widen the "
                   "universe, lower the lookback, or extend the dates.")
        return
    res = etf_pf.run_portfolio(panel, ws, cfg)

    t_design, t_back, t_cmp, t_overlap, t_econ = st.tabs(
        ["DESIGN", "BACKTEST", "COMPARE", "OVERLAP", "FUND ECONOMICS"])

    # --- DESIGN: current holdings + weights ------------------------------------
    with t_design:
        st.markdown(section("Current holdings (latest rebalance)", 0), unsafe_allow_html=True)
        latest = ws.iloc[-1]
        latest = latest[latest != 0].sort_values(key=abs, ascending=False)
        if latest.empty:
            st.info("No holdings at the latest rebalance.")
        else:
            hold = pd.DataFrame({"Weight %": (latest * 100).round(2),
                                 "Side": np.where(latest > 0, "LONG", "SHORT")})
            st.dataframe(hold, use_container_width=True, height=320)
            wfig = go.Figure(go.Bar(x=latest.index.astype(str), y=latest.values * 100,
                marker_color=np.where(latest > 0, THEME.teal, THEME.coral)))
            wfig.update_layout(title="HOLDING WEIGHTS (%)", yaxis_title="Weight %")
            st.plotly_chart(style_fig(wfig, height=320), use_container_width=True)
        st.caption(f"Universe loaded: {panel.shape[1]} tickers · {len(rebal_dts)} "
                   f"{spec.rebalance.lower()} rebalances · weighting: {spec.weighting}.")

    # --- BACKTEST --------------------------------------------------------------
    with t_back:
        log_y = st.checkbox("Log scale", False, key="etf_log", help=HELP["logscale"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity, mode="lines",
                                 name=spec.name, line=dict(color=THEME.teal, width=2)))
        for b in bench:
            if b in panel:
                beq = (1 + panel[b].pct_change().fillna(0)).cumprod() * cfg.capital
                fig.add_trace(go.Scatter(x=beq.index, y=beq, mode="lines",
                              name=f"{b} (B&H)", line=dict(dash="dot")))
        fig.update_layout(title=f"EQUITY CURVE — {spec.name}")
        st.plotly_chart(style_fig(fig, log_y=log_y), use_container_width=True)

        cc = st.columns(4)
        cc[0].metric("Final equity", f"${res.stats['FinalEquity']:,.0f}")
        cc[1].metric("Total return", f"{res.stats['TotalReturn%']:.1f}%")
        cc[2].metric("CAGR", f"{res.stats['CAGR']*100:.1f}%")
        cc[3].metric("Sharpe", f"{res.stats['Sharpe']:.2f}")
        cc2 = st.columns(4)
        cc2[0].metric("Ann. vol", f"{res.stats['AnnVol%']:.1f}%")
        cc2[1].metric("Max drawdown", f"{res.stats['MaxDD']*100:.1f}%")
        cc2[2].metric("Avg turnover/rebal", f"{res.stats['AvgTurnover']*100:.0f}%")
        cc2[3].metric("Total cost paid", f"{res.stats['TotalCost%']:.2f}%")

        st.markdown(section("Weights over time", 1), unsafe_allow_html=True)
        nz = ws.columns[(ws != 0).any()]
        afig = go.Figure()
        for i, t in enumerate(nz):
            wseries = res.weights[t].reindex(res.weights.index).fillna(0) * 100
            afig.add_trace(go.Scatter(x=wseries.index, y=wseries, mode="lines",
                stackgroup=None, name=t,
                line=dict(width=1, color=THEME.series[i % len(THEME.series)])))
        afig.update_layout(title="HOLDING WEIGHTS OVER TIME (%)", yaxis_title="Weight %")
        st.plotly_chart(style_fig(afig, height=360), use_container_width=True)

        st.markdown(section("Drawdown", 2), unsafe_allow_html=True)
        dd = metrics.drawdown_series(res.returns) * 100
        ddf = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                        line=dict(color=THEME.coral)))
        ddf.update_layout(title="DRAWDOWN (%)", yaxis_title="Drawdown %")
        st.plotly_chart(style_fig(ddf, height=300), use_container_width=True)

    # --- COMPARE ---------------------------------------------------------------
    with t_cmp:
        st.markdown(section("Fund vs benchmarks", 0), unsafe_allow_html=True)
        series = {spec.name: res.returns}
        for b in bench:
            if b in panel:
                series[b] = panel[b].pct_change()
        rows = []
        bench_ret0 = next((panel[b].pct_change() for b in bench if b in panel),
                          market_close.pct_change())
        for name, r in series.items():
            r = r.dropna()
            rows.append({
                "Fund": name, "CAGR %": metrics.cagr(r) * 100,
                "Sharpe": metrics.sharpe(r), "Ann Vol %": metrics.annual_vol(r) * 100,
                "MaxDD %": metrics.max_drawdown(r) * 100,
                "Up capture %": metrics.upside_capture(r, bench_ret0),
                "Down capture %": metrics.downside_capture(r, bench_ret0),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Fund").style.format("{:.2f}"),
                     use_container_width=True)

        st.markdown(section("Return correlation", 1), unsafe_allow_html=True)
        corr = pd.DataFrame(series).dropna(how="all").corr()
        if corr.shape[0] >= 2:
            labels = [str(c) for c in corr.columns]
            hm = go.Figure(go.Heatmap(z=corr.values, x=labels, y=labels, colorscale="RdBu",
                zmid=0, zmin=-1, zmax=1, text=np.round(corr.values, 2),
                texttemplate="%{text}", colorbar=dict(title="r")))
            hm.update_layout(title="RETURN CORRELATION")
            st.plotly_chart(style_fig(hm), use_container_width=True)

    # --- OVERLAP ---------------------------------------------------------------
    with t_overlap:
        st.markdown(section("Holdings overlap", 0), unsafe_allow_html=True)
        st.caption("Weighted overlap (Σ min weight) between your fund and each enabled "
                   "preset. Third-party ETF holdings aren't free, so this compares funds "
                   "built here.")
        my_w = {t: float(v) for t, v in ws.iloc[-1].items() if v != 0}
        st.markdown("**Your fund's current holdings**")
        st.dataframe(pd.Series({t: round(w * 100, 2) for t, w in my_w.items()},
                     name="Weight %").to_frame(), use_container_width=True, height=240)
        compare = st.checkbox("Compare overlap against the preset funds "
                              "(fetches their universes — may be slow)", False,
                              key="etf_overlap_go")
        baskets = {spec.name: my_w}
        for p in (etf_presets.PRESETS if compare else []):
            if not p.enabled:
                continue
            try:
                pneed = set(etf_screens.resolve_universe(p.selection.universe))
                if p.selection.explicit:
                    pneed = set(p.selection.explicit)
                ppanel = _etf_panel(tuple(pneed | {"SPY"}), start, end)
                prd = etf_pf.rebalance_dates(ppanel.index, p.rebalance)
                pws = etf_screens.build_weight_schedule(p, ppanel, prd[-1:],
                        ppanel["SPY"] if "SPY" in ppanel else ppanel.iloc[:, 0])
                baskets[p.name] = {t: float(v) for t, v in pws.iloc[-1].items() if v != 0}
            except Exception:
                continue
        if len(baskets) >= 2:
            m = etf_overlap.overlap_matrix(baskets)
            hm = go.Figure(go.Heatmap(z=m.values.astype(float),
                x=[str(c) for c in m.columns], y=[str(i) for i in m.index],
                colorscale="Viridis", zmin=0, zmax=1, text=np.round(m.values.astype(float), 2),
                texttemplate="%{text}"))
            hm.update_layout(title="WEIGHTED HOLDINGS OVERLAP")
            st.plotly_chart(style_fig(hm, height=420), use_container_width=True)
        else:
            st.info("Need at least two funds to compare overlap.")

    # --- FUND ECONOMICS --------------------------------------------------------
    with t_econ:
        st.markdown(section("If you ran this fund…", 0), unsafe_allow_html=True)
        st.caption("Issuer economics — all figures are planning estimates.")
        c1, c2, c3 = st.columns(3)
        aum = c1.number_input("AUM ($)", 1e6, 1e12, 100e6, 1e6, key="etf_aum")
        er_pct = c2.number_input("Expense ratio (%/yr)", 0.0, 3.0,
                                 cfg.expense_ratio_annual * 100, 0.05, key="etf_econ_er") / 100
        growth = c3.number_input("Annual AUM growth (%)", -50.0, 200.0, 10.0, 5.0,
                                 key="etf_growth") / 100
        c4, c5, c6 = st.columns(3)
        startup = c4.number_input("Startup cost ($)", 0.0, 1e8, 400_000.0, 50_000.0, key="etf_startup")
        maint = c5.number_input("Annual maintenance ($)", 0.0, 1e8, 150_000.0, 25_000.0, key="etf_maint")
        lic = c6.number_input("Index licensing (bps)", 0.0, 50.0, 3.0, 0.5, key="etf_lic")
        econ_cfg = etf_econ.FundEconConfig(aum=aum, expense_ratio=er_pct,
            expected_aum_growth=growth, startup_cost=startup, annual_maintenance=maint,
            index_licensing_bps=lic, num_holdings=int((ws.iloc[-1] != 0).sum()))
        years = int(st.slider("Years", 1, 15, 5, key="etf_years"))
        edf = etf_econ.economics(econ_cfg, years)
        ec = st.columns(3)
        ec[0].metric("Yr-1 fee revenue", f"${edf.loc[1,'Revenue']:,.0f}")
        ec[1].metric(f"Cumulative net (yr {years})", f"${edf.loc[years,'CumNetProfit']:,.0f}")
        be = etf_econ.breakeven_aum(econ_cfg)
        ec[2].metric("Breakeven AUM", "∞" if be == float("inf") else f"${be:,.0f}")
        st.dataframe(edf.style.format("${:,.0f}"), use_container_width=True)
        st.markdown(section("Investor fee drag", 1), unsafe_allow_html=True)
        drag = etf_econ.investor_fee_drag(max(res.stats["CAGR"], 0.0), er_pct, years)
        dc = st.columns(3)
        dc[0].metric("Gross CAGR", f"{res.stats['CAGR']*100:.1f}%")
        dc[1].metric("Net of fees", f"{drag['net_return_annual']*100:.1f}%")
        dc[2].metric(f"Fee cost / $10k ({years}y)", f"${drag['fee_cost']:,.0f}")


if __name__ == "__main__":
    main()
