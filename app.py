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
from ghost.config import BacktestConfig, RiskConfig, THEME
from ghost.data import providers, synthetic
from ghost.data.providers import TIMEFRAMES, resample_ohlcv
from ghost.data.universe import UNIVERSES, master_categories, master_tickers
from ghost.backtest.engine import run_single
from ghost.backtest.spectrum import run_spectrum, run_spectrum_2d, make_spectrum
from ghost.backtest.diagnostics import (
    return_correlation, walk_forward, beta_and_correlation,
)
from ghost.backtest.trades import extract_trades
from ghost.backtest import montecarlo
from ghost.data.providers import clean_ticker
from ghost.auth import require_password
from ghost.ui_theme import CSS, BANNER, section, style_fig

st.set_page_config(page_title="PARALLAX // quant lab", layout="wide", page_icon="◧")
st.markdown(CSS, unsafe_allow_html=True)

# Password gate (open locally when no secret is set; required once deployed).
require_password()

st.markdown(BANNER, unsafe_allow_html=True)

# Resource caps to keep a public instance responsive under load.
MAX_TICKERS = 12

# Hover-help definitions for features.
HELP = {
    "mode": "Real pulls actual price history from Yahoo Finance. Synthetic generates fake data with known behavior to validate strategies.",
    "vol_target": "Volatility targeting scales each position so the strategy aims for a constant risk level (target vol). Turn off for fixed-notional sizing where high-vol periods carry more risk.",
    "target_vol": "Annualized volatility the strategy targets, e.g. 0.20 = 20%/yr. Higher = bigger positions and swings.",
    "direction": "Restrict trades: 'both' allows long & short, 'long' only buys, 'short' only sells.",
    "atr_stop": "Average True Range stop-loss: exit when price moves k×ATR against the entry. ATR adapts the stop to recent volatility.",
    "atr_tp": "ATR take-profit: exit when price moves k×ATR in your favor.",
    "timeframe": "Bar frequency for the backtest. Resamples daily data into weekly/monthly candles.",
    "cost": "Transaction cost charged on traded notional, in basis points (1 bp = 0.01%).",
    "slippage": "Estimated execution slippage (half-spread) in basis points.",
    "sweep": "Run the strategy across a range of one (or two) parameters at once. Look for a broad plateau of good performance (robust) rather than a single lucky spike (overfit).",
    "seed": "Random seed for synthetic data — change it to draw a different fake price path.",
}

PARAM_HELP = {
    "fast": "Fast (short) moving-average span — reacts quickly to price.",
    "slow": "Slow (long) moving-average span — the trend baseline.",
    "lookback": "Number of bars the rule looks back over.",
    "skip": "Most-recent bars to skip (controls short-term reversal).",
    "smooth": "Smoothing window applied to the raw signal.",
    "speed": "Scales all Guppy ribbon spans up/down together.",
}


def label_to_key(label: str) -> str:
    for k, cls in REGISTRY.items():
        if cls.label == label:
            return k
    raise KeyError(label)


def render_param_controls(cls, prefix: str) -> dict:
    """Typeable number inputs (with help tooltips) for every parameter."""
    values = {}
    for name, (default, lo, hi, step) in cls.params.items():
        is_int = isinstance(default, int) and isinstance(step, int)
        key = f"{prefix}_{cls.key}_{name}"
        help_txt = PARAM_HELP.get(name, f"{name} parameter")
        if is_int:
            values[name] = int(st.number_input(
                name, min_value=int(lo), max_value=int(hi), value=int(default),
                step=int(step), key=key, help=help_txt))
        else:
            values[name] = float(st.number_input(
                name, min_value=float(lo), max_value=float(hi), value=float(default),
                step=float(step), key=key, help=help_txt))
    return values


def instantiate(cls, values: dict):
    return cls(**values)


INSTRUMENT_HOLDER = {"name": None}


def forecast_for(cls, values, ohlcv, panel):
    strat = instantiate(cls, values)
    if cls.cross_sectional:
        return strat.forecast_panel(panel)[INSTRUMENT_HOLDER["name"]], strat
    return strat.forecast(ohlcv), strat


def sb_header(label, idx):
    st.sidebar.markdown(section(label, idx), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def sidebar():
    cfg = {}
    sb_header("Data source", 0)
    cfg["mode"] = st.sidebar.radio("Mode", ["Real (yfinance)", "Synthetic"],
                                   index=0, help=HELP["mode"])

    if cfg["mode"].startswith("Real"):
        src = st.sidebar.radio("Ticker source", ["Curated", "ETF master list"], index=0)
        if src == "Curated":
            uni = st.sidebar.selectbox("Universe", list(UNIVERSES), index=0)
            options = UNIVERSES[uni]
        else:
            cat = st.sidebar.selectbox("Category", master_categories())
            options = master_tickers(cat)
        # never auto-select multiple: default to the first single ticker
        cfg["tickers"] = st.sidebar.multiselect(
            "Tickers", options, default=options[:1] if options else [],
            help="Pick one or more tickers. Defaults to a single ticker.")
        custom = st.sidebar.text_input("Add tickers (comma-sep)", "")
        if custom.strip():
            for raw in custom.split(","):
                if raw.strip():
                    try:
                        cfg["tickers"].append(clean_ticker(raw))
                    except ValueError:
                        st.sidebar.warning(f"Ignored invalid ticker: {raw.strip()!r}")
        # de-dupe and cap the number of tickers for responsiveness
        cfg["tickers"] = list(dict.fromkeys(cfg["tickers"]))[:MAX_TICKERS]
    else:
        cfg["kind"] = st.sidebar.selectbox(
            "Synthetic kind",
            ["trending", "mean_reverting", "gbm", "regime", "fat_tailed"])
        cfg["n_days"] = st.sidebar.number_input("Days", 300, 6000, 1500, 100)
        cfg["seed"] = int(st.sidebar.number_input("Seed", value=42, step=1, help=HELP["seed"]))
        cfg["n_assets"] = st.sidebar.number_input("Synthetic assets", 1, 12, 4)

    sb_header("Timeframe", 1)
    c1, c2 = st.sidebar.columns(2)
    cfg["start"] = c1.text_input("Start", "2015-01-01")
    cfg["end"] = c2.text_input("End", "")
    cfg["tf"] = st.sidebar.selectbox("Bar frequency", list(TIMEFRAMES), index=0,
                                     help=HELP["timeframe"])

    sb_header("Strategies", 2)
    cfg["selected"] = st.sidebar.multiselect(
        "Active rules", [REGISTRY[k].label for k in sorted(REGISTRY)],
        default=[REGISTRY["ema"].label])
    cfg["params"] = {}
    for lbl in cfg["selected"]:
        k = label_to_key(lbl)
        with st.sidebar.expander(f"{lbl} parameters", expanded=False):
            cfg["params"][k] = render_param_controls(REGISTRY[k], prefix="bt")

    sb_header("Position sizing", 3)
    bt = BacktestConfig()
    bt.use_vol_target = st.sidebar.checkbox("Volatility targeting", True,
                                            help=HELP["vol_target"])
    if bt.use_vol_target:
        bt.target_vol = st.sidebar.slider("Target vol", 0.05, 0.50, 0.20, 0.01,
                                          help=HELP["target_vol"])
    bt.direction = st.sidebar.radio("Direction", ["both", "long", "short"],
                                    horizontal=True, help=HELP["direction"])

    sb_header("Risk overlay (ATR)", 4)
    risk = RiskConfig()
    risk.use_atr_stop = st.sidebar.checkbox("ATR stop-loss", False, help=HELP["atr_stop"])
    risk.use_atr_tp = st.sidebar.checkbox("ATR take-profit", False, help=HELP["atr_tp"])
    if risk.use_atr_stop or risk.use_atr_tp:
        risk.atr_period = st.sidebar.number_input("ATR period", 5, 60, 14)
        risk.atr_stop_mult = st.sidebar.number_input("Stop k·ATR", 1.0, 10.0, 3.0, 0.5)
        risk.atr_tp_mult = st.sidebar.number_input("TP k·ATR", 1.0, 20.0, 6.0, 0.5)
        risk.trailing_stop = st.sidebar.checkbox("Trailing stop", True)

    sb_header("Costs", 5)
    bt.cost_bps = st.sidebar.number_input("Cost (bps)", 0.0, 20.0, 1.0, 0.5, help=HELP["cost"])
    bt.slippage_bps = st.sidebar.number_input("Slippage (bps)", 0.0, 20.0, 0.5, 0.5,
                                              help=HELP["slippage"])
    cfg["bt"], cfg["risk"] = bt, risk
    return cfg


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_real(tickers, start, end):
    return {t: providers.get_prices(t, start=start or None, end=end or None)
            for t in tickers}


def load_data(cfg) -> dict[str, pd.DataFrame]:
    if cfg["mode"].startswith("Real"):
        if not cfg["tickers"]:
            return {}
        data = load_real(tuple(cfg["tickers"]), cfg["start"], cfg["end"])
    else:
        data = synthetic.generate_panel(
            n_assets=int(cfg["n_assets"]), kind=cfg["kind"],
            n_days=int(cfg["n_days"]), seed=cfg["seed"],
            start=cfg["start"] or "2015-01-01")
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


def add_trade_markers(fig, price, position):
    sign = np.sign(position).fillna(0.0)
    prev = sign.shift(1).fillna(0.0)
    le = (sign > 0) & (prev <= 0)
    se = (sign < 0) & (prev >= 0)
    fig.add_trace(go.Scatter(
        x=price.index[le], y=price[le], mode="markers", name="Long entry",
        marker=dict(symbol="triangle-up", size=12, color=THEME.long_color,
                    line=dict(width=1, color="#fff"))))
    fig.add_trace(go.Scatter(
        x=price.index[se], y=price[se], mode="markers", name="Short entry",
        marker=dict(symbol="triangle-down", size=12, color=THEME.short_color,
                    line=dict(width=1, color="#fff"))))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
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
    bt, risk = cfg["bt"], cfg["risk"]

    def run_one(k):
        cls = REGISTRY[k]
        fc, strat = forecast_for(cls, cfg["params"].get(k, {}), ohlcv, data)
        res = run_single(fc, ohlcv, bt=bt, risk=risk, label=strat.describe())
        return res, strat

    tab_run, tab_spec, tab_diag, tab_mc = st.tabs(
        ["BACKTEST", "SPECTRUM", "DIAGNOSTICS", "MONTE CARLO"])

    # --- Backtest ----------------------------------------------------------
    with tab_run:
        results = {REGISTRY[k].label: run_one(k) for k in keys}

        fig = go.Figure()
        for label, (res, _) in results.items():
            fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity,
                                     mode="lines", name=label))
        bench = (1 + ohlcv["close"].pct_change().fillna(0)).cumprod() * bt.capital
        fig.add_trace(go.Scatter(x=bench.index, y=bench, mode="lines",
                                 name=f"Buy&Hold {instrument}",
                                 line=dict(color=THEME.muted, dash="dot")))
        fig.update_layout(title=f"EQUITY CURVE — {instrument}")
        st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown(section("Signals & indicators", 0), unsafe_allow_html=True)
        sig_label = st.selectbox("Strategy to inspect", list(results))
        sig_res, sig_strat = results[sig_label]
        pfig = go.Figure()
        pfig.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv["close"], mode="lines",
                                  name=instrument, line=dict(color="#ffffff", width=1.5)))
        # indicator overlays (MA lines / guppy / bands)
        lines = sig_strat.indicator_lines(ohlcv)
        for i, (name, series) in enumerate(lines.items()):
            color = THEME.series[(i + 1) % len(THEME.series)]
            pfig.add_trace(go.Scatter(x=series.index, y=series, mode="lines",
                                      name=name, line=dict(width=1, color=color),
                                      opacity=0.85))
        add_trade_markers(pfig, ohlcv["close"], sig_res.position)
        pfig.update_layout(title=f"{instrument} — price, indicators & entries ({sig_label})")
        st.plotly_chart(style_fig(pfig, height=480), use_container_width=True)

        # always-on trade ledger (entry+exit, accurate prices)
        st.markdown(section("Trade details", 1), unsafe_allow_html=True)
        ledger = extract_trades(sig_res.position, ohlcv["close"])
        if ledger.empty:
            st.info("No trades for this strategy/period.")
        else:
            st.dataframe(ledger, use_container_width=True, height=260)
            st.download_button("Export trades CSV", ledger.to_csv(index=False),
                               file_name=f"trades_{instrument}_{sig_label}.csv")

        st.markdown(section("Metrics", 2), unsafe_allow_html=True)
        table = pd.DataFrame({lbl: r.stats for lbl, (r, _) in results.items()}).T
        table["FinalEquity"] = [r.equity.iloc[-1] for (r, _) in results.values()]
        st.dataframe(table.style.format("{:.3f}"), use_container_width=True)

    # --- Spectrum ----------------------------------------------------------
    with tab_spec:
        st.markdown(section("Parameter sweep", 0), unsafe_allow_html=True)
        st.caption(HELP["sweep"])
        spec_key = st.selectbox("Strategy", keys,
                                format_func=lambda k: REGISTRY[k].label)
        cls = REGISTRY[spec_key]
        if cls.cross_sectional:
            st.warning("Spectrum view supports single-instrument rules for now.")
        else:
            base_params = cfg["params"].get(spec_key, {})
            sweep_params = st.multiselect(
                "Parameter(s) to sweep (pick 1 or 2)", list(cls.params),
                default=[cls.spectrum_param],
                help="Pick one for a bar chart, or two (e.g. fast & slow) for a heatmap.")

            if len(sweep_params) == 0:
                st.info("Select one or two parameters to sweep.")
            elif len(sweep_params) == 1:
                p = sweep_params[0]
                d, lo, hi, _ = cls.params[p]
                c1, c2, c3 = st.columns(3)
                v_lo = c1.number_input(f"{p} min", value=float(lo))
                v_hi = c2.number_input(f"{p} max", value=float(hi))
                n = c3.number_input("Variants", 4, 30, 12)
                vals = make_spectrum(v_lo, v_hi, int(n), integer=isinstance(d, int))
                fixed = {k: v for k, v in base_params.items() if k != p}
                res_map, tbl = run_spectrum(cls, ohlcv, param=p, values=vals,
                                            bt=bt, risk=risk, fixed_params=fixed)
                fig = go.Figure()
                for lbl, res in res_map.items():
                    fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity,
                                             mode="lines", name=lbl, opacity=0.85))
                fig.update_layout(title=f"SPECTRUM EQUITY — {cls.label} (sweep {p})")
                st.plotly_chart(style_fig(fig), use_container_width=True)
                bar = go.Figure(go.Bar(x=tbl.index.astype(str), y=tbl["Sharpe"],
                                       marker_color=THEME.teal))
                bar.update_layout(title=f"SHARPE vs {p}  (DSR over {len(vals)} trials)")
                st.plotly_chart(style_fig(bar, height=320), use_container_width=True)
                st.dataframe(tbl.style.format("{:.3f}"), use_container_width=True)
            else:
                px, py = sweep_params[0], sweep_params[1]
                dx, lox, hix, _ = cls.params[px]
                dy, loy, hiy, _ = cls.params[py]
                c1, c2, c3 = st.columns(3)
                n = c1.number_input("Grid size (per axis)", 4, 16, 8)
                vx = make_spectrum(float(lox), float(hix), int(n), integer=isinstance(dx, int))
                vy = make_spectrum(float(loy), float(hiy), int(n), integer=isinstance(dy, int))
                fixed = {k: v for k, v in base_params.items() if k not in (px, py)}
                grid = run_spectrum_2d(cls, ohlcv, px, vx, py, vy, bt=bt, risk=risk,
                                       fixed_params=fixed)
                hm = go.Figure(go.Heatmap(
                    z=grid.values.astype(float), x=[str(c) for c in grid.columns],
                    y=[str(i) for i in grid.index], colorscale="Viridis",
                    colorbar=dict(title="Sharpe")))
                hm.update_layout(title=f"SHARPE HEATMAP — {px} (x) vs {py} (y)",
                                 xaxis_title=px, yaxis_title=py)
                st.plotly_chart(style_fig(hm, height=480), use_container_width=True)
                st.caption("Bright contiguous regions = robust parameter zones.")

    # --- Diagnostics -------------------------------------------------------
    with tab_diag:
        st.markdown(section("Return correlation", 0), unsafe_allow_html=True)
        chosen = st.multiselect("Strategies to compare",
                                [REGISTRY[k].label for k in keys],
                                default=[REGISTRY[k].label for k in keys])
        ck = [label_to_key(l) for l in chosen
              if not REGISTRY[label_to_key(l)].cross_sectional]
        if len(ck) >= 2:
            res = {REGISTRY[k].label: run_one(k)[0] for k in ck}
            corr = return_correlation(res)
            hm = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                      colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                                      text=np.round(corr.values, 2), texttemplate="%{text}"))
            hm.update_layout(title="RETURN CORRELATION")
            st.plotly_chart(style_fig(hm), use_container_width=True)
        else:
            st.info("Select 2+ single-instrument strategies to see correlations.")

        st.markdown(section(f"Beta & correlation to {instrument}", 1), unsafe_allow_html=True)
        rows = []
        for k in keys:
            if REGISTRY[k].cross_sectional:
                continue
            r = run_one(k)[0]
            bc = beta_and_correlation(r.returns, ohlcv["close"])
            rows.append({"Strategy": REGISTRY[k].label, "Beta": bc["beta"],
                         "Correlation": bc["correlation"]})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Strategy").style.format("{:.3f}"),
                         use_container_width=True)

        st.markdown(section("Walk-forward (IS vs OOS)", 2), unsafe_allow_html=True)
        wf_keys = [k for k in keys if not REGISTRY[k].cross_sectional]
        if wf_keys:
            wf_key = st.selectbox("Rule", wf_keys,
                                  format_func=lambda k: REGISTRY[k].label, key="wf")
            strat = instantiate(REGISTRY[wf_key], cfg["params"].get(wf_key, {}))
            wf = walk_forward(strat, ohlcv, n_splits=4, bt=bt, risk=risk)
            if not wf.empty:
                st.dataframe(wf.style.format({"IS_Sharpe": "{:.2f}", "OOS_Sharpe": "{:.2f}"}),
                             use_container_width=True)

    # --- Monte Carlo -------------------------------------------------------
    with tab_mc:
        st.markdown(section("Bootstrap distribution", 0), unsafe_allow_html=True)
        st.caption("Block-bootstrap the return stream for a distribution of "
                   "outcomes, not a single lucky path.")
        mc_keys = [k for k in keys if not REGISTRY[k].cross_sectional] or keys
        mc_key = st.selectbox("Rule", mc_keys,
                              format_func=lambda k: REGISTRY[k].label, key="mc")
        if not REGISTRY[mc_key].cross_sectional:
            res = run_one(mc_key)[0]
            boot = montecarlo.bootstrap(res.returns, n_sims=800)
            if not boot.empty:
                fig = go.Figure(go.Histogram(x=boot["Sharpe"], nbinsx=40,
                                             marker_color=THEME.teal))
                fig.add_vline(x=res.stats["Sharpe"], line_color=THEME.coral,
                              annotation_text="actual")
                fig.update_layout(title="BOOTSTRAP SHARPE DISTRIBUTION")
                st.plotly_chart(style_fig(fig), use_container_width=True)
                summ = montecarlo.summarize(boot)
                cols = st.columns(3)
                for i, (m, q) in enumerate(summ.items()):
                    cols[i % 3].metric(f"{m} p5/p50/p95",
                                       f"{q['p5']:.2f} / {q['p50']:.2f} / {q['p95']:.2f}")


if __name__ == "__main__":
    main()
