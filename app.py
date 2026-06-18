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
from ghost.data.universe import (
    UNIVERSES, BENCHMARK, master_categories, master_tickers,
)
from ghost.backtest.engine import run_single
from ghost.backtest.spectrum import run_spectrum, make_spectrum
from ghost.backtest.diagnostics import (
    return_correlation, walk_forward, beta_and_correlation,
)
from ghost.backtest import montecarlo
from ghost.ui_theme import CSS, BANNER, style_fig

st.set_page_config(page_title="PARALLAX // quant lab", layout="wide", page_icon="◧")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(BANNER, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def label_to_key(label: str) -> str:
    for k, cls in REGISTRY.items():
        if cls.label == label:
            return k
    raise KeyError(label)


def render_param_controls(cls, prefix: str) -> dict:
    """Render a control per parameter for a strategy; return chosen values."""
    values = {}
    for name, (default, lo, hi, step) in cls.params.items():
        is_int = isinstance(default, int) and isinstance(step, int)
        key = f"{prefix}_{cls.key}_{name}"
        if is_int:
            values[name] = st.slider(name, int(lo), int(hi), int(default),
                                     int(step), key=key)
        else:
            values[name] = st.slider(name, float(lo), float(hi), float(default),
                                     float(step), key=key)
    return values


def instantiate(cls, values: dict):
    return cls(**values)


def forecast_for(cls, values, ohlcv, panel):
    """Return a forecast series for one instrument, handling cross-sectional."""
    strat = instantiate(cls, values)
    if cls.cross_sectional:
        fcs = strat.forecast_panel(panel)
        return fcs[INSTRUMENT_HOLDER["name"]], strat
    return strat.forecast(ohlcv), strat


INSTRUMENT_HOLDER = {"name": None}  # set per-render so xsmom can pick its column


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def sidebar():
    cfg = {}
    st.sidebar.header("Data source")
    cfg["mode"] = st.sidebar.radio("Mode", ["Real (yfinance)", "Synthetic"], index=0)

    if cfg["mode"].startswith("Real"):
        src = st.sidebar.radio("Ticker source", ["Curated", "ETF master list"], index=0)
        if src == "Curated":
            uni = st.sidebar.selectbox("Universe", list(UNIVERSES), index=0)
            options = UNIVERSES[uni]
            default = options[:3]
        else:
            cat = st.sidebar.selectbox("Category", master_categories())
            options = master_tickers(cat)
            default = options[:3]
        cfg["tickers"] = st.sidebar.multiselect("Tickers", options, default=default)
        custom = st.sidebar.text_input("Add tickers (comma-sep)", "")
        if custom.strip():
            cfg["tickers"] += [t.strip().upper() for t in custom.split(",") if t.strip()]
    else:
        cfg["kind"] = st.sidebar.selectbox(
            "Synthetic kind",
            ["trending", "mean_reverting", "gbm", "regime", "fat_tailed"],
        )
        cfg["n_days"] = st.sidebar.slider("Days", 300, 3000, 1500, 100)
        cfg["seed"] = int(st.sidebar.number_input("Seed", value=42, step=1))
        cfg["n_assets"] = st.sidebar.slider("Synthetic assets", 1, 8, 4)

    st.sidebar.header("Timeframe")
    c1, c2 = st.sidebar.columns(2)
    cfg["start"] = c1.text_input("Start", "2015-01-01")
    cfg["end"] = c2.text_input("End", "")
    cfg["tf"] = st.sidebar.selectbox("Bar frequency", list(TIMEFRAMES), index=0)

    st.sidebar.header("Strategies")
    cfg["selected"] = st.sidebar.multiselect(
        "Active rules",
        [REGISTRY[k].label for k in sorted(REGISTRY)],
        default=[REGISTRY["ema"].label, REGISTRY["tsmom"].label],
    )
    # per-strategy parameter controls
    cfg["params"] = {}
    for lbl in cfg["selected"]:
        k = label_to_key(lbl)
        with st.sidebar.expander(f"⚙ {lbl} parameters", expanded=False):
            cfg["params"][k] = render_param_controls(REGISTRY[k], prefix="bt")

    st.sidebar.header("Position sizing")
    bt = BacktestConfig()
    bt.use_vol_target = st.sidebar.checkbox("Volatility targeting", True)
    if bt.use_vol_target:
        bt.target_vol = st.sidebar.slider("Target vol", 0.05, 0.50, 0.20, 0.01)
    bt.direction = st.sidebar.radio(
        "Direction", ["both", "long", "short"], horizontal=True
    )

    st.sidebar.header("Risk overlay (ATR)")
    risk = RiskConfig()
    risk.use_atr_stop = st.sidebar.checkbox("ATR stop-loss", False)
    risk.use_atr_tp = st.sidebar.checkbox("ATR take-profit", False)
    if risk.use_atr_stop or risk.use_atr_tp:
        risk.atr_period = st.sidebar.slider("ATR period", 5, 40, 14)
        risk.atr_stop_mult = st.sidebar.slider("Stop k·ATR", 1.0, 8.0, 3.0, 0.5)
        risk.atr_tp_mult = st.sidebar.slider("TP k·ATR", 1.0, 15.0, 6.0, 0.5)
        risk.trailing_stop = st.sidebar.checkbox("Trailing stop", True)

    st.sidebar.header("Costs")
    bt.cost_bps = st.sidebar.slider("Cost (bps)", 0.0, 10.0, 1.0, 0.5)
    bt.slippage_bps = st.sidebar.slider("Slippage (bps)", 0.0, 10.0, 0.5, 0.5)

    cfg["bt"] = bt
    cfg["risk"] = risk
    return cfg


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_real(tickers, start, end):
    out = {}
    for t in tickers:
        out[t] = providers.get_prices(t, start=start or None, end=end or None)
    return out


def load_data(cfg) -> dict[str, pd.DataFrame]:
    if cfg["mode"].startswith("Real"):
        if not cfg["tickers"]:
            return {}
        data = load_real(tuple(cfg["tickers"]), cfg["start"], cfg["end"])
    else:
        data = synthetic.generate_panel(
            n_assets=cfg["n_assets"], kind=cfg["kind"],
            n_days=cfg["n_days"], seed=cfg["seed"], start=cfg["start"] or "2015-01-01",
        )
    # apply timeframe resample + date window
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


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def add_trade_markers(fig, price: pd.Series, position: pd.Series):
    """Mark long and short entries (sign flips) on a price series."""
    sign = np.sign(position).fillna(0.0)
    prev = sign.shift(1).fillna(0.0)
    long_entry = (sign > 0) & (prev <= 0)
    short_entry = (sign < 0) & (prev >= 0)
    fig.add_trace(go.Scatter(
        x=price.index[long_entry], y=price[long_entry], mode="markers",
        name="Long entry", marker=dict(symbol="triangle-up", size=11,
                                       color=THEME.long_color, line=dict(width=1, color="#000")),
    ))
    fig.add_trace(go.Scatter(
        x=price.index[short_entry], y=price[short_entry], mode="markers",
        name="Short entry", marker=dict(symbol="triangle-down", size=11,
                                        color=THEME.short_color, line=dict(width=1, color="#000")),
    ))


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
        vals = cfg["params"].get(k, {})
        fc, strat = forecast_for(cls, vals, ohlcv, data)
        return run_single(fc, ohlcv, bt=bt, risk=risk, label=strat.describe())

    tab_run, tab_spec, tab_diag, tab_mc = st.tabs(
        ["BACKTEST", "SPECTRUM", "DIAGNOSTICS", "MONTE CARLO"]
    )

    # --- Backtest ----------------------------------------------------------
    with tab_run:
        results = {REGISTRY[k].label: run_one(k) for k in keys}

        fig = go.Figure()
        for label, res in results.items():
            fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity,
                                     mode="lines", name=label))
        bench_eq = (1 + ohlcv["close"].pct_change().fillna(0)).cumprod() * bt.capital
        fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, mode="lines",
                                 name=f"Buy&Hold {instrument}",
                                 line=dict(color=THEME.muted, dash="dot")))
        fig.update_layout(title=f"EQUITY CURVE — {instrument}")
        st.plotly_chart(style_fig(fig), use_container_width=True)

        # price chart with long/short entry markers for a chosen strategy
        st.subheader("Trade signals on price")
        sig_label = st.selectbox("Show entries for", list(results))
        sig_res = results[sig_label]
        pfig = go.Figure()
        pfig.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv["close"], mode="lines",
                                  name=instrument, line=dict(color=THEME.navy)))
        add_trade_markers(pfig, ohlcv["close"], sig_res.position)
        pfig.update_layout(title=f"{instrument} price — {sig_label}")
        st.plotly_chart(style_fig(pfig), use_container_width=True)

        st.subheader("Metrics")
        table = pd.DataFrame({lbl: r.stats for lbl, r in results.items()}).T
        table["FinalEquity"] = [r.equity.iloc[-1] for r in results.values()]
        st.dataframe(table.style.format("{:.3f}"), use_container_width=True)

        for lbl, res in results.items():
            if not res.events.empty:
                with st.expander(f"Trade blotter — {lbl} ({len(res.events)} events)"):
                    st.dataframe(res.events, use_container_width=True)
                    st.download_button(f"Export {lbl} blotter CSV",
                                       res.events.to_csv(index=False),
                                       file_name=f"blotter_{instrument}_{lbl}.csv")

    # --- Spectrum ----------------------------------------------------------
    with tab_spec:
        st.caption("Sweep ONE parameter across a family; hold the others fixed. "
                   "Look for a robust *plateau*, not an overfit *spike*.")
        spec_key = st.selectbox("Strategy", keys,
                                format_func=lambda k: REGISTRY[k].label)
        cls = REGISTRY[spec_key]
        if cls.cross_sectional:
            st.warning("Spectrum view supports single-instrument rules for now.")
        else:
            sweep = st.selectbox("Parameter to sweep", list(cls.params))
            default, lo, hi, _ = cls.params[sweep]
            c1, c2, c3 = st.columns(3)
            v_lo = c1.number_input(f"{sweep} min", value=float(lo))
            v_hi = c2.number_input(f"{sweep} max", value=float(hi))
            n = c3.slider("Variants", 4, 24, 12)
            values = make_spectrum(v_lo, v_hi, n, integer=isinstance(default, int))
            fixed = {k: v for k, v in cfg["params"].get(spec_key, {}).items() if k != sweep}
            res_map, tbl = run_spectrum(cls, ohlcv, param=sweep, values=values,
                                        bt=bt, risk=risk, fixed_params=fixed)

            fig = go.Figure()
            for lbl, res in res_map.items():
                fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity,
                                         mode="lines", name=lbl, opacity=0.8))
            fig.update_layout(title=f"SPECTRUM EQUITY — {cls.label} (sweeping {sweep})")
            st.plotly_chart(style_fig(fig), use_container_width=True)

            bar = go.Figure(go.Bar(x=tbl.index.astype(str), y=tbl["Sharpe"],
                                   marker_color=THEME.teal))
            bar.update_layout(title=f"SHARPE vs {sweep}  (DSR-penalized, {n} trials)")
            st.plotly_chart(style_fig(bar, height=320), use_container_width=True)
            st.dataframe(tbl.style.format("{:.3f}"), use_container_width=True)

    # --- Diagnostics -------------------------------------------------------
    with tab_diag:
        st.subheader("Strategy return correlation")
        chosen = st.multiselect(
            "Strategies to compare",
            [REGISTRY[k].label for k in keys],
            default=[REGISTRY[k].label for k in keys],
        )
        chosen_keys = [label_to_key(l) for l in chosen
                       if not REGISTRY[label_to_key(l)].cross_sectional]
        if len(chosen_keys) >= 2:
            res = {REGISTRY[k].label: run_one(k) for k in chosen_keys}
            corr = return_correlation(res)
            hm = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                      colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                                      text=np.round(corr.values, 2),
                                      texttemplate="%{text}"))
            hm.update_layout(title="RETURN CORRELATION")
            st.plotly_chart(style_fig(hm), use_container_width=True)
            st.caption("High correlation = rules that look different but aren't "
                       "diversifying.")
        else:
            st.info("Select 2+ single-instrument strategies to see correlations.")

        st.subheader(f"Beta & correlation to underlying ({instrument})")
        rows = []
        for k in keys:
            if REGISTRY[k].cross_sectional:
                continue
            r = run_one(k)
            bc = beta_and_correlation(r.returns, ohlcv["close"])
            rows.append({"Strategy": REGISTRY[k].label,
                         "Beta": bc["beta"], "Correlation": bc["correlation"]})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Strategy")
                         .style.format("{:.3f}"), use_container_width=True)

        st.subheader("Walk-forward (in-sample vs out-of-sample)")
        wf_keys = [k for k in keys if not REGISTRY[k].cross_sectional]
        if wf_keys:
            wf_key = st.selectbox("Rule", wf_keys,
                                  format_func=lambda k: REGISTRY[k].label, key="wf")
            strat = instantiate(REGISTRY[wf_key], cfg["params"].get(wf_key, {}))
            wf = walk_forward(strat, ohlcv, n_splits=4, bt=bt, risk=risk)
            if not wf.empty:
                st.dataframe(
                    wf.style.format({"IS_Sharpe": "{:.2f}", "OOS_Sharpe": "{:.2f}"}),
                    use_container_width=True)

    # --- Monte Carlo -------------------------------------------------------
    with tab_mc:
        st.caption("Block-bootstrap the return stream for a *distribution* of "
                   "outcomes, not a single lucky path.")
        mc_keys = [k for k in keys if not REGISTRY[k].cross_sectional] or keys
        mc_key = st.selectbox("Rule", mc_keys,
                              format_func=lambda k: REGISTRY[k].label, key="mc")
        if not REGISTRY[mc_key].cross_sectional:
            res = run_one(mc_key)
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
