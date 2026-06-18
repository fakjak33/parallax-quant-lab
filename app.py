"""GHOST — Streamlit dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ghost import strategies
from ghost.strategies import REGISTRY
from ghost.config import BacktestConfig, RiskConfig, THEME
from ghost.data import providers, synthetic
from ghost.data.universe import UNIVERSES, BENCHMARK
from ghost.backtest.engine import run_strategy
from ghost.backtest.spectrum import run_spectrum, make_spectrum
from ghost.backtest.diagnostics import return_correlation, walk_forward
from ghost.backtest import montecarlo, metrics
from ghost.ui_theme import CSS, BANNER, style_fig

st.set_page_config(page_title="GHOST // quant lab", layout="wide", page_icon="◓")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(BANNER, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Sidebar: data source, universe, strategies, risk, costs
# ----------------------------------------------------------------------------
def sidebar():
    cfg = {}
    st.sidebar.header("▌ Data source")
    cfg["mode"] = st.sidebar.radio("Mode", ["Real (yfinance)", "Synthetic"], index=0)

    if cfg["mode"].startswith("Real"):
        uni = st.sidebar.selectbox("Universe", list(UNIVERSES), index=0)
        cfg["tickers"] = st.sidebar.multiselect(
            "Tickers", UNIVERSES[uni], default=UNIVERSES[uni][:3]
        )
        custom = st.sidebar.text_input("Add tickers (comma-sep)", "")
        if custom.strip():
            cfg["tickers"] += [t.strip().upper() for t in custom.split(",") if t.strip()]
        c1, c2 = st.sidebar.columns(2)
        cfg["start"] = c1.text_input("Start", "2015-01-01")
        cfg["end"] = c2.text_input("End", "")
        cfg["force"] = st.sidebar.checkbox("Force re-download", False)
    else:
        cfg["kind"] = st.sidebar.selectbox(
            "Synthetic kind",
            ["trending", "mean_reverting", "gbm", "regime", "fat_tailed"],
        )
        cfg["n_days"] = st.sidebar.slider("Days", 300, 3000, 1500, 100)
        cfg["seed"] = st.sidebar.number_input("Seed", value=42, step=1)
        cfg["n_assets"] = st.sidebar.slider("Synthetic assets", 1, 8, 4)

    st.sidebar.header("▌ Strategies")
    cfg["selected"] = st.sidebar.multiselect(
        "Active rules",
        [REGISTRY[k].label for k in sorted(REGISTRY)],
        default=[REGISTRY["ema"].label, REGISTRY["tsmom"].label],
    )

    st.sidebar.header("▌ Risk overlay (ATR)")
    risk = RiskConfig()
    risk.use_atr_stop = st.sidebar.checkbox("ATR stop-loss", False)
    risk.use_atr_tp = st.sidebar.checkbox("ATR take-profit", False)
    if risk.use_atr_stop or risk.use_atr_tp:
        risk.atr_period = st.sidebar.slider("ATR period", 5, 40, 14)
        risk.atr_stop_mult = st.sidebar.slider("Stop k·ATR", 1.0, 8.0, 3.0, 0.5)
        risk.atr_tp_mult = st.sidebar.slider("TP k·ATR", 1.0, 15.0, 6.0, 0.5)
        risk.trailing_stop = st.sidebar.checkbox("Trailing stop", True)
    cfg["risk"] = risk

    st.sidebar.header("▌ Backtest")
    bt = BacktestConfig()
    bt.target_vol = st.sidebar.slider("Target vol", 0.05, 0.50, 0.20, 0.01)
    bt.cost_bps = st.sidebar.slider("Cost (bps)", 0.0, 10.0, 1.0, 0.5)
    bt.slippage_bps = st.sidebar.slider("Slippage (bps)", 0.0, 10.0, 0.5, 0.5)
    cfg["bt"] = bt
    return cfg


def label_to_key(label: str) -> str:
    for k, cls in REGISTRY.items():
        if cls.label == label:
            return k
    raise KeyError(label)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_real(tickers, start, end, force):
    out = {}
    for t in tickers:
        out[t] = providers.get_prices(t, start=start or None, end=end or None,
                                      force_refresh=force)
    return out


def load_data(cfg) -> dict[str, pd.DataFrame]:
    if cfg["mode"].startswith("Real"):
        if not cfg["tickers"]:
            return {}
        return load_real(tuple(cfg["tickers"]), cfg["start"], cfg["end"], cfg["force"])
    panel = synthetic.generate_panel(
        n_assets=cfg["n_assets"], kind=cfg["kind"],
        n_days=cfg["n_days"], seed=int(cfg["seed"]),
    )
    return panel


# ----------------------------------------------------------------------------
# Main panels
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
    ohlcv = data[instrument]

    tab_run, tab_spec, tab_diag, tab_mc = st.tabs(
        ["◓ BACKTEST", "▦ SPECTRUM", "◈ DIAGNOSTICS", "∿ MONTE CARLO"]
    )

    # --- Backtest tab ------------------------------------------------------
    with tab_run:
        results = {}
        for k in keys:
            cls = REGISTRY[k]
            if cls.cross_sectional:
                fcs = cls().forecast_panel(data)
                from ghost.backtest.engine import run_single
                res = run_single(fcs[instrument], ohlcv, bt=cfg["bt"],
                                 risk=cfg["risk"], label=cls.label)
            else:
                res = run_strategy(cls(), ohlcv, bt=cfg["bt"], risk=cfg["risk"])
            results[res.label] = res

        fig = go.Figure()
        for label, res in results.items():
            fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity,
                                     mode="lines", name=label))
        # benchmark overlay
        bench_eq = (1 + ohlcv["close"].pct_change().fillna(0)).cumprod() * cfg["bt"].capital
        fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, mode="lines",
                                 name=f"Buy&Hold {instrument}",
                                 line=dict(color=THEME.muted, dash="dot")))
        fig.update_layout(title=f"EQUITY CURVE // {instrument}")
        st.plotly_chart(style_fig(fig), use_container_width=True)

        st.subheader("Metrics")
        table = pd.DataFrame({lbl: res.stats for lbl, res in results.items()}).T
        st.dataframe(table.style.format("{:.3f}"), use_container_width=True)

        # ATR blotter
        for lbl, res in results.items():
            if not res.events.empty:
                with st.expander(f"Trade blotter — {lbl} ({len(res.events)} events)"):
                    st.dataframe(res.events, use_container_width=True)
                    st.download_button(
                        f"Export {lbl} blotter CSV",
                        res.events.to_csv(index=False),
                        file_name=f"blotter_{instrument}_{lbl}.csv",
                    )

    # --- Spectrum tab ------------------------------------------------------
    with tab_spec:
        st.caption("Test a whole family of one parameter at once. Look for a "
                   "*plateau* of good Sharpe (robust), not a lone *spike* (overfit).")
        spec_key = st.selectbox("Strategy", keys,
                                format_func=lambda k: REGISTRY[k].label)
        cls = REGISTRY[spec_key]
        if cls.cross_sectional:
            st.warning("Spectrum view supports single-instrument rules for now.")
        else:
            param = cls.spectrum_param
            default, lo, hi, _ = cls.params[param]
            c1, c2, c3 = st.columns(3)
            v_lo = c1.number_input(f"{param} min", value=float(lo))
            v_hi = c2.number_input(f"{param} max", value=float(hi))
            n = c3.slider("Variants", 4, 24, 12)
            values = make_spectrum(v_lo, v_hi, n, integer=isinstance(default, int))
            res_map, tbl = run_spectrum(cls, ohlcv, param=param, values=values,
                                        bt=cfg["bt"], risk=cfg["risk"])

            fig = go.Figure()
            for lbl, res in res_map.items():
                fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity,
                                         mode="lines", name=lbl, opacity=0.75))
            fig.update_layout(title=f"SPECTRUM EQUITY // {cls.label}")
            st.plotly_chart(style_fig(fig), use_container_width=True)

            hm = go.Figure(go.Bar(x=tbl.index.astype(str), y=tbl["Sharpe"],
                                  marker_color=THEME.teal))
            hm.update_layout(title=f"SHARPE vs {param}  (DSR-penalized over {n} trials)")
            st.plotly_chart(style_fig(hm, height=320), use_container_width=True)
            st.dataframe(tbl.style.format("{:.3f}"), use_container_width=True)

    # --- Diagnostics tab ---------------------------------------------------
    with tab_diag:
        single = {k: REGISTRY[k] for k in keys if not REGISTRY[k].cross_sectional}
        if len(single) >= 2:
            res = {cls.label: run_strategy(cls(), ohlcv, bt=cfg["bt"], risk=cfg["risk"])
                   for cls in single.values()}
            corr = return_correlation(res)
            fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                       colorscale="Teal", zmid=0,
                                       text=np.round(corr.values, 2),
                                       texttemplate="%{text}"))
            fig.update_layout(title="STRATEGY RETURN CORRELATION")
            st.plotly_chart(style_fig(fig), use_container_width=True)
            st.caption("High correlation = rules that look different but aren't "
                       "diversifying. Carver combines low-correlation rules.")
        else:
            st.info("Select 2+ single-instrument strategies to see correlations.")

        st.subheader("Walk-forward (in-sample vs out-of-sample)")
        wf_key = st.selectbox("Rule", [k for k in keys if not REGISTRY[k].cross_sectional]
                              or keys, format_func=lambda k: REGISTRY[k].label, key="wf")
        if not REGISTRY[wf_key].cross_sectional:
            wf = walk_forward(REGISTRY[wf_key](), ohlcv, n_splits=4,
                              bt=cfg["bt"], risk=cfg["risk"])
            if not wf.empty:
                st.dataframe(wf.style.format({"IS_Sharpe": "{:.2f}", "OOS_Sharpe": "{:.2f}"}),
                             use_container_width=True)

    # --- Monte Carlo tab ---------------------------------------------------
    with tab_mc:
        st.caption("Block-bootstrap the return stream to get a *distribution* of "
                   "outcomes, not a single lucky path.")
        mc_key = st.selectbox("Rule", [k for k in keys if not REGISTRY[k].cross_sectional]
                              or keys, format_func=lambda k: REGISTRY[k].label, key="mc")
        if not REGISTRY[mc_key].cross_sectional:
            res = run_strategy(REGISTRY[mc_key](), ohlcv, bt=cfg["bt"], risk=cfg["risk"])
            boot = montecarlo.bootstrap(res.returns, n_sims=800)
            if not boot.empty:
                fig = go.Figure(go.Histogram(x=boot["Sharpe"], nbinsx=40,
                                             marker_color=THEME.teal))
                fig.add_vline(x=res.stats["Sharpe"], line_color=THEME.amber,
                              annotation_text="actual")
                fig.update_layout(title="BOOTSTRAP SHARPE DISTRIBUTION")
                st.plotly_chart(style_fig(fig), use_container_width=True)
                summ = montecarlo.summarize(boot)
                cols = st.columns(3)
                for i, (m, q) in enumerate(summ.items()):
                    cols[i % 3].metric(f"{m} p5 / p50 / p95",
                                       f"{q['p5']:.2f} / {q['p50']:.2f} / {q['p95']:.2f}")


if __name__ == "__main__":
    main()
