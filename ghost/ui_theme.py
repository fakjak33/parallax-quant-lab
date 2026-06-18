"""Ghost in the Shell visual theme: CSS injection + Plotly layout helpers."""

from __future__ import annotations

import plotly.graph_objects as go

from .config import THEME

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;700&display=swap');

.stApp {{
    background:
        radial-gradient(circle at 20% 0%, #0a1c20 0%, {THEME.bg} 55%),
        repeating-linear-gradient(0deg, transparent 0 3px, rgba(35,224,208,0.025) 3px 4px);
    color: {THEME.text};
    font-family: {THEME.font_mono};
}}
h1, h2, h3, h4 {{
    font-family: {THEME.font_mono};
    color: {THEME.teal};
    letter-spacing: 0.12em;
    text-transform: uppercase;
    text-shadow: 0 0 8px rgba(35,224,208,0.45);
}}
h1 {{ border-bottom: 1px solid {THEME.grid}; padding-bottom: 0.3em; }}
section[data-testid="stSidebar"] {{
    background: {THEME.panel};
    border-right: 1px solid {THEME.grid};
}}
.stButton>button {{
    background: transparent;
    color: {THEME.teal};
    border: 1px solid {THEME.teal};
    border-radius: 0;
    font-family: {THEME.font_mono};
    letter-spacing: 0.15em;
    text-transform: uppercase;
    transition: all 0.15s ease;
}}
.stButton>button:hover {{
    background: {THEME.teal};
    color: {THEME.bg};
    box-shadow: 0 0 14px rgba(35,224,208,0.6);
}}
[data-testid="stMetricValue"] {{ color: {THEME.cyan}; font-family: {THEME.font_mono}; }}
[data-testid="stMetricLabel"] {{ color: {THEME.muted}; letter-spacing: 0.1em; }}
.ghost-tag {{
    color: {THEME.amber};
    font-size: 0.75em;
    letter-spacing: 0.3em;
    text-transform: uppercase;
}}
.stDataFrame {{ border: 1px solid {THEME.grid}; }}
</style>
"""

BANNER = f"""
<div style="font-family:{THEME.font_mono}; line-height:1.15; color:{THEME.teal};
            text-shadow:0 0 10px rgba(35,224,208,0.5); margin-bottom:0.2em;">
<span style="font-size:2.4em; font-weight:700;">G H O S T</span><br>
<span class="ghost-tag">// systematic strategy R&amp;D lab &nbsp;·&nbsp; section-9 quant division</span>
</div>
"""


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply the dark cyber theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=THEME.bg,
        plot_bgcolor=THEME.panel,
        font=dict(family="JetBrains Mono, monospace", color=THEME.text, size=12),
        colorway=list(THEME.series),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=THEME.grid, borderwidth=1),
        xaxis=dict(gridcolor=THEME.grid, zerolinecolor=THEME.grid),
        yaxis=dict(gridcolor=THEME.grid, zerolinecolor=THEME.grid),
    )
    return fig
