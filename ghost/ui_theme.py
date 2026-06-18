"""Parallax visual theme: modernist, minimalist, black + retro pastel.

CSS injection plus Plotly layout helpers. Bold Archivo display type, clean
Inter body, near-black canvas, retro-pantone accents.
"""

from __future__ import annotations

import plotly.graph_objects as go

from .config import THEME

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800;900&family=Inter:wght@400;500;600&display=swap');

.stApp {{
    background: {THEME.bg};
    color: {THEME.text};
    font-family: {THEME.font_body};
}}
h1, h2, h3, h4 {{
    font-family: {THEME.font_display};
    font-weight: 800;
    color: {THEME.text};
    letter-spacing: -0.02em;
    text-transform: none;
}}
h1 {{ font-weight: 900; }}
section[data-testid="stSidebar"] {{
    background: {THEME.panel};
    border-right: 1px solid {THEME.grid};
}}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {THEME.teal};
    font-weight: 700;
}}
.stButton>button {{
    background: {THEME.teal};
    color: {THEME.bg};
    border: none;
    border-radius: 2px;
    font-family: {THEME.font_display};
    font-weight: 700;
    letter-spacing: 0.02em;
    transition: transform 0.1s ease, filter 0.1s ease;
}}
.stButton>button:hover {{ filter: brightness(1.1); transform: translateY(-1px); }}
[data-testid="stMetricValue"] {{
    color: {THEME.text};
    font-family: {THEME.font_display};
    font-weight: 800;
}}
[data-testid="stMetricLabel"] {{
    color: {THEME.muted};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.7rem;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {THEME.grid}; }}
.stTabs [data-baseweb="tab"] {{
    font-family: {THEME.font_display};
    font-weight: 700;
    letter-spacing: 0.02em;
    color: {THEME.muted};
    background: transparent;
}}
.stTabs [aria-selected="true"] {{ color: {THEME.text}; border-bottom: 2px solid {THEME.coral}; }}
.parallax-tag {{
    color: {THEME.muted};
    font-size: 0.8rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 500;
}}
.stDataFrame {{ border: 1px solid {THEME.grid}; }}
</style>
"""

BANNER = f"""
<div style="margin-bottom:0.4em;">
  <div style="display:flex; align-items:baseline; gap:0.6rem; font-family:{THEME.font_display};">
    <span style="font-size:3rem; font-weight:900; letter-spacing:-0.04em; color:{THEME.text};">PARALLAX</span>
    <span style="display:inline-block; width:14px; height:14px; background:{THEME.coral};"></span>
    <span style="display:inline-block; width:14px; height:14px; background:{THEME.mustard};"></span>
    <span style="display:inline-block; width:14px; height:14px; background:{THEME.teal};"></span>
  </div>
  <div class="parallax-tag">systematic strategy R&amp;D lab</div>
</div>
"""


def style_fig(fig: go.Figure, height: int = 440) -> go.Figure:
    """Apply the Parallax dark-modernist theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=THEME.bg,
        plot_bgcolor=THEME.panel,
        font=dict(family="Inter, sans-serif", color=THEME.text, size=12),
        title_font=dict(family="Archivo, sans-serif", size=16, color=THEME.text),
        colorway=list(THEME.series),
        margin=dict(l=55, r=24, t=48, b=44),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=THEME.grid, borderwidth=1),
        xaxis=dict(gridcolor=THEME.grid, zerolinecolor=THEME.grid),
        yaxis=dict(gridcolor=THEME.grid, zerolinecolor=THEME.grid),
    )
    return fig
