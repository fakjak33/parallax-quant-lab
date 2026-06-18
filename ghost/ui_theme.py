"""Parallax visual theme: brutalist, modernist, pure black + retro pantone.

CSS injection, an inline vector logo, section-header helpers, and Plotly layout
helpers. Square/hard-edged everything, bright white text, geometric font.
"""

from __future__ import annotations

import plotly.graph_objects as go

from .config import THEME

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* unified modernist font + pure black canvas + bright white text */
html, body, .stApp, [class*="css"] {{
    font-family: {THEME.font_display} !important;
}}
.stApp {{ background: {THEME.bg}; color: {THEME.text}; }}
.stApp, .stMarkdown, p, label, span, div {{ color: {THEME.text}; }}

h1, h2, h3, h4 {{
    font-family: {THEME.font_display};
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.01em;
}}

/* ---- BRUTALIST: square, hard-bordered inputs & boxes ---- */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input,
.stDateInput input,
div[data-baseweb="select"] {{
    border-radius: 0 !important;
    border: 2px solid {THEME.border} !important;
    background: {THEME.bg} !important;
    color: #ffffff !important;
    font-weight: 600;
}}
.stNumberInput button {{ border-radius: 0 !important; border: 2px solid {THEME.border} !important; }}

/* sidebar */
section[data-testid="stSidebar"] {{
    background: {THEME.panel};
    border-right: 2px solid {THEME.border};
}}

/* section headers get cycled accent colors via .parallax-sec */
.parallax-sec {{
    font-family: {THEME.font_display};
    font-weight: 700;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin: 0.4rem 0 0.2rem 0;
    padding: 0.2rem 0.5rem;
    border-left: 4px solid var(--sec, {THEME.teal});
    color: var(--sec, {THEME.teal}) !important;
}}

/* buttons: square, bordered, palette-varied */
.stButton>button, .stDownloadButton>button {{
    border-radius: 0 !important;
    border: 2px solid {THEME.border} !important;
    background: {THEME.bg};
    color: #ffffff;
    font-family: {THEME.font_display};
    font-weight: 700;
    letter-spacing: 0.02em;
    transition: all 0.1s ease;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    background: {THEME.teal}; color: #000; border-color: {THEME.teal} !important;
}}

/* multiselect tags cycle palette colors */
.stMultiSelect span[data-baseweb="tag"] {{ border-radius: 0 !important; color:#000 !important; font-weight:700; }}
.stMultiSelect span[data-baseweb="tag"]:nth-child(6n+1) {{ background: {THEME.teal} !important; }}
.stMultiSelect span[data-baseweb="tag"]:nth-child(6n+2) {{ background: {THEME.mustard} !important; }}
.stMultiSelect span[data-baseweb="tag"]:nth-child(6n+3) {{ background: {THEME.coral} !important; }}
.stMultiSelect span[data-baseweb="tag"]:nth-child(6n+4) {{ background: {THEME.mauve} !important; }}
.stMultiSelect span[data-baseweb="tag"]:nth-child(6n+5) {{ background: {THEME.navy} !important; }}
.stMultiSelect span[data-baseweb="tag"]:nth-child(6n+6) {{ background: {THEME.mint} !important; }}

/* expanders: square */
.streamlit-expanderHeader, details, [data-testid="stExpander"] {{
    border-radius: 0 !important;
    border: 2px solid {THEME.border} !important;
}}

/* ---- TABS: spaced out, square, palette underlines ---- */
.stTabs [data-baseweb="tab-list"] {{ gap: 18px; border-bottom: 2px solid {THEME.border}; }}
.stTabs [data-baseweb="tab"] {{
    font-family: {THEME.font_display};
    font-weight: 700;
    letter-spacing: 0.06em;
    color: {THEME.muted};
    background: {THEME.panel};
    border: 2px solid {THEME.border};
    border-bottom: none;
    border-radius: 0 !important;
    padding: 10px 22px;
    margin-right: 4px;
}}
.stTabs [aria-selected="true"] {{ color: #000 !important; background: {THEME.teal}; }}

/* metrics */
[data-testid="stMetric"] {{ border: 2px solid {THEME.border}; padding: 0.6rem; background: {THEME.panel}; }}
[data-testid="stMetricValue"] {{ color: #ffffff; font-family: {THEME.font_display}; font-weight: 700; }}
[data-testid="stMetricLabel"] {{
    color: {THEME.muted}; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.7rem; font-weight: 600;
}}

.stDataFrame {{ border: 2px solid {THEME.border}; }}
.parallax-tag {{
    color: {THEME.muted}; font-size: 0.85rem; letter-spacing: 0.2em;
    text-transform: uppercase; font-weight: 600;
}}
</style>
"""

# --- Inline vector logo: concentric "parallax" P in white, transparent bg ---
def _logo_svg(size: int = 60) -> str:
    """Nested, offset rounded-square 'P' giving the parallax-tunnel effect,
    rendered in white with graduated opacity (transparent background)."""
    layers = []
    n = 6
    for i in range(n):
        op = 0.30 + 0.70 * (i / (n - 1))      # fade outer -> inner
        inset = 4 + i * 5
        sw = 3.2
        # a 'P' built from a rounded-rect bowl; concentric insets create depth
        layers.append(
            f'<rect x="{inset}" y="{inset}" width="{100-2*inset}" height="{100-2*inset}" '
            f'rx="10" ry="10" fill="none" stroke="#ffffff" stroke-opacity="{op:.2f}" '
            f'stroke-width="{sw}"/>'
        )
    # the descending stem of the P (left vertical), white solid
    stem = ('<rect x="20" y="20" width="9" height="68" fill="#ffffff"/>')
    inner = "".join(layers)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" '
        f'xmlns="http://www.w3.org/2000/svg" fill="none">{inner}{stem}</svg>'
    )


BANNER = f"""
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.4em;">
  <div style="line-height:0;">{_logo_svg(64)}</div>
  <div>
    <div style="font-family:{THEME.font_display}; font-size:3rem; font-weight:700;
                letter-spacing:-0.03em; color:#ffffff; line-height:1;">PARALLAX</div>
    <div class="parallax-tag">systematic strategy R&amp;D lab</div>
  </div>
</div>
"""


def section(label: str, idx: int = 0) -> str:
    """Return HTML for a palette-accented section header (cycles colors)."""
    color = THEME.section_colors[idx % len(THEME.section_colors)]
    return f'<div class="parallax-sec" style="--sec:{color}">{label}</div>'


def style_fig(fig: go.Figure, height: int = 440, transparent: bool = True) -> go.Figure:
    """Apply the Parallax theme. Charts are transparent (pure-black canvas
    shows through), leaving only axes, gridlines, and data visible."""
    bg = "rgba(0,0,0,0)" if transparent else THEME.bg
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Space Grotesk, sans-serif", color=THEME.text, size=12),
        title_font=dict(family="Space Grotesk, sans-serif", size=16, color="#ffffff"),
        colorway=list(THEME.series),
        margin=dict(l=55, r=24, t=48, b=44),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=THEME.grid, borderwidth=1),
        xaxis=dict(gridcolor=THEME.grid, zerolinecolor=THEME.grid, linecolor=THEME.muted),
        yaxis=dict(gridcolor=THEME.grid, zerolinecolor=THEME.grid, linecolor=THEME.muted),
    )
    return fig
