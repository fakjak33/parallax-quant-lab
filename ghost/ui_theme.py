"""Parallax visual theme: brutalist, modernist, pure black + retro pantone.

CSS injection, an inline vector logo, section-header helpers, and Plotly layout
helpers. Square/hard-edged everything, bright white text, geometric font.
"""

from __future__ import annotations

import plotly.graph_objects as go

from .config import THEME

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=VT323&family=Space+Mono:wght@400;700&family=Share+Tech+Mono&display=swap');

/* unified vintage monospace 'code' font + pure black canvas + bright text */
html, body, .stApp, [class*="css"],
input, button, select, textarea, .stMarkdown, p, label, span, div, td, th {{
    font-family: {THEME.font_body} !important;
}}
/* IMPORTANT: re-assert Streamlit's Material icon font so dropdown/expander
   chevrons & dataframe sort icons render as glyphs (not literal text). */
[data-testid="stIconMaterial"], span.material-icons, span.material-icons-outlined,
[class*="material-symbols"], [class*="material-icons"] {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons', 'Material Icons Outlined' !important;
}}
.stApp {{ background: {THEME.bg}; color: {THEME.text}; }}
.stApp, .stMarkdown, p, label, span, div {{ color: {THEME.text}; }}

/* CRT grain / scanline overlay for a vintage, grainy feel */
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 1;
    opacity: 0.04;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}}
.stApp::after {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 1;
    opacity: 0.25;
    background: repeating-linear-gradient(
        0deg, rgba(0,0,0,0) 0px, rgba(0,0,0,0) 2px,
        rgba(0,0,0,0.25) 3px, rgba(0,0,0,0) 4px);
}}

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
.stNumberInput button {{ border-radius: 0 !important; border: 1px solid {THEME.border} !important; }}

/* sub-section inputs (number/date/text) square + thinner 1px border */
.stNumberInput div[data-baseweb="input"], .stDateInput div[data-baseweb="input"],
.stNumberInput div[data-baseweb="base-input"], [data-baseweb="spinner"],
.stNumberInput input, .stDateInput input {{
    border-radius: 0 !important; border-width: 1px !important;
}}

/* consistent "?" help emblems — muted, no box, uniform (cover all variants) */
[data-testid="stTooltipIcon"], [data-testid="stTooltipIcon"] svg,
[data-testid="stTooltipHoverTarget"], [data-testid="stTooltipHoverTarget"] svg,
[data-testid="stWidgetLabel"] svg, label svg, [aria-label="Help"], .stTooltipIcon {{
    border: none !important; background: transparent !important; box-shadow: none !important;
    outline: none !important; border-radius: 0 !important;
    color: {THEME.muted} !important; fill: {THEME.muted} !important;
}}
/* some help icons sit inside a span that picks up the input border — strip it */
[data-testid="stTooltipHoverTarget"] > div, [data-testid="stTooltipIcon"] > div {{
    border: none !important; background: transparent !important;
}}

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

/* buttons: square, bordered, gradient on hover */
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
    background: linear-gradient(90deg, {THEME.navy}, {THEME.mauve}, {THEME.coral});
    color: #fff; border-color: {THEME.mauve} !important;
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
.stTabs [aria-selected="true"] {{
    color: #ffffff !important;
    background: linear-gradient(120deg, {THEME.navy}, {THEME.mauve}, {THEME.coral});
    border-color: {THEME.mauve} !important;
}}

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

# --- Inline vector logo: concentric "parallax" tunnel, palette gradient -----
def _logo_svg(size: int = 64) -> str:
    """Nested concentric rounded-square outlines forming a parallax-tunnel
    'P', stroked with the retro palette (transparent background, no fill).

    Built purely from open strokes so the centre stays transparent — no solid
    rectangle in the middle.
    """
    palette = [THEME.teal, THEME.mint, THEME.mustard, THEME.orange,
               THEME.coral, THEME.mauve]
    n = len(palette)
    layers = []
    for i, color in enumerate(palette):
        inset = 5 + i * 6.5            # concentric inward steps
        w = 100 - 2 * inset
        # offset each ring slightly up-left to fake 3D depth (parallax)
        ox, oy = -i * 1.2, -i * 1.2
        layers.append(
            f'<rect x="{inset + ox:.1f}" y="{inset + oy:.1f}" '
            f'width="{w:.1f}" height="{w:.1f}" rx="11" ry="11" fill="none" '
            f'stroke="{color}" stroke-width="3.4" stroke-linejoin="round"/>'
        )
    inner = "".join(layers)
    return (
        f'<svg width="{size}" height="{size}" viewBox="-6 -6 112 112" '
        f'xmlns="http://www.w3.org/2000/svg" fill="none">{inner}</svg>'
    )


BANNER = f"""
<div style="display:flex; align-items:center; gap:1.1rem; margin-bottom:0.4em;">
  <div style="line-height:0;">{_logo_svg(96)}</div>
  <div style="display:flex; flex-direction:column; gap:0.1rem;">
    <div style="font-family:{THEME.font_display}; font-size:5rem; font-weight:400;
                letter-spacing:0.16em; color:#ffffff; line-height:0.9;
                -webkit-text-stroke:1.2px #ffffff; text-shadow:0 0 1px #fff;">PARALLAX</div>
    <div class="parallax-tag">systematic strategy R&amp;D lab</div>
  </div>
</div>
"""


def section(label: str, idx: int = 0) -> str:
    """Return HTML for a gradient palette-accented section header (cycles)."""
    cols = THEME.section_colors
    c1 = cols[idx % len(cols)]
    c2 = cols[(idx + 1) % len(cols)]
    grad = f"linear-gradient(90deg, {c1}, {c2})"
    return (
        f'<div class="parallax-sec" style="--sec:{c1}; border-image:{grad} 1; '
        f'background:{grad}; -webkit-background-clip:text; background-clip:text; '
        f'-webkit-text-fill-color:transparent;">{label}</div>'
    )


def style_fig(fig: go.Figure, height: int = 440, transparent: bool = True,
              log_y: bool = False) -> go.Figure:
    """Apply the Parallax theme. Charts are transparent (pure-black canvas
    shows through), leaving only axes, gridlines, and data visible.

    ``log_y`` switches the y-axis to a logarithmic scale.
    """
    bg = "rgba(0,0,0,0)" if transparent else THEME.bg
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Space Mono, monospace", color=THEME.text, size=12),
        title_font=dict(family="Space Mono, monospace", size=17, color="#ffffff"),
        colorway=list(THEME.series),
        margin=dict(l=55, r=24, t=48, b=44),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=THEME.grid, borderwidth=1),
        xaxis=dict(gridcolor=THEME.grid, zerolinecolor=THEME.grid, linecolor=THEME.muted),
        yaxis=dict(gridcolor=THEME.grid, zerolinecolor=THEME.grid, linecolor=THEME.muted,
                   type="log" if log_y else "linear"),
    )
    return fig
