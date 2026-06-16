"""
FLOW — Flood Level Observation Warning System
UI Module: Streamlit dashboard styling and reusable components
"""

import streamlit as st
from typing import Dict, List, Optional


# ─── Theme Definitions ─────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "--bg-primary":     "#0a0e1a",
        "--bg-secondary":   "#111827",
        "--bg-card":        "#1a2235",
        "--bg-card-hover":  "#1e2940",
        "--border-color":   "#1e3a5f",
        "--text-primary":   "#e8f4fd",
        "--text-secondary": "#7ba3cc",
        "--text-muted":     "#4a6b8a",
        "--accent-cyan":    "#00d4ff",
        "--accent-blue":    "#0096c7",
        "--accent-green":   "#00e676",
        "--accent-orange":  "#ff9800",
        "--accent-red":     "#f44336",
        "--gauge-bg":       "#1a2235",
        "--shadow":         "0 4px 24px rgba(0,0,0,0.5)",
        "--glow-cyan":      "0 0 20px rgba(0,212,255,0.3)",
    },
    "light": {
        "--bg-primary":     "#f0f4f8",
        "--bg-secondary":   "#ffffff",
        "--bg-card":        "#ffffff",
        "--bg-card-hover":  "#f7faff",
        "--border-color":   "#c8d8e8",
        "--text-primary":   "#0d1b2a",
        "--text-secondary": "#3a6186",
        "--text-muted":     "#7896b2",
        "--accent-cyan":    "#0096c7",
        "--accent-blue":    "#023e8a",
        "--accent-green":   "#00897b",
        "--accent-orange":  "#ef6c00",
        "--accent-red":     "#c62828",
        "--gauge-bg":       "#e8f0fe",
        "--shadow":         "0 2px 12px rgba(0,80,150,0.12)",
        "--glow-cyan":      "0 0 12px rgba(0,150,199,0.2)",
    }
}


def _get_nevera_b64() -> str:
    """Return base64-encoded Nevera-Regular.otf, looked up relative to this file."""
    import base64, os
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Nevera-Regular.otf")
    with open(font_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def inject_styles(theme: str = "dark"):
    """Inject global CSS for the FLOW dashboard."""
    t = THEMES.get(theme, THEMES["dark"])
    vars_block = "\n".join(f"    {k}: {v};" for k, v in t.items())
    nevera_b64 = _get_nevera_b64()

    # Extra glass tokens — dark vs light
    if theme == "light":
        glass_bg          = "rgba(255,255,255,0.45)"
        glass_bg_hover    = "rgba(255,255,255,0.60)"
        glass_border      = "rgba(255,255,255,0.75)"
        glass_border_hi   = "rgba(0,150,199,0.55)"
        glass_shadow      = "0 8px 32px rgba(0,80,150,0.14), 0 1.5px 0 rgba(255,255,255,0.7) inset"
        glass_blur        = "blur(18px) saturate(160%)"
        mesh_a            = "radial-gradient(circle at 18% 28%, rgba(79,70,229,0.18) 0%, transparent 45%)"
        mesh_b            = "radial-gradient(circle at 82% 72%, rgba(0,150,199,0.18) 0%, transparent 45%)"
        sidebar_glass_bg  = "rgba(255,255,255,0.35)"
        topline_opacity   = "0.5"
    else:
        glass_bg          = "rgba(255,255,255,0.04)"
        glass_bg_hover    = "rgba(255,255,255,0.07)"
        glass_border      = "rgba(255,255,255,0.10)"
        glass_border_hi   = "rgba(0,212,255,0.45)"
        glass_shadow      = "0 8px 32px rgba(0,0,0,0.45), 0 1px 0 rgba(255,255,255,0.06) inset"
        glass_blur        = "blur(18px) saturate(140%)"
        mesh_a            = "radial-gradient(circle at 18% 28%, rgba(0,212,255,0.12) 0%, transparent 45%)"
        mesh_b            = "radial-gradient(circle at 82% 72%, rgba(79,70,229,0.14) 0%, transparent 45%)"
        sidebar_glass_bg  = "rgba(255,255,255,0.03)"
        topline_opacity   = "0.7"

    css = f"""
<style>
/* ─── Custom Font ────────────────────────────────────────── */
@font-face {{
    font-family: 'Nevera';
    src: url('data:font/otf;base64,{nevera_b64}') format('opentype');
    font-weight: normal;
    font-style: normal;
}}

/* ─── CSS Variables ─────────────────────────────────────── */
:root {{
{vars_block}
    /* Glass tokens */
    --glass-bg:         {glass_bg};
    --glass-bg-hover:   {glass_bg_hover};
    --glass-border:     {glass_border};
    --glass-border-hi:  {glass_border_hi};
    --glass-shadow:     {glass_shadow};
    --glass-blur:       {glass_blur};
}}

/* ─── Animated Mesh Background ──────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: var(--bg-primary) !important;
    background-image: {mesh_a}, {mesh_b} !important;
    background-attachment: fixed !important;
    font-family: -apple-system, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
    color: var(--text-primary);
}}

/* ─── Sidebar — glass panel ──────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {sidebar_glass_bg} !important;
    backdrop-filter: {glass_blur};
    -webkit-backdrop-filter: {glass_blur};
    border-right: 1px solid var(--glass-border) !important;
}}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(.stButton span),
[data-testid="stSidebar"] div:not(.stButton div) {{
    color: var(--text-primary) !important;
}}

/* ─── Header ─────────────────────────────────────────────── */
.flow-header {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 0 16px;
    border-bottom: 1px solid var(--glass-border);
    margin-bottom: 24px;
}}

.flow-logo {{
    font-family: 'Nevera', -apple-system, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.flow-subtitle {{
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
}}

/* ─── Status Badge ───────────────────────────────────────── */
.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}}

.status-live {{
    background: rgba(0, 230, 118, 0.12);
    color: var(--accent-green);
    border: 1px solid rgba(0, 230, 118, 0.35);
}}

.status-offline {{
    background: rgba(244, 67, 54, 0.10);
    color: var(--accent-red);
    border: 1px solid rgba(244, 67, 54, 0.30);
}}

/* ═══════════════════════════════════════════════════════════
   LIQUID-GLASS CARD  — all .metric-card containers
═══════════════════════════════════════════════════════════ */
.metric-card {{
    background: var(--glass-bg);
    backdrop-filter: {glass_blur};
    -webkit-backdrop-filter: {glass_blur};
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: background 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    box-shadow: var(--glass-shadow);
}}

/* Glossy top-edge sheen */
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 12px; right: 12px;
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(255,255,255,0.55) 30%,
        rgba(255,255,255,0.55) 70%,
        transparent 100%);
    opacity: {topline_opacity};
    border-radius: 50%;
}}

/* Cyan accent top bar */
.metric-card::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue));
    opacity: 0.55;
    border-radius: 20px 20px 0 0;
}}

.metric-card:hover {{
    background: var(--glass-bg-hover);
    border-color: var(--glass-border-hi);
    box-shadow: var(--glass-shadow), 0 0 28px rgba(0,212,255,0.12);
}}

.metric-label {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}}

.metric-value {{
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -1px;
    color: var(--text-primary);
}}

.metric-unit {{
    font-size: 13px;
    color: var(--text-secondary);
    margin-left: 2px;
    font-weight: 400;
}}

/* ─── Risk Display ───────────────────────────────────────── */
.risk-low    {{ color: #2ecc71; }}
.risk-medium {{ color: #f39c12; }}
.risk-high   {{ color: #e74c3c; }}

.risk-badge {{
    display: inline-block;
    padding: 8px 20px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-align: center;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}}

.risk-badge-low    {{ background: rgba(46,204,113,0.13);  color: #2ecc71; border: 1px solid rgba(46,204,113,0.40); }}
.risk-badge-medium {{ background: rgba(243,156,18,0.13);  color: #f39c12; border: 1px solid rgba(243,156,18,0.40); }}
.risk-badge-high   {{ background: rgba(231,76,60,0.13);   color: #e74c3c; border: 1px solid rgba(231,76,60,0.40); }}

/* ─── Progress Bar ───────────────────────────────────────── */
.prog-wrap {{
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin: 6px 0;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
}}
.prog-fill {{
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
    box-shadow: 0 0 8px currentColor;
}}

/* ─── Alert Cards — glass tint ───────────────────────────── */
.alert-card {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 12px 14px;
    border-radius: 12px;
    margin-bottom: 8px;
    font-size: 13px;
    line-height: 1.45;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}}

.alert-critical {{
    background: rgba(231, 76, 60, 0.10);
    border: 1px solid rgba(231, 76, 60, 0.25);
    border-left: 3px solid #e74c3c;
    color: var(--text-primary);
}}

.alert-warning {{
    background: rgba(243, 156, 18, 0.10);
    border: 1px solid rgba(243, 156, 18, 0.25);
    border-left: 3px solid #f39c12;
    color: var(--text-primary);
}}

.alert-info {{
    background: rgba(52, 152, 219, 0.10);
    border: 1px solid rgba(52, 152, 219, 0.25);
    border-left: 3px solid #3498db;
    color: var(--text-primary);
}}

.alert-time {{
    font-size: 10px;
    color: var(--text-muted);
    font-weight: 500;
    margin-top: 2px;
    letter-spacing: 0.5px;
}}

/* ─── Section Headers ────────────────────────────────────── */
.section-header {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    padding-bottom: 8px;
    border-bottom: 1px solid var(--glass-border);
    margin-bottom: 14px;
}}

/* ─── ROI Counts ─────────────────────────────────────────── */
.roi-item {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid var(--glass-border);
    font-size: 13px;
}}

.roi-label {{ color: var(--text-secondary); font-weight: 500; }}
.roi-count {{
    font-weight: 700;
    font-size: 15px;
    color: var(--accent-cyan);
    min-width: 28px;
    text-align: right;
}}

/* ─── Sidebar Controls — glass tiles ────────────────────── */
.sidebar-section {{
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 14px;
    box-shadow: var(--glass-shadow);
}}

.sidebar-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 10px;
}}

/* ─── Button Overrides — glass style ────────────────────── */
.stButton > button {{
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--glass-border) !important;
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    color: var(--text-primary) !important;
    box-shadow: var(--glass-shadow) !important;
}}
.stButton > button:hover {{
    background: var(--glass-bg-hover) !important;
    border-color: var(--glass-border-hi) !important;
    color: var(--text-primary) !important;
    box-shadow: var(--glass-shadow), 0 0 18px rgba(0,212,255,0.15) !important;
}}
.stButton > button:active,
.stButton > button:focus {{
    background: var(--glass-bg-hover) !important;
    color: var(--text-primary) !important;
    box-shadow: 0 0 0 2px var(--accent-cyan), var(--glass-shadow) !important;
}}
/* Primary buttons */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(0,150,199,0.25)) !important;
    color: var(--accent-cyan) !important;
    border-color: var(--accent-cyan) !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, rgba(0,212,255,0.38), rgba(0,150,199,0.38)) !important;
    color: #ffffff !important;
    border-color: var(--accent-blue) !important;
}}

/* ─── Slider Overrides ───────────────────────────────────── */
.stSlider [data-testid="stThumb"] {{
    background: var(--accent-cyan) !important;
    box-shadow: 0 0 10px rgba(0,212,255,0.5) !important;
}}

/* ─── Selectbox / Dropdown Overrides ────────────────────── */
[data-testid="stSelectbox"] > div > div {{
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}}
[data-testid="stSelectbox"] > div > div > div {{
    color: var(--text-primary) !important;
}}
[data-testid="stSelectbox"] svg {{
    fill: var(--text-secondary) !important;
    stroke: var(--text-secondary) !important;
}}
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="select"] [data-baseweb="menu"] {{
    background: rgba(15,20,40,0.85) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
}}
[data-baseweb="popover"] [role="option"],
[data-baseweb="select"] [role="option"] {{
    background: transparent !important;
    color: var(--text-primary) !important;
}}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="select"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"],
[data-baseweb="select"] [aria-selected="true"] {{
    background: rgba(0,212,255,0.10) !important;
    color: var(--accent-cyan) !important;
}}

/* ─── Expander overrides — glass panel ───────────────────── */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] > div:first-child {{
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg {{
    color: var(--text-primary) !important;
    fill: var(--text-primary) !important;
}}
[data-testid="stExpanderDetails"] {{
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}}
[data-testid="stExpanderDetails"] p,
[data-testid="stExpanderDetails"] span,
[data-testid="stExpanderDetails"] label,
[data-testid="stExpanderDetails"] div {{
    color: var(--text-primary) !important;
}}

/* ─── Streamlit native metric widget — glass tint ────────── */
[data-testid="stMetric"] {{
    background: var(--glass-bg) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 14px 18px !important;
    box-shadow: var(--glass-shadow) !important;
}}

/* ─── Hide Streamlit chrome ──────────────────────────────── */
#MainMenu, footer, .stDeployButton {{ display: none !important; }}

/* ─── Webcam frame ───────────────────────────────────────── */
[data-testid="stImage"] img {{
    border-radius: 14px;
    border: 1px solid var(--glass-border);
    box-shadow: var(--glass-shadow);
}}

/* ─── Pulse animation ────────────────────────────────────── */
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
}}
.pulse {{ animation: pulse 1.8s infinite ease-in-out; }}

/* ─── Shimmer / refraction animation ────────────────────── */
@keyframes glass-shimmer {{
    0%   {{ background-position: -200% center; }}
    100% {{ background-position:  200% center; }}
}}

/* ─── Toast / notification — glass ──────────────────────── */
.flow-toast {{
    position: fixed;
    bottom: 20px; right: 20px;
    max-width: 360px;
    z-index: 9999;
    background: var(--glass-bg);
    backdrop-filter: {glass_blur};
    -webkit-backdrop-filter: {glass_blur};
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: var(--glass-shadow);
    font-size: 13px;
    color: var(--text-primary);
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def render_header(is_monitoring: bool):
    dot = '<span class="pulse" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#00e676;margin-right:4px;"></span>' if is_monitoring else '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f44336;margin-right:4px;"></span>'
    status_class = "status-live" if is_monitoring else "status-offline"
    status_text = "LIVE" if is_monitoring else "OFFLINE"

    st.markdown(f"""
<div class="flow-header">
    <div>
        <div class="flow-logo">FLOW</div>
        <div class="flow-subtitle">Flood Level Observation Warning System</div>
    </div>
    <div style="margin-left:auto;">
        <span class="status-badge {status_class}">{dot}{status_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, unit: str = "", accent_color: str = "var(--accent-cyan)"):
    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value" style="color:{accent_color};">
        {value}<span class="metric-unit">{unit}</span>
    </div>
</div>
""", unsafe_allow_html=True)


def render_blockage_bar(blockage_pct: float):
    if blockage_pct < 30:
        color = "#2ecc71"
    elif blockage_pct < 60:
        color = "#f39c12"
    elif blockage_pct < 80:
        color = "#e67e22"
    else:
        color = "#e74c3c"

    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">River Blockage</div>
    <div class="metric-value" style="color:{color};">{blockage_pct:.1f}<span class="metric-unit">%</span></div>
    <div class="prog-wrap" style="margin-top:10px;">
        <div class="prog-fill" style="width:{min(blockage_pct,100):.1f}%;background:{color};"></div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_risk_panel(risk: str, confidence: float, probabilities: Dict):
    risk_key = risk.replace(" ", "-").lower()
    badge_class = f"risk-badge-{'high' if 'High' in risk else 'medium' if 'Medium' in risk else 'low'}"
    conf_pct = confidence * 100

    prob_bars = ""
    for label, prob in probabilities.items():
        r_key = label.split()[0].lower()
        c = "#2ecc71" if r_key=="low" else "#f39c12" if r_key=="medium" else "#e74c3c"
        prob_bars += f"""
<div style="margin-bottom:6px;">
    <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-secondary);margin-bottom:3px;">
        <span>{label}</span><span style="font-weight:600;">{prob*100:.1f}%</span>
    </div>
    <div class="prog-wrap"><div class="prog-fill" style="width:{prob*100:.1f}%;background:{c};"></div></div>
</div>"""

    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Flood Risk Prediction</div>
    <div style="text-align:center;margin:12px 0;">
        <span class="risk-badge {badge_class}">{risk}</span>
    </div>
    <div style="font-size:11px;color:var(--text-muted);text-align:center;margin-bottom:14px;">
        Model Confidence: <strong style="color:var(--text-primary);">{conf_pct:.1f}%</strong>
    </div>
    {prob_bars}
</div>
""", unsafe_allow_html=True)


def render_roi_counts(counts: Dict[str, int]):
    # Icon pool — known classes get a specific emoji, any new class from best.pt
    # automatically falls back to a cycling set of distinct symbols.
    KNOWN_ICONS: Dict[str, str] = {
        # Original classes
        "bottle":       "🍶",
        "plastic_waste":"🛍",
        "log":          "🪵",
        "branch":       "🌿",
        "trash":        "🗑",
        "river_debris": "🌊",
        # Common new classes — add more here as needed
        "tire":         "🔵",
        "tyre":         "🔵",
        "can":          "🥫",
        "bag":          "👜",
        "foam":         "🟦",
        "wood":         "🪵",
        "cloth":        "🧣",
        "paper":        "📄",
        "metal":        "🔩",
        "glass":        "🪟",
        "carton":       "📦",
        "styrofoam":    "🟦",
        "polystyrene":  "🟦",
        "tin":          "🥫",
        "aluminium":    "🥫",
        "aluminum":     "🥫",
        "food_container": "🍱",
        "food container": "🍱",
        "plastic":      "🛍",
    }
    FALLBACK_ICONS = ["◆", "▲", "●", "■", "★", "◉", "◈", "◇", "△", "○"]

    # Assign icons: known classes get their emoji, unknown classes cycle through fallbacks
    unknown_labels = [l for l in sorted(counts.keys()) if l.lower() not in KNOWN_ICONS]
    fallback_map = {
        label: FALLBACK_ICONS[i % len(FALLBACK_ICONS)]
        for i, label in enumerate(unknown_labels)
    }

    total = sum(counts.values())
    items_html = ""
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        icon = KNOWN_ICONS.get(label.lower(), fallback_map.get(label, "●"))
        items_html += f"""
<div class="roi-item">
    <span class="roi-label">{icon} {label.replace('_', ' ').title()}</span>
    <span class="roi-count">{count}</span>
</div>"""

    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">ROI Object Count</div>
    <div style="font-size:26px;font-weight:700;color:var(--accent-cyan);margin-bottom:12px;">{total} <span style="font-size:13px;color:var(--text-muted);font-weight:400;">objects in zone</span></div>
    {items_html if items_html else '<div style="color:var(--text-muted);font-size:13px;text-align:center;padding:10px 0;">No objects detected in ROI</div>'}
</div>
""", unsafe_allow_html=True)


def render_alerts(alerts: List[Dict]):
    if not alerts:
        st.markdown("""
<div style="text-align:center;padding:20px;color:var(--text-muted);font-size:13px;">
    ✓ No active alerts — system nominal
</div>""", unsafe_allow_html=True)
        return

    alerts_html = ""
    for a in alerts[-8:]:
        css_class = f"alert-{a['severity'].lower()}"
        alerts_html += f"""
<div class="alert-card {css_class}">
    <span style="font-size:16px;flex-shrink:0;">{a['icon']}</span>
    <div>
        <div>{a['message']}</div>
        <div class="alert-time">{a['timestamp']} · {a['severity']}</div>
    </div>
</div>"""

    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label" style="margin-bottom:12px;">Active Alerts ({len(alerts)})</div>
    <div style="max-height:260px;overflow-y:auto;padding-right:4px;">
        {alerts_html}
    </div>
</div>
""", unsafe_allow_html=True)


def render_rain_panel(is_rain: bool, rain_intensity: float,
                      rain_category: str = "", use_live_weather: bool = False):
    """
    Render the rain status metric card.

    Parameters
    ----------
    is_rain          : True when rain simulation overlay is active.
    rain_intensity   : Normalised 0-1 intensity value.
    rain_category    : Human-readable category name (e.g. 'Light Drizzle').
                       When provided this overrides the internal label logic.
    use_live_weather : When True, titles the card as 'Live Rain Condition'
                       instead of 'Rain Simulation'.
    """
    # ── Resolve label ──────────────────────────────────────────────────────────
    if rain_category:
        intensity_label = rain_category
    elif is_rain:
        intensity_label = (
            "Light Drizzle"   if rain_intensity < 0.2
            else "Slight Rain"    if rain_intensity < 0.4
            else "Moderate Rain"  if rain_intensity < 0.6
            else "Heavy Rain"     if rain_intensity < 0.8
            else "Violent Showers"
        )
    else:
        intensity_label = "No Rain"

    # ── Resolve icon + color ───────────────────────────────────────────────────
    # Map is keyed on substrings of the WMO condition label so it works for both
    # live weather labels (e.g. "Slight Showers") and simulation labels.
    _LABEL_STYLE = [
        # (substring,          emoji,  color)
        ("Clear",              "☀️",   "var(--text-muted)"),
        ("Mainly Clear",       "🌤️",   "#2ecc71"),
        ("Partly Cloudy",      "⛅",   "var(--text-muted)"),
        ("Overcast",           "☁️",   "var(--text-muted)"),
        ("Fog",                "🌫️",   "var(--text-muted)"),
        ("Icy Fog",            "🌫️",   "var(--text-muted)"),
        ("Light Drizzle",      "🌦️",   "#2ecc71"),
        ("Moderate Drizzle",   "🌦️",   "#2ecc71"),
        ("Heavy Drizzle",      "🌧️",   "#f39c12"),
        ("Slight Rain",        "🌦️",   "#2ecc71"),
        ("Slight Shower",      "🌦️",   "#2ecc71"),
        ("Moderate Rain",      "🌧️",   "#f39c12"),
        ("Moderate Shower",    "🌧️",   "#f39c12"),
        ("Heavy Rain",         "🌧️",   "#e67e22"),
        ("Violent Shower",     "⛈️",   "#e74c3c"),
        ("Violent",            "⛈️",   "#e74c3c"),
        ("Thunderstorm",       "⛈️",   "#e74c3c"),
        ("Snow",               "🌨️",   "#7ba3cc"),
        ("No Rain",            "☀️",   "var(--text-muted)"),
        ("No Rainfall",        "☀️",   "var(--text-muted)"),
        ("Unavailable",        "⚠️",   "var(--text-muted)"),
    ]

    drop_anim = "🌧️"          # safe default
    color     = "#f39c12"

    label_lower = intensity_label.lower()
    for substring, emoji, clr in _LABEL_STYLE:
        if substring.lower() in label_lower:
            drop_anim = emoji
            color     = clr
            break

    # For simulation mode, "No Rain" should also catch intensity == 0
    if not use_live_weather and rain_intensity <= 0.0:
        drop_anim = "☀️"
        color     = "var(--text-muted)"

    panel_title = "Live Rain Condition" if use_live_weather else "Rain Simulation"

    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">{panel_title}</div>
    <div style="font-size:24px;margin:8px 0;">{drop_anim}</div>
    <div style="font-size:15px;font-weight:600;color:{color};">{intensity_label}</div>
    <div style="font-size:12px;color:var(--text-muted);margin-top:4px;">Intensity: {rain_intensity:.3f}</div>
</div>
""", unsafe_allow_html=True)


def render_polygon_editor_html(
    bg_b64: str,
    bg_w: int,
    bg_h: int,
    existing_points: list,
    cam_index: int = 0,
    draw_target: str = "debris",
    show_save_btn: bool = True,
) -> str:
    """
    Return a self-contained HTML string for the interactive polygon editor.

    The canvas:
      • Live webcam feed as background (getUserMedia).
      • Left-click → add point · Right-click → undo · R reset · Z undo last.
      • Polygon overlay with numbered dots and coordinates.
      • Apply / Save / Cancel buttons write the ROI directly back to the
        Streamlit text-input bridge via DOM — no manual copy-paste required.

    Parameters
    ----------
    bg_b64         : unused (kept for API compatibility).
    bg_w / bg_h    : pixel dimensions of the canvas.
    existing_points: list of (x, y) tuples to pre-populate.
    cam_index      : browser camera device index (0 = default).
    draw_target    : "debris" | "gauge"  — shapes button labels and payload.
    show_save_btn  : whether to show the "Save to Config" button.
    """
    pts_js          = str([[p[0], p[1]] for p in existing_points])
    apply_label     = "✅ Apply Gauge ROI" if draw_target == "gauge" else "✅ Apply Polygon ROI"
    save_display    = "block" if show_save_btn else "none"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0e1a; font-family: 'SF Mono', 'Fira Code', monospace; }}

  #wrap {{ position:relative; display:inline-block; width:100%; }}
  #liveVideo {{ display:none; }}
  #bgCanvas {{
    display:block; width:100%; height:auto;
    cursor:crosshair; border:1px solid #1e3a5f; border-radius:8px;
  }}
  #cam-status {{ font-size:11px; padding:4px 0 2px; color:#7ba3cc; }}

  #infobar {{
    display:flex; align-items:center; gap:18px;
    padding:6px 12px; background:#111827;
    border:1px solid #1e3a5f; border-radius:6px;
    margin-bottom:6px; font-size:11px; color:#7ba3cc; flex-wrap:wrap;
  }}
  #infobar span {{ display:flex; align-items:center; gap:5px; }}
  #infobar b {{ color:#00d4ff; }}
  #pt-count {{ margin-left:auto; color:#00e676; font-weight:700; font-size:12px; }}

  #coords-wrap {{ margin-top:6px; }}
  #coords-label {{ font-size:10px; color:#4a6b8a; margin-bottom:3px;
                   letter-spacing:1px; text-transform:uppercase; }}
  #coords-out {{
    width:100%; background:#111827; border:1px solid #1e3a5f; border-radius:6px;
    color:#00d4ff; font-family:inherit; font-size:11px;
    padding:7px 10px; resize:none; outline:none;
  }}

  #btn-row {{ display:flex; gap:8px; margin-top:8px; flex-wrap:wrap; }}
  .roi-btn {{
    flex:1; min-width:110px; padding:9px 12px; border:none; border-radius:6px;
    font-size:12px; font-weight:700; cursor:pointer; letter-spacing:0.5px;
    transition:background 0.15s, opacity 0.15s;
  }}
  #apply-btn  {{ background:#00b4d8; color:#fff; }}
  #apply-btn:hover:not(:disabled) {{ background:#0096c7; }}
  #save-btn   {{ background:#1a6b3a; color:#fff; display:{save_display}; }}
  #save-btn:hover:not(:disabled) {{ background:#27ae60; }}
  #cancel-btn {{ background:#1e2a3a; color:#aaa; flex:0 0 auto; min-width:88px; }}
  #cancel-btn:hover {{ background:#2c3e50; color:#ddd; }}
  .roi-btn:disabled {{ background:#1e3a5f !important; color:#4a6b8a !important;
                       cursor:not-allowed; opacity:0.6; }}

  #status-msg {{
    font-size:11px; margin-top:5px; padding:4px 8px;
    border-radius:4px; display:none;
  }}
  #status-msg.ok  {{ color:#00e676; background:rgba(0,230,118,0.08);
                     border:1px solid rgba(0,230,118,0.2); }}
  #status-msg.err {{ color:#f44336; background:rgba(244,67,54,0.08);
                     border:1px solid rgba(244,67,54,0.2); }}
</style>
</head>
<body>

<div id="cam-status">📷 Requesting webcam access…</div>

<div id="infobar">
  <span>🖱 <b>Left-click</b> add point</span>
  <span>🖱 <b>Right-click</b> undo</span>
  <span>⌨ <b>R</b> reset</span>
  <span>⌨ <b>Z</b> undo last</span>
  <div id="pt-count">0 points</div>
</div>

<div id="wrap">
  <video id="liveVideo" autoplay playsinline muted></video>
  <canvas id="bgCanvas"></canvas>
</div>

<div id="coords-wrap">
  <div id="coords-label">Current ROI Coordinates (auto-updated)</div>
  <textarea id="coords-out" rows="2" readonly>[]</textarea>
</div>

<div id="btn-row">
  <button id="apply-btn"  class="roi-btn" disabled>{apply_label}</button>
  <button id="save-btn"   class="roi-btn" disabled>💾 Save to Config</button>
  <button id="cancel-btn" class="roi-btn">✕ Cancel</button>
</div>
<div id="status-msg"></div>

<script>
(function() {{
  const CANVAS_W   = {bg_w};
  const CANVAS_H   = {bg_h};
  const CAM_INDEX  = {cam_index};
  const INIT_PTS   = {pts_js};
  const DRAW_TARGET = "{draw_target}";

  const video      = document.getElementById("liveVideo");
  const canvas     = document.getElementById("bgCanvas");
  const ctx        = canvas.getContext("2d");
  const ptCount    = document.getElementById("pt-count");
  const coordsOut  = document.getElementById("coords-out");
  const applyBtn   = document.getElementById("apply-btn");
  const saveBtn    = document.getElementById("save-btn");
  const cancelBtn  = document.getElementById("cancel-btn");
  const statusMsg  = document.getElementById("status-msg");
  const camStatus  = document.getElementById("cam-status");

  canvas.width  = CANVAS_W;
  canvas.height = CANVAS_H;

  let points     = INIT_PTS.map(p => ({{ x: p[0], y: p[1] }}));
  let hover      = null;
  let videoReady = false;

  // ── Webcam ────────────────────────────────────────────────────────────────
  async function startCamera() {{
    try {{
      const initialStream = await navigator.mediaDevices.getUserMedia({{
        video: {{ width: CANVAS_W, height: CANVAS_H }}
      }});

      let finalStream = initialStream;

      if (CAM_INDEX > 0) {{
        try {{
          const devices = await navigator.mediaDevices.enumerateDevices();
          const cams    = devices.filter(d => d.kind === "videoinput");
          const deviceId = cams[CAM_INDEX] ? cams[CAM_INDEX].deviceId : null;
          if (deviceId) {{
            initialStream.getTracks().forEach(t => t.stop());
            finalStream = await navigator.mediaDevices.getUserMedia({{
              video: {{ deviceId: {{ exact: deviceId }}, width: CANVAS_W, height: CANVAS_H }}
            }});
          }}
        }} catch (switchErr) {{
          console.warn("[FLOW] Camera switch failed:", switchErr.message);
        }}
      }}

      video.srcObject = finalStream;
      window._flowStream = finalStream;

      video.addEventListener("playing", () => {{
        videoReady = true;
        camStatus.style.color = "#00e676";
        camStatus.textContent = "📷 Live — draw your polygon";
      }});
      await video.play();
    }} catch(err) {{
      camStatus.style.color = "#e74c3c";
      const denied = err.name === "NotAllowedError" || err.name === "PermissionDeniedError";
      camStatus.textContent = denied
        ? "⚠ Camera permission denied — allow camera in your browser address bar, then reload"
        : "⚠ Camera error: " + err.message;
    }}
  }}

  startCamera();

  window.addEventListener("beforeunload", () => {{
    if (window._flowStream) {{
      window._flowStream.getTracks().forEach(t => t.stop());
      window._flowStream = null;
    }}
  }});

  // ── Animation loop ─────────────────────────────────────────────────────────
  function loop() {{
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    if (videoReady && video.readyState >= 2) {{
      ctx.drawImage(video, 0, 0, CANVAS_W, CANVAS_H);
    }} else {{
      ctx.fillStyle = "#0a1628";
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.fillStyle = "#4a6b8a";
      ctx.font = "16px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for webcam...", CANVAS_W/2, CANVAS_H/2);
      ctx.textAlign = "left";
    }}
    drawPolygon();
    requestAnimationFrame(loop);
  }}
  requestAnimationFrame(loop);

  // ── Coordinate utilities ───────────────────────────────────────────────────
  function getScale() {{ return CANVAS_W / canvas.getBoundingClientRect().width; }}
  function canvasXY(e) {{
    const r = canvas.getBoundingClientRect();
    const s = getScale();
    return {{ x: Math.round((e.clientX - r.left) * s),
              y: Math.round((e.clientY - r.top)  * s) }};
  }}

  // ── Refresh coordinate display + button state ──────────────────────────────
  // Does NOT trigger any Streamlit rerun — reruns only happen on button click.
  function refreshDisplay() {{
    const arr = points.map(p => [p.x, p.y]);
    coordsOut.value = JSON.stringify(arr);
    const n = points.length;
    ptCount.textContent = n + " point" + (n !== 1 ? "s" : "")
      + (n >= 3 ? "  ✓" : "  (need ≥3)");
    const ok = n >= 3;
    applyBtn.disabled = !ok;
    saveBtn.disabled  = !ok;
  }}

  // ── Send action to Streamlit via the hidden text-input bridge ─────────────
  // html() iframes share the same origin (localhost:8501) with the parent, so
  // window.parent.document is accessible and the React setter trick works.
  function sendActionToStreamlit(action) {{
    const payload = JSON.stringify({{
      action: action,
      target: DRAW_TARGET,
      pts: points.map(p => [p.x, p.y])
    }});

    try {{
      // Find the hidden sync input by its distinctive label "FLOW_ROI_SYNC"
      const labels = window.parent.document.querySelectorAll("label");
      let syncInput = null;
      for (let i = 0; i < labels.length; i++) {{
        if (labels[i].textContent.trim() === "FLOW_ROI_SYNC") {{
          // Walk up to the nearest Streamlit text-input container
          const container =
            labels[i].closest('[data-testid="stTextInputRootElement"]') ||
            labels[i].closest('[data-testid="stTextInput"]') ||
            labels[i].parentElement;
          if (container) {{
            syncInput = container.querySelector('input[type="text"]');
          }}
          break;
        }}
      }}

      if (!syncInput) {{
        // Fallback: any text input whose current value looks like JSON
        const all = window.parent.document.querySelectorAll('input[type="text"]');
        for (let i = 0; i < all.length; i++) {{
          const v = all[i].value;
          if (v === "[]" || v.startsWith("{{") || v.startsWith("[[")) {{
            syncInput = all[i];
            break;
          }}
        }}
      }}

      if (syncInput) {{
        // Use React's internal value setter so React recognises the change
        const nativeSetter = Object.getOwnPropertyDescriptor(
          window.parent.HTMLInputElement.prototype, "value"
        ).set;
        nativeSetter.call(syncInput, payload);
        syncInput.dispatchEvent(new Event("input",  {{ bubbles: true }}));
        syncInput.dispatchEvent(new Event("change", {{ bubbles: true }}));
        // Simulate Enter to trigger Streamlit widget submission
        ["keydown", "keypress", "keyup"].forEach(evType => {{
          syncInput.dispatchEvent(new KeyboardEvent(evType, {{
            key: "Enter", keyCode: 13, which: 13, bubbles: true, cancelable: true
          }}));
        }});
        showStatus("⏳ Applying…", "ok");
      }} else {{
        showStatus("⚠ Could not reach Streamlit input — please use the buttons below.", "err");
      }}
    }} catch(err) {{
      showStatus("⚠ Error: " + err.message, "err");
      console.warn("[FLOW ROI] DOM bridge failed:", err);
    }}
  }}

  function showStatus(msg, cls) {{
    statusMsg.textContent   = msg;
    statusMsg.className     = "status-msg " + cls;
    statusMsg.style.display = "";
  }}

  // ── Polygon rendering ──────────────────────────────────────────────────────
  function drawPolygon() {{
    if (points.length === 0) return;

    if (points.length >= 3) {{
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
      ctx.closePath();
      ctx.fillStyle   = "rgba(0,212,255,0.18)";
      ctx.fill();
      ctx.strokeStyle = "#00d4ff";
      ctx.lineWidth   = 2;
      ctx.setLineDash([]);
      ctx.stroke();
    }} else {{
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
      ctx.strokeStyle = "#00d4ff";
      ctx.lineWidth   = 2;
      ctx.setLineDash([]);
      ctx.stroke();
    }}

    if (hover && points.length >= 1) {{
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(0,212,255,0.5)";
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(points[points.length-1].x, points[points.length-1].y);
      ctx.lineTo(hover.x, hover.y);
      ctx.stroke();
      if (points.length >= 2) {{
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        ctx.lineTo(hover.x, hover.y);
        ctx.stroke();
      }}
      ctx.setLineDash([]);
    }}

    const DOT_COLORS = ["#00e676","#00d4ff","#ff9800","#e040fb",
                        "#f44336","#ffeb3b","#69f0ae","#40c4ff"];
    for (let i = 0; i < points.length; i++) {{
      const p = points[i], col = DOT_COLORS[i % DOT_COLORS.length];

      ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, Math.PI*2);
      ctx.fillStyle = "rgba(0,0,0,0.55)"; ctx.fill();

      ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI*2);
      ctx.fillStyle = col; ctx.fill();

      ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, Math.PI*2);
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke();

      const lbl = "P" + (i+1);
      ctx.font = "bold 11px 'SF Mono', monospace";
      const tw = ctx.measureText(lbl).width;
      const lx = p.x + 13, ly = p.y - 6;
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillRect(lx - 2, ly - 11, tw + 6, 14);
      ctx.fillStyle = col;
      ctx.fillText(lbl, lx + 1, ly);

      const coord = "(" + p.x + ", " + p.y + ")";
      ctx.font = "10px 'SF Mono', monospace";
      ctx.fillStyle = "rgba(200,220,255,0.85)";
      ctx.fillText(coord, lx + 1, ly + 12);
    }}
  }}

  // ── Canvas events (drawing — no Streamlit reruns during draw) ─────────────
  canvas.addEventListener("mousemove", e => {{ hover = canvasXY(e); }});
  canvas.addEventListener("mouseleave", () => {{ hover = null; }});

  canvas.addEventListener("click", e => {{
    e.preventDefault();
    points.push(canvasXY(e));
    refreshDisplay();
  }});

  canvas.addEventListener("contextmenu", e => {{
    e.preventDefault();
    if (points.length > 0) {{ points.pop(); refreshDisplay(); }}
  }});

  document.addEventListener("keydown", e => {{
    if (e.key === "r" || e.key === "R") {{ points = []; refreshDisplay(); }}
    if (e.key === "z" || e.key === "Z") {{
      if (points.length > 0) {{ points.pop(); refreshDisplay(); }}
    }}
  }});

  // ── Action buttons ─────────────────────────────────────────────────────────
  applyBtn.addEventListener("click",  () => sendActionToStreamlit("apply"));
  saveBtn.addEventListener("click",   () => sendActionToStreamlit("save"));
  cancelBtn.addEventListener("click", () => sendActionToStreamlit("cancel"));

  refreshDisplay();
}})();
</script>
</body>
</html>"""
    return html
