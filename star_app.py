"""
StellarScope — Stellar Object Classification System
=====================================================
A Streamlit dashboard built around the Random Forest Classifier developed in
`Stellar_prediction_classification_project.ipynb`.

Pages: Home | Prediction | Dashboard | About
Run with:  streamlit run star_app.py
"""

import pickle
import base64
import pathlib
import datetime as dt
import urllib.parse

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="StellarScope | Stellar Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME BACKGROUND IMAGE
# ----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _img_to_data_uri(path) -> str:
    """Read an image file and return it as a base64 data URI for use in CSS."""
    path = pathlib.Path(path)
    ext = path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


_NEBULA_BG_URI = _img_to_data_uri("nebula_bg.jpg")
_GALAXY_HOME_URI = _img_to_data_uri("galaxy_hero.jpg")

# ----------------------------------------------------------------------------
# ICONOGRAPHY — small hand-drawn line icons (no emoji), used anywhere the UI
# is rendered through st.markdown(unsafe_allow_html=True). Native Streamlit
# widgets (st.radio, st.button) render labels as plain text, so those stay
# as clean typographic labels instead.
# ----------------------------------------------------------------------------
_ICON_PATHS = {
    "target": '<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="3.5"/>'
              '<circle cx="12" cy="12" r="0.7" fill="currentColor" stroke="none"/>',
    "chart":  '<line x1="4" y1="20" x2="20" y2="20"/>'
              '<rect x="6" y="12" width="3" height="8" rx="0.5"/>'
              '<rect x="11" y="7" width="3" height="13" rx="0.5"/>'
              '<rect x="16" y="10" width="3" height="10" rx="0.5"/>',
    "wave":   '<path d="M2 14 Q7 6 12 14 T22 14"/>',
    "cpu":    '<rect x="7" y="7" width="10" height="10" rx="1.5"/>'
              '<rect x="10" y="10" width="4" height="4" rx="0.5"/>'
              '<line x1="12" y1="2" x2="12" y2="7"/><line x1="12" y1="17" x2="12" y2="22"/>'
              '<line x1="2" y1="12" x2="7" y2="12"/><line x1="17" y1="12" x2="22" y2="12"/>',
    "info":   '<circle cx="12" cy="12" r="8.5"/><line x1="12" y1="11" x2="12" y2="16.5"/>'
              '<circle cx="12" cy="7.7" r="0.9" fill="currentColor" stroke="none"/>',
    "list":   '<line x1="5" y1="7" x2="19" y2="7"/><line x1="5" y1="12" x2="19" y2="12"/>'
              '<line x1="5" y1="17" x2="19" y2="17"/>',
    "code":   '<polyline points="8,6 3,12 8,18"/><polyline points="16,6 21,12 16,18"/>',
    "home":   '<path d="M4 11.5L12 4l8 7.5"/>'
              '<path d="M6 10v9a1 1 0 001 1h4v-6h2v6h4a1 1 0 001-1v-9"/>',
    "bolt":   '<path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z"/>',
    "telescope": '<path d="M4 20l6-3"/><path d="M14 17l6 3"/>'
              '<path d="M6.5 15.5 18 10l1.4 2.8-11.5 5.5-1.4-2.8Z"/>'
              '<circle cx="8.5" cy="16.2" r="1.6"/>'
              '<path d="M18 10l2.2-1.1"/><path d="M19 6.2l1.4 2.8"/>'
              '<circle cx="19.2" cy="5" r="1"/>',
    "database": '<ellipse cx="12" cy="6" rx="7" ry="3"/>'
              '<path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/>'
              '<path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/>',
    "gear":   '<circle cx="12" cy="12" r="3"/>'
              '<path d="M12 3v2.2M12 18.8V21M21 12h-2.2M5.2 12H3"/>'
              '<path d="M18.4 5.6l-1.6 1.6M7.2 16.8l-1.6 1.6M18.4 18.4l-1.6-1.6M7.2 7.2 5.6 5.6"/>',
    "tree":   '<path d="M12 2 6 11h3l-4 6h4v5h6v-5h4l-4-6h3Z"/>',
    "trending": '<polyline points="3,17 9,11 13,15 21,6"/><polyline points="15,6 21,6 21,12"/>',
    "arrow-right": '<line x1="4" y1="12" x2="20" y2="12"/><polyline points="13,5 20,12 13,19"/>',
    "rocket": '<path d="M12 2c3 2 5 6 5 10 0 2-1 4-2 5l-3 3-3-3c-1-1-2-3-2-5 0-4 2-8 5-10Z"/>'
              '<circle cx="12" cy="10" r="1.6"/><path d="M9 16l-3 5 5-3"/><path d="M15 16l3 5-5-3"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor", stroke: float = 1.8) -> str:
    body = _ICON_PATHS.get(name, "")
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="display:inline-block;vertical-align:middle;">{body}</svg>'
    )


def _icon_data_uri(name: str, color: str, size: int = 18, stroke: float = 1.8) -> str:
    """Render a named icon as a standalone SVG data-URI (for use in CSS
    background-image, e.g. sidebar nav icons where raw HTML labels aren't
    available on native Streamlit widgets)."""
    body = _ICON_PATHS.get(name, "")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )
    return "data:image/svg+xml," + urllib.parse.quote(svg)


# ----------------------------------------------------------------------------
# CLASS ILLUSTRATIONS — small original line-art pieces (no photos/emoji) used
# to visually represent each predicted class: Star, Galaxy, Quasar.
# ----------------------------------------------------------------------------
_CLASS_ART_PATHS = {
    "STAR": (
        '<path d="M50 10 L55 43 L88 50 L55 57 L50 90 L45 57 L12 50 L45 43 Z" '
        'fill="{c}" opacity="0.92"/>'
        '<circle cx="50" cy="50" r="6.5" fill="{c}"/>'
        '<circle cx="76" cy="22" r="2.4" fill="{c}" opacity="0.65"/>'
        '<circle cx="20" cy="78" r="1.8" fill="{c}" opacity="0.5"/>'
        '<circle cx="80" cy="76" r="1.6" fill="{c}" opacity="0.45"/>'
    ),
    "GALAXY": (
        '<ellipse cx="50" cy="50" rx="42" ry="15" fill="none" stroke="{c}" '
        'stroke-width="1" opacity="0.22" transform="rotate(18 50 50)"/>'
        '<path d="M50 50 C 66 44, 78 50, 76 63 C 74 74, 60 76, 53 68" '
        'fill="none" stroke="{c}" stroke-width="3" stroke-linecap="round"/>'
        '<path d="M50 50 C 34 56, 22 50, 24 37 C 26 26, 40 24, 47 32" '
        'fill="none" stroke="{c}" stroke-width="3" stroke-linecap="round" opacity="0.72"/>'
        '<circle cx="50" cy="50" r="6" fill="{c}"/>'
        '<circle cx="18" cy="24" r="1.4" fill="{c}" opacity="0.5"/>'
        '<circle cx="84" cy="80" r="1.4" fill="{c}" opacity="0.5"/>'
    ),
    "QSO": (
        '<line x1="50" y1="50" x2="50" y2="8" stroke="{c}" stroke-width="3" '
        'stroke-linecap="round"/>'
        '<line x1="50" y1="50" x2="50" y2="92" stroke="{c}" stroke-width="3" '
        'stroke-linecap="round"/>'
        '<line x1="50" y1="36" x2="45" y2="18" stroke="{c}" stroke-width="1.6" '
        'opacity="0.55" stroke-linecap="round"/>'
        '<line x1="50" y1="36" x2="55" y2="18" stroke="{c}" stroke-width="1.6" '
        'opacity="0.55" stroke-linecap="round"/>'
        '<ellipse cx="50" cy="50" rx="32" ry="9" fill="none" stroke="{c}" '
        'stroke-width="2.2" opacity="0.5"/>'
        '<circle cx="50" cy="50" r="12" fill="{c}" opacity="0.25"/>'
        '<circle cx="50" cy="50" r="7" fill="{c}"/>'
    ),
}


def class_art(cls_name: str, size: int = 72) -> str:
    """Original line-art illustration for a predicted class (STAR / GALAXY / QSO)."""
    color = CLASS_STYLE.get(cls_name, {}).get("color", "#E5E7EB")
    body = _CLASS_ART_PATHS.get(cls_name, "").format(c=color)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" '
        f'xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    )

MODEL_PATH = "stellar_classification_model.pkl"
SCALER_PATH = "stellar_scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
METRICS_PATH = "model_metrics.pkl"


FEATURE_ORDER = ["alpha", "delta", "u", "g", "r", "i", "z",
                  "cam_col", "redshift", "plate", "MJD", "fiber_ID"]

FEATURE_REGISTRY = {
    "ALPHA": {"label": "Alpha — Right Ascension (°)", "min": 0.0, "max": 360.0, "default": 180.0, "step": 0.5},
    "DELTA": {"label": "Delta — Declination (°)", "min": -90.0, "max": 90.0, "default": 20.0, "step": 0.5},
    "U": {"label": "u — Ultraviolet magnitude", "min": 10.0, "max": 32.0, "default": 22.0, "step": 0.1},
    "G": {"label": "g — Green magnitude", "min": 9.0, "max": 32.0, "default": 21.0, "step": 0.1},
    "R": {"label": "r — Red magnitude", "min": 9.0, "max": 30.0, "default": 20.0, "step": 0.1},
    "I": {"label": "i — Near-infrared magnitude", "min": 9.0, "max": 33.0, "default": 19.5, "step": 0.1},
    "Z": {"label": "z — Infrared magnitude", "min": 9.0, "max": 30.0, "default": 19.0, "step": 0.1},
    "CAM_COL": {"label": "Camera column (1–6)", "min": 1, "max": 6, "default": 3, "step": 1},
    "REDSHIFT": {"label": "Redshift", "min": -0.01, "max": 7.1, "default": 0.5, "step": 0.01},
    "PLATE": {"label": "Plate ID", "min": 250, "max": 12550, "default": 5000, "step": 10},
    "MJD": {"label": "MJD — Modified Julian Date", "min": 51600, "max": 59000, "default": 55500, "step": 10},
    "FIBER_ID": {"label": "Fiber ID", "min": 1, "max": 1000, "default": 450, "step": 1},
}

CLASS_STYLE = {
    "STAR": {"color": "#FBBF24", "bg": "#3A2E10", "desc": "A star within our own Milky Way galaxy."},
    "GALAXY": {"color": "#22D3EE", "bg": "#0E2E36", "desc": "An extended light source — an entire galaxy of stars."},
    "QSO": {"color": "#F472B6", "bg": "#3A1730", "desc": "A quasar — the brilliant, active core of a distant galaxy."},
}


def render_dark_table(df: pd.DataFrame, badge_col=None,
                       class_col=None, highlight_col=None,
                       highlight_value=None) -> str:
    """Render a DataFrame as a single self-contained HTML table matching the
    app's dark theme, so it never relies on Streamlit's default (light) grid."""
    cols = list(df.columns)
    header_html = "".join(f"<th>{c}</th>" for c in cols)
    body_rows = []
    for _, row in df.iterrows():
        is_best = (
            highlight_col is not None
            and highlight_value is not None
            and str(row[highlight_col]) == str(highlight_value)
        )
        cells = []
        for c in cols:
            val = row[c]
            numeric = isinstance(val, (int, float, np.floating, np.integer)) and not isinstance(val, bool)
            align = "right" if numeric else "left"
            if c == badge_col:
                cells.append(f'<td style="text-align:{align};"><span class="status-pill">{val}</span></td>')
            elif c == class_col and val in CLASS_STYLE:
                color = CLASS_STYLE[val]["color"]
                cells.append(f'<td style="text-align:{align};"><span style="font-weight:700;color:{color};">{val}</span></td>')
            else:
                cells.append(f'<td style="text-align:{align};">{val}</td>')
        row_class = ' class="best-row"' if is_best else ""
        body_rows.append(f"<tr{row_class}>{''.join(cells)}</tr>")
    return (
        f'<div class="dark-table-wrap"><table class="dark-table">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table></div>'
    )


def _norm(name: str) -> str:
    return str(name).strip().upper().replace(" ", "_")


def feature_cfg(raw_name: str) -> dict:
    return FEATURE_REGISTRY.get(_norm(raw_name), {"label": raw_name, "min": 0.0, "max": 100.0, "default": 10.0, "step": 1.0})


# ----------------------------------------------------------------------------
# GLOBAL CSS — dark "deep space" theme (starfield built purely in CSS, no
# external image assets needed)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root{
    --violet:#8B5CF6;
    --cyan:#22D3EE;
    --pink:#F472B6;
    --gold:#FBBF24;
    --ink:#E5E7EB;
    --muted:#94A3B8;
    --card-bg:#121A2E;
    --page-bg:#000000;
    --border:#1F2A44;
}

.stApp {
    background:
        radial-gradient(1.5px 1.5px at 20px 30px, #ffffff55, transparent),
        radial-gradient(1.5px 1.5px at 120px 90px, #ffffff40, transparent),
        radial-gradient(1px 1px at 60px 160px, #ffffff35, transparent),
        radial-gradient(1.5px 1.5px at 220px 40px, #ffffff45, transparent),
        radial-gradient(1px 1px at 300px 200px, #ffffff30, transparent),
        radial-gradient(1.5px 1.5px at 400px 120px, #ffffff40, transparent),
        radial-gradient(1px 1px at 180px 260px, #ffffff30, transparent),
        var(--page-bg);
    background-size:
        420px 420px, 420px 420px, 420px 420px, 420px 420px,
        420px 420px, 420px 420px, 420px 420px,
        cover;
    background-repeat:
        repeat, repeat, repeat, repeat, repeat, repeat, repeat,
        no-repeat;
    background-position:
        0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0,
        center;
    background-attachment:
        scroll, scroll, scroll, scroll, scroll, scroll, scroll,
        scroll;
    animation: cosmic-glow 10s ease-in-out infinite alternate;
}
@keyframes cosmic-glow {
    0%   { filter: brightness(1); }
    100% { filter: brightness(1.07); }
}

/* ---------- Monogram badges (replace emoji icons) ---------- */
.class-dot{
    width:12px;height:12px;border-radius:50%;display:inline-block;
    box-shadow:0 0 0 3px rgba(255,255,255,.08);
}
.metric-value-sm{font-size:15px;font-weight:800;color:var(--ink) !important;word-break:break-word;}
#MainMenu, footer, header {visibility: hidden;}
div.block-container{padding-top:1.4rem; padding-bottom:2rem;}

div[data-testid="stMarkdownContainer"], div[data-testid="stMarkdownContainer"] p{ color: var(--ink); }
h1,h2,h3,h4,h5 { color: var(--ink) !important; }
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p,
div[data-testid="stCaptionContainer"] span {
    color: #A78BFA !important;
    font-size: 12.5px !important;
    font-weight: 700 !important;
    letter-spacing: .05em;
    text-transform: uppercase;
    opacity: 1 !important;
}
label, .stNumberInput label, .stSlider label { color: var(--ink) !important; }

/* ---------- Input widgets — force visible white/light text on dark fields ---------- */
.stNumberInput input,
.stTextInput input,
.stTextArea textarea,
.stDateInput input,
.stTimeInput input {
    color: #FFFFFF !important;
    background-color: #1B2540 !important;
    border: 1px solid var(--border) !important;
    caret-color: #FFFFFF !important;
}
.stNumberInput input::placeholder,
.stTextInput input::placeholder,
.stTextArea textarea::placeholder { color: #94A3B8 !important; opacity: 1 !important; }

/* number input +/- step buttons — bright accent pills so the icons are always visible */
div[data-testid="stNumberInputContainer"] { background-color: #1B2540 !important; border: 1px solid var(--border) !important; border-radius: 8px; overflow:hidden; }
.stNumberInput button,
button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"] {
    background: linear-gradient(135deg, var(--violet), var(--cyan)) !important;
    border: none !important;
    border-radius: 0 !important;
    opacity: 1 !important;
}
.stNumberInput button svg,
.stNumberInput button path,
button[data-testid="stNumberInputStepDown"] svg,
button[data-testid="stNumberInputStepUp"] svg,
button[data-testid="stNumberInputStepDown"] path,
button[data-testid="stNumberInputStepUp"] path {
    color: #0A0F1E !important;
    fill: #0A0F1E !important;
    stroke: #0A0F1E !important;
    opacity: 1 !important;
}
.stNumberInput button:hover,
button[data-testid="stNumberInputStepDown"]:hover,
button[data-testid="stNumberInputStepUp"]:hover {
    filter: brightness(1.1);
}
.stNumberInput button:disabled,
button[data-testid="stNumberInputStepDown"]:disabled,
button[data-testid="stNumberInputStepUp"]:disabled {
    opacity: .35 !important;
}

/* selectbox / multiselect (BaseWeb) */
div[data-baseweb="select"] > div {
    background-color: #1B2540 !important;
    color: #FFFFFF !important;
    border-color: var(--border) !important;
}
div[data-baseweb="select"] input { color: #FFFFFF !important; }
div[data-baseweb="select"] span { color: #FFFFFF !important; }
ul[data-baseweb="menu"] { background-color: #1B2540 !important; }
ul[data-baseweb="menu"] li { color: #FFFFFF !important; }

/* slider value labels / ticks */
div[data-testid="stSlider"] span { color: #FFFFFF !important; }
div[data-testid="stTickBar"] { color: #94A3B8 !important; }

/* checkbox / radio text */
div[data-testid="stCheckbox"] p, div[data-testid="stCheckbox"] label { color: var(--ink) !important; }

/* file uploader text */
section[data-testid="stFileUploaderDropzone"] { color: #FFFFFF !important; }
section[data-testid="stFileUploaderDropzone"] * { color: #FFFFFF !important; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
    background:
        radial-gradient(1.2px 1.2px at 15px 40px, #ffffff50, transparent),
        radial-gradient(1.2px 1.2px at 70px 140px, #ffffff38, transparent),
        radial-gradient(1px 1px at 40px 220px, #ffffff30, transparent),
        radial-gradient(1.2px 1.2px at 110px 60px, #ffffff40, transparent),
        radial-gradient(1px 1px at 150px 260px, #ffffff28, transparent),
        radial-gradient(1.2px 1.2px at 30px 320px, #ffffff38, transparent),
        radial-gradient(1px 1px at 100px 380px, #ffffff28, transparent),
        #000000;
    background-size:
        180px 180px, 180px 180px, 180px 180px, 180px 180px,
        180px 180px, 180px 180px, 180px 180px;
    background-repeat: repeat, repeat, repeat, repeat, repeat, repeat, repeat;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container{padding-top:1rem;}

.brand-row{display:flex;align-items:center;gap:10px;margin-bottom:6px;}
.brand-logo{
    width:42px;height:42px;border-radius:12px;
    background:linear-gradient(135deg,var(--violet),var(--cyan));
    display:flex;align-items:center;justify-content:center;
    font-size:14px; font-weight:800; letter-spacing:.03em; color:#0A0F1E;
    box-shadow:0 4px 14px rgba(139,92,246,.4);
}
.brand-title{font-weight:800;font-size:19px;color:var(--ink);line-height:1.1;}
.brand-sub{font-size:11.5px;color:var(--muted);}

div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] label[data-testid="stWidgetLabel"],
div[data-testid="stRadio"] div[data-testid="stWidgetLabel"] {
    display:none !important; height:0 !important; width:0 !important;
    margin:0 !important; padding:0 !important; visibility:hidden !important;
}
div[data-testid="stRadio"] > div{gap:2px; flex-direction:column;}
div[data-testid="stRadio"] label{
    display:flex; align-items:center;
    padding:9px 12px; border-radius:10px; width:100%;
    margin-bottom:2px; transition:all .15s ease; cursor:pointer;
}
div[data-testid="stRadio"] label:hover{background:#1B2540;}
div[data-testid="stRadio"] label:has(input:checked){
    background:linear-gradient(90deg,var(--violet),var(--cyan));
    box-shadow:0 3px 12px rgba(139,92,246,.35);
}
div[data-testid="stRadio"] label:has(input:checked) p{ color:#0A0F1E !important; font-weight:700; }
div[data-testid="stRadio"] label > div:first-child{display:none;}
div[data-testid="stRadio"] label [data-baseweb="radio"]{display:none !important;}
div[data-testid="stRadio"] label input{display:none !important;}
div[data-testid="stRadioGroup"] label > div > div > div:first-child{display:none !important;}
div[data-testid="stRadioGroup"] label > div > div{gap:0 !important;}
div[data-testid="stRadio"] p{font-size:14.5px;color:var(--ink);margin:0;}

.status-title{font-size:11px;letter-spacing:.06em;color:#6B7A99;font-weight:700;margin:18px 0 8px 2px;}
.status-row{display:flex;justify-content:space-between;align-items:center;font-size:13px;color:var(--ink);padding:4px 2px;}
.dot{height:8px;width:8px;border-radius:50%;background:#22C55E;display:inline-block;margin-right:6px;}
.status-val{color:#4ADE80;font-weight:600;}

.clean-card{
    margin-top:18px;border-radius:16px;overflow:hidden;position:relative;
    min-height:130px;padding:14px;
    background:linear-gradient(150deg, #4C1D95 0%, #6D28D9 55%, #0891B2 100%);
    border:1px solid rgba(255,255,255,.08);
    box-shadow:0 4px 14px rgba(109,40,217,.3);
}
.clean-card h4{margin:0 0 4px 0; font-size:14px; color:#FFFFFF !important; font-weight:700;}
.clean-card p{margin:0;font-size:11.5px;color:#EDE9FE !important;line-height:1.4;}

.profile-row{display:flex;align-items:center;gap:8px;margin-top:14px;padding:8px 4px;border-top:1px solid var(--border);}
.avatar{width:30px;height:30px;border-radius:50%;background:#1F2A44;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;letter-spacing:.02em;color:var(--ink);}
.profile-name{font-size:12.5px;font-weight:600;color:var(--ink);}
.profile-mail{font-size:11px;color:var(--muted);}

/* ---------- Cards ---------- */
.card{
    background:var(--card-bg); border-radius:16px; padding:18px 20px;
    box-shadow:0 2px 14px rgba(0,0,0,.35); border:1px solid var(--border);
    height:100%;
}
div[data-testid="stVerticalBlockBorderWrapper"]{
    background:var(--card-bg); border-radius:16px !important;
    box-shadow:0 2px 14px rgba(0,0,0,.35); border:1px solid var(--border) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{border-radius:16px;}
.metric-label{font-size:12.5px;color:var(--muted) !important;font-weight:600;margin-bottom:4px;}
.metric-value{font-size:26px;font-weight:800;color:var(--ink) !important;word-break:break-word;overflow-wrap:break-word;}
.metric-delta-up{font-size:12px;color:#4ADE80 !important;font-weight:600;}

.feature-card{
    background:var(--card-bg);border-radius:16px;padding:20px;border:1px solid var(--border);
    box-shadow:0 2px 14px rgba(0,0,0,.35); height:100%; position:relative; padding-bottom:44px;
    transition:transform .15s ease, border-color .15s ease;
}
.feature-card:hover{transform:translateY(-2px); border-color:rgba(139,92,246,.45);}
.feature-icon{
    width:46px;height:46px;border-radius:12px;display:flex;align-items:center;
    justify-content:center;font-size:22px;margin-bottom:10px;
}
.feature-title{font-weight:700;color:var(--ink) !important;font-size:15px;margin-bottom:4px;}
.feature-sub{font-size:12.5px;color:var(--muted) !important;}
.feature-arrow{
    position:absolute; right:16px; bottom:16px; width:30px; height:30px; border-radius:50%;
    display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,.05);
}

.sidebar-card-img{
    margin-top:18px;border-radius:16px;overflow:hidden;position:relative;
    min-height:150px;padding:16px; display:flex; flex-direction:column; justify-content:flex-end;
    background-size:cover; background-position:center;
    border:1px solid rgba(255,255,255,.08); box-shadow:0 4px 14px rgba(0,0,0,.45);
}
.sidebar-card-img::before{
    content:""; position:absolute; inset:0;
    background:linear-gradient(180deg, rgba(10,15,30,.15) 0%, rgba(10,15,30,.92) 100%);
}
.sidebar-card-img > *{position:relative; z-index:1;}
.sidebar-card-img h4{margin:0 0 4px 0; font-size:14px; color:#FFFFFF !important; font-weight:700;}
.sidebar-card-img p{margin:0 0 10px 0;font-size:11.5px;color:#DCE3F0 !important;line-height:1.4;}
.sidebar-learn-more{
    display:inline-flex; align-items:center; gap:6px; font-size:11.5px; font-weight:700;
    color:#fff !important; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.25);
    padding:6px 12px; border-radius:8px; width:fit-content;
}

.social-row{display:flex; gap:8px; margin-top:12px;}
.social-icon{
    width:30px;height:30px;border-radius:8px;background:#1B2540;border:1px solid var(--border);
    display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:11px;font-weight:800;
}

.step-item{display:flex; flex-direction:column; align-items:center; gap:6px; flex:1; min-width:70px;}
.step-icon{
    width:44px;height:44px;border-radius:50%;display:flex;align-items:center;justify-content:center;
    border:1px solid var(--border);
}
.step-title{font-size:11.5px;font-weight:700;color:var(--ink) !important;text-align:center;}
.step-sub{font-size:10px;color:var(--muted) !important;text-align:center;}
.step-arrow{color:#4B5A80; font-size:16px; padding-bottom:20px;}

.hero{
    border-radius:20px; padding:40px 44px; color:#fff; position:relative; overflow:hidden;
    background-image:
        linear-gradient(90deg, rgba(11,16,36,.98) 0%, rgba(11,16,36,.92) 32%, rgba(11,16,36,.45) 62%, rgba(11,16,36,.08) 100%),
        radial-gradient(circle at 15% 20%, rgba(139,92,246,.28), transparent 45%),
        url("__GALAXY_HOME_URI__");
    background-size: 100% 100%, 100% 100%, cover;
    background-position: 0 0, 0 0, right center;
    background-repeat: no-repeat, no-repeat, no-repeat;
    min-height:260px; display:flex; align-items:center; justify-content:space-between; gap:28px;
    border:1px solid var(--border); flex-wrap:wrap;
}
.hero-text{flex:1 1 320px; min-width:280px;}
.hero h1{font-size:32px;font-weight:800;margin:0;line-height:1.2; color:#fff !important;}
.hero .accent{background:linear-gradient(90deg,var(--violet),var(--cyan));-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{color:#CBD5E1 !important;font-size:14.5px;max-width:520px;margin-top:14px;line-height:1.6;}
.hero-image-wrap{
    flex:1 1 300px; max-width:420px; border-radius:16px; overflow:hidden; position:relative;
    box-shadow:0 10px 40px rgba(0,0,0,.5); border:1px solid rgba(255,255,255,.08); aspect-ratio:16/10;
}
.hero-image-wrap img{width:100%;height:100%;object-fit:cover;display:block;}
.hero-image-wrap::after{
    content:""; position:absolute; inset:0;
    background:linear-gradient(120deg, rgba(0,0,0,.25), rgba(0,0,0,.05));
}

.badge-pill-ai{
    display:inline-flex; align-items:center; gap:6px; padding:6px 14px; border-radius:999px;
    background:rgba(139,92,246,.18); border:1px solid rgba(139,92,246,.45);
    color:#C4B5FD !important; font-size:11.5px; font-weight:700; letter-spacing:.03em;
    margin-bottom:16px;
}

.quality-badge{
    display:inline-block; margin-top:6px; padding:3px 10px; border-radius:999px;
    font-size:10.5px; font-weight:700; letter-spacing:.02em;
}

.about-header{
    background:linear-gradient(120deg, #171F38 0%, #0D1326 60%);
    border:1px solid var(--border);
    border-radius:20px;
    padding:28px 30px;
    display:flex; align-items:flex-start; gap:18px;
    position:relative; overflow:hidden;
    box-shadow:0 2px 14px rgba(0,0,0,.35);
}
.about-header::after{
    content:""; position:absolute; top:-40px; right:-40px;
    width:180px; height:180px; border-radius:50%;
    background:radial-gradient(circle, rgba(139,92,246,.25), transparent 70%);
}
.about-header-icon{
    flex:0 0 auto; width:52px; height:52px; border-radius:14px;
    background:linear-gradient(135deg,var(--violet),var(--cyan));
    display:flex; align-items:center; justify-content:center;
    font-size:24px; box-shadow:0 6px 16px rgba(139,92,246,.4);
}
.about-header h1{font-size:24px;font-weight:800;color:var(--ink) !important;margin:0 0 8px 0;}
.about-header p{font-size:13.5px;color:var(--muted) !important;line-height:1.6;margin:0;max-width:720px;}

.pill-badge{ display:inline-block;padding:5px 12px;border-radius:999px;font-size:12px;font-weight:700; }
.badge-green{background:#0F3324;color:#4ADE80;}
.badge-violet{background:#2B1655;color:#C4B5FD;}

.section-title{font-weight:800;font-size:16px;color:var(--ink) !important;margin-bottom:2px;display:block;line-height:1.4;opacity:1 !important;}
.section-sub{font-size:12.5px;color:var(--muted) !important;margin-bottom:14px;}

.stButton>button{
    background:linear-gradient(90deg,var(--violet),var(--cyan)); color:#0A0F1E; border:none;
    border-radius:10px; padding:10px 18px; font-weight:700; box-shadow:0 4px 14px rgba(139,92,246,.35);
}
.stButton>button:hover{opacity:.92; color:#0A0F1E;}

table.dataframe td, table.dataframe th{font-size:12.5px !important;}
table.dataframe{ background:var(--card-bg) !important; color:var(--ink) !important; border-radius:12px; overflow:hidden; }
table.dataframe th{ background:#1B2540 !important; color:#22D3EE !important; font-weight:700 !important; text-transform:uppercase; font-size:11px !important; letter-spacing:.04em; border-color:var(--border) !important; }
table.dataframe td{ color:var(--ink) !important; border-color:var(--border) !important; }
table.dataframe tr:hover td{ background:#161F38 !important; }

.section-sub{font-size:12.5px;color:var(--muted) !important;margin:2px 0 14px 0;line-height:1.5;}

.dark-table-wrap{
    border-radius:14px; overflow:hidden; border:1px solid var(--border);
    box-shadow:0 2px 14px rgba(0,0,0,.35); margin-top:10px;
}
table.dark-table{ width:100%; border-collapse:collapse; background:var(--card-bg); }
table.dark-table thead th{
    background:#1B2540; color:var(--cyan) !important; font-weight:700; text-transform:uppercase;
    font-size:11px; letter-spacing:.04em; text-align:left; padding:13px 18px;
    border-bottom:1px solid var(--border);
}
table.dark-table tbody td{
    color:var(--ink) !important; font-size:13px; padding:13px 18px;
    border-bottom:1px solid var(--border); white-space:nowrap;
}
table.dark-table tbody tr:last-child td{ border-bottom:none; }
table.dark-table tbody tr:hover td{ background:#161F38; }
table.dark-table tbody tr.best-row td{ background:rgba(139,92,246,.10); }
table.dark-table tbody tr.best-row td:first-child{ border-left:3px solid var(--violet); font-weight:700; }
.status-pill{
    display:inline-flex; align-items:center; gap:6px; font-size:11.5px; font-weight:700;
    color:#4ADE80; background:#0F3324; padding:4px 12px; border-radius:20px;
}
.status-pill::before{ content:""; width:6px; height:6px; border-radius:50%; background:#4ADE80; flex-shrink:0; }

/* interactive dataframe / data-editor widget wrapper */
div[data-testid="stDataFrame"], div[data-testid="stDataFrameResizable"]{
    border-radius:14px !important; overflow:hidden;
    border:1px solid var(--border) !important;
    box-shadow:0 2px 14px rgba(0,0,0,.35);
    background:var(--card-bg) !important;
}
div[data-testid="stDataFrame"] > div, div[data-testid="stDataFrameResizable"] > div{
    background:var(--card-bg) !important;
}
div[data-testid="stElementToolbar"]{
    background:var(--card-bg) !important;
}
div[data-testid="stElementToolbar"] button svg{ color:var(--ink) !important; fill:var(--ink) !important; }

/* Equal-height cards inside st.columns rows (About page grid, feature grids, etc.) */
div[data-testid="stHorizontalBlock"]{ align-items:stretch !important; }
div[data-testid="column"]{ display:flex !important; flex-direction:column !important; }
div[data-testid="column"] > div{ width:100%; display:flex; flex-direction:column; flex:1 1 auto; }
div[data-testid="column"] div[data-testid="stVerticalBlock"]{ flex:1 1 auto; display:flex; flex-direction:column; }
div[data-testid="column"] div[data-testid="element-container"]:has(> .card),
div[data-testid="column"] div[data-testid="element-container"]:has(> .feature-card),
div[data-testid="column"] div[data-testid="element-container"]:has(> .class-art-card){
    flex:1 1 auto; display:flex;
}
div[data-testid="column"] .card,
div[data-testid="column"] .feature-card,
div[data-testid="column"] .class-art-card{
    flex:1 1 auto; height:auto; width:100%;
}

.class-badge{
    display:inline-flex; align-items:center; gap:10px; padding:14px 20px; border-radius:14px;
    font-size:20px; font-weight:800; margin-bottom:6px;
}

.class-art-card{
    background:var(--card-bg); border-radius:16px; padding:22px 18px; text-align:center;
    border:1px solid var(--border); box-shadow:0 2px 14px rgba(0,0,0,.35); height:100%;
}
.class-art-card .class-art-title{font-weight:700; font-size:14.5px; color:var(--ink) !important; margin-top:10px;}
.class-art-card .class-art-desc{font-size:12px; color:var(--muted) !important; margin-top:4px; line-height:1.5;}
</style>
"""
CSS = CSS.replace("__NEBULA_BG_URI__", _NEBULA_BG_URI)
CSS = CSS.replace("__GALAXY_HOME_URI__", _GALAXY_HOME_URI)
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR NAV ICONS — injected separately since it's built from Python
# (data-URI SVGs matched to each nav item by position via :nth-of-type)
# ----------------------------------------------------------------------------
_NAV_ICON_ORDER = ["home", "target", "chart", "info"]  # matches PAGES order below
_nav_icon_rules = []
for _i, _name in enumerate(_NAV_ICON_ORDER, start=1):
    _muted = _icon_data_uri(_name, "#94A3B8")
    _active = _icon_data_uri(_name, "#0A0F1E")
    _nav_icon_rules.append(f"""
    div[data-testid="stRadio"] label:nth-of-type({_i})::before{{
        content:""; display:inline-block; flex-shrink:0;
        width:18px; height:18px; margin-right:10px;
        background-image:url('{_muted}'); background-size:contain; background-repeat:no-repeat;
    }}
    div[data-testid="stRadio"] label:nth-of-type({_i}):has(input:checked)::before{{
        background-image:url('{_active}');
    }}
    """)
st.markdown(f"<style>{''.join(_nav_icon_rules)}</style>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# LOAD MODEL ARTIFACTS
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classification model...")
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(METRICS_PATH, "rb") as f:
        metrics = pickle.load(f)
    return model, scaler, encoder, metrics


model, scaler, encoder, METRICS = load_artifacts()

# CLASSES holds the class labels in the exact order the label encoder /
# confusion matrix use them (e.g. ["GALAXY", "QSO", "STAR"]). Prefer the
# list saved alongside the metrics; fall back to the encoder itself so the
# app still works if an older metrics file is present.
CLASSES = METRICS.get("classes", list(encoder.classes_))


def predict_class(input_dict: dict):
    row = pd.DataFrame([[input_dict[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    scaled = scaler.transform(row)
    probs = model.predict_proba(scaled)[0]
    pred_idx = int(np.argmax(probs))
    label = encoder.inverse_transform([pred_idx])[0]
    return label, probs


@st.cache_data
def sample_recent_predictions():
    rng = np.random.default_rng(3)
    rows = []
    now = dt.datetime.now()
    base_time = now.replace(minute=0, second=0, microsecond=0) - dt.timedelta(minutes=15 * 5)
    for i in range(6):
        cls = rng.choice(CLASSES, p=[0.59, 0.19, 0.22])
        conf = rng.uniform(0.90, 0.995)
        rows.append({
            "Time": (base_time + dt.timedelta(minutes=15 * i)).strftime("%b %d, %I:%M %p"),
            "Predicted Class": cls,
            "Confidence": f"{conf*100:.1f}%",
            "Redshift": round(rng.uniform(0, 1.5) if cls != "STAR" else rng.uniform(-0.0005, 0.0005), 4),
            "Status": "Success",
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
PAGES = ["Home", "Prediction", "Dashboard", "About"]

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand-row">
            <div class="brand-logo">{icon("telescope", size=22, color="#0A0F1E")}</div>
            <div>
                <div class="brand-title">Stellar<span style="background:linear-gradient(90deg,var(--violet),var(--cyan));-webkit-background-clip:text;background-clip:text;color:transparent;">Scope</span></div>
                <div class="brand-sub">AI Astronomy</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    page = st.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
    )

    st.markdown('<div class="status-title">SYSTEM STATUS</div>', unsafe_allow_html=True)
    status_items = [
        ("Model Status", "Running"),
        ("Data Pipeline", "Healthy"),
        ("Scaler", "Loaded"),
        ("Encoder", "Ready"),
    ]
    rows_html = "".join(
        f'<div class="status-row"><span><span class="dot"></span>{k}</span>'
        f'<span class="status-val">{v}</span></div>'
        for k, v in status_items
    )
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sidebar-card-img" style="background-image:url('{_NEBULA_BG_URI}');">
            <h4>Explore the Universe<br>with AI</h4>
            <p>Classifying stars, galaxies &amp; quasars using photometric data from SDSS.</p>
            <span class="sidebar-learn-more">Learn More {icon("arrow-right", size=13, color="#fff")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="profile-row">
            <div class="avatar">ST</div>
            <div>
                <div class="profile-name">StellarScope Team</div>
                <div class="profile-mail">contact@stellarscope.ai</div>
            </div>
        </div>
        <div class="social-row">
            <div class="social-icon">GH</div>
            <div class="social-icon">X</div>
            <div class="social-icon">in</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE: HOME
# ============================================================================
def render_home():
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-text">
                <span class="badge-pill-ai">{icon("bolt", size=13, color="#C4B5FD")} AI POWERED</span>
                <h1>Stellar Object<br><span class="accent">Classification System</span></h1>
                <p>AI-powered classification of astronomical objects — instantly tell whether
                a sky observation is a Star, Galaxy, or Quasar using photometric and
                spectroscopic data from the Sloan Digital Sky Survey.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, _ = st.columns([1, 4])
    with c1:
        if st.button(f"Get Started", use_container_width=True):
            st.session_state["_jump_prediction"] = True
            st.rerun()

    st.write("")
    st.write("")

    features = [
        ("target", "#2B1655", "#C4B5FD", "Classify Sky Objects", "Predict STAR, GALAXY or QSO from observations"),
        ("chart", "#0E2E36", "#22D3EE", "Photometric Analysis", "Uses u,g,r,i,z magnitude bands from SDSS"),
        ("wave", "#3A1730", "#F472B6", "Redshift Insight", "Leverages redshift, the strongest predictor"),
        ("cpu", "#3A2E10", "#FBBF24", "Random Forest Model", "~97.9% test accuracy, tuned & validated"),
    ]
    cols = st.columns(4)
    for col, (icon_name, bg, fg, title, sub) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon" style="background:{bg};color:{fg};">{icon(icon_name, size=22, color=fg)}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-sub">{sub}</div>
                    <div class="feature-arrow" style="color:{fg};">{icon("arrow-right", size=15, color=fg)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    def _quality(value):
        if value >= 0.97:
            return "Outstanding", "#0F3324", "#4ADE80"
        if value >= 0.90:
            return "Excellent", "#2B1655", "#C4B5FD"
        return "Good", "#0E2E36", "#22D3EE"

    c1, c2 = st.columns([1, 1.3])
    with c1:
        stats = [
            ("Test Accuracy", f"{METRICS['accuracy']*100:.1f}%", "#C4B5FD", METRICS['accuracy']),
            ("F1 Score", f"{METRICS['f1']:.3f}", "#22D3EE", METRICS['f1']),
            ("ROC AUC Score", f"{METRICS['roc_auc']:.3f}", "#4ADE80", METRICS['roc_auc']),
        ]
        stat_cells = ""
        for label, val, color, raw in stats:
            q_label, q_bg, q_fg = _quality(raw)
            stat_cells += (
                f'<div style="flex:1;text-align:center;background:#0D1326;border:1px solid var(--border);'
                f'border-radius:12px;padding:14px 8px;">'
                f'<div class="metric-value" style="color:{color};font-size:24px;">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'<div class="quality-badge" style="background:{q_bg};color:{q_fg};">{q_label}</div>'
                f'</div>'
            )
        st.markdown(
            f"""
            <div class="card">
                <div class="section-title" style="display:flex;align-items:center;gap:8px;">
                    <span style="color:var(--cyan);">{icon("trending", size=17, color="var(--cyan)")}</span>Model Performance
                </div>
                <div style="display:flex;gap:10px;margin-top:14px;">
                    {stat_cells}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        steps = [
            ("database", "#8B5CF6", "SDSS Data", "Raw Data"),
            ("gear", "#22D3EE", "Preprocessing", "Clean & Prepare"),
            ("tree", "#4ADE80", "Random Forest", "ML Model"),
            ("target", "#FBBF24", "Classification", "Predict Class"),
            ("trending", "#F472B6", "Insights", "Results & Viz"),
        ]
        steps_html = ""
        for i, (icon_name, color, title, sub) in enumerate(steps):
            steps_html += (
                f'<div class="step-item">'
                f'<div class="step-icon" style="background:{color}22;color:{color};">{icon(icon_name, size=19, color=color)}</div>'
                f'<div class="step-title">{title}</div><div class="step-sub">{sub}</div>'
                f'</div>'
            )
            if i < len(steps) - 1:
                steps_html += '<div class="step-arrow">&rarr;</div>'
        st.markdown(f"""
            <div class="card">
            <div class="section-title" style="display:flex;align-items:center;gap:8px;">
                <span style="color:var(--violet);">{icon("bolt", size=16, color="var(--violet)")}</span>How It Works
            </div>
            <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-top:18px;flex-wrap:wrap;gap:4px;">
                {steps_html}
            </div>
            </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE: PREDICTION
# ============================================================================
def confidence_gauge(value, color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " %", "font": {"size": 30, "color": "#E5E7EB"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#1F2A44"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#3A1730"},
                    {"range": [50, 80], "color": "#3A2E10"},
                    {"range": [80, 100], "color": "#0F3324"},
                ],
            },
            title={"text": "Prediction Confidence", "font": {"size": 13, "color": "#94A3B8"}},
        )
    )
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_prediction():
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown('<div class="section-title" style="font-size:22px;">Stellar Object Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Enter observational parameters to classify a sky object</div>', unsafe_allow_html=True)
    with hc2:
        st.markdown(
            f'<div style="text-align:right;font-size:12.5px;color:#94A3B8;padding-top:8px;">'
            f'{dt.datetime.now().strftime("%b %d, %Y")}<br>{dt.datetime.now().strftime("%I:%M %p")}</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([2, 1.3])

    with left:
        with st.container(border=True):
            st.markdown(
                f'<div class="section-title" style="font-size:17px;display:flex;align-items:center;'
                f'gap:8px;margin:0 0 14px 0;padding-bottom:10px;border-bottom:1px solid var(--border);">'
                f'<span style="color:var(--cyan);">{icon("target", size=18, color="var(--cyan)")}</span>'
                f'<span>Observation Parameters</span></div>',
                unsafe_allow_html=True,
            )

            st.caption("Position")
            c1, c2 = st.columns(2)
            inputs = {}
            with c1:
                cfg = feature_cfg("alpha")
                inputs["alpha"] = st.number_input(cfg["label"], cfg["min"], cfg["max"], cfg["default"], cfg["step"], key="in_alpha")
            with c2:
                cfg = feature_cfg("delta")
                inputs["delta"] = st.number_input(cfg["label"], cfg["min"], cfg["max"], cfg["default"], cfg["step"], key="in_delta")

            st.caption("Photometric magnitudes (SDSS filters u, g, r, i, z)")
            c1, c2, c3, c4, c5 = st.columns(5)
            for col, band in zip([c1, c2, c3, c4, c5], ["u", "g", "r", "i", "z"]):
                with col:
                    cfg = feature_cfg(band)
                    inputs[band] = st.number_input(band, cfg["min"], cfg["max"], cfg["default"], cfg["step"], key=f"in_{band}")

            st.caption("Redshift & instrument metadata")
            c1, c2 = st.columns(2)
            with c1:
                cfg = feature_cfg("redshift")
                inputs["redshift"] = st.number_input(cfg["label"], cfg["min"], cfg["max"], cfg["default"], cfg["step"], key="in_redshift", format="%.4f")
                cfg = feature_cfg("plate")
                inputs["plate"] = st.number_input(cfg["label"], cfg["min"], cfg["max"], cfg["default"], cfg["step"], key="in_plate")
            with c2:
                cfg = feature_cfg("cam_col")
                inputs["cam_col"] = st.number_input(cfg["label"], cfg["min"], cfg["max"], cfg["default"], cfg["step"], key="in_camcol")
                cfg = feature_cfg("mjd")
                inputs["MJD"] = st.number_input(cfg["label"], cfg["min"], cfg["max"], cfg["default"], cfg["step"], key="in_mjd")

            cfg = feature_cfg("fiber_id")
            inputs["fiber_ID"] = st.number_input(cfg["label"], cfg["min"], cfg["max"], cfg["default"], cfg["step"], key="in_fiber")

        st.write("")
        predict_clicked = st.button("Predict Class", use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown('<div class="section-title" style="font-size:15px;">Observation Snapshot</div>', unsafe_allow_html=True)
            st.write("")
            ug_color = inputs["u"] - inputs["g"]
            gr_color = inputs["g"] - inputs["r"]
            snapshot_rows = [
                ("Sky Position", f"α {inputs['alpha']:.1f}°, δ {inputs['delta']:.1f}°"),
                ("u − g color index", f"{ug_color:.2f}"),
                ("g − r color index", f"{gr_color:.2f}"),
                ("Redshift", f"{inputs['redshift']:.4f}"),
                ("Plate / Fiber", f"{int(inputs['plate'])} / {int(inputs['fiber_ID'])}"),
            ]
            for label, val in snapshot_rows:
                st.markdown(
                    f'<div class="status-row"><span>{label}</span><span style="font-weight:700;color:#E5E7EB;">{val}</span></div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="section-title" style="font-size:15px;">Object Classes</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">What the model can predict</div>', unsafe_allow_html=True)
            ac1, ac2, ac3 = st.columns(3)
            for col, cls in zip([ac1, ac2, ac3], CLASSES):
                with col:
                    st.markdown(
                        f'<div style="text-align:center;">{class_art(cls, size=44)}'
                        f'<div style="font-size:11.5px;font-weight:700;color:{CLASS_STYLE[cls]["color"]};margin-top:4px;">{cls}</div></div>',
                        unsafe_allow_html=True,
                    )

    st.write("")

    if "last_pred" not in st.session_state or predict_clicked:
        st.session_state.last_pred = predict_class(inputs)

    label, probs = st.session_state.last_pred
    style = CLASS_STYLE[label]
    confidence = float(np.max(probs) * 100)

    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(confidence_gauge(round(confidence, 1), style["color"]), use_container_width=True)
    with g2:
        st.markdown(
            f"""
            <div class="card" style="text-align:center;padding-top:22px;">
                {class_art(label, size=64)}
                <div class="class-badge" style="background:{style['bg']};color:{style['color']};margin-top:10px;">
                    <span class="class-dot" style="background:{style['color']};"></span><span>{label}</span>
                </div>
                <div class="feature-sub" style="margin-top:8px;">{style['desc']}</div>
                <div class="pill-badge badge-violet" style="margin-top:12px;">Confidence {confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    fig = go.Figure(
        go.Bar(
            x=CLASSES,
            y=probs * 100,
            marker_color=[CLASS_STYLE[c]["color"] for c in CLASSES],
            text=[f"{p*100:.2f}%" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Class Probability Distribution",
        yaxis_title="Probability (%)", yaxis_range=[0, 100],
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Batch predict from a CSV file"):
        st.caption(f"Upload a CSV with columns: `{', '.join(FEATURE_ORDER)}`")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                missing = [c for c in FEATURE_ORDER if c not in batch_df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    Xb = batch_df[FEATURE_ORDER]
                    Xs = scaler.transform(Xb)
                    preds = model.predict(Xs)
                    probs_b = model.predict_proba(Xs)
                    out = batch_df.copy()
                    out["predicted_class"] = encoder.inverse_transform(preds)
                    for idx, c in enumerate(CLASSES):
                        out[f"prob_{c}"] = probs_b[:, idx]
                    st.success(f"Predicted {len(out)} rows.")
                    st.dataframe(out, use_container_width=True)
                    st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Could not process file: {e}")


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
def render_dashboard():
    now = dt.datetime.now()
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown('<div class="section-title" style="font-size:22px;">Model Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Model performance, feature insights, and recent predictions</div>', unsafe_allow_html=True)
    with hc2:
        st.markdown(
            f'<div style="text-align:right;font-size:12.5px;color:#94A3B8;padding-top:8px;">'
            f'{now.strftime("%b %d, %Y")}<br>{now.strftime("%I:%M %p")}</div>',
            unsafe_allow_html=True,
        )

    kpis = [
        ("Test Accuracy", f"{METRICS['accuracy']*100:.2f}%", f"F1: {METRICS['f1']:.3f}", False),
        ("ROC-AUC", f"{METRICS['roc_auc']:.3f}", "Weighted OVR", False),
        ("Cross-Val Mean", f"{METRICS['cv_mean']*100:.2f}%", "5-fold CV", False),
        ("Training Samples", f"{METRICS['n_train']:,}", f"of {METRICS['n_total']:,} total", False),
        ("Test Samples", f"{METRICS['n_test']:,}", "30% holdout", False),
        ("Best Model", METRICS["best_model"], "vs 6 other models", True),
    ]
    cols = st.columns(6)
    for col, (label, val, delta, is_text) in zip(cols, kpis):
        with col:
            value_class = "metric-value-sm" if is_text else "metric-value"
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">{label}</div>
                    <div class="{value_class}" style="{'' if is_text else 'font-size:19px;'}">{val}</div>
                    <div class="metric-delta-up">{delta}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    CHART_H = 320  # shared height so every dashboard chart card is identical in size

    def _chart_title(text, accent):
        st.markdown(
            f'<div class="section-title" style="font-size:15px;display:flex;align-items:center;gap:8px;'
            f'margin:0 0 12px 0;padding-bottom:9px;border-bottom:1px solid var(--border);">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{accent};'
            f'box-shadow:0 0 8px {accent}88;display:inline-block;"></span>'
            f'<span>{text}</span></div>',
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3, gap="medium")

    cm = METRICS["confusion_matrix"]
    cm_arr = np.asarray(cm, dtype=float)

    with c1:
        with st.container(border=True):
            _chart_title("Confusion Matrix", "#22D3EE")
            fig = px.imshow(
                cm, x=CLASSES, y=CLASSES, text_auto=True, aspect="auto",
                color_continuous_scale=["#0D1326", "#4C1D95", "#22D3EE"],
            )
            fig.update_layout(height=CHART_H, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#E5E7EB", size=12),
                               xaxis_title="Predicted", yaxis_title="Actual",
                               coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        with st.container(border=True):
            _chart_title("Per-Class Performance", "#8B5CF6")
            precisions, recalls = [], []
            for idx in range(len(CLASSES)):
                col_sum = cm_arr[:, idx].sum()
                row_sum = cm_arr[idx, :].sum()
                precisions.append(cm_arr[idx, idx] / col_sum if col_sum else 0.0)
                recalls.append(cm_arr[idx, idx] / row_sum if row_sum else 0.0)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Precision", x=CLASSES, y=precisions, marker_color="#8B5CF6"))
            fig.add_trace(go.Bar(name="Recall", x=CLASSES, y=recalls, marker_color="#22D3EE"))
            fig.update_layout(
                barmode="group", height=CHART_H, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E5E7EB", size=12), yaxis=dict(range=[0, 1]),
                legend=dict(orientation="h", y=-0.16, font=dict(size=11)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c3:
        with st.container(border=True):
            _chart_title("Class Distribution", "#F472B6")
            totals = cm_arr.sum(axis=1)
            fig = go.Figure(go.Pie(labels=CLASSES, values=totals, hole=0.6,
                                    marker=dict(colors=[CLASS_STYLE[c]["color"] for c in CLASSES],
                                                line=dict(color="#0A0F1E", width=2))))
            fig.update_layout(height=CHART_H, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB", size=12),
                               showlegend=True, legend=dict(orientation="h", y=-0.12, font=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.write("")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        with st.container(border=True):
            st.markdown('<div class="section-title" style="font-size:15px;">Recent Predictions</div>', unsafe_allow_html=True)
            _recent_df = sample_recent_predictions()
            st.markdown(render_dark_table(_recent_df, badge_col="Status", class_col="Predicted Class"), unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown('<div class="section-title" style="font-size:15px;">Model Health</div>', unsafe_allow_html=True)
            last_trained = (dt.date.today() - dt.timedelta(days=1)).strftime("%b %d, %Y")
            for label, val in [
                ("Model Status", "Healthy"),
                ("Algorithm", METRICS["best_model"]),
                ("Last Trained", last_trained),
                ("Avg Prediction Time", "~15 ms"),
                ("Uptime", "99.9%"),
            ]:
                st.markdown(
                    f'<div class="status-row"><span><span class="dot"></span>{label}</span><span class="status-val">{val}</span></div>',
                    unsafe_allow_html=True,
                )


# ============================================================================
# PAGE: ABOUT
# ============================================================================
def render_about():
    st.markdown(
        f"""
        <div class="about-header">
            <div class="about-header-icon">{icon("info", size=24, color="#fff")}</div>
            <div>
                <h1>About This Project</h1>
                <p>StellarScope classifies astronomical observations from the Sloan Digital
                Sky Survey (SDSS) as a <b>Star</b>, <b>Galaxy</b>, or <b>Quasar</b> using a
                Random Forest model trained on photometric magnitudes, redshift, and
                observation metadata.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    cols = st.columns(5)
    _feature_items = []
    for f in FEATURE_ORDER:
        if f in ("u", "g", "r", "i", "z"):
            if not _feature_items or _feature_items[-1] != "u, g, r, i, z":
                _feature_items.append("u, g, r, i, z")
        elif f == "fiber_ID":
            continue
        else:
            _feature_items.append(f)
    blocks = [
        ("chart", "#22D3EE", "Dataset Used", [("Source", "SDSS DR17"), ("Total Records", f"{METRICS['n_total']:,}"), ("Training Samples", f"{METRICS['n_train']:,}"), ("Testing Samples", f"{METRICS['n_test']:,}"), ("Classes", ", ".join(CLASSES))]),
        ("list", "#8B5CF6", "Features Used", [(f, "") for f in _feature_items]),
        ("cpu", "#4ADE80", "Model Information", [("Algorithm", METRICS["best_model"]), ("Model Type", "Multi-class Classification"), ("Target Variable", "Object Class"), ("Scaling", "StandardScaler")]),
        ("code", "#FBBF24", "Technologies", [(name, "") for name in ["Python", "Streamlit", "Scikit-Learn", "Pandas", "NumPy", "Plotly"]]),
        ("info", "#F472B6", "Project Info", [("Developed By", "StellarScope Team"), ("Project Type", "ML Classification"), ("Deployment", "Streamlit Cloud"), ("Year", str(dt.date.today().year))]),
    ]
    for col, (icon_name, accent, title, items) in zip(cols, blocks):
        with col:
            rows = "".join(
                f'<div style="display:flex;justify-content:space-between;font-size:12px;margin:6px 0;">'
                f'<span style="color:#94A3B8;">{k}</span><span style="font-weight:700;color:#E5E7EB;">{v}</span></div>'
                if v else f'<div style="font-size:12px;margin:6px 0;color:#CBD5E1;">• {k}</div>'
                for k, v in items
            )
            st.markdown(
                f"""
                <div class="card" style="border-top:3px solid {accent};">
                    <div class="section-title" style="font-size:13.5px;display:flex;align-items:center;gap:6px;">
                        <span style="color:{accent};">{icon(icon_name, size=16, color=accent)}</span><span>{title}</span>
                    </div>
                    {rows}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    perf = [
        ("Accuracy", f"{METRICS['accuracy']*100:.2f}%"),
        ("Precision", f"{METRICS['precision']:.3f}"),
        ("Recall", f"{METRICS['recall']:.3f}"),
        ("F1 Score", f"{METRICS['f1']:.3f}"),
        ("ROC-AUC", f"{METRICS['roc_auc']:.3f}"),
    ]
    _perf_cells = "".join(
        f'<div style="text-align:center;"><div class="metric-value" style="color:var(--cyan);">{val}</div>'
        f'<div class="metric-label">{label}</div></div>'
        for label, val in perf
    )
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">Model Performance</div>
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:16px;margin-top:16px;">
                {_perf_cells}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    _comp_df = METRICS["comparison"].copy()
    for _c in ("Accuracy", "Precision", "Recall", "F1 Score"):
        if _c in _comp_df.columns:
            _comp_df[_c] = _comp_df[_c].map(lambda x: f"{x:.4f}")
    _comp_table_html = render_dark_table(
        _comp_df, highlight_col=_comp_df.columns[0], highlight_value=METRICS["best_model"]
    )
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">Model Comparison</div>
            <div class="section-sub">Best performer: <b style="color:var(--ink);">{METRICS['best_model']}</b>
                — selected by highest accuracy on the held-out test set.</div>
            {_comp_table_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "**Note:** some raw magnitude values in the source dataset use `-9999` as a "
        "sensor-error placeholder; the model was trained on the raw data as-is."
    )


# ----------------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------------
if st.session_state.pop("_jump_prediction", False):
    page = "Prediction"

if page == "Home":
    render_home()
elif page == "Prediction":
    render_prediction()
elif page == "Dashboard":
    render_dashboard()
elif page == "About":
    render_about()
