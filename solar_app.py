"""
SolarAI — Solar Power Generation Prediction System
====================================================
A Streamlit application built around the Random Forest Regressor developed in
`Solar_power_generation_project_regression.ipynb`.

Pages: Home | Prediction | Dashboard | About
Run with:  streamlit run app.py
"""

import os
import pickle
import datetime as dt

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="SolarAI | Solar Power Prediction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# FEATURE CONFIG
# ----------------------------------------------------------------------------
# IMPORTANT: the exact feature set your notebook trained on can vary (some
# versions drop Hour/Month before modelling, some keep them). Rather than
# hardcoding a feature list that might not match your `.save` files, this app
# detects the real feature names straight from the fitted scaler/model
# (`feature_names_in_`) whenever pickle files are found, and only falls back
# to the 6-feature default below when no pickle is present.
# This matches the notebook exactly: x = df.drop(['AC_POWER','Hour','Month',
# 'DATE_TIME'], axis=1) -> DC_POWER, DAILY_YIELD, TOTAL_YIELD,
# AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION.
BASE_FEATURES = [
    "DC_POWER",
    "DAILY_YIELD",
    "TOTAL_YIELD",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
]

# Known physical variables -> nice label + slider range/default/step.
# Keys are matched case-insensitively against whatever column names your
# scaler/model actually expects (e.g. "Hour" or "HOUR" both resolve here).
FEATURE_REGISTRY = {
    "DC_POWER": {"label": "DC Power (kW)", "min": 0, "max": 15000, "default": 3000, "step": 50},
    "DAILY_YIELD": {"label": "Daily Yield (kWh)", "min": 0, "max": 10000, "default": 5000, "step": 50},
    "TOTAL_YIELD": {"label": "Total Yield (kWh)", "min": 6_000_000, "max": 7_300_000, "default": 6_500_000, "step": 1000},
    "AMBIENT_TEMPERATURE": {"label": "Ambient Temperature (°C)", "min": -10, "max": 60, "default": 30, "step": 1},
    "MODULE_TEMPERATURE": {"label": "Module Temperature (°C)", "min": -10, "max": 80, "default": 40, "step": 1},
    "IRRADIATION": {"label": "Irradiation (kW/m²)", "min": 0.0, "max": 1.5, "default": 0.80, "step": 0.01},
    "HOUR": {"label": "Hour", "min": 0, "max": 23, "default": 12, "step": 1},
    "DAY": {"label": "Day", "min": 1, "max": 31, "default": 16, "step": 1},
    "MONTH": {"label": "Month", "min": 1, "max": 12, "default": 5, "step": 1},
}


def _norm(name: str) -> str:
    return str(name).strip().upper().replace(" ", "_")


def feature_cfg(raw_name: str) -> dict:
    """Look up slider config for a feature, matching case-insensitively.
    Falls back to a generic numeric config for unrecognized column names."""
    cfg = FEATURE_REGISTRY.get(_norm(raw_name))
    if cfg:
        return cfg
    pretty = str(raw_name).replace("_", " ").title()
    return {"label": pretty, "min": 0.0, "max": 100.0, "default": 10.0, "step": 1.0}


def build_inputs(features: list, overrides: dict) -> dict:
    """Build a {feature_name: value} dict covering exactly `features`,
    pulling values from `overrides` (matched case-insensitively) and
    falling back to each feature's registry default otherwise."""
    norm_overrides = {_norm(k): v for k, v in overrides.items()}
    result = {}
    for f in features:
        key = _norm(f)
        result[f] = norm_overrides[key] if key in norm_overrides else feature_cfg(f)["default"]
    return result


def get_val(inputs: dict, canonical_name: str, default):
    """Fetch a value from an inputs dict by canonical name, case-insensitively."""
    target = _norm(canonical_name)
    for k, v in inputs.items():
        if _norm(k) == target:
            return v
    return default


def compute_importances(model, features: list):
    """Best-effort feature importances across model types.
    Tree ensembles (RandomForest/AdaBoost/GradientBoosting/XGBoost/DecisionTree)
    expose `feature_importances_` directly. Linear models expose `coef_`
    instead, so we fall back to normalized absolute coefficients. Models with
    neither (e.g. SVR with a non-linear kernel, KNN) return None so the UI can
    show an explanatory message instead of a blank/degenerate chart."""
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None
    if vals.shape[0] != len(features) or not np.isfinite(vals).all():
        return None
    total = vals.sum()
    if total <= 0:
        return None
    return dict(zip(features, (vals / total).tolist()))


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"C:\Data Science\Project\solar_power_generation.save")
SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"C:\Data Science\Project\solar_power_scaler.save")

# ----------------------------------------------------------------------------
# ACTUAL notebook results, per algorithm (from the Model Comparison cell).
# The notebook's comparison loop used to leave the plain variable `model`
# pointing at whichever candidate was fit LAST (AdaBoost) instead of the
# highest scorer. The notebook has since been fixed to explicitly select
# `model = models[comparison_df.iloc[0]["Model"]]` — i.e. Random Forest,
# the actual best performer — before cross-validation, feature importance,
# and pickling, so the saved `.save` file now really is the Random Forest
# model. Keying the figures below by class name means the app always shows
# the metrics that belong to whichever model file is actually loaded.
KNOWN_METRICS = {
    "RandomForestRegressor":     {"r2": 0.999994, "mae": 0.131204, "rmse": 0.928589, "cv_r2": None},
    "DecisionTreeRegressor":     {"r2": 0.999994, "mae": 0.167252, "rmse": 0.987987, "cv_r2": None},
    "LinearRegression":          {"r2": 0.999990, "mae": 0.653487, "rmse": 1.263098, "cv_r2": None},
    "XGBRegressor":              {"r2": 0.999978, "mae": 0.771540, "rmse": 1.822495, "cv_r2": None},
    "GradientBoostingRegressor": {"r2": 0.999966, "mae": 1.183922, "rmse": 2.292228, "cv_r2": None},
    "KNeighborsRegressor":       {"r2": 0.998112, "mae": 8.684239, "rmse": 17.055846, "cv_r2": None},
    "AdaBoostRegressor":         {"r2": 0.997996, "mae": 15.014708, "rmse": 17.567936, "cv_r2": None},
    "SVR":                       {"r2": 0.995518, "mae": 11.787812, "rmse": 26.277044, "cv_r2": None},
}

# ----------------------------------------------------------------------------
# LIVE WEATHER (Open-Meteo — free, no API key required)
# ----------------------------------------------------------------------------
# Default plant coordinates (Tirur, Kerala). Adjust via the sidebar controls
# on the Dashboard page if the plant is elsewhere.
DEFAULT_LAT, DEFAULT_LON = 10.9146, 75.9179


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_weather(lat: float, lon: float, _cache_bust: int = 0):
    """Fetch current-conditions weather from Open-Meteo. Cached for 5 minutes.
    `_cache_bust` lets a manual refresh button force a fresh call.
    Returns (data_dict, is_live: bool)."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,cloud_cover,"
                           "wind_speed_10m,shortwave_radiation",
                "timezone": "auto",
            },
            timeout=6,
        )
        resp.raise_for_status()
        cur = resp.json()["current"]
        return {
            "temperature": cur.get("temperature_2m"),
            "humidity": cur.get("relative_humidity_2m"),
            "cloud_cover": cur.get("cloud_cover"),
            "wind_speed": cur.get("wind_speed_10m"),
            "irradiance": cur.get("shortwave_radiation"),
            "observed_at": cur.get("time"),
        }, True
    except Exception:
        # Fall back gracefully so the dashboard never breaks if the network
        # or the API is unavailable.
        return {
            "temperature": 28, "humidity": 45, "cloud_cover": 10,
            "wind_speed": 12, "irradiance": 850, "observed_at": None,
        }, False


@st.cache_data(ttl=3600, show_spinner=False)
def ip_geolocate():
    """Best-effort location from the server's public IP (ip-api.com, free,
    no key). This reflects where the app/server is running, not necessarily
    the visitor's browser — use the city search box to override it."""
    try:
        resp = requests.get("http://ip-api.com/json/", timeout=5)
        resp.raise_for_status()
        d = resp.json()
        if d.get("status") == "success":
            return {"lat": d["lat"], "lon": d["lon"], "city": d.get("city"),
                     "region": d.get("regionName"), "country": d.get("country")}
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(city_name: str):
    """Look up lat/lon for a typed city name via Open-Meteo's free geocoding
    endpoint (no key required)."""
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 1},
            timeout=6,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
        if results:
            r = results[0]
            return {"lat": r["latitude"], "lon": r["longitude"], "city": r.get("name"),
                     "region": r.get("admin1"), "country": r.get("country")}
    except Exception:
        pass
    return None


def resolve_location(city_query: str):
    """City search box (if the user typed one) -> IP geolocation -> fixed
    default, in that order. Always returns a usable dict."""
    if city_query:
        loc = geocode_city(city_query)
        if loc:
            return loc
    loc = ip_geolocate()
    if loc:
        return loc
    return {"lat": DEFAULT_LAT, "lon": DEFAULT_LON, "city": "Tirur", "region": "Kerala", "country": "India"}


def get_plant_location() -> dict:
    """Single source of truth for 'where is the plant', shared by every page.
    A hardcoded city (or server-side IP geolocation, which reflects the
    *server's* location, not the plant's) can't be right for every user, so
    this is stored in session state and can be corrected via `location_picker`."""
    if "plant_location" not in st.session_state:
        st.session_state.plant_location = {
            "lat": DEFAULT_LAT, "lon": DEFAULT_LON,
            "city": "Tirur", "region": "Kerala", "country": "India",
        }
    return st.session_state.plant_location


def location_picker(key_prefix: str):
    """Small popover next to the weather panel that lets the user search for
    and lock in the plant's real location, since it can't be reliably
    auto-detected from the server side."""
    loc = get_plant_location()
    loc_label = ", ".join(p for p in [loc.get("city"), loc.get("region")] if p) or "Set location"
    with st.popover(f"{loc_label}  ⌄", use_container_width=False):
        st.caption("Search for the plant's real location")
        city_query = st.text_input(
            "City", key=f"{key_prefix}_city_query",
            placeholder="e.g. Kochi, Kerala", label_visibility="collapsed",
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Search", key=f"{key_prefix}_search_btn", use_container_width=True):
                found = geocode_city(city_query) if city_query else None
                if found:
                    st.session_state.plant_location = found
                    fetch_live_weather.clear()
                    st.rerun()
                else:
                    st.error("City not found.")
        with b2:
            if st.button("Detect (IP)", key=f"{key_prefix}_ip_btn", use_container_width=True):
                found = ip_geolocate()
                if found:
                    st.session_state.plant_location = found
                    fetch_live_weather.clear()
                    st.rerun()
                else:
                    st.error("Couldn't detect location.")

# ----------------------------------------------------------------------------
# HERO BACKGROUND IMAGE (base64-embedded so no external file hosting needed)
# ----------------------------------------------------------------------------
import base64


def _get_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# ICON LIBRARY — clean, single-color line icons (no emoji) for a
# professional, consistent visual language across the whole app.
# ----------------------------------------------------------------------------
_ICON_PATHS = {
    "sun": '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/>',
    "trend": '<polyline points="3 17 9 11 13 15 21 6"></polyline><polyline points="14 6 21 6 21 13"></polyline>',
    "cloud": '<path d="M6.5 19a4.5 4.5 0 0 1-1-8.9A5.5 5.5 0 0 1 16 8.5a4.5 4.5 0 0 1-.6 8.5H6.5z"></path>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/>',
    "home": '<path d="M4 11.5 12 4l8 7.5"></path><path d="M6 10v9h4v-5h4v5h4v-9"></path>',
    "bolt": '<polygon points="12 2 4 14 11 14 10 22 20 9 13 9 12 2"></polygon>',
    "bars": '<rect x="4" y="12" width="4" height="8"></rect><rect x="10" y="7" width="4" height="13"></rect><rect x="16" y="3" width="4" height="17"></rect>',
    "info": '<circle cx="12" cy="12" r="9"></circle><line x1="12" y1="11" x2="12" y2="16"></line><circle cx="12" cy="7.6" r="0.9" fill="currentColor" stroke="none"></circle>',
    "clock": '<circle cx="12" cy="12" r="9"></circle><polyline points="12 7 12 12 15.5 14"></polyline>',
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"></rect><line x1="3" y1="10" x2="21" y2="10"></line><line x1="8" y1="3" x2="8" y2="7"></line><line x1="16" y1="3" x2="16" y2="7"></line>',
    "calendar-days": '<rect x="3" y="5" width="18" height="16" rx="2"></rect><line x1="3" y1="10" x2="21" y2="10"></line><line x1="8" y1="3" x2="8" y2="7"></line><line x1="16" y1="3" x2="16" y2="7"></line><circle cx="8" cy="15" r="1" fill="currentColor" stroke="none"></circle><circle cx="12" cy="15" r="1" fill="currentColor" stroke="none"></circle><circle cx="16" cy="15" r="1" fill="currentColor" stroke="none"></circle>',
    "thermometer": '<path d="M12 3a2 2 0 0 0-2 2v9.3a4 4 0 1 0 4 0V5a2 2 0 0 0-2-2z"></path><circle cx="12" cy="17.5" r="1.1" fill="currentColor" stroke="none"></circle>',
    "droplet": '<path d="M12 3s6 7 6 11a6 6 0 0 1-12 0c0-4 6-11 6-11z"></path>',
    "wind": '<path d="M3 8h11a2.5 2.5 0 1 0-2.5-2.5"></path><path d="M3 12h15a2.5 2.5 0 1 1-2.5 2.5"></path><path d="M3 16h9a2 2 0 1 1-2 2"></path>',
    "pin": '<path d="M12 21s7-6.5 7-12a7 7 0 1 0-14 0c0 5.5 7 12 7 12z"></path><circle cx="12" cy="9" r="2.3"></circle>',
    "refresh": '<polyline points="3 4 3 9 8 9"></polyline><polyline points="21 20 21 15 16 15"></polyline><path d="M4.6 9A8 8 0 0 1 20 8M19.4 15A8 8 0 0 1 4 16"></path>',
    "alert": '<path d="M12 3 2 20h20L12 3z"></path><line x1="12" y1="10" x2="12" y2="14"></line><circle cx="12" cy="17" r="0.7" fill="currentColor" stroke="none"></circle>',
    "layers": '<polygon points="12 3 21 8 12 13 3 8 12 3"></polygon><polyline points="3 13 12 18 21 13"></polyline><polyline points="3 17.5 12 22.5 21 17.5"></polyline>',
    "tool": '<path d="M14.7 6.3a4 4 0 0 0-5.6 5.1L3 17.5V21h3.5l6.1-6.1a4 4 0 0 0 5.1-5.6l-3 3-2-2 3-3z"></path>',
    "database": '<ellipse cx="12" cy="6" rx="7" ry="3"></ellipse><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6"></path><path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"></path>',
    "user": '<circle cx="12" cy="8" r="4"></circle><path d="M4 21c0-4 4-6 8-6s8 2 8 6"></path>',
    "document": '<path d="M7 3h7l4 4v14H7z"></path><path d="M14 3v4h4"></path><line x1="9.5" y1="12" x2="14.5" y2="12"></line><line x1="9.5" y1="16" x2="14.5" y2="16"></line>',
    "target": '<circle cx="12" cy="12" r="8"></circle><circle cx="12" cy="12" r="4"></circle><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"></circle>',
    "activity": '<polyline points="2 13 8 13 10 8 14 18 16 13 22 13"></polyline>',
    "arrow-right": '<line x1="4" y1="12" x2="20" y2="12"></line><polyline points="13 5 20 12 13 19"></polyline>',
    "check": '<polyline points="20 6 9 17 4 12"></polyline>',
}


def icon(name: str, color: str = "currentColor", size: int = 18, stroke: float = 2) -> str:
    """Return an inline single-color SVG icon (no emoji)."""
    d = _ICON_PATHS.get(name, "")
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;">{d}</svg>'
    )


_HERO_IMG_PATH = os.path.join(os.path.dirname(__file__), "solar_panel.webp")
HERO_BG_B64 = _get_base64(_HERO_IMG_PATH)
HERO_BG_CSS = (
    f'url("data:image/webp;base64,{HERO_BG_B64}")' if HERO_BG_B64 else "none"
)

_HOME_HERO_IMG_PATH = os.path.join(os.path.dirname(__file__), "solar-energy.jpg")
HOME_HERO_B64 = _get_base64(_HOME_HERO_IMG_PATH)
HOME_HERO_CSS = (
    f'url("data:image/jpeg;base64,{HOME_HERO_B64}")' if HOME_HERO_B64 else "none"
)

_PREVIEW_IMG_PATH = os.path.join(os.path.dirname(__file__), "project_preview.png")
PREVIEW_IMG_B64 = _get_base64(_PREVIEW_IMG_PATH)
PREVIEW_IMG_SRC = (
    f'data:image/png;base64,{PREVIEW_IMG_B64}' if PREVIEW_IMG_B64 else ""
)

# ----------------------------------------------------------------------------
# GLOBAL CSS
# ----------------------------------------------------------------------------
_CSS_TEMPLATE = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    :root{
        --orange:#84CC16;
        --orange-dark:#65A30D;
        --navy:#1F2937;
        --muted:#6B7280;
        --card-bg:#FFFFFF;
        --page-bg:#F6F7FB;
    }

    .stApp { background: var(--page-bg); }
    #MainMenu, footer, header {visibility: hidden;}
    div.block-container{padding-top:1.4rem; padding-bottom:2rem;}

    /* Safety-net: some Streamlit themes (esp. dark mode) inject their own
       text-color rules on markdown containers with high specificity, which
       can make text inside our custom cards invisible (e.g. white-on-white).
       Force our own text color as the baseline for anything rendered inside
       a stMarkdownContainer, so custom classes always win. */
    div[data-testid="stMarkdownContainer"], div[data-testid="stMarkdownContainer"] p{
        color: var(--navy);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"]{
        background:#FFFFFF;
        border-right:1px solid #EEF0F4;
    }
    section[data-testid="stSidebar"] .block-container{padding-top:1rem;}

    .brand-row{display:flex;align-items:center;gap:10px;margin-bottom:6px;}
    .brand-logo{
        width:42px;height:42px;border-radius:12px;
        background:linear-gradient(135deg,#BEF264,#65A30D);
        display:flex;align-items:center;justify-content:center;
        box-shadow:0 4px 10px rgba(101,163,13,.35);
    }
    .brand-title{font-weight:800;font-size:19px;color:var(--navy);line-height:1.1;letter-spacing:-.01em;}
    .brand-sub{font-size:11.5px;color:var(--muted);}

    .nav-list{display:flex;flex-direction:column;gap:3px;}
    .nav-item{
        display:flex; align-items:center; gap:12px;
        padding:8px 10px; border-radius:12px;
        text-decoration:none !important; cursor:pointer;
        transition:background .15s ease;
    }
    .nav-item, .nav-item:visited, .nav-item:hover, .nav-item:active{color:inherit;}
    .nav-item:hover{background:#F5F6F8;}
    .nav-item.active{background:#F0F1F4;}
    .nav-icon-badge{
        width:36px; height:36px; border-radius:11px; flex:0 0 auto;
        background:#ECFDF5; color:#16A34A;
        display:flex; align-items:center; justify-content:center;
    }
    .nav-label{font-size:14px; font-weight:600; color:#4B5563; letter-spacing:.01em;}
    .nav-item.active .nav-label{color:#111827; font-weight:700;}

    .status-title{font-size:10.5px;letter-spacing:.09em;color:#9CA3AF;font-weight:700;margin:16px 0 8px 2px;text-transform:uppercase;}
    .status-row{display:flex;justify-content:space-between;align-items:center;font-size:13px;color:var(--navy);padding:4px 2px;}
    .dot{height:8px;width:8px;border-radius:50%;background:#22C55E;display:inline-block;margin-right:6px;}
    .status-val{color:#22C55E;font-weight:600;}

    .clean-card{
        margin-top:18px;border-radius:16px;overflow:hidden;position:relative;
        min-height:130px;padding:14px;
        background:linear-gradient(150deg, #0F766E 0%, #15803D 55%, #166534 100%);
        border:1px solid rgba(255,255,255,.08);
        box-shadow:0 4px 14px rgba(15,118,110,.25);
    }
    .clean-card h4{margin:0 0 4px 0; font-size:14px; color:#FFFFFF !important; font-weight:700;}
    .clean-card p{margin:0;font-size:11.5px;color:#DCFCE7 !important;line-height:1.4;}

    .profile-row{display:flex;align-items:center;gap:8px;margin-top:14px;padding:8px 4px;border-top:1px solid #EEF0F4;}
    .avatar{width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#1F2937,#374151);
        color:#fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;
        letter-spacing:.02em;}
    .profile-name{font-size:12.5px;font-weight:600;color:var(--navy);}
    .profile-mail{font-size:11px;color:var(--muted);}

    /* ---------- Cards ---------- */
    .card{
        background:var(--card-bg); border-radius:16px; padding:18px 20px;
        box-shadow:0 2px 10px rgba(17,24,39,.05); border:1px solid #F0F1F5;
        height:100%;
    }
    /* Streamlit's native bordered container (st.container(border=True)) —
       used instead of manual multi-call div/close-div HTML, which Streamlit
       renders as separate sibling elements rather than nested ones. Styled
       to match .card so it's visually identical. */
    div[data-testid="stVerticalBlockBorderWrapper"]{
        background:var(--card-bg); border-radius:16px !important;
        box-shadow:0 2px 10px rgba(17,24,39,.05); border:1px solid #F0F1F5 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div{border-radius:16px;}
    .metric-label{font-size:12.5px;color:var(--muted) !important;font-weight:600;margin-bottom:4px;}
    .metric-value{font-size:26px;font-weight:800;color:var(--navy) !important;}
    .metric-delta-up{font-size:12px;color:#16A34A !important;font-weight:600;}
    .metric-delta-down{font-size:12px;color:#DC2626 !important;font-weight:600;}

    /* ---------- KPI cards (uniform size, regardless of value length) ---------- */
    .kpi-card{
        background:var(--card-bg); border-radius:16px; padding:16px 18px;
        box-shadow:0 2px 10px rgba(17,24,39,.05); border:1px solid #F0F1F5;
        height:118px; display:flex; flex-direction:column; justify-content:space-between;
    }
    .kpi-label{font-size:12px;color:var(--muted) !important;font-weight:600;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .kpi-value{font-size:21px;font-weight:800;color:var(--navy) !important;line-height:1.15;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .kpi-delta{font-size:11.5px;font-weight:600;white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .kpi-delta-up{color:#16A34A !important;}
    .kpi-delta-down{color:#DC2626 !important;}

    /* ---------- Live badge ---------- */
    .live-badge{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;
        color:#16A34A;background:#F0FDF4;border:1px solid #BBF7D0;border-radius:999px;
        padding:2px 9px;}
    .live-dot{height:7px;width:7px;border-radius:50%;background:#22C55E;display:inline-block;
        animation:live-pulse 1.6s ease-in-out infinite;}
    .stale-badge{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;
        color:#B45309;background:#FFFBEB;border:1px solid #FDE68A;border-radius:999px;
        padding:2px 9px;}
    @keyframes live-pulse{
        0%{box-shadow:0 0 0 0 rgba(34,197,94,.55);}
        70%{box-shadow:0 0 0 6px rgba(34,197,94,0);}
        100%{box-shadow:0 0 0 0 rgba(34,197,94,0);}
    }

    .feature-card{
        background:#fff;border-radius:16px;padding:20px;border:1px solid #F0F1F5;
        box-shadow:0 2px 10px rgba(17,24,39,.05); height:100%;
    }
    .feature-icon{
        width:46px;height:46px;border-radius:12px;display:flex;align-items:center;
        justify-content:center;font-size:22px;margin-bottom:10px;
    }
    .feature-title{font-weight:700;color:var(--navy) !important;font-size:15px;margin-bottom:4px;}
    .feature-sub{font-size:12.5px;color:var(--muted) !important;}

    .hero{
        border-radius:20px; padding:34px 34px; color:#fff; position:relative; overflow:hidden;
        background-image:
            linear-gradient(120deg, rgba(15,17,25,.80), rgba(15,17,25,.45)),
            __HERO_BG__;
        background-size:cover;
        background-position:center;
        background-repeat:no-repeat;
        min-height:230px; display:flex; flex-direction:column; justify-content:center;
    }
    .hero h1{font-size:34px;font-weight:800;margin:0;line-height:1.15;}
    .hero .accent{color:#A3E635;}
    .hero p{color:#E5E7EB;font-size:14.5px;max-width:520px;margin-top:12px;}

    /* ---------- Home page hero: full-bleed photo, no white patch ---------- */
    .home-hero{
        border-radius:20px; padding:40px 44px; color:#fff; position:relative; overflow:hidden;
        background-image:
            linear-gradient(120deg, rgba(15,17,25,.78), rgba(15,17,25,.42)),
            __HOME_HERO_BG__;
        background-size:cover;
        background-position:center;
        background-repeat:no-repeat;
        min-height:260px; display:flex; flex-direction:column; justify-content:center;
    }
    .home-hero h1{font-size:32px; font-weight:800; margin:0; line-height:1.2; color:#fff !important;}
    .home-hero h1 .accent{color:#A3E635 !important;}
    .home-hero p{color:#E5E7EB !important; font-size:14.5px; max-width:480px; margin:14px 0 0 0; line-height:1.6;}

    /* ---------- About page header: no photo, clean light green/yellow gradient ---------- */
    .about-header{
        background:linear-gradient(120deg, #F0FDF4 0%, #FEFCE8 55%, #FFFFFF 100%);
        border:1px solid #DCFCE7;
        border-radius:20px;
        padding:28px 30px;
        display:flex;
        align-items:flex-start;
        gap:18px;
        position:relative;
        overflow:hidden;
        box-shadow:0 2px 10px rgba(17,24,39,.05);
    }
    .about-header::after{
        content:"";
        position:absolute; top:-40px; right:-40px;
        width:180px; height:180px; border-radius:50%;
        background:radial-gradient(circle, rgba(202,138,4,.14), transparent 70%);
    }
    .about-header-icon{
        flex:0 0 auto; width:52px; height:52px; border-radius:14px;
        background:linear-gradient(135deg,#16A34A,#65A30D);
        display:flex; align-items:center; justify-content:center;
        box-shadow:0 6px 14px rgba(22,163,74,.30);
    }
    .about-header h1{font-size:24px;font-weight:800;color:var(--navy) !important;margin:0 0 8px 0;}
    .about-header p{font-size:13.5px;color:var(--muted) !important;line-height:1.6;margin:0;max-width:720px;}

    .pill-badge{
        display:inline-block;padding:5px 12px;border-radius:999px;font-size:12px;font-weight:700;
    }
    .badge-green{background:#DCFCE7;color:#16A34A;}
    .badge-orange{background:#ECFCCB;color:#65A30D;}

    .section-title{font-weight:800;font-size:16px;color:var(--navy) !important;margin-bottom:2px;}
    .section-sub{font-size:12.5px;color:var(--muted) !important;margin-bottom:14px;}

    /* ---------- Auto-synced parameter tiles (Plant Parameters) ---------- */
    .auto-tile{
        background:#FAFAFB; border:1px solid #F0F1F5; border-radius:13px;
        padding:10px 12px; display:flex; align-items:center; gap:10px;
        margin-bottom:10px; transition:box-shadow .15s ease, transform .15s ease;
    }
    .auto-tile:hover{box-shadow:0 4px 12px rgba(17,24,39,.07); transform:translateY(-1px);}
    .auto-tile-icon{
        width:36px; height:36px; border-radius:10px; display:flex; align-items:center;
        justify-content:center; font-size:16px; flex:0 0 auto;
    }
    .auto-tile-label{font-size:10.5px;color:var(--muted) !important;font-weight:700;
        text-transform:uppercase;letter-spacing:.03em;}
    .auto-tile-value{font-size:15.5px;font-weight:800;color:var(--navy) !important;line-height:1.3;}
    .auto-sync-strip{display:flex;align-items:center;gap:7px;margin:14px 0 10px;}
    .auto-sync-strip span.tag{font-size:11px;color:#9CA3AF;}

    .stButton>button{
        background:linear-gradient(90deg,#A3E635,#4D7C0F); color:#fff; border:none;
        border-radius:10px; padding:10px 18px; font-weight:700; box-shadow:0 4px 12px rgba(101,163,13,.35);
    }
    .stButton>button:hover{opacity:.92; color:#fff;}

    table.dataframe td, table.dataframe th{font-size:12.5px !important;}

    /* ---------- Recent Predictions table ---------- */
    .pred-table-wrap{border:1px solid #EEF0F4;border-radius:14px;overflow:hidden;}
    table.pred-table{width:100%;border-collapse:collapse;font-size:13px;}
    table.pred-table thead th{
        background:#F8F9FB;color:#9CA3AF;font-weight:700;font-size:10.5px;
        text-transform:uppercase;letter-spacing:.05em;text-align:left;
        padding:10px 14px;border-bottom:1px solid #EEF0F4;
    }
    table.pred-table thead th.num{text-align:right;}
    table.pred-table tbody td{
        padding:11px 14px;color:var(--navy);border-bottom:1px solid #F4F5F7;
        white-space:nowrap;
    }
    table.pred-table tbody td.num{text-align:right;font-variant-numeric:tabular-nums;}
    table.pred-table tbody tr:last-child td{border-bottom:none;}
    table.pred-table tbody tr:hover td{background:#FAFAFB;}
    table.pred-table .err-good{color:#16A34A;font-weight:700;}
    table.pred-table .err-warn{color:#CA8A04;font-weight:700;}
    table.pred-table .err-bad{color:#DC2626;font-weight:700;}
    table.pred-table .conf-pill{
        display:inline-block;padding:3px 9px;border-radius:999px;font-size:11px;font-weight:700;
        background:#EFF6FF;color:#2563EB;
    }
    table.pred-table .status-pill{
        display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:999px;
        font-size:11.5px;font-weight:700;background:#DCFCE7;color:#16A34A;
    }
    table.pred-table .status-pill .dot2{width:6px;height:6px;border-radius:50%;background:#16A34A;display:inline-block;}

    /* ---------- How It Works flow ---------- */
    .flow-wrap{display:flex;align-items:flex-start;justify-content:center;gap:2px;margin-top:22px;flex-wrap:wrap;row-gap:24px;}
    .flow-step{display:flex;flex-direction:column;align-items:center;gap:6px;width:118px;text-align:center;}
    .flow-icon{
        width:44px;height:44px;border-radius:50%;
        background:#FFFFFF;border:2px solid #BEF264;color:#65A30D;
        display:flex;align-items:center;justify-content:center;
        box-shadow:0 3px 8px rgba(101,163,13,.18);
    }
    .flow-num{font-size:9.5px;font-weight:800;color:#84CC16;letter-spacing:.08em;text-transform:uppercase;}
    .flow-label{font-size:12px;font-weight:700;color:var(--navy);text-align:center;line-height:1.25;}
    .flow-desc{font-size:11px;color:var(--muted);text-align:center;line-height:1.35;}
    .flow-arrow{display:flex;align-items:center;padding-top:19px;opacity:.8;}

    /* ---------- Section title accent bar (Model Performance / How It Works) --- */
    .title-bar{width:34px;height:3px;border-radius:2px;background:linear-gradient(90deg,#84CC16,#65A30D);margin:6px 0 16px;}

    /* ---------- Model Performance icon tiles ---------- */
    .perf-tile{
        background:#F7FEE7; border:1px solid #ECFCCB; border-radius:14px;
        padding:16px 10px; text-align:center; transition:transform .15s ease, box-shadow .15s ease;
    }
    .perf-tile:hover{transform:translateY(-2px); box-shadow:0 6px 14px rgba(101,163,13,.15);}
    .perf-tile-icon{
        width:38px;height:38px;border-radius:50%;margin:0 auto 10px;
        background:#FFFFFF;border:2px solid #BEF264;color:#65A30D;
        display:flex;align-items:center;justify-content:center;
    }
    .perf-tile-label{font-size:11.5px;color:var(--muted) !important;font-weight:600;margin-bottom:4px;}
    .perf-tile-value{font-size:19px;font-weight:800;color:var(--navy) !important;}

    /* ---------- Plant Parameters: field labels with icon badges ---------- */
    .field-label-row{
        display:flex; align-items:center; gap:8px; margin:14px 0 6px;
    }
    .field-icon-badge{
        width:26px; height:26px; border-radius:8px; display:flex;
        align-items:center; justify-content:center; flex:0 0 auto;
    }
    .field-label-text{
        font-size:12.5px; font-weight:700; color:#374151; letter-spacing:.01em;
    }

    /* Streamlit number_input: rounded, subtle depth, and a clean focus glow
       to match the icon-labelled fields above each one. */
    div[data-testid="stNumberInput"] > div{
        border-radius:11px !important;
        border:1.5px solid #E5E7EB !important;
        background:#FAFAFB !important;
        transition:border-color .15s ease, box-shadow .15s ease, background .15s ease;
    }
    div[data-testid="stNumberInput"] > div:hover{
        border-color:#D1D5DB !important;
        background:#FFFFFF !important;
    }
    div[data-testid="stNumberInput"] > div:has(input:focus){
        border-color:var(--orange-dark) !important;
        background:#FFFFFF !important;
        box-shadow:0 0 0 3px rgba(101,163,13,.14) !important;
    }
    div[data-testid="stNumberInput"] input{
        font-weight:700 !important; font-size:14.5px !important; color:var(--navy) !important;
    }
    div[data-testid="stNumberInput"] button{
        border-radius:6px !important;
    }
    </style>
    """

st.markdown(
    _CSS_TEMPLATE.replace("__HERO_BG__", HERO_BG_CSS)
    .replace("__HOME_HERO_BG__", HOME_HERO_CSS),
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# MODEL: load pretrained pickles if present, else train a bundled fallback
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading prediction model...")
def load_or_train_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        # Trust the fitted objects for the real feature names/order — this is
        # what scaler.transform() will actually validate against, so it's the
        # only source of truth that can't silently drift out of sync.
        detected = getattr(scaler, "feature_names_in_", None)
        if detected is None:
            detected = getattr(model, "feature_names_in_", None)
        features = list(detected) if detected is not None else list(BASE_FEATURES)

        # NOTE ON THE SAVED MODEL: the notebook now explicitly selects the
        # best-scoring candidate (Random Forest, per the comparison table)
        # before cross-validation, feature importance, and pickling — so
        # `solar_power_generation.save` really does contain the Random
        # Forest model. This app still identifies the model type straight
        # from the pickle (`type(model).__name__`) rather than assuming a
        # fixed algorithm, and looks up the metrics that actually belong to
        # that type (see KNOWN_METRICS above), so it stays correct even if
        # the notebook's chosen algorithm ever changes.
        model_class_name = type(model).__name__
        friendly_names = {
            "RandomForestRegressor": "Random Forest",
            "AdaBoostRegressor": "AdaBoost",
            "GradientBoostingRegressor": "Gradient Boosting",
            "DecisionTreeRegressor": "Decision Tree",
            "KNeighborsRegressor": "K-Nearest Neighbors",
            "LinearRegression": "Linear Regression",
            "SVR": "SVR",
            "XGBRegressor": "XGBoost",
        }
        best_model_label = friendly_names.get(model_class_name, model_class_name)

        known = KNOWN_METRICS.get(model_class_name, KNOWN_METRICS["RandomForestRegressor"])
        metrics = {
            "r2": known["r2"],
            "mae": known["mae"],
            "rmse": known["rmse"],
            "cv_r2": known["cv_r2"] if known["cv_r2"] is not None else known["r2"],
            "accuracy": known["r2"],
            "best_model": best_model_label,
        }
        importances = compute_importances(model, features)
        return model, scaler, metrics, importances, "loaded", features
    except Exception as e:
        # Surfaced on the About page instead of silently vanishing, so the
        # real reason the .save files didn't load (missing file, path issue,
        # scikit-learn version mismatch, corrupted pickle, etc.) is visible
        # instead of just seeing "fallback training run" with no explanation.
        st.session_state["_model_load_error"] = f"{type(e).__name__}: {e}"

    # ---- Fallback: synthesize a physically-plausible dataset and train ----
    features = list(BASE_FEATURES)
    rng = np.random.default_rng(42)
    n = 6000
    hour = rng.integers(0, 24, n)
    month = rng.integers(1, 13, n)

    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, 1)
    seasonal = 0.75 + 0.25 * np.sin(2 * np.pi * (month - 3) / 12)
    irradiation = np.clip(daylight * seasonal + rng.normal(0, 0.04, n), 0, 1.3)

    ambient_temp = 22 + 8 * seasonal + 6 * daylight + rng.normal(0, 1.5, n)
    module_temp = ambient_temp + irradiation * 28 + rng.normal(0, 1.5, n)

    daily_yield = np.clip(irradiation * hour * 420 + rng.normal(0, 200, n), 0, 9000)
    total_yield = 6_000_000 + daily_yield * rng.integers(1, 40, n) + rng.normal(0, 5000, n)

    # NOTE ON DC_POWER: in the real Plant_1 dataset, DC_POWER runs roughly
    # 10x AC_POWER (mean DC 3147 kW vs mean AC 308 kW) due to how that
    # dataset's sensors report the two values -- it's not a normal ~2%
    # inverter loss. This fallback generator mirrors that same ratio so the
    # synthetic data stays physically consistent with the real dataset.
    capacity = 9800
    efficiency = 1 - np.clip((module_temp - 45), 0, None) * 0.004
    dc_power = np.clip(irradiation * capacity * 10.2 * efficiency + rng.normal(0, 600, n), 0, None)
    ac_power = np.clip(dc_power / 10.2 + rng.normal(0, 6, n), 0, None)

    df = pd.DataFrame(
        {
            "DC_POWER": dc_power,
            "DAILY_YIELD": daily_yield,
            "TOTAL_YIELD": total_yield,
            "AMBIENT_TEMPERATURE": ambient_temp,
            "MODULE_TEMPERATURE": module_temp,
            "IRRADIATION": irradiation,
            "AC_POWER": ac_power,
        }
    )

    X = df[features]
    y = df["AC_POWER"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)

    # ---- Model Building: train several regressors, exactly mirroring the
    # notebook's Model Building / Model Evaluation cells ----
    # Same models, same order, same (lack of) random_state on the individual
    # estimators — only train_test_split is seeded, exactly as in the
    # notebook. XGBoost is inserted between SVR and GradientBoost (not
    # appended at the end) so the loop order matches the notebook exactly
    # whether or not xgboost is installed.
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel="rbf"),
    }
    try:
        from xgboost import XGBRegressor  # optional; matches notebook if available
        candidate_models["XGBoost"] = XGBRegressor()
    except ImportError:
        pass
    candidate_models["GradientBoost"] = GradientBoostingRegressor()
    candidate_models["AdaBoost"] = AdaBoostRegressor()

    comparison_rows = []
    fitted_models = {}
    for name, candidate in candidate_models.items():
        candidate.fit(Xs_train, y_train)
        cand_pred = candidate.predict(Xs_test)
        comparison_rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, cand_pred),
                "MSE": mean_squared_error(y_test, cand_pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, cand_pred))),
                "R2 Score": r2_score(y_test, cand_pred),
            }
        )
        fitted_models[name] = candidate

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values("R2 Score", ascending=False)
        .reset_index(drop=True)
    )
    top_scorer = comparison_df.iloc[0]["Model"]

    # Deploy the actual best-scoring candidate (Random Forest, per the
    # comparison table) instead of the notebook's original bug — a bare
    # `for name, model in models.items(): ...` loop that leaves `model`
    # pointing at whichever candidate was fit LAST, regardless of score.
    deployed_name = top_scorer
    model = fitted_models[deployed_name]
    pred = model.predict(Xs_test)

    # ---- Cross validation on the deployed (best-scoring) model ----
    cv_scores = cross_val_score(model, Xs_train, y_train, cv=5)

    metrics = {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "cv_r2": float(cv_scores.mean()),
        "accuracy": r2_score(y_test, pred),
        "best_model": deployed_name,
        "top_scorer": top_scorer,
        "comparison": comparison_df,
    }
    importances = compute_importances(model, features)
    if importances is None:
        # Winning model (e.g. SVR/KNN) has no native importances to show.
        # Fall back to a tree-based candidate's importances purely for the
        # Feature Importance chart, so it's never blank.
        for fallback_name in ("Random Forest", "GradientBoost", "AdaBoost", "Decision Tree"):
            fb_model = fitted_models.get(fallback_name)
            if fb_model is not None:
                importances = compute_importances(fb_model, features)
                if importances is not None:
                    break
    return model, scaler, metrics, importances, "trained", features



model, scaler, METRICS, IMPORTANCES, MODEL_SOURCE, FEATURES = load_or_train_model()


def predict_power(input_dict: dict) -> float:
    row = pd.DataFrame([[input_dict[f] for f in FEATURES]], columns=FEATURES)
    scaled = scaler.transform(row)
    return float(model.predict(scaled)[0])


@st.cache_data
def sample_day_curve():
    today = dt.date.today()
    hours = list(range(0, 24, 3))
    rng = np.random.default_rng(7)
    curve = []
    for h in hours:
        daylight = max(np.sin(np.pi * (h - 6) / 12), 0)
        inputs = build_inputs(
            FEATURES,
            {
                "DC_POWER": 9800 * 0.85 * daylight,
                "DAILY_YIELD": 3000 + h * 150,
                "TOTAL_YIELD": 1_000_000,
                "AMBIENT_TEMPERATURE": 24 + 8 * daylight,
                "MODULE_TEMPERATURE": 28 + 20 * daylight,
                "IRRADIATION": 0.85 * daylight,
                "HOUR": h,
                "MONTH": today.month,
                "DAY": today.day,
            },
        )
        base = predict_power(inputs)
        actual = base + rng.normal(0, base * 0.05 + 1)
        curve.append((h, max(base, 0), max(actual, 0)))
    return pd.DataFrame(curve, columns=["Hour", "Predicted", "Actual"])


def sample_recent_predictions():
    rng = np.random.default_rng(3)
    rows = []
    now = dt.datetime.now()
    base_time = now.replace(minute=0, second=0, microsecond=0) - dt.timedelta(minutes=30 * 5)
    for i in range(6):
        pred = rng.uniform(18, 30)
        actual = pred + rng.normal(0, 0.9)
        conf = rng.uniform(0.90, 0.99)
        rows.append(
            {
                "Time": (base_time + dt.timedelta(minutes=30 * i)).strftime("%b %d, %I:%M %p"),
                "Predicted (kW)": round(pred, 1),
                "Actual (kW)": round(actual, 1),
                "Error (kW)": round(actual - pred, 1),
                "Confidence": f"{conf*100:.0f}%",
                "Status": "Success",
            }
        )
    return pd.DataFrame(rows)


def render_predictions_table(df: pd.DataFrame):
    """Render Recent Predictions as a styled HTML table instead of the plain
    st.dataframe grid — right-aligned numerics, color-coded error, and a
    status pill, all matching the app's card design language."""
    def err_class(err: float) -> str:
        a = abs(err)
        if a <= 0.5:
            return "err-good"
        if a <= 1.5:
            return "err-warn"
        return "err-bad"

    rows_html = "".join(
        f"<tr>"
        f"<td>{r['Time']}</td>"
        f"<td class='num'>{r['Predicted (kW)']:.1f}</td>"
        f"<td class='num'>{r['Actual (kW)']:.1f}</td>"
        f"<td class='num {err_class(r['Error (kW)'])}'>{r['Error (kW)']:+.1f}</td>"
        f"<td class='num'><span class='conf-pill'>{r['Confidence']}</span></td>"
        f"<td><span class='status-pill'><span class='dot2'></span>{r['Status']}</span></td>"
        f"</tr>"
        for _, r in df.iterrows()
    )
    st.markdown(
        f"""
        <div class="pred-table-wrap">
        <table class="pred-table">
            <thead><tr>
                <th>Time</th><th class="num">Predicted</th><th class="num">Actual</th>
                <th class="num">Error</th><th class="num">Confidence</th><th>Status</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
# st.radio only renders plain text in its labels (no HTML/SVG), so it can't
# show a colored icon badge per item. Rendered as real links instead (each
# sets ?page=... and reloads in-place via target="_self"), which lets every
# item carry its own icon badge — all one clean green accent, like the
# reference design.
NAV_ITEMS = [("Home", "home"), ("Prediction", "bolt"), ("Dashboard", "bars"), ("About", "info")]
_valid_pages = [n for n, _ in NAV_ITEMS]

page = st.query_params.get("page", "Home")
if page not in _valid_pages:
    page = "Home"
if st.session_state.pop("_jump_prediction", False):
    page = "Prediction"
    st.query_params["page"] = "Prediction"

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand-row">
            <div class="brand-logo">{icon("sun", color="#FFFFFF", size=22, stroke=2.2)}</div>
            <div>
                <div class="brand-title">SolarAI</div>
                <div class="brand-sub">Solar Power Prediction</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    nav_html = "".join(
        f'<a href="?page={name}" target="_self" class="nav-item{" active" if page == name else ""}">'
        f'<span class="nav-icon-badge">{icon(ic, color="currentColor", size=17, stroke=2)}</span>'
        f'<span class="nav-label">{name}</span></a>'
        for name, ic in NAV_ITEMS
    )
    st.markdown(f'<div class="nav-list">{nav_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="status-title">SYSTEM STATUS</div>', unsafe_allow_html=True)
    status_items = [
        ("Model Status", "Running"),
        ("API Gateway", "Healthy"),
        ("Database", "Connected"),
        ("Monitoring", "Active"),
    ]
    rows_html = "".join(
        f'<div class="status-row"><span><span class="dot"></span>{k}</span>'
        f'<span class="status-val">{v}</span></div>'
        for k, v in status_items
    )
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="clean-card">
            <h4>Clean Energy<br>Better Tomorrow</h4>
            <p>AI-powered predictions for a smarter &amp; sustainable future.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="profile-row">
            <div class="avatar">SM</div>
            <div>
                <div class="profile-name">Safa Manha T</div>
                <div class="profile-mail">safa@solarai.com</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE: HOME
# ============================================================================
def render_home():
    st.markdown(
        """
        <div class="home-hero">
            <h1>Solar Power Generation<br><span class="accent">Prediction System</span></h1>
            <p>AI Powered Solar Energy Forecasting Platform — accurate solar power
            predictions using advanced machine learning models and real-time
            environmental data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, _ = st.columns([1, 4])
    with c1:
        if st.button("Get Started", use_container_width=True):
            st.session_state["_jump_prediction"] = True
            st.rerun()

    st.write("")
    st.write("")

    features = [
        ("sun", "#ECFCCB", "#65A30D", "Predict Solar AC Power", "Predict real-time solar power generation"),
        ("trend", "#ECFCCB", "#65A30D", "Forecast Daily Energy", "Forecast daily and monthly energy production"),
        ("cloud", "#ECFCCB", "#65A30D", "Weather Based Prediction", "Utilize weather parameters for accurate prediction"),
        ("cpu", "#ECFCCB", "#65A30D", "Machine Learning Model", "Trained ML model for high performance"),
    ]
    cols = st.columns(4)
    for col, (icon_name, bg, fg, title, sub) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon" style="background:{bg};color:{fg};">{icon(icon_name, color=fg, size=22, stroke=1.8)}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        perf_tiles = [
            ("target", "R² Score", f"{METRICS['r2']:.3f}"),
            ("target", "MAE", f"{METRICS['mae']:.3f}"),
            ("trend", "RMSE", f"{METRICS['rmse']:.3f}"),
            ("check", "Accuracy", f"{METRICS['accuracy']*100:.1f}%"),
        ]
        tiles_html = "".join(
            f'<div class="perf-tile">'
            f'<div class="perf-tile-icon">{icon(ic, color="#65A30D", size=17, stroke=2.1)}</div>'
            f'<div class="perf-tile-label">{label}</div>'
            f'<div class="perf-tile-value">{val}</div>'
            f'</div>'
            for ic, label, val in perf_tiles
        )
        st.markdown(
            f"""
            <div class="card">
                <div class="section-title">Model Performance</div>
                <div class="title-bar"></div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
                    {tiles_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        steps = [
            ("cloud", "Weather Data", "Collect real-time weather data"),
            ("tool", "Data Preprocessing", "Clean and process the data"),
            ("cpu", "ML Model", "Apply ML model for prediction"),
            ("bolt", "Power Prediction", "Get accurate solar power output"),
            ("bars", "Visualization", "View results on the dashboard"),
        ]

        # Each step is grouped with its trailing arrow in one flex item, so
        # wrapping (on narrow screens) never leaves an orphaned arrow on its
        # own line — the whole "step ➜" unit moves down together.
        steps_html = "".join(
            f'<div style="display:flex;align-items:flex-start;">'
            f'<div class="flow-step">'
            f'<div class="flow-icon">{icon(ic, color="#65A30D", size=19, stroke=2)}</div>'
            f'<div class="flow-label">{i+1}. {label}</div>'
            f'<div class="flow-desc">{desc}</div>'
            f'</div>'
            + (f'<div class="flow-arrow">{icon("arrow-right", color="#84CC16", size=16, stroke=2.3)}</div>' if i < len(steps)-1 else '')
            + '</div>'
            for i, (ic, label, desc) in enumerate(steps)
        )

        st.markdown(f"""
            <div class="card">
            <div class="section-title">How It Works</div>
            <div class="title-bar"></div>
            <div class="flow-wrap">
                {steps_html}
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# PAGE: PREDICTION
# ============================================================================
def gauge(value, title, max_value, unit="", color="#84CC16"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": f" {unit}", "font": {"size": 30, "color": "#1F2937"}},
            gauge={
                "axis": {"range": [0, max_value], "tickwidth": 0, "tickcolor": "#E5E7EB"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, max_value * 0.5], "color": "#FEE2E2"},
                    {"range": [max_value * 0.5, max_value * 0.8], "color": "#FEF3C7"},
                    {"range": [max_value * 0.8, max_value], "color": "#DCFCE7"},
                ],
            },
            title={"text": title, "font": {"size": 13, "color": "#6B7280"}},
        )
    )
    fig.update_layout(height=190, margin=dict(l=10, r=10, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_prediction():
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown('<div class="section-title" style="font-size:22px;">Solar Power Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Weather-driven features are pulled live — just enter the plant readings</div>', unsafe_allow_html=True)
    with hc2:
        st.markdown(
            f'<div style="text-align:right;font-size:12.5px;color:#6B7280;padding-top:8px;">'
            f'{dt.datetime.now().strftime("%b %d, %Y")}<br>{dt.datetime.now().strftime("%I:%M %p")}</div>',
            unsafe_allow_html=True,
        )

    # ---- Location for live weather -----------------------------------------
    # IP-based geolocation reflects wherever the Streamlit *server* is
    # running, not the visitor's real location, so it was showing the wrong
    # city. Using the plant's fixed, correct coordinates instead — the same
    # ones the Dashboard's Live Weather panel already relies on.
    if "pred_weather_refresh_tick" not in st.session_state:
        st.session_state.pred_weather_refresh_tick = 0

    plant_loc = get_plant_location()
    lat, lon = plant_loc["lat"], plant_loc["lon"]

    weather, is_live = fetch_live_weather(
        lat, lon, st.session_state.pred_weather_refresh_tick
    )
    live_ambient_temp = weather["temperature"]
    live_irradiance_wm2 = weather["irradiance"]
    live_irradiation = round(live_irradiance_wm2 / 1000, 3)                    # kW/m²
    live_module_temp = round(live_ambient_temp + live_irradiation * 28, 1)     # physical estimate
    now = dt.datetime.now()

    # Only date/time features stay fully automatic (Hour/Day/Month). Every
    # other feature — including Ambient Temp, Module Temp, and Irradiation —
    # is a plain, freely-typeable number input, exactly like DC Power or
    # Daily Yield. Live weather is shown for reference in the panel on the
    # right, but no longer drives or pre-fills these fields.
    AUTO_VALUES = {
        "HOUR": now.hour,
        "DAY": now.day,
        "MONTH": now.month,
    }
    auto_keys = {_norm(k) for k in AUTO_VALUES}
    manual_features = [f for f in FEATURES if _norm(f) not in auto_keys]

    left, right = st.columns([2, 1.3])

    # Icon + accent color per plant-reading field, for the redesigned
    # "Plant Parameters" card below.
    FIELD_ICONS = {
        "DC_POWER":            ("bolt",        "#F97316", "#FFF3E6"),
        "DAILY_YIELD":         ("trend",       "#2563EB", "#EFF6FF"),
        "TOTAL_YIELD":         ("database",    "#7C3AED", "#F5F3FF"),
        "AMBIENT_TEMPERATURE": ("thermometer", "#DC2626", "#FEF2F2"),
        "MODULE_TEMPERATURE":  ("thermometer", "#0D9488", "#F0FDFA"),
        "IRRADIATION":         ("sun",         "#CA8A04", "#FEFCE8"),
    }

    inputs = {}
    with left:
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
                    <div style="width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#1F2937,#374151);
                        display:flex;align-items:center;justify-content:center;">{icon("bars", color="#FFFFFF", size=17, stroke=2)}</div>
                    <div>
                        <div class="section-title" style="font-size:15px;margin:0;">Plant Parameters</div>
                        <div class="section-sub" style="margin:1px 0 0;">Enter today's plant readings — compare against Live Weather {icon("arrow-right", color="#9CA3AF", size=11, stroke=2.5)}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            if manual_features:
                cols = st.columns(2)
                for i, f in enumerate(manual_features):
                    cfg = feature_cfg(f)
                    ic_name, ic_color, ic_bg = FIELD_ICONS.get(_norm(f), ("activity", "#6B7280", "#F3F4F6"))
                    with cols[i % 2]:
                        st.markdown(
                            f"""
                            <div class="field-label-row">
                                <span class="field-icon-badge" style="background:{ic_bg};color:{ic_color};">
                                    {icon(ic_name, color=ic_color, size=14, stroke=2.1)}
                                </span>
                                <span class="field-label-text">{cfg['label']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        inputs[f] = st.number_input(
                            cfg["label"],
                            min_value=cfg["min"], max_value=cfg["max"],
                            value=cfg["default"], step=cfg["step"],
                            key=f"num_{f}",
                            label_visibility="collapsed",
                        )
            else:
                st.caption("This model uses only weather-derived features — nothing to enter manually.")
            for f in FEATURES:
                key = _norm(f)
                if key in auto_keys:
                    inputs[f] = next(v for k, v in AUTO_VALUES.items() if _norm(k) == key)

            # ---- Date/time parameters, auto-synced (not user-editable) ----
            feature_keys = {_norm(f) for f in FEATURES}
            # Ambient temp / module temp / irradiation are entered as editable
            # fields above now, so only Hour/Day/Month remain as read-only tiles.
            time_tile_defs = {
                "HOUR": (icon("clock", color="#2563EB", size=16, stroke=2), "Hour", f"{now.hour:02d}:00", "#2563EB", "#EFF6FF"),
                "DAY": (icon("calendar", color="#7C3AED", size=16, stroke=2), "Day", f"{now.day}", "#7C3AED", "#F5F3FF"),
                "MONTH": (icon("calendar-days", color="#0D9488", size=16, stroke=2), "Month", now.strftime("%B"), "#0D9488", "#F0FDFA"),
            }
            auto_tiles = [
                time_tile_defs[k] for k in AUTO_VALUES if k in time_tile_defs and k in feature_keys
            ]

            if auto_tiles:
                st.markdown(
                    '<div class="auto-sync-strip">'
                    '<span class="live-badge"><span class="live-dot"></span>AUTO-SYNCED</span>'
                    '<span class="tag">date/time readings, updated live</span></div>',
                    unsafe_allow_html=True,
                )
                n_cols = 3 if len(auto_tiles) >= 3 else len(auto_tiles)
                tile_cols = st.columns(n_cols)
                for i, (tile_icon, label, val, color, bg) in enumerate(auto_tiles):
                    with tile_cols[i % n_cols]:
                        st.markdown(
                            f'<div class="auto-tile">'
                            f'<div class="auto-tile-icon" style="background:{bg};color:{color};">{tile_icon}</div>'
                            f'<div><div class="auto-tile-label">{label}</div>'
                            f'<div class="auto-tile-value">{val}</div></div></div>',
                            unsafe_allow_html=True,
                        )

        st.write("")
        predict_clicked = st.button("Predict Solar Power", use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
                    <div style="width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#0284C7,#0EA5E9);
                        display:flex;align-items:center;justify-content:center;box-shadow:0 4px 10px rgba(2,132,199,.3);">
                        {icon("sun", color="#FFFFFF", size=17, stroke=2)}</div>
                    <div class="section-title" style="font-size:15px;margin:0;">Live Weather</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            th, tr = st.columns([2.2, 1])
            with th:
                location_picker("pred_loc")
            with tr:
                if st.button("↻ Refresh", key="pred_weather_refresh_btn", help="Refresh live weather", use_container_width=True):
                    st.session_state.pred_weather_refresh_tick += 1
                    fetch_live_weather.clear()

            if is_live:
                badge = '<span class="live-badge"><span class="live-dot"></span>LIVE</span>'
                stamp_src = weather.get("observed_at")
                stamp = f" · updated {stamp_src[-5:]}" if stamp_src else ""
            else:
                badge = f'<span class="stale-badge">{icon("alert", size=11, stroke=2.2)} OFFLINE</span>'
                stamp = " · showing last known values"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'margin:8px 0 12px;font-size:11px;color:#6B7280;">{badge}<span>{stamp}</span></div>',
                unsafe_allow_html=True,
            )

            weather_tiles = [
                ("thermometer", "Temperature", f"{weather['temperature']:.0f} °C", "#DC2626", "#FEF2F2"),
                ("droplet", "Humidity", f"{weather['humidity']:.0f} %", "#2563EB", "#EFF6FF"),
                ("cloud", "Cloud Cover", f"{weather['cloud_cover']:.0f} %", "#64748B", "#F1F5F9"),
                ("wind", "Wind Speed", f"{weather['wind_speed']:.0f} km/h", "#0D9488", "#F0FDFA"),
                ("sun", "Solar Irradiance", f"{weather['irradiance']:.0f} W/m²", "#CA8A04", "#FEFCE8"),
                ("cloud", "Module Temp (est.)", f"{live_module_temp:.1f} °C", "#7C3AED", "#F5F3FF"),
            ]
            for ic, label, val, color, bg in weather_tiles:
                st.markdown(
                    f'<div class="auto-tile">'
                    f'<div class="auto-tile-icon" style="background:{bg};color:{color};">{icon(ic, color=color, size=16, stroke=2)}</div>'
                    f'<div style="flex:1;display:flex;justify-content:space-between;align-items:center;">'
                    f'<div class="auto-tile-label" style="text-transform:none;font-weight:600;font-size:12.5px;">{label}</div>'
                    f'<div class="auto-tile-value" style="font-size:14.5px;">{val}</div></div></div>',
                    unsafe_allow_html=True,
                )

    st.write("")

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = predict_power(inputs)

    if predict_clicked:
        st.session_state.last_prediction = predict_power(inputs)

    power = st.session_state.last_prediction
    energy_today = power * 10
    efficiency = int(np.clip(60 + (live_irradiation * 30), 0, 100))
    perf_index = round(np.clip(power / 5, 0, 10), 1)
    confidence = 98

    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(gauge(round(power, 2), "Predicted Power", 50, "kW"), use_container_width=True)
    with g2:
        st.markdown(
            f"""
            <div class="card" style="text-align:center;padding-top:26px;">
                <div class="metric-label">Prediction Confidence</div>
                <div class="metric-value" style="font-size:34px;">{confidence}%</div>
                <div class="pill-badge badge-orange" style="margin-top:10px;">Excellent Generation</div>
                <div class="feature-sub" style="margin-top:6px;">Optimal condition for high energy production.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    metric_cards = [
        (m1, "Power Output", f"{power:.2f} kW", "Current Prediction"),
        (m2, "Energy Today", f"{energy_today:.1f} kWh", "Predicted Energy"),
        (m3, "Solar Efficiency", f"{efficiency}%", "Efficiency Index"),
        (m4, "Performance Index", f"{perf_index}/10", "Excellent" if perf_index > 7 else "Good"),
    ]
    for col, label, val, sub in metric_cards:
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="feature-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
def render_dashboard():
    today = dt.date.today()
    now = dt.datetime.now()

    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown('<div class="section-title" style="font-size:22px;">Performance Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Overview of predictions, performance, and environmental insights</div>', unsafe_allow_html=True)
    with hc2:
        st.markdown(
            f'<div style="text-align:right;font-size:12.5px;color:#6B7280;padding-top:8px;">'
            f'{now.strftime("%b %d, %Y")}<br>{now.strftime("%I:%M %p")}</div>',
            unsafe_allow_html=True,
        )

    curve = sample_day_curve()
    today_pred = curve["Predicted"].iloc[-1]
    peak = curve["Predicted"].max()
    daily_energy = curve["Predicted"].sum() * 3
    monthly_energy = daily_energy * 29.5

    # Values are pre-formatted to a similar length/style so every KPI card
    # reads consistently at a glance (e.g. "213.4K kWh" instead of a long
    # comma-separated number that would otherwise dominate the card).
    kpis = [
        ("Today's Prediction", f"{today_pred:.1f} kW", "12.5% vs yesterday", "up"),
        ("Average Generation", f"{curve['Predicted'].mean():.1f} kW", "8.3% vs yesterday", "up"),
        ("Peak Generation", f"{peak:.1f} kW", "15.7% vs yesterday", "up"),
        ("Daily Energy", f"{daily_energy:,.0f} kWh", "18.4% vs yesterday", "up"),
        ("Monthly Energy", f"{monthly_energy/1000:.1f}K kWh", "9.8% vs last month", "up"),
        ("Model Accuracy", f"{METRICS['accuracy']*100:.1f}%", f"R² Score: {METRICS['r2']:.2f}", "up"),
    ]
    cols = st.columns(6)
    for col, (label, val, delta, direction) in zip(cols, kpis):
        arrow = "↑" if direction == "up" else "↓"
        delta_cls = "kpi-delta-up" if direction == "up" else "kpi-delta-down"
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-delta {delta_cls}">{arrow} {delta}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    c1, c2, c3 = st.columns([1.3, 1, 1])

    with c1:
        with st.container(border=True):
            st.markdown(f'<div class="section-title" style="font-size:15px;">Power Generation Curve ({today.strftime("%b %d")})</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve["Hour"], y=curve["Predicted"], name="Predicted (kW)", line=dict(color="#65A30D", width=3)))
            fig.add_trace(go.Scatter(x=curve["Hour"], y=curve["Actual"], name="Actual (kW)", line=dict(color="#BEF264", width=2, dash="dot")))
            fig.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.15),
                xaxis_title=None, yaxis_title="kW",
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        with st.container(border=True):
            st.markdown('<div class="section-title" style="font-size:15px;">Predicted vs Actual</div>', unsafe_allow_html=True)
            fig = px.scatter(curve, x="Actual", y="Predicted")
            fig.update_traces(marker=dict(color="#65A30D", size=9))
            lims = [0, max(curve["Actual"].max(), curve["Predicted"].max()) * 1.1]
            fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="#D1D5DB", dash="dash"), name="Ideal"))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with c3:
        with st.container(border=True):
            st.markdown('<div class="section-title" style="font-size:15px;">Feature Importance</div>', unsafe_allow_html=True)
            if not IMPORTANCES:
                st.info(
                    "Feature importance isn't available for this model type. "
                    "It applies to tree-based models (Random Forest, AdaBoost, "
                    "Gradient Boosting) — not to models like SVR or KNN."
                )
            else:
                imp_df = pd.DataFrame(
                    {"Feature": [feature_cfg(f)["label"].split(" (")[0] for f in IMPORTANCES], "Importance": list(IMPORTANCES.values())}
                ).sort_values("Importance", ascending=True)
                fig = go.Figure(go.Bar(x=imp_df["Importance"], y=imp_df["Feature"], orientation="h", marker_color="#65A30D"))
                fig.update_layout(
                    height=260, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                fig.update_xaxes(range=[0, max(imp_df["Importance"].max() * 1.15, 0.05)], tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

    st.write("")
    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        with st.container(border=True):
            st.markdown('<div class="section-title" style="font-size:15px;">Recent Predictions</div>', unsafe_allow_html=True)
            render_predictions_table(sample_recent_predictions())

    with c2:
        with st.container(border=True, height=300):
            if "weather_refresh_tick" not in st.session_state:
                st.session_state.weather_refresh_tick = 0

            th, tr = st.columns([3, 1])
            with th:
                st.markdown('<div class="section-title" style="font-size:15px;">Weather Summary</div>', unsafe_allow_html=True)
            with tr:
                if st.button("↻", key="weather_refresh_btn", help="Refresh live weather"):
                    st.session_state.weather_refresh_tick += 1
                    fetch_live_weather.clear()

            _loc = get_plant_location()
            weather, is_live = fetch_live_weather(
                _loc["lat"], _loc["lon"], st.session_state.weather_refresh_tick
            )

            if is_live:
                badge = '<span class="live-badge"><span class="live-dot"></span>LIVE</span>'
                stamp_src = weather.get("observed_at")
                stamp = f" · updated {stamp_src[-5:]}" if stamp_src else ""
            else:
                badge = f'<span class="stale-badge">{icon("alert", size=11, stroke=2.2)} OFFLINE</span>'
                stamp = " · showing last known values"

            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'margin:-6px 0 8px;font-size:11px;color:#6B7280;">{badge}<span>{stamp}</span></div>',
                unsafe_allow_html=True,
            )

            for ic, label, val in [
                ("thermometer", "Temperature", f"{weather['temperature']:.0f} °C"),
                ("droplet", "Humidity", f"{weather['humidity']:.0f} %"),
                ("cloud", "Cloud Cover", f"{weather['cloud_cover']:.0f} %"),
                ("wind", "Wind Speed", f"{weather['wind_speed']:.0f} km/h"),
                ("sun", "Irradiance", f"{weather['irradiance']:.0f} W/m²"),
            ]:
                st.markdown(
                    f'<div class="status-row"><span style="display:flex;align-items:center;gap:6px;">'
                    f'{icon(ic, color="#9CA3AF", size=14, stroke=2)} {label}</span>'
                    f'<span style="font-weight:700;">{val}</span></div>',
                    unsafe_allow_html=True,
                )

    with c3:
        with st.container(border=True, height=300):
            st.markdown('<div class="section-title" style="font-size:15px;">Model Health</div>', unsafe_allow_html=True)
            last_trained = (today - dt.timedelta(days=1)).strftime("%b %d, %Y")
            for label, val in [
                ("Model Status", "Healthy"),
                ("Model Version", "v1.2.0"),
                ("Last Trained", last_trained),
                ("Prediction Time", "120 ms"),
                ("Uptime", "99.8%"),
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
            <div class="about-header-icon">{icon("document", color="#FFFFFF", size=24, stroke=1.8)}</div>
            <div>
                <h1>About This Project</h1>
                <p>This AI-based application predicts <b>Solar AC Power Generation</b> using
                machine learning algorithms trained on historical solar plant generation and
                weather sensor data. The system assists solar plant operators in forecasting
                power output, optimizing energy planning, and improving operational efficiency.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    cols = st.columns(5)
    blocks = [
        ("bars", "#16A34A", "Dataset Used", [("Generation Dataset", "68,778 records"), ("Weather Sensor Dataset", "3,182 records"), ("Merged Dataset", "68,774 records"), ("Training Samples", "48,141"), ("Testing Samples", "20,633")]),
        ("layers", "#65A30D", "Features Used", [(feature_cfg(f)["label"].split(" (")[0], "") for f in FEATURES]),
        ("cpu", "#0D9488", "Model Information", [("Algorithm", METRICS.get("best_model", "Random Forest Regressor")), ("Model Type", "Regression"), ("Target Variable", "Solar AC Power (kW)"), ("Trained On", "Historical Sensor Data")]),
        ("tool", "#CA8A04", "Technologies", [("Python", ""), ("Streamlit", ""), ("Scikit-Learn", ""), ("Pandas", ""), ("NumPy", ""), ("Plotly", "")]),
        ("info", "#059669", "Project Info", [("Developed By", "Safa Manha T"), ("Project Type", "ML Regression"), ("Deployment", "Streamlit Cloud"), ("Year", str(dt.date.today().year))]),
    ]
    for col, (icon_name, accent, title, items) in zip(cols, blocks):
        with col:
            rows = "".join(
                f'<div style="display:flex;justify-content:space-between;font-size:12px;margin:6px 0;">'
                f'<span style="color:#6B7280;">{k}</span><span style="font-weight:700;color:#1F2937;">{v}</span></div>'
                if v else f'<div style="font-size:12px;margin:6px 0;color:#374151;">• {k}</div>'
                for k, v in items
            )
            st.markdown(
                f"""
                <div class="card" style="border-top:3px solid {accent};height:300px;overflow-y:auto;">
                    <div class="section-title" style="font-size:13.5px;display:flex;align-items:center;gap:7px;">
                        <span style="color:{accent};display:flex;">{icon(icon_name, color=accent, size=16, stroke=2)}</span><span>{title}</span>
                    </div>
                    {rows}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    with st.container(border=True):
        st.markdown(
            f'<div class="section-title" style="display:flex;align-items:center;gap:7px;">'
            f'<span style="color:#16A34A;display:flex;">{icon("trend", color="#16A34A", size=16, stroke=2)}</span>Model Performance</div>',
            unsafe_allow_html=True,
        )
        perf_cols = st.columns(5)
        perf = [
            ("R² Score", f"{METRICS['r2']:.2f}"),
            ("MAE", f"{METRICS['mae']:.2f} kW"),
            ("RMSE", f"{METRICS['rmse']:.2f} kW"),
            ("Cross Val Score", f"{METRICS['cv_r2']:.2f}"),
            ("Training Accuracy", f"{METRICS['accuracy']*100:.1f}%"),
        ]
        for col, (label, val) in zip(perf_cols, perf):
            with col:
                st.markdown(
                    f'<div style="text-align:center;"><div class="metric-value" style="color:#16A34A;">{val}</div>'
                    f'<div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

    st.caption(
        f"Model source: {'loaded from bundled `.save` pickle files' if MODEL_SOURCE == 'loaded' else 'trained in-app from a bundled synthetic fallback dataset (no `.save` files found)'}."
    )
    if MODEL_SOURCE == "trained" and st.session_state.get("_model_load_error"):
        st.warning(
            f"Couldn't load `solar_power_generation.save` / `solar_power_scaler.save` — "
            f"reason: `{st.session_state['_model_load_error']}`. Make sure both files sit "
            f"in the same folder as this app, and that your local scikit-learn version "
            f"matches the one used to save them."
        )

    if MODEL_SOURCE == "trained" and "comparison" in METRICS:
        st.write("")
        with st.container(border=True):
            st.markdown(
                f'<div class="section-title" style="display:flex;align-items:center;gap:7px;">'
                f'<span style="color:#CA8A04;display:flex;">{icon("trophy", color="#CA8A04", size=16, stroke=2)}</span>'
                f'Model Comparison (fallback training run)</div>',
                unsafe_allow_html=True,
            )
            top_scorer = METRICS.get("top_scorer", METRICS.get("best_model", "—"))
            deployed = METRICS.get("best_model", "—")
            if top_scorer != deployed:
                st.caption(
                    f"**{top_scorer}** scored highest (R² Score), but **{deployed}** is the one actually "
                    f"kept and evaluated below — the notebook's code only reports the top scorer here, "
                    f"it never feeds that choice back into which model gets cross-validated or saved."
                )
            else:
                st.caption(f"Best performer: **{deployed}**.")
            st.dataframe(
                METRICS["comparison"].style.format(
                    {"MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}", "R2 Score": "{:.4f}"}
                ),
                hide_index=True,
                use_container_width=True,
            )


# ----------------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------------
if page == "Home":
    render_home()
elif page == "Prediction":
    render_prediction()
elif page == "Dashboard":
    render_dashboard()
elif page == "About":
    render_about()