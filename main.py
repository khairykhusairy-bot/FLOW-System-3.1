"""
FLOW — Flood Level Observation Warning System
Main Application: Streamlit Dashboard Entry Point

Run with:
    streamlit run main.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import threading
import base64
import json as _json
from datetime import datetime
from typing import Dict, List, Optional

# FLOW modules
from database import init_db, log_monitoring_data, log_alert, get_recent_logs, get_stats_summary
from detection import DebrisDetector
from polygon_roi import PolygonROI
from tracking import CentroidTracker
from prediction import FloodPredictor
from alerts import AlertManager
from ui import (
    inject_styles, render_header, render_metric_card,
    render_blockage_bar, render_risk_panel,
    render_roi_counts, render_alerts, render_rain_panel,
    render_polygon_editor_html,
)
from utils import resize_frame, get_timestamp
from datetime import timezone, timedelta
MYT = timezone(timedelta(hours=8))
from weather import WeatherService, render_weather_sidebar, rain_intensity_to_category
from config import WEATHER_LOCATIONS
from telegram_notify import TelegramNotifier
from flood_risk_engine import FloodRiskEngine

# ── Water Level Module ────────────────────────────────────────────────────────
from water_level import WaterLevelMonitor

# ── WebRTC smooth camera integration ─────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    from webrtc_processor import FLOWVideoProcessor
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FLOW — Flood Level Observation Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Initialise Modules ────────────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    return DebrisDetector(model_path="best.pt", confidence=0.35)

@st.cache_resource
def load_predictor():
    return FloodPredictor()

@st.cache_resource
def get_roi():
    # Automatically picks up ROI_POLYGON from config.py if set by setup_polygon.py
    return PolygonROI(load_from_config=True)

@st.cache_resource
def get_tracker():
    return CentroidTracker()

@st.cache_resource
def get_alert_manager():
    return AlertManager(cooldown_seconds=12.0)

@st.cache_resource
def get_weather_service():
    # Use the first real (non-custom) preset as the startup default
    default_name, default_coords = next(
        (name, coords)
        for name, coords in WEATHER_LOCATIONS.items()
        if coords is not None
    )
    lat, lon = default_coords
    return WeatherService(
        latitude=lat,
        longitude=lon,
        location_name=default_name,
        cache_ttl=300,
    )

@st.cache_resource
def get_water_level_monitor():
    """Single WaterLevelMonitor shared across all Streamlit reruns."""
    monitor = WaterLevelMonitor()
    monitor.calibration.load()   # no-op if calibration.json doesn't exist yet
    return monitor

@st.cache_resource
def get_telegram_notifier():
    return TelegramNotifier()

@st.cache_resource
def get_flood_risk_engine():
    """Persistent FloodRiskEngine — runs weather-based scoring even before START."""
    return FloodRiskEngine()

# Initialise DB once
init_db()

detector     = load_detector()
predictor    = load_predictor()
roi          = get_roi()
tracker      = get_tracker()
alert_mgr    = get_alert_manager()
weather_svc  = get_weather_service()
wl_monitor   = get_water_level_monitor()
telegram     = get_telegram_notifier()
risk_engine  = get_flood_risk_engine()

# ─── Session State Defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "monitoring":       False,
        "theme":            "dark",
        "rain_enabled":     False,
        "rain_intensity":   0.200,
        "use_live_weather": True,
        "confidence_thr":   0.35,
        "show_trails":      True,
        "show_labels":      True,
        "frame_count":      0,
        "last_log_time":    0,
        "roi_counts":       {},
        "blockage_pct":     0.0,
        "flood_result":     {"risk": "Low Risk", "confidence": 0.92,
                             "probabilities": {"Low Risk": 0.92, "Medium Risk": 0.06, "High Risk": 0.02},
                             "risk_score": 0.05, "color": "#2ecc71"},
        "alert_list":       [],
        "history_blockage": [],
        "history_risk":     [],
        "cam_source":       0,
        "total_detections": 0,
        # Polygon draw mode  (draw_mode_target: "debris" | "gauge")
        "draw_mode":              False,
        "draw_mode_target":       "debris",   # which polygon we're drawing
        "roi_draw_capture_requested": False,
        "roi_editor_points":  [],
        "roi_editor_result":  "",
        "roi_editor_saved":   False,
        # Water level
        "water_level_enabled": True,
        "wl_gauge_roi":        [],   # persisted gauge ROI polygon points
        "wl_level_cm":         None,
        "wl_rise_rate":        0.0,
        "wl_trend":            "Stable",
        "wl_risk_status":      "Normal",
        "wl_history":          [],
        # Water level thresholds (cm)
        "wl_thresh_normal":    50,
        "wl_thresh_warning":   100,
        "wl_thresh_danger":    150,
        "wl_thresh_critical":  180,
        # Telegram notifications
        "tg_enabled":          False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── Apply Theme ───────────────────────────────────────────────────────────────
inject_styles(st.session_state.theme)

# ─── Background Worker State ───────────────────────────────────────────────────
# Defined here (before the sidebar) so stop_worker/start_worker can be called
# from button handlers without forward-reference errors.
@st.cache_resource
def get_worker_state():
    """Persistent shared state between the camera thread and the UI thread."""
    return {
        "lock":          threading.Lock(),
        "running":       False,
        "frame_rgb":     None,
        "fps":           0.0,
        "frame_count":   0,
        "blockage_pct":  0.0,
        "roi_counts":    {},
        "total_roi":     0,
        "flood_result":  {
            "risk": "Low Risk", "confidence": 0.92,
            "probabilities": {"Low Risk": 0.92, "Medium Risk": 0.06, "High Risk": 0.02},
            "risk_score": 0.05, "color": "#2ecc71",
        },
        "alert_list":    [],
        "history_blockage": [],
        "history_risk":     [],
        "total_detections": 0,
        "cam_source":       0,
        "rain_enabled":     False,
        "rain_intensity":   0.0,
        "use_live_weather": True,
        "show_labels":      True,
        "show_trails":      True,
        "blockage_warn_th": 50,
        "roi_warn_th":      10,
        # Water level
        "wl_level_cm":      None,
        "wl_rise_rate":     0.0,
        "wl_trend":         "Stable",
        "wl_risk_status":   "Normal",
        "wl_history":       [],
        "wl_enabled":       True,
        "wl_gauge_roi":     [],   # gauge ROI polygon — set by Draw Gauge ROI
        # Water level thresholds
        "wl_thresh_normal":    50,
        "wl_thresh_warning":   100,
        "wl_thresh_danger":    150,
        "wl_thresh_critical":  180,
        # Telegram
        "tg_enabled":       False,
    }

worker_state = get_worker_state()
LOG_INTERVAL = 15

@st.cache_resource
def get_worker_thread_ref():
    """Stores a reference to the running worker thread across Streamlit reruns."""
    return {"thread": None}

thread_ref = get_worker_thread_ref()

def stop_worker():
    """Signal the background thread to stop and release the camera."""
    worker_state["running"] = False
    t = thread_ref["thread"]
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
    thread_ref["thread"] = None
    tracker.reset()
    telegram.reset_state()

def start_worker():
    """Spawn the background thread if not already running."""
    t = thread_ref["thread"]
    if t is not None and t.is_alive():
        return  # already running
    worker_state.update({
        "running":          True,
        "cam_source":       st.session_state.cam_source,
        "rain_enabled":     st.session_state.rain_enabled,
        "rain_intensity":   st.session_state.rain_intensity,
        "use_live_weather": st.session_state.use_live_weather,
        "show_labels":      st.session_state.show_labels,
        "show_trails":      st.session_state.show_trails,
        "blockage_warn_th": worker_state.get("blockage_warn_th", 50),
        "roi_warn_th":      worker_state.get("roi_warn_th", 10),
        "wl_enabled":       st.session_state.get("water_level_enabled", True),
        "tg_enabled":       st.session_state.get("tg_enabled", False),
        "wl_gauge_roi":     st.session_state.get("wl_gauge_roi", []),
    })
    t = threading.Thread(target=camera_worker, args=(worker_state,), daemon=True)
    t.start()
    thread_ref["thread"] = t

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
<div style="padding:16px 0 10px;text-align:center;">
    <div style="font-family:'Nevera',-apple-system,'SF Pro Display','Helvetica Neue',Arial,sans-serif;font-size:26px;font-weight:800;background:linear-gradient(135deg,#00d4ff,#0096c7);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-1px;">
        FLOW
    </div>
    <div style="font-size:9px;letter-spacing:2px;color:#4a6b8a;margin-top:2px;">
        FLOOD MONITORING SYSTEM
    </div>
</div>
<hr style="border:none;border-top:1px solid #1e3a5f;margin:0 0 16px;">
""", unsafe_allow_html=True)

    # Theme toggle
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("☀ Light", use_container_width=True):
            st.session_state.theme = "light"
            st.rerun()
    with col_b:
        if st.button("🌙 Dark", use_container_width=True):
            st.session_state.theme = "dark"
            st.rerun()

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # ── Monitoring Controls ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">📷 MONITORING</div>', unsafe_allow_html=True)

    cam_options = {"Webcam (0)": 0, "Webcam (1)": 1, "Demo Mode": "demo"}
    cam_choice = st.selectbox("Camera Source", list(cam_options.keys()), key="cam_select")
    st.session_state.cam_source = cam_options[cam_choice]

    # ── WebRTC toggle ─────────────────────────────────────────────────────────
    if WEBRTC_AVAILABLE:
        use_webrtc = st.checkbox(
            "📡 Smooth stream (WebRTC)",
            value=st.session_state.get("use_webrtc", True),
            key="use_webrtc_chk",
            help="Uses streamlit-webrtc for a smoother camera feed at full frame rate. "
                 "Uncheck to fall back to the original OpenCV thread.",
        )
        st.session_state["use_webrtc"] = use_webrtc
    else:
        st.session_state["use_webrtc"] = False
        st.markdown(
            '<div style="font-size:10px;color:#f39c12;">'
            '⚠ streamlit-webrtc not installed — pip install streamlit-webrtc</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button(
            "▶ START" if not st.session_state.monitoring else "■ STOP",
            use_container_width=True,
            type="primary" if not st.session_state.monitoring else "secondary",
        )
    with col2:
        reset_btn = st.button("↺ Reset", use_container_width=True)

    if start_btn:
        st.session_state.monitoring = not st.session_state.monitoring
        if not st.session_state.monitoring:
            stop_worker()
            tracker.reset()
        st.rerun()
    if reset_btn:
        st.session_state.monitoring = False
        st.session_state.roi_counts = {}
        st.session_state.blockage_pct = 0.0
        st.session_state.alert_list = []
        st.session_state.history_blockage = []
        st.session_state.history_risk = []
        st.session_state.frame_count = 0
        alert_mgr.clear_all()
        stop_worker()   # explicitly release camera + kill thread
        tracker.reset()
        st.rerun()

    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)

    # ── Detection Settings ────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">🎯 DETECTION</div>', unsafe_allow_html=True)

    conf_val = st.slider(
        "Confidence Threshold", 0.10, 0.90,
        st.session_state.confidence_thr, 0.05,
        key="conf_slider"
    )
    st.session_state.confidence_thr = conf_val
    detector.set_confidence(conf_val)

    st.session_state.show_labels = st.checkbox("Show Labels", value=st.session_state.show_labels)
    st.session_state.show_trails = st.checkbox("Show Trails", value=st.session_state.show_trails)

    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)

    # ── ROI Polygon ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">🔷 POLYGON ROI</div>', unsafe_allow_html=True)

    # Show current polygon status
    n_pts = len(roi.get_polygon())
    if n_pts >= 3:
        st.markdown(
            f'<div style="font-size:11px;color:#00e676;margin-bottom:8px;">'
            f'✓ Active polygon: {n_pts} points</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:11px;color:#f39c12;margin-bottom:8px;">'
            '⚠ No polygon set — full frame monitored</div>',
            unsafe_allow_html=True,
        )

    draw_mode_btn = st.button(
        "✏ Draw Polygon" if not st.session_state.get("draw_mode") else "✕ Cancel Drawing",
        use_container_width=True,
        key="toggle_draw_mode_btn",
        type="primary" if not st.session_state.get("draw_mode") else "secondary",
    )
    if draw_mode_btn:
        st.session_state.draw_mode = not st.session_state.get("draw_mode", False)
        if st.session_state.draw_mode:
            st.session_state.draw_mode_target = "debris"
            st.session_state.roi_draw_capture_requested = True
        st.rerun()

    if st.button("🗑 Clear Polygon", use_container_width=True, key="clear_poly_btn"):
        roi.polygon = []
        roi._area = 1.0
        st.session_state.roi_editor_points = []
        st.session_state.roi_editor_result = "[]"
        st.session_state.draw_mode = False
        st.rerun()

    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)

    # ── Live Weather + Rain Simulation ───────────────────────────────────────
    render_weather_sidebar(weather_svc)

    # ── Feed live weather into FloodRiskEngine (runs always, even idle) ───────
    try:
        _wx_now = weather_svc.get_current()
        _live_mm_h = float(_wx_now.get("rain_mm", 0.0))
        _forecast_entries = weather_svc.get_forecast(hours=6)
        _forecast_acc = FloodRiskEngine.estimate_forecast_accumulation(_forecast_entries)
        risk_engine.update_weather(_live_mm_h, forecast_mm_next6h=_forecast_acc)
    except Exception:
        pass

    # ── Weather Risk Summary (visible even before START) ──────────────────────
    _wx_risk = risk_engine.get_weather_risk()
    _cat  = _wx_risk["rainfall_category"]
    _sc   = _wx_risk["score_result"]
    st.markdown(
        f'<div style="background:rgba(0,0,0,0.18);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:8px;padding:8px 10px;margin:8px 0 4px;">'
        f'<div style="font-size:10px;letter-spacing:1px;color:#4a6b8a;margin-bottom:5px;">🌧 WEATHER FLOOD RISK</div>'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:12px;font-weight:700;color:{_cat["color"]};">'
        f'{_cat["label"]}</span>'
        f'<span style="font-size:11px;color:#aaa;">{_cat["mm_h"]} mm/h</span>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#888;margin-top:3px;">'
        f'<span>Score: <b style="color:{_sc["color"]};">{_sc["score"]:.1f}</b> → <b>{_sc["category"]}</b></span>'
        f'<span>⏱ {_cat["continuous_hours"]:.1f}h</span>'
        f'</div>'
        + (f'<div style="font-size:9px;color:#f39c12;margin-top:3px;">⚠ {_cat["upgrade_reason"]}</div>'
           if _cat.get("upgrade_reason") else '')
        + f'</div>',
        unsafe_allow_html=True,
    )

    # Live weather toggle — when ON, use live rain data; when OFF, use slider
    use_live_weather = st.checkbox(
        "📡 Use Live Weather for Prediction",
        value=st.session_state.get("use_live_weather", True),
        key="use_live_weather_chk",
        help="When enabled, flood prediction uses live rain intensity from the weather API. "
             "Uncheck to manually simulate rain intensity.",
    )
    st.session_state["use_live_weather"] = use_live_weather

    if use_live_weather:
        # Pull rain intensity from live weather service
        try:
            wx = weather_svc.get_current()
            live_rain_mm_h = float(wx.get("rain_mm", 0.0))
            live_rain_norm = wx.get("rain_intensity", 0.0)   # already 0-1 from _rain_to_intensity
            # Also keep risk_engine in sync with manual-mode mm/h value
            risk_engine.update_weather(live_rain_mm_h)
        except Exception:
            live_rain_mm_h = 0.0
            live_rain_norm = 0.0
        st.session_state.rain_enabled   = False   # no visual simulation overlay
        st.session_state.rain_intensity = live_rain_norm
        st.markdown(
            f'<div style="font-size:11px;color:#00d4ff;margin:4px 0 2px;">'
            f'🌧 Live rain: <b>{live_rain_mm_h:.1f} mm/h</b> · intensity <b>{live_rain_norm:.3f}</b> '
            f'(used for flood prediction)</div>',
            unsafe_allow_html=True,
        )
    else:
        # Manual rain simulation
        st.markdown(
            '<div style="font-size:11px;color:#f39c12;margin:6px 0 4px;">'
            '🌧 <b>Rain Simulation</b> — drag to set intensity</div>',
            unsafe_allow_html=True,
        )
        rain_val = st.slider(
            "Rain Intensity", 0.000, 1.000,
            float(st.session_state.rain_intensity) if st.session_state.rain_intensity else 0.200,
            0.001,
            format="%.3f",
            key="rain_sim_slider",
            help="0 = no rain · 0.5 = moderate · 1.0 = extreme. "
                 "Enables rain animation overlay on the video feed.",
        )
        st.session_state.rain_intensity = rain_val
        st.session_state.rain_enabled   = rain_val > 0.0

        # Intensity label — names aligned with WMO live weather categories
        if rain_val == 0.0:
            label, color = "No Rain", "#4a6b8a"
        elif rain_val < 0.2:
            label, color = "Light Drizzle 🌦", "#2ecc71"
        elif rain_val < 0.4:
            label, color = "Slight Rain 🌦", "#2ecc71"
        elif rain_val < 0.6:
            label, color = "Moderate Rain 🌧", "#f39c12"
        elif rain_val < 0.8:
            label, color = "Heavy Rain 🌧", "#e67e22"
        else:
            label, color = "Violent Showers ⛈", "#e74c3c"
        st.markdown(
            f'<div style="font-size:11px;color:{color};margin-top:2px;">{label}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)

    # ── Water Level Estimation ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">💧 WATER LEVEL</div>', unsafe_allow_html=True)

    wl_enabled = st.checkbox(
        "Enable Water Level Estimation",
        value=st.session_state.get("water_level_enabled", True),
        key="wl_enabled_chk",
    )
    st.session_state["water_level_enabled"] = wl_enabled
    wl_monitor.enabled = wl_enabled

    if wl_enabled:
        # ── Gauge ROI selector ────────────────────────────────────────────────
        _wl_roi_pts = st.session_state.get("wl_gauge_roi", [])
        if len(_wl_roi_pts) >= 3:
            st.markdown(
                f'<div style="font-size:11px;color:#00e676;margin-bottom:4px;">'
                f'✓ Gauge ROI: {len(_wl_roi_pts)} points active</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:11px;color:#f39c12;margin-bottom:4px;">'
                '⚠ No gauge ROI — full frame scanned (false positives likely)</div>',
                unsafe_allow_html=True,
            )

        _gauge_draw_active = (
            st.session_state.get("draw_mode", False)
            and st.session_state.get("draw_mode_target") == "gauge"
        )
        wl_draw_btn = st.button(
            "✏ Draw Gauge ROI" if not _gauge_draw_active else "✕ Cancel Gauge Drawing",
            use_container_width=True,
            key="wl_toggle_draw_mode_btn",
            type="primary" if not _gauge_draw_active else "secondary",
            help="Draw a polygon around the flood gauge / ruler on the video feed. "
                 "The waterline detector will only search inside this area.",
        )
        if wl_draw_btn:
            if _gauge_draw_active:
                # cancel
                st.session_state.draw_mode = False
            else:
                st.session_state.draw_mode = True
                st.session_state.draw_mode_target = "gauge"
                st.session_state.roi_draw_capture_requested = True
                st.session_state.roi_editor_points = []
                st.session_state.roi_editor_result = "[]"
            st.rerun()

        if st.button("🗑 Clear Gauge ROI", use_container_width=True, key="wl_clear_roi_btn"):
            st.session_state.wl_gauge_roi = []
            wl_monitor.set_gauge_roi(None)
            worker_state["wl_gauge_roi"] = []
            st.rerun()

        if wl_monitor.is_calibrated:
            cal = wl_monitor.calibration
            st.markdown(
                f'<div style="font-size:11px;color:#00e676;margin-bottom:6px;">'
                f'✓ Calibrated: {cal.h_min_cm:.0f}–{cal.h_max_cm:.0f} cm</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:11px;color:#f39c12;margin-bottom:6px;">'
                '⚠ Not calibrated — using defaults</div>',
                unsafe_allow_html=True,
            )
        with st.expander("⚙ Calibration", expanded=False):

            # ── Gauge range (top of the scale) ────────────────────────────────
            _h_max_range = int(wl_monitor.calibration.h_max_cm) or 200

            # ── Initialise threshold session state on first run ───────────────
            for _k, _default in [
                ("wl_thresh_normal",   int(_h_max_range * 0.25)),
                ("wl_thresh_warning",  int(_h_max_range * 0.50)),
                ("wl_thresh_danger",   int(_h_max_range * 0.75)),
                ("wl_thresh_critical", int(_h_max_range * 0.90)),
            ]:
                if _k not in st.session_state:
                    st.session_state[_k] = _default

            st.markdown(
                '<div style="font-size:10px;color:#4a6b8a;margin:0 0 6px;">'
                'Drag sliders to set alert thresholds (cm):</div>',
                unsafe_allow_html=True,
            )

            # ── Four native sliders — each writes directly to session state ───
            wl_thresh_normal = st.slider(
                "🟢 Normal threshold (cm)",
                min_value=0, max_value=_h_max_range,
                value=st.session_state["wl_thresh_normal"],
                step=1, key="wl_n_slider",
                help="Below this level the system reports Normal status.",
            )
            wl_thresh_warning = st.slider(
                "🟡 Warning threshold (cm)",
                min_value=0, max_value=_h_max_range,
                value=max(st.session_state["wl_thresh_warning"], wl_thresh_normal + 1),
                step=1, key="wl_w_slider",
                help="Above this level the system raises a Warning.",
            )
            wl_thresh_danger = st.slider(
                "🟠 Danger threshold (cm)",
                min_value=0, max_value=_h_max_range,
                value=max(st.session_state["wl_thresh_danger"], wl_thresh_warning + 1),
                step=1, key="wl_d_slider",
                help="Above this level the system raises a Danger alert.",
            )
            wl_thresh_critical = st.slider(
                "🔴 Critical threshold (cm)",
                min_value=0, max_value=_h_max_range,
                value=max(st.session_state["wl_thresh_critical"], wl_thresh_danger + 1),
                step=1, key="wl_c_slider",
                help="At or above this level the system raises a Critical alert.",
            )

            # Enforce strict ordering (guard against edge cases)
            wl_thresh_warning  = max(wl_thresh_warning,  wl_thresh_normal  + 1)
            wl_thresh_danger   = max(wl_thresh_danger,   wl_thresh_warning + 1)
            wl_thresh_critical = max(wl_thresh_critical, wl_thresh_danger  + 1)

            # Persist to session state
            st.session_state["wl_thresh_normal"]   = wl_thresh_normal
            st.session_state["wl_thresh_warning"]  = wl_thresh_warning
            st.session_state["wl_thresh_danger"]   = wl_thresh_danger
            st.session_state["wl_thresh_critical"] = wl_thresh_critical

            # Push immediately to worker + monitor so the live overlay updates
            # without requiring an explicit Apply click
            worker_state["wl_thresh_normal"]   = wl_thresh_normal
            worker_state["wl_thresh_warning"]  = wl_thresh_warning
            worker_state["wl_thresh_danger"]   = wl_thresh_danger
            worker_state["wl_thresh_critical"] = wl_thresh_critical
            wl_monitor.thresholds = {
                "normal":   wl_thresh_normal,
                "warning":  wl_thresh_warning,
                "danger":   wl_thresh_danger,
                "critical": wl_thresh_critical,
            }

            # ── Save button (persists calibration.json) ───────────────────────
            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
            if st.button("💾 Save calibration", use_container_width=True, key="wl_cal_save"):
                if wl_monitor.calibration.save():
                    st.success("Calibration saved.")
                else:
                    st.error("Save failed.")
        # Live reading
        lv    = st.session_state.get("wl_level_cm")
        lv_txt = f"{lv:.1f} cm" if lv is not None else "--"
        rt    = st.session_state.get("wl_rise_rate", 0.0)
        rt_sign = "+" if rt > 0 else ""
        st.markdown(
            f'<div style="font-size:13px;color:#00d4ff;font-weight:700;">'
            f'💧 {lv_txt} &nbsp;'
            f'<span style="font-size:11px;color:#aaa;">({rt_sign}{rt:.2f} cm/min)</span>'
            f'</div>'
            f'<div style="font-size:11px;color:#aaa;margin-top:2px;">'
            f'{st.session_state.get("wl_trend","Stable")} · '
            f'{st.session_state.get("wl_risk_status","Normal")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)

    # ── Alert Thresholds ──────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">🔔 ALERT THRESHOLDS</div>', unsafe_allow_html=True)
    blockage_warn_th = st.slider("Blockage Warning (%)", 20, 90, 50, 5)
    roi_warn_th = st.slider("ROI Count Warning", 5, 30, 10, 1)

    st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)

    # ── Telegram Notifications ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-label">📲 TELEGRAM ALERTS</div>', unsafe_allow_html=True)

    tg_enabled = st.checkbox(
        "Enable Telegram Notifications",
        value=st.session_state.get("tg_enabled", False),
        key="tg_enabled_chk",
    )
    st.session_state["tg_enabled"] = tg_enabled
    worker_state["tg_enabled"]     = tg_enabled

    # Subscriber count (live from the notifier's polling thread)
    sub_count = telegram.subscriber_count
    sub_color = "#00e676" if sub_count > 0 else "#f39c12"
    st.markdown(
        f'<div style="font-size:11px;color:{sub_color};margin-bottom:6px;">'
        f'👥 {sub_count} subscriber(s) active</div>',
        unsafe_allow_html=True,
    )

    # How-to instructions
    _QR_B64 = '/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAMhAywDASIAAhEBAxEB/8QAHAABAAIDAQEBAAAAAAAAAAAAAAcIAQUGBAMC/8QAVhAAAQMCAgQHCQwIBQIGAgMBAQACAwQFBhEHEiExE0FRYXGBkRQXIlWUobGy0QgVMjZCUlRzdJPB0hYjNDVTVmJyMzeSs+FDgiQlosLi8GTxJ0SDY//EABsBAAEFAQEAAAAAAAAAAAAAAAACAwQFBgEH/8QAPhEAAgEDAQMHCwQBBAMBAQEAAAECAwQRBRIhMQYTQVFxkaEUFTI0UlNhgbHB0RYiM+HwIzVy8SRCQ2KCY//aAAwDAQACEQMRAD8AuWiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiIgAiL8ySRxjOSRrByuOSOIH6ReY19ADka2mH/APq32p74UH06m+9b7UrYl1CdqPWelF5vfCg+nU33rfanvhQfTqb71vtXdiXUG3HrPSi83vhQfTqb71vtT3woPp1N9632o2JdQbces9KLze+FB9OpvvW+1PfCg+nU33rfajYl1Btx6z0ovN74UH06m+9b7U98KD6dTfet9qNiXUG3HrPSi+MdVSyHKOpheTxNeCvsktNcTqafAIiLh0IiIAIiIAIiIAIiIAIiIAIiIAIi+ElbRxuLZKuBjhvDpACupN8DjaXE+6Lze+FB9OpvvW+1PfCg+nU33rfalbEuo5tx6z0ovN74UH06m+9b7U98KD6dTfet9qNiXUG3HrPSi83vhQfTqb71vtT3woPp1N9632rmxLqDbj1npRfmN7JGh8b2vaeNpzC/SSKCIiACIsOcGtLnEADeSUAZReY19ADka2m+9b7U98KD6dTfet9qVsS6hO1HrPSi83vhQfTqb71vtX6jrKORwbHVQPcdwbICUbEuoNqPWfdERJFBERABERABERABF+JZYoW60sjI28rnABIpYpm60UjJG8rXAhdw8ZOZXA/aIi4dCIiACIviyqppH8Gyohc/5oeCV1JvgcykfZERcOhERABERABERABEX5e9kbdZ72tHKTkgD9IvOa+hByNbTA88rfase+FB9OpvvW+1L2JdQnbj1npReb3woPp1N9632p74UH06m+9b7Uc3LqDbj1npReb3woPp1N9632p74UH06m+9b7Uc3LqDbj1npReb3woPp1N9632p74UH06m+9b7Uc3LqDbj1npReb3woPp1N9632p74UH06m+9b7Uc3LqDbj1npRfBlZSPOTKqBx5pAV90lpridTT4BERcOhERABERABERABERABERABERABEXyrKmno6WWqqpWQwRNLnvccg0LqTbwjjeN7PqSACSQAN5K4fFWkiz2l76agHvjVN2Hg3ZRtPO7j6lwmP8e1l9kkorc9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e08oO8FbBEqE5QkpReGjqbW9EbVGh2wvkLobjcYWcTdZrsushfLvN2fxxceyP8qk5FZrW79f/V+A7z9TrIx7zdn8cXHsj/Knebs/ji49kf5VJyI8+X/vH4fgPKKnWRj3m7P44uPZH+VO83Z/HFx7I/yqTkR58v8A3j8PwHlFTrIx7zdn8cXHsj/Knebs/ji49kf5VJyI8+X/ALx+H4Dyip1kY95uz+OLj2R/lTvN2fxxceyP8qk5EefL/wB4/D8Bz9TrOAtOibDVHO2aqfV1+qcwyZ4DT0hoGa7yCKKCFkMMbY42DVaxoyDRyAL9oodzeV7p5rTbG5TlLiwiIowkLV4jw/acQ0gprrSNma34Dtz2HlBG0LaIl06k6clKDw11HU2nlEb957D/AA+v3fceCz/w9du7pyzXZ4bw9acPUhprVSNha74b973nlJO0raopVxqN1cR2as20KlUlJYbCIihCAvxNFHNC+GaNskbxqua4ZgjkIX7RCeAOAu2ifDVZO6amdVUGscyyF4LeoOByWwwxo7w5YqltXHBJV1LDmySodrap5QNwK69FPnqt5Onzcqjx2jjqzaxkIiKANhERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERABERAH//2Q=='
    st.markdown(
        '<div style="font-size:11px;color:var(--text-muted);background:var(--bg-secondary);'
        'border-radius:8px;padding:8px 10px;margin-bottom:8px;'
        'border:1px solid var(--border-color);">'
        '<div style="display:flex;gap:10px;align-items:flex-start;">'
        '<div style="flex:1;">'
        '📱 <b style="color:var(--text-primary);">How to subscribe:</b><br>'
        '1. Open Telegram<br>'
        '2. Search <b><a href="https://t.me/Aiflowsystembot" '
        'target="_blank" style="color:var(--accent-cyan);">@Aiflowsystembot</a></b><br>'
        '3. Send <code style="background:var(--bg-card);color:var(--text-primary);padding:1px 4px;border-radius:3px;">/start</code><br>'
        '4. Done — alerts arrive automatically!<br><br>'
        'Commands: <code style="background:var(--bg-card);color:var(--text-primary);padding:1px 4px;border-radius:3px;">/start</code> · '
        '<code style="background:var(--bg-card);color:var(--text-primary);padding:1px 4px;border-radius:3px;">/stop</code> · '
        '<code style="background:var(--bg-card);color:var(--text-primary);padding:1px 4px;border-radius:3px;">/status</code>'
        '</div>'
        f'<div style="flex-shrink:0;">'
        f'<img src="data:image/png;base64,{_QR_B64}" '
        'style="width:90px;height:90px;border-radius:6px;display:block;" '
        'alt="QR Code"/>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if tg_enabled:
        col_tg1, col_tg2 = st.columns(2)
        with col_tg1:
            if st.button("🔔 Test", use_container_width=True, key="tg_test_btn"):
                ok, result_msg = telegram.send_test()
                if ok:
                    st.success(result_msg)
                else:
                    st.warning(result_msg)
        with col_tg2:
            st.markdown(
                '<div style="font-size:11px;color:#00e676;'
                'padding:8px 0;text-align:center;">● Auto-mode</div>',
                unsafe_allow_html=True,
            )
        errs = telegram.last_errors
        if errs:
            with st.expander("⚠ Send errors", expanded=False):
                for e in errs:
                    st.caption(e)
    else:
        st.markdown(
            '<div style="font-size:11px;color:#4a6b8a;">Enable above to activate '
            'auto-broadcast alerts to all subscribers.</div>',
            unsafe_allow_html=True,
        )

    # ── Model Status ──────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="font-size:10px;color:#4a6b8a;text-align:center;padding:4px 0;">
    Model: {detector.status}<br>
    Frames: {st.session_state.frame_count:,}
</div>
""", unsafe_allow_html=True)

# ─── Demo Frame Generator (defined early — used by ROI editor & worker) ────────
def generate_demo_frame(tick: int, width: int = 960, height: int = 540) -> np.ndarray:
    """
    Generate an animated demo frame simulating a river feed.
    Width/height match the actual webcam resolution when known.
    """
    w, h = width, height
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # ── Sky gradient (top 35% of frame) ───────────────────────────────────────
    sky_end = int(h * 0.35)
    for y in range(sky_end):
        t = y / max(sky_end, 1)
        b = int(60 + t * 30)
        g = int(40 + t * 30)
        r = int(20 + t * 15)
        frame[y, :] = (b, g, r)

    # ── Riverbanks ────────────────────────────────────────────────────────────
    bank_top    = sky_end
    bank_h      = max(int(h * 0.05), 8)
    bank_bot_y  = h - max(int(h * 0.05), 8)
    frame[bank_top : bank_top + bank_h, :] = (40, 90, 30)   # top bank
    frame[bank_bot_y:,                   :] = (35, 80, 25)   # bottom bank

    # ── River body ────────────────────────────────────────────────────────────
    river_top = bank_top + bank_h
    river_bot = bank_bot_y
    river_h   = river_bot - river_top
    for y in range(river_top, river_bot):
        depth = (y - river_top) / max(river_h, 1)
        b = int(110 - depth * 30)
        g = int(75  - depth * 20)
        r = int(35  - depth * 10)
        frame[y, :] = (max(b, 20), max(g, 10), max(r, 5))

    # ── Animated water ripples ─────────────────────────────────────────────────
    n_ripples = max(w // 55, 8)
    for i in range(n_ripples):
        phase   = (tick * 2 + i * 47) % 360
        x_base  = int((i + 0.5) * w / n_ripples)
        x_off   = int(20 * np.sin(np.radians(phase)))
        y_line  = river_top + int(river_h * (0.25 + 0.5 * ((i * 13) % 100) / 100))
        y_wave  = y_line + int(6 * np.sin(np.radians(phase * 2 + i * 30)))
        rx      = min(max(int(w * 0.04), 20), 60)
        ry      = max(int(h * 0.012), 4)
        cx      = min(max(x_base + x_off, rx), w - rx)
        alpha_r = int(80 + 50 * np.sin(np.radians(phase)))
        cv2.ellipse(frame, (cx, y_wave), (rx, ry), 0, 0, 180,
                    (alpha_r, alpha_r - 20, 60), 1, cv2.LINE_AA)

    # ── Flowing current streaks ────────────────────────────────────────────────
    np.random.seed(42)
    for i in range(18):
        sx   = int(np.random.rand() * w)
        sy   = river_top + int(np.random.rand() * river_h)
        dx   = int(40 * np.sin(np.radians((tick * 4 + i * 71) % 360)))
        ex   = (sx + dx) % w
        cv2.line(frame, (sx, sy), (ex, sy), (100, 80, 55), 1, cv2.LINE_AA)

    # ── Trees on top bank ─────────────────────────────────────────────────────
    tree_spacing = max(w // 9, 60)
    for i in range(0, w, tree_spacing):
        sway = int(6 * np.sin(np.radians(tick * 1.5 + i * 20)))
        tx   = i + sway + tree_spacing // 2
        ty   = bank_top - max(int(h * 0.05), 15)
        tr   = max(int(h * 0.04), 12)
        cv2.circle(frame, (tx, ty), tr,           (25, 110, 20), -1)
        cv2.circle(frame, (tx - tr//3, ty - tr//3), tr // 2, (30, 130, 25), -1)
        cv2.rectangle(frame, (tx - 3, ty + tr - 2), (tx + 3, ty + tr + bank_h), (55, 38, 18), -1)

    # ── Debris objects drifting downstream ────────────────────────────────────
    debris_seed = tick // 6
    np.random.seed(debris_seed)
    for i in range(5):
        dx_base  = int((i * w / 5 + tick * 3) % w)
        dy_base  = river_top + int(river_h * (0.2 + 0.6 * np.random.rand()))
        dy_wave  = dy_base + int(4 * np.sin(np.radians(tick * 3 + i * 60)))
        size     = max(int(h * 0.018), 6)
        col_idx  = i % 3
        dcolor   = [(0, 120, 200), (20, 80, 150), (40, 100, 170)][col_idx]
        cv2.ellipse(frame, (dx_base, dy_wave), (size, size // 2), 0, 0, 360,
                    dcolor, -1, cv2.LINE_AA)

    # ── Rain animation (driven by session state simulation) ───────────────────
    rain_intensity = 0.0
    try:
        import streamlit as _st
        if not _st.session_state.get("use_live_weather", True):
            rain_intensity = float(_st.session_state.get("rain_intensity", 0.0))
    except Exception:
        pass

    font_scale = max(w / 1920, 0.38)   # defined here — used by rain badge & labels below

    if rain_intensity > 0.001:
        n_drops = max(1, int(rain_intensity * 600))
        rng = np.random.default_rng(42)
        xs_base = rng.integers(0, w, n_drops).astype(np.int32)
        ys_base = rng.integers(0, max(1, h - 18), n_drops).astype(np.int32)

        shift   = int(tick * 6) % max(w, 1)
        xs_anim = (xs_base + shift) % w
        xe_anim = (xs_anim - 3).clip(0, w - 1)
        ye_anim = (ys_base + 15).clip(0, h - 1)

        rain_layer = np.zeros((h, w), dtype=np.uint8)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            px = (xs_anim + t * (xe_anim - xs_anim)).astype(np.int32).clip(0, w - 1)
            py = (ys_base + t * (ye_anim - ys_base)).astype(np.int32).clip(0, h - 1)
            rain_layer[py, px] = 255

        kernel = np.ones((2, 1), np.uint8)
        rain_layer = cv2.dilate(rain_layer, kernel, iterations=1)

        rain_color = np.zeros_like(frame)
        rain_color[:, :, 0] = rain_layer
        rain_color[:, :, 1] = rain_layer
        rain_color[:, :, 2] = rain_layer
        alpha = 0.30 + rain_intensity * 0.35
        cv2.addWeighted(rain_color, alpha, frame, 1.0, 0, frame)

        # Dark overlay to simulate overcast sky in heavy rain
        if rain_intensity > 0.5:
            dark = np.zeros_like(frame)
            cv2.addWeighted(dark, rain_intensity * 0.25, frame, 1.0, 0, frame)

        # Rain intensity badge
        badge_txt = f"SIM RAIN: {rain_intensity:.2f}"
        badge_col = (80, 160, 255)
        cv2.putText(frame, badge_txt,
                    (w - max(int(w * 0.22), 140), max(int(h * 0.045), 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, badge_col, 1, cv2.LINE_AA)

    # ── Timestamp & camera label ───────────────────────────────────────────────
    ts = datetime.now(MYT).strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, f"FLOW DEMO FEED  {ts}",
                (12, max(int(h * 0.045), 18)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "CAM-01  RIVER GATE A  [DEMO]",
                (12, max(int(h * 0.08), 32)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, (120, 160, 200), 1, cv2.LINE_AA)

    return frame


# ─── Main Dashboard ─────────────────────────────────────────────────────────────
render_header(st.session_state.monitoring)

# ── Top row: Webcam + Key Metrics ─────────────────────────────────────────────
feed_col, metrics_col = st.columns([5, 2], gap="medium")

with feed_col:
    st.markdown('<div class="section-header">Live Webcam Feed</div>', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    fps_placeholder = st.empty()

with metrics_col:
    st.markdown('<div class="section-header">System Metrics</div>', unsafe_allow_html=True)
    metric_ph_blockage = st.empty()
    metric_ph_rain     = st.empty()
    metric_ph_risk     = st.empty()

# ── Bottom row: ROI Counts | Risk | Alerts ─────────────────────────────────────
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
roi_col, risk_col, alert_col = st.columns([2, 2, 3], gap="medium")

with roi_col:
    st.markdown('<div class="section-header">ROI Object Count</div>', unsafe_allow_html=True)
    roi_count_ph = st.empty()

with risk_col:
    st.markdown('<div class="section-header">Flood Prediction</div>', unsafe_allow_html=True)
    risk_ph = st.empty()

with alert_col:
    st.markdown('<div class="section-header">Alert Center</div>', unsafe_allow_html=True)
    alert_ph = st.empty()

# ── History Charts ─────────────────────────────────────────────────────────────
with st.expander("📊 Monitoring History", expanded=False):
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        blockage_chart_ph = st.empty()
    with chart_col2:
        risk_chart_ph = st.empty()

# ── Data Log Table ─────────────────────────────────────────────────────────────
with st.expander("🗄 Data Log (Recent 100 entries)", expanded=False):
    log_ph = st.empty()

# ── ROI Polygon Draw Mode — overlays on the webcam feed when active ────────────
if st.session_state.get("draw_mode", False):
    _draw_target = st.session_state.get("draw_mode_target", "debris")

    # Capture background frame once when draw mode is activated
    if st.session_state.get("roi_draw_capture_requested", False):
        st.session_state.roi_draw_capture_requested = False
        st.session_state.roi_editor_points = []
        st.session_state.roi_editor_result = "[]"
        with worker_state["lock"]:
            _wf = worker_state.get("frame_rgb")
        if _wf is not None:
            _bg = cv2.cvtColor(_wf, cv2.COLOR_RGB2BGR)
        elif st.session_state.cam_source != "demo":
            try:
                _cap_tmp = cv2.VideoCapture(int(st.session_state.cam_source))
                _cap_tmp.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                _ret, _raw = _cap_tmp.read()
                _cap_tmp.release()
                _bg = resize_frame(_raw, width=960) if _ret and _raw is not None else generate_demo_frame(0)
            except Exception:
                _bg = generate_demo_frame(0)
        else:
            _bg = generate_demo_frame(0)
        _bg_rgb = cv2.cvtColor(_bg, cv2.COLOR_BGR2RGB)
        _, _enc = cv2.imencode(".jpg", _bg_rgb, [cv2.IMWRITE_JPEG_QUALITY, 88])
        st.session_state.roi_bg_b64    = base64.b64encode(_enc.tobytes()).decode()
        st.session_state.roi_bg_shape  = (_bg_rgb.shape[1], _bg_rgb.shape[0])

    bg_b64  = st.session_state.get("roi_bg_b64", "")
    bg_w, bg_h = st.session_state.get("roi_bg_shape", (960, 540))
    existing_pts = st.session_state.get("roi_editor_points", [])

    # Draw the interactive canvas in the same feed_col placeholder
    with feed_col:
        if _draw_target == "gauge":
            st.markdown(
                '<div style="font-size:12px;color:#00d4ff;padding:6px 0 8px;">'
                '💧 <b>Gauge ROI Draw Mode</b> — left-click to add points · right-click to undo · '
                'R to reset · Draw a polygon around the <b>flood gauge / ruler</b> · '
                'Click <b>Apply Gauge ROI</b> when done</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:12px;color:#f39c12;padding:6px 0 8px;">'
                '✏ <b>Draw Mode</b> — left-click to add points · right-click to undo · '
                'R to reset · Click <b>Apply</b> when done</div>',
                unsafe_allow_html=True,
            )

        # ── Render canvas — Apply/Save/Cancel are inside the HTML ────────────
        _cam_idx = 0
        try:
            _cam_idx = int(st.session_state.cam_source)
        except (TypeError, ValueError):
            _cam_idx = 0

        import streamlit.components.v1 as _components
        _component_html = render_polygon_editor_html(
            bg_b64=bg_b64, bg_w=bg_w, bg_h=bg_h,
            existing_points=existing_pts,
            cam_index=_cam_idx,
            draw_target=_draw_target,
            show_save_btn=(_draw_target != "gauge"),
        )
        _components.html(_component_html, height=bg_h + 200, scrolling=False)

        # ── Hidden sync input — canvas JS writes action+coords here via DOM ──
        # Label "FLOW_ROI_SYNC" lets the canvas JS locate this specific input.
        _raw_sync = st.text_input(
            "FLOW_ROI_SYNC",
            value=st.session_state.get("roi_editor_result", "[]"),
            key="roi_pts_readback",
            label_visibility="hidden",
        )

        # ── Parse action payload written by the canvas Apply/Save/Cancel btns ─
        _action = ""
        _pts: list = []
        try:
            _parsed = _json.loads(_raw_sync) if _raw_sync.strip() else {}
            if isinstance(_parsed, dict) and "action" in _parsed:
                _action = _parsed.get("action", "")
                _pts    = [tuple(int(v) for v in p) for p in _parsed.get("pts", [])]
            elif isinstance(_parsed, list):
                # Legacy coordinate-only list: just update session state silently
                st.session_state.roi_editor_points = [
                    tuple(int(v) for v in p) for p in _parsed if len(p) == 2
                ]
        except Exception:
            pass

        if _action == "apply":
            if len(_pts) >= 3:
                if _draw_target == "gauge":
                    st.session_state.wl_gauge_roi = _pts
                    wl_monitor.set_gauge_roi(_pts)
                    worker_state["wl_gauge_roi"] = _pts
                    st.session_state.draw_mode   = False
                    st.session_state.roi_editor_result = "[]"
                    st.success(
                        f"✅ Gauge ROI applied — {len(_pts)} points. "
                        "Water level detector is now restricted to this area."
                    )
                else:
                    roi.set_polygon(_pts)
                    st.session_state.draw_mode   = False
                    st.session_state.roi_editor_result = "[]"
                    st.success(f"✅ Polygon ROI applied — {len(_pts)} points active.")
                st.rerun()
            else:
                st.warning("Need at least 3 points to define the ROI area.")

        elif _action == "save" and _draw_target != "gauge":
            if len(_pts) >= 3:
                roi.set_polygon(_pts)
                from setup_polygon import save_polygon_to_config
                try:
                    import config as _cfg_mod
                    _cfg_path = _cfg_mod.CONFIG_PATH
                except ImportError:
                    import os as _os
                    _cfg_path = _os.path.join(_os.path.dirname(__file__), "config.py")
                if save_polygon_to_config(_pts, _cfg_path):
                    st.session_state.draw_mode   = False
                    st.session_state.roi_editor_result = "[]"
                    st.success(f"💾 Saved & applied {len(_pts)}-point polygon to config.")
                else:
                    st.error("Could not write to config.py — check file permissions.")
                st.rerun()
            else:
                st.warning("Need at least 3 points.")

        elif _action == "cancel":
            st.session_state.draw_mode   = False
            st.session_state.roi_editor_result = "[]"
            st.rerun()


# ─── Background Camera / Processing Thread ──────────────────────────────────────
def camera_worker(state: dict):
    """
    Optimised camera worker — targets ~25 FPS by:
      1. Running YOLO only every DETECT_EVERY frames (not every frame).
      2. Reusing the last detection result on skipped frames.
      3. Setting explicit 30 FPS + resolution hints on VideoCapture.
      4. Fully vectorised rain overlay (no Python for-loop over drops).
      5. Encoding at 640px width for Streamlit (upscaled by browser).
    """
    cam_src        = state["cam_source"]
    use_demo       = (cam_src == "demo")
    cap_local      = None
    tick           = 0
    last_log       = 0.0
    fps_count      = 0
    t_fps          = time.time()
    demo_w, demo_h = 960, 540

    # ── How often to run YOLO (every N frames) ────────────────────────────────
    # 1 = every frame (slowest), 3 = every 3rd frame (~3× faster), etc.
    # At 25 FPS pipeline, DETECT_EVERY=3 → ~8 detections/sec which is plenty.
    DETECT_EVERY = 3

    # Cached results reused on non-detection frames
    last_detections   = []
    last_inside_dets  = []
    last_outside_dets = []
    last_blockage_pct = 0.0
    last_roi_counts   = {}
    last_total_roi    = 0
    last_flood_result = {
        "risk": "Low Risk", "confidence": 0.92,
        "probabilities": {"Low Risk": 0.92, "Medium Risk": 0.06, "High Risk": 0.02},
        "risk_score": 0.05, "color": "#2ecc71",
    }
    last_new_alerts   = []

    if not use_demo:
        # ── Use DSHOW backend on Windows to avoid MSMF grab errors ───────────
        # MSMF (Media Foundation) has a known bug where it drops frames and
        # logs "can't grab frame. Error: -2147483638". DSHOW is more stable.
        import platform
        if platform.system() == "Windows":
            cap_local = cv2.VideoCapture(int(cam_src), cv2.CAP_DSHOW)
        else:
            cap_local = cv2.VideoCapture(int(cam_src))

        # ── Explicit camera hints for maximum FPS ──────────────────────────────
        cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # don't queue stale frames
        cap_local.set(cv2.CAP_PROP_FPS, 30)             # request 30 FPS from driver
        cap_local.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap_local.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPG >> YUY2 speed

        if not cap_local.isOpened():
            print("[Worker] Camera not available — switching to demo mode.")
            use_demo = True
        else:
            # Warm up: discard first few frames (MSMF/DSHOW often sends blank frames)
            for _ in range(3):
                cap_local.read()
            _ret, _probe = cap_local.read()
            if _ret and _probe is not None:
                _probe_r = resize_frame(_probe, width=960)
                demo_h, demo_w = _probe_r.shape[:2]

    # Pre-build rain drop position arrays once per intensity change
    _last_rain_intensity = -1.0
    _rain_xs = _rain_ys = _rain_xe = _rain_ye = None

    # Periodic tracker reset — every ~10 min at 25 FPS (15 000 frames)
    TRACKER_RESET_INTERVAL = 15_000

    try:
        while state["running"]:
            tick += 1
            fps_count += 1

            # ── Periodic tracker reset (memory hygiene)
            if tick % TRACKER_RESET_INTERVAL == 0:
                tracker.reset()

            # ── Acquire frame ─────────────────────────────────────────────────
            if use_demo:
                frame = generate_demo_frame(tick, width=demo_w, height=demo_h)
            else:
                ret, raw_frame = cap_local.read()
                if not ret:
                    # Transient grab failure (common with MSMF/DSHOW on Windows).
                    # Retry once immediately before sleeping — usually recovers.
                    ret, raw_frame = cap_local.read()
                    if not ret:
                        time.sleep(0.01)
                        continue
                frame = resize_frame(raw_frame, width=960)
                demo_h, demo_w = frame.shape[:2]

            h, w = frame.shape[:2]

            # ── Read UI settings ──────────────────────────────────────────────
            show_labels    = state["show_labels"]
            show_trails    = state["show_trails"]
            rain_on        = state["rain_enabled"]          # True only in simulation mode
            # rain_intensity is always set (live or simulated) — used for prediction
            rain_intensity = state["rain_intensity"]        # 0.0 when no rain/simulation

            # ── Detection (every DETECT_EVERY frames only) ────────────────────
            run_detection = (tick % DETECT_EVERY == 0)

            if run_detection:
                detections        = detector.detect(frame)
                tracked           = tracker.update(detections)
                inside_dets, outside_dets = roi.classify_detections(detections, frame.shape)
                blockage_pct      = roi.calculate_blockage(inside_dets, frame.shape)
                roi_counts        = roi.count_by_class(inside_dets)
                total_roi         = sum(roi_counts.values())

                # ── Water Level Estimation ────────────────────────────────────
                wl_enabled = state.get("wl_enabled", True)
                if wl_enabled:
                    # Sync gauge ROI whenever the sidebar updates it
                    _new_gauge_roi = state.get("wl_gauge_roi") or []
                    _cur_gauge_roi = wl_monitor.gauge_roi or []
                    if _new_gauge_roi != _cur_gauge_roi:
                        wl_monitor.set_gauge_roi(_new_gauge_roi if _new_gauge_roi else None)

                    # If Apply was clicked, sync the updated calibration into the monitor
                    new_cal = state.get("wl_calibration")
                    if new_cal:
                        wl_monitor.calibration.y_min_px = int(new_cal.get("y_min_px", wl_monitor.calibration.y_min_px))
                        wl_monitor.calibration.y_max_px = int(new_cal.get("y_max_px", wl_monitor.calibration.y_max_px))
                        wl_monitor.calibration.h_min_cm = float(new_cal.get("h_min_cm", wl_monitor.calibration.h_min_cm))
                        wl_monitor.calibration.h_max_cm = float(new_cal.get("h_max_cm", wl_monitor.calibration.h_max_cm))
                        state["wl_calibration"] = None  # consume it — don't re-apply every frame
                    # Keep wl_monitor thresholds in sync with the sidebar sliders
                    wl_monitor.thresholds = {
                        "normal":   state.get("wl_thresh_normal",   50),
                        "warning":  state.get("wl_thresh_warning",  100),
                        "danger":   state.get("wl_thresh_danger",   150),
                        "critical": state.get("wl_thresh_critical", 180),
                    }
                    wl_result = wl_monitor.process(frame)
                    water_level_norm = wl_result["level_norm"]
                    for wl_msg in wl_result.get("alerts", []):
                        sev = "CRITICAL" if "CRITICAL" in wl_msg else "WARNING"
                        alert_mgr._trigger("water_level", wl_msg, sev)
                else:
                    wl_result = None
                    water_level_norm = min(
                        1.0, blockage_pct / 100 * 0.6 + rain_intensity * 0.4
                    )

                flood_result = predictor.predict_fused(
                    roi_count=total_roi,
                    blockage_pct=blockage_pct,
                    rain_intensity=rain_intensity,
                    water_level=water_level_norm,
                    risk_engine=risk_engine,
                )

                custom_th = {
                    "blockage_warning":  state["blockage_warn_th"],
                    "roi_count_warning": state["roi_warn_th"],
                }
                new_alerts = alert_mgr.evaluate(
                    blockage_pct=blockage_pct,
                    rain_intensity=rain_intensity,
                    roi_count=total_roi,
                    flood_risk=flood_result["risk"],
                    custom_thresholds=custom_th,
                )
                for a in new_alerts:
                    log_alert(a.alert_type, a.message, a.severity)

                # ── Telegram notification ─────────────────────────────────────
                if state.get("tg_enabled", False):
                    telegram.set_location(weather_svc.location_name)
                    telegram.evaluate(
                        flood_result   = flood_result,
                        blockage_pct   = blockage_pct,
                        rain_intensity = rain_intensity,
                        roi_count      = total_roi,
                        water_level_cm = wl_result.get("level_cm") if wl_result else None,
                        wl_trend       = wl_result.get("trend", "Stable") if wl_result else "Stable",
                    )

                # Cache for non-detection frames
                last_detections   = detections
                last_inside_dets  = inside_dets
                last_outside_dets = outside_dets
                last_blockage_pct = blockage_pct
                last_roi_counts   = roi_counts
                last_total_roi    = total_roi
                last_flood_result = flood_result
                last_new_alerts   = new_alerts
            else:
                # Reuse cached values — just update tracker with empty (keeps IDs alive)
                tracked       = tracker.update([])
                inside_dets   = last_inside_dets
                outside_dets  = last_outside_dets
                blockage_pct  = last_blockage_pct
                roi_counts    = last_roi_counts
                total_roi     = last_total_roi
                flood_result  = last_flood_result
                new_alerts    = []
                wl_result     = wl_monitor.last_result if state.get("wl_enabled", True) else None

            # ── Draw on frame ─────────────────────────────────────────────────
            display_frame = roi.draw_on_frame(
                frame, inside_dets, outside_dets, show_labels=show_labels
            )
            if show_trails:
                display_frame = tracker.draw_trails(display_frame, tracked)

            # ── Water Level Overlay ───────────────────────────────────────────
            if state.get("wl_enabled", True) and wl_result:
                display_frame = wl_monitor.draw(display_frame, wl_result)

            # ── Rain overlay — pure numpy, no Python loop ─────────────────────
            if rain_on and rain_intensity > 0:
                # Rebuild base drop positions only when frame size or intensity changes
                if rain_intensity != _last_rain_intensity or _rain_xs is None:
                    n_drops = max(1, int(rain_intensity * 500))
                    rng = np.random.default_rng(42)
                    _rain_xs = rng.integers(0, w,          n_drops).astype(np.int32)
                    _rain_ys = rng.integers(0, max(1, h - 15), n_drops).astype(np.int32)
                    _last_rain_intensity = rain_intensity

                # Animate: shift x positions each tick so drops move across frame
                shift    = int(tick * 5) % max(w, 1)
                xs_anim  = (_rain_xs + shift) % w          # shape (n,)
                xe_anim  = (xs_anim - 2).clip(0, w - 1)
                ys_anim  = _rain_ys
                ye_anim  = (_rain_ys + 13).clip(0, h - 1)

                # Build rain mask via numpy index assignment (no cv2 loop)
                # Draw diagonal streaks: set pixels along each drop line
                rain_layer = np.zeros((h, w), dtype=np.uint8)

                # Vectorised Bresenham-style: each drop is a short diagonal segment
                # Sample 4 pixels per drop (t=0, 0.33, 0.66, 1.0)
                for t in (0.0, 0.33, 0.66, 1.0):
                    px = (xs_anim + t * (xe_anim - xs_anim)).astype(np.int32).clip(0, w - 1)
                    py = (ys_anim + t * (ye_anim - ys_anim)).astype(np.int32).clip(0, h - 1)
                    rain_layer[py, px] = 255

                # Dilate 1px so single-pixel drops are visible
                kernel = np.ones((2, 1), np.uint8)
                rain_layer = cv2.dilate(rain_layer, kernel, iterations=1)

                # Colour the layer (light blue-white) and blend onto frame
                rain_color  = np.zeros_like(display_frame)
                rain_color[:, :, 0] = rain_layer          # B
                rain_color[:, :, 1] = rain_layer          # G
                rain_color[:, :, 2] = rain_layer          # R  → white-ish
                alpha = 0.35 + rain_intensity * 0.30      # more intense = more opaque
                cv2.addWeighted(rain_color, alpha, display_frame, 1.0, 0, display_frame)

            # ── HUD ───────────────────────────────────────────────────────────
            cv2.putText(
                display_frame,
                f"Blockage: {blockage_pct:.1f}%  |  ROI: {total_roi}  |  Risk: {flood_result['risk']}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA,
            )
            if alert_mgr.has_critical():
                cv2.rectangle(display_frame, (0, 0), (w, 36), (0, 0, 200), -1)
                cv2.putText(display_frame, "⚠ CRITICAL ALERT — CHECK ALERT CENTER",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # BGR → RGB for Streamlit
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # ── FPS ───────────────────────────────────────────────────────────
            elapsed = time.time() - t_fps
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                t_fps = time.time()
            else:
                fps = state.get("fps", 0.0)

            # ── DB logging (every LOG_INTERVAL seconds) ───────────────────────
            now = time.time()
            if now - last_log >= LOG_INTERVAL:
                # Determine rain category name and numeric value for the log.
                # - Live weather: use the WMO condition label from the API
                #   (e.g. "Moderate Rain", "Thunderstorm") so the category name
                #   matches exactly what the live weather sidebar shows.
                # - Simulation: derive category from rain_intensity_to_category()
                #   so it matches the slider labels shown in the sidebar.
                use_live = state.get("use_live_weather", False)
                if use_live:
                    try:
                        wx = weather_svc.get_current()
                        log_humidity    = wx.get("humidity", None)
                        log_wind        = wx.get("wind_speed", None)
                        log_temperature = wx.get("temperature", None)
                        log_feels       = wx.get("feels_like", None)
                        rain_category   = wx.get("condition_label",
                                                  rain_intensity_to_category(rain_intensity))
                    except Exception:
                        log_humidity = log_wind = log_temperature = log_feels = None
                        rain_category = rain_intensity_to_category(rain_intensity)
                else:
                    log_humidity = log_wind = log_temperature = log_feels = None
                    rain_category = rain_intensity_to_category(rain_intensity)

                log_monitoring_data({
                    "timestamp":            get_timestamp(),
                    # Dynamic per-class counts — all classes from best.pt are included
                    "roi_counts":           roi_counts,
                    "total_roi_objects":    total_roi,
                    "blockage_percentage":  blockage_pct,
                    "rain_intensity":       rain_category,
                    "rain_intensity_value": round(rain_intensity, 4),
                    "humidity":             log_humidity,
                    "wind_speed":           log_wind,
                    "temperature":          log_temperature,
                    "feels_like":           log_feels,
                    "flood_risk":           flood_result["risk"],
                    "confidence":           flood_result["confidence"],
                    "alert_triggered":      len(new_alerts) > 0,
                    "alert_message":        "; ".join(a.message for a in new_alerts[:3]),
                    # Water level fields
                    "water_level_cm":       wl_result.get("level_cm") if wl_result else None,
                    "water_level_trend":    wl_result.get("trend") if wl_result else None,
                    "water_level_status":   wl_result.get("risk_status") if wl_result else None,
                    "water_rise_rate":      wl_result.get("rise_rate", 0.0) if wl_result else None,
                    # Location
                    "location":             weather_svc.location_name,
                })
                last_log = now

            # ── Write results under lock ──────────────────────────────────────
            history_max = 60
            with state["lock"]:
                state["frame_rgb"]        = frame_rgb
                state["fps"]              = round(fps, 1)
                state["frame_count"]      = tick
                state["blockage_pct"]     = blockage_pct
                state["roi_counts"]       = roi_counts
                state["total_roi"]        = total_roi
                state["flood_result"]     = flood_result
                state["alert_list"]       = alert_mgr.get_active_alerts()
                state["total_detections"] += len(last_detections)
                state["history_blockage"] = (state["history_blockage"] + [blockage_pct])[-history_max:]
                state["history_risk"]     = (state["history_risk"] + [flood_result["risk_score"] * 100])[-history_max:]
                # Water level
                if wl_result:
                    state["wl_level_cm"]    = wl_result.get("level_cm")
                    state["wl_rise_rate"]   = wl_result.get("rise_rate", 0.0)
                    state["wl_trend"]       = wl_result.get("trend", "Stable")
                    state["wl_risk_status"] = wl_result.get("risk_status", "Normal")
                    state["wl_history"]     = wl_result.get("history_cm", [])

    finally:
        if cap_local is not None:
            cap_local.release()
        state["running"] = False
        print("[Worker] Camera thread stopped.")





# ─── Monitoring Render Loop ──────────────────────────────────────────────────────
if st.session_state.monitoring:

    # Push live UI settings to worker on every Streamlit rerun
    worker_state["rain_enabled"]     = st.session_state.rain_enabled
    worker_state["rain_intensity"]   = st.session_state.rain_intensity
    worker_state["use_live_weather"] = st.session_state.use_live_weather
    worker_state["show_labels"]      = st.session_state.show_labels
    worker_state["show_trails"]      = st.session_state.show_trails
    worker_state["blockage_warn_th"] = blockage_warn_th
    worker_state["roi_warn_th"]      = roi_warn_th
    worker_state["wl_gauge_roi"]     = st.session_state.get("wl_gauge_roi", [])

    start_worker()

    # ── Draw mode — keep thread alive but stop the render loop ────────────────
    # The render loop's continuous reruns destroy the interactive polygon canvas.
    # When draw_mode is active we skip the loop entirely; the background thread
    # keeps capturing frames so the camera never actually stops.
    if st.session_state.get("draw_mode", False):
        if not st.session_state.monitoring:
            stop_worker()   # release camera if monitoring was stopped while in draw mode
        st.stop()

    # ── Lightweight Streamlit render loop ─────────────────────────────────────
    # This loop ONLY reads pre-computed results and renders them.
    # All heavy work (detection, tracking, prediction) happens in the thread.
    RENDER_INTERVAL = 0.04   # ~25 UI refreshes per second
    METRICS_EVERY   = 5      # Update metrics panel every N render ticks
    render_tick     = 0

    while st.session_state.monitoring and worker_state["running"]:
        render_tick += 1

        # Snapshot shared state atomically
        with worker_state["lock"]:
            frame_rgb       = worker_state["frame_rgb"]
            fps             = worker_state["fps"]
            frame_count     = worker_state["frame_count"]
            blockage_pct    = worker_state["blockage_pct"]
            roi_counts      = worker_state["roi_counts"]
            total_roi       = worker_state["total_roi"]
            flood_result    = worker_state["flood_result"]
            alert_list      = worker_state["alert_list"]
            hist_blockage   = list(worker_state["history_blockage"])
            hist_risk       = list(worker_state["history_risk"])
            wl_level_cm     = worker_state.get("wl_level_cm")
            wl_rise_rate    = worker_state.get("wl_rise_rate", 0.0)
            wl_trend        = worker_state.get("wl_trend", "Stable")
            wl_risk_status  = worker_state.get("wl_risk_status", "Normal")
            wl_history      = list(worker_state.get("wl_history", []))

        # Update session state for idle display after stop
        st.session_state.frame_count      = frame_count
        st.session_state.blockage_pct     = blockage_pct
        st.session_state.roi_counts       = roi_counts
        st.session_state.flood_result     = flood_result
        st.session_state.alert_list       = alert_list
        st.session_state.history_blockage = hist_blockage
        st.session_state.history_risk     = hist_risk
        st.session_state["wl_level_cm"]   = wl_level_cm
        st.session_state["wl_rise_rate"]  = wl_rise_rate
        st.session_state["wl_trend"]      = wl_trend
        st.session_state["wl_risk_status"]= wl_risk_status
        st.session_state["wl_history"]    = wl_history

        # ── Render frame ──────────────────────────────────────────────────────
        if frame_rgb is not None:
            frame_placeholder.image(frame_rgb, width="stretch")

        fps_placeholder.markdown(
            f'<div style="font-size:11px;color:#4a6b8a;text-align:right;">'
            f'⚡ {fps:.1f} FPS · Frame #{frame_count:,}</div>',
            unsafe_allow_html=True,
        )

        # ── Metric panels (throttled — no need to redraw every render tick) ──
        if render_tick % METRICS_EVERY == 1:
            with metric_ph_blockage:
                render_blockage_bar(blockage_pct)
            with metric_ph_rain:
                if st.session_state.use_live_weather:
                    try:
                        _wx = weather_svc.get_current()
                        _live_rain_category = _wx.get("condition_label",
                                                       rain_intensity_to_category(st.session_state.rain_intensity))
                    except Exception:
                        _live_rain_category = rain_intensity_to_category(st.session_state.rain_intensity)
                else:
                    _live_rain_category = rain_intensity_to_category(st.session_state.rain_intensity)
                render_rain_panel(
                    st.session_state.rain_enabled,
                    st.session_state.rain_intensity,
                    rain_category=_live_rain_category,
                    use_live_weather=st.session_state.use_live_weather,
                )
            with metric_ph_risk:
                risk_lbl   = flood_result["risk"].split()[0]
                risk_color = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}.get(risk_lbl, "white")
                render_metric_card(
                    "Flood Risk Score",
                    f"{flood_result['risk_score']*100:.0f}",
                    "/100", risk_color,
                )
            with roi_count_ph:
                render_roi_counts(roi_counts)
            with risk_ph:
                render_risk_panel(
                    flood_result["risk"],
                    flood_result["confidence"],
                    flood_result["probabilities"],
                )
            with alert_ph:
                render_alerts(alert_list)

        # ── History charts (heavy; update infrequently) ────────────────────
        if render_tick % 25 == 1 and len(hist_blockage) > 2:
            import pandas as pd
            with blockage_chart_ph:
                st.line_chart(
                    pd.DataFrame({"Blockage %": hist_blockage}),
                    use_container_width=True, height=160, color="#00d4ff",
                )
            with risk_chart_ph:
                st.line_chart(
                    pd.DataFrame({"Risk Score": hist_risk}),
                    use_container_width=True, height=160, color="#ff6b35",
                )

        # ── DB log table (update every ~10 s) ─────────────────────────────
        if render_tick % 250 == 1:
            with log_ph:
                recent = get_recent_logs(100)
                if recent:
                    import pandas as pd
                    df_log = pd.DataFrame(recent)
                    cols = ["timestamp", "location", "total_roi_objects", "blockage_percentage",
                            "rain_intensity_value", "rain_intensity",
                            "temperature", "feels_like",
                            "humidity", "wind_speed",
                            "water_level_cm", "water_level_trend",
                            "water_level_status", "water_rise_rate",
                            "flood_risk", "alert_triggered"]
                    df_log = df_log[[c for c in cols if c in df_log.columns]]
                    df_log = df_log.rename(columns={
                        "rain_intensity_value": "rain intensity (value)",
                        "rain_intensity":       "rain intensity (category)",
                        "water_level_cm":       "water level (cm)",
                        "water_level_trend":    "water trend",
                        "water_level_status":   "water status",
                        "water_rise_rate":      "rise rate (cm/min)",
                    })
                    st.dataframe(df_log, use_container_width=True, height=400)

        time.sleep(RENDER_INTERVAL)

    # User clicked STOP or thread died
    stop_worker()

else:
    # ── IDLE STATE ─────────────────────────────────────────────────────────────
    idle_frame = generate_demo_frame(0)
    # Only draw polygon overlay if one is actually set
    if roi.has_polygon():
        idle_frame = roi.draw_on_frame(idle_frame, [], [])
    idle_rgb = cv2.cvtColor(idle_frame, cv2.COLOR_BGR2RGB)
    if not st.session_state.get("draw_mode", False):
        frame_placeholder.image(idle_rgb, width="stretch")
    fps_placeholder.markdown(
        '<div style="font-size:11px;color:#4a6b8a;text-align:right;">— Monitoring paused —</div>',
        unsafe_allow_html=True
    )

    with metric_ph_blockage:
        render_blockage_bar(st.session_state.blockage_pct)
    with metric_ph_rain:
        if st.session_state.use_live_weather:
            try:
                _wx = weather_svc.get_current()
                _live_rain_category = _wx.get("condition_label",
                                               rain_intensity_to_category(st.session_state.rain_intensity))
            except Exception:
                _live_rain_category = rain_intensity_to_category(st.session_state.rain_intensity)
        else:
            _live_rain_category = rain_intensity_to_category(st.session_state.rain_intensity)
        render_rain_panel(
            st.session_state.rain_enabled,
            st.session_state.rain_intensity,
            rain_category=_live_rain_category,
            use_live_weather=st.session_state.use_live_weather,
        )
    with metric_ph_risk:
        # Show weather-driven risk score even when idle — same .metric-card box as START state
        _wr = risk_engine.get_weather_risk()
        _wsc = _wr["score_result"]
        _wcat = _wr["rainfall_category"]
        render_metric_card(
            "Weather Flood Risk",
            f'{_wsc["score"]:.1f}',
            f' / {_wsc["category"]}',
            _wsc["color"],
        )
    with roi_count_ph:
        render_roi_counts(st.session_state.roi_counts)
    with risk_ph:
        # Idle: weather-driven risk — same render_risk_panel box as START state
        _wr2 = risk_engine.get_weather_risk()
        _wsc2 = _wr2["score_result"]
        # Map score category → risk label used by render_risk_panel
        _score_to_risk = {"Low": "Low Risk", "Moderate": "Medium Risk",
                          "High": "High Risk", "Severe": "High Risk"}
        _idle_risk_label = _score_to_risk.get(_wsc2["category"], "Low Risk")
        # Synthetic probabilities derived from score for the progress bars
        _s_norm = min(_wsc2["score"] / 40.0, 1.0)
        _idle_probs = {
            "Low Risk":    round(max(0.0, 1.0 - _s_norm * 1.5), 4),
            "Medium Risk": round(min(0.5, _s_norm), 4),
            "High Risk":   round(max(0.0, _s_norm - 0.3), 4),
        }
        _prob_sum = sum(_idle_probs.values()) or 1.0
        _idle_probs = {k: round(v / _prob_sum, 4) for k, v in _idle_probs.items()}
        _idle_conf = max(_idle_probs.values())
        render_risk_panel(_idle_risk_label, _idle_conf, _idle_probs)
    with alert_ph:
        render_alerts(st.session_state.alert_list)

    # Stats summary
    with log_ph:
        stats = get_stats_summary()
        if stats and stats.get("total_records", 0) > 0:
            import pandas as pd
            recent = get_recent_logs(100)
            if recent:
                df_log = pd.DataFrame(recent)
                cols = ["timestamp", "location", "total_roi_objects", "blockage_percentage",
                        "rain_intensity_value", "rain_intensity",
                        "temperature", "feels_like",
                        "humidity", "wind_speed",
                        "water_level_cm", "water_level_trend",
                        "water_level_status", "water_rise_rate",
                        "flood_risk", "alert_triggered"]
                df_log = df_log[[c for c in cols if c in df_log.columns]]
                df_log = df_log.rename(columns={
                    "rain_intensity_value": "rain intensity (value)",
                    "rain_intensity":       "rain intensity (category)",
                    "water_level_cm":       "water level (cm)",
                    "water_level_trend":    "water trend",
                    "water_level_status":   "water status",
                    "water_rise_rate":      "rise rate (cm/min)",
                })
                st.dataframe(df_log, use_container_width=True, height=400)
        else:
            st.info("No monitoring data yet. Click ▶ START to begin.")

    # Idle hint
    st.markdown("""
<div style="text-align:center;padding:30px 0;color:var(--text-muted);">
    <div style="font-size:40px;margin-bottom:12px;">🌊</div>
    <div style="font-size:16px;font-weight:600;color:var(--text-secondary);margin-bottom:6px;">
        System Ready
    </div>
    <div style="font-size:13px;">
        Select a camera source and click <strong>▶ START</strong> to begin monitoring
    </div>
</div>
""", unsafe_allow_html=True)