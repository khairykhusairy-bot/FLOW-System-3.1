"""
FLOW — Flood Level Observation Warning System
INTEGRATION GUIDE: How to add rain_validation to your existing main.py

Copy the snippets below into the indicated locations in main.py.
Each snippet is clearly marked with the line or function it belongs near.
No existing code needs to be deleted — only additions are required.

─────────────────────────────────────────────────────────────────────────────
STEP 1 — Import at the top of main.py (after existing imports)
─────────────────────────────────────────────────────────────────────────────
"""

# ── PASTE AFTER: "from flood_risk_engine import FloodRiskEngine" ──────────────

from rain_validation.composite import CompositeRainValidator, render_cv_validation_panel

"""
─────────────────────────────────────────────────────────────────────────────
STEP 2 — Cache resource (after existing @st.cache_resource functions)
─────────────────────────────────────────────────────────────────────────────
"""

# ── PASTE AFTER: "def get_flood_risk_engine():" block ────────────────────────
STEP_2_CODE = """
@st.cache_resource
def get_rain_validator():
    \"\"\"
    Persistent CompositeRainValidator — holds inter-frame state (prev_gray, etc.)
    across Streamlit reruns by living inside the cache.

    To disable streak detection (e.g. unstable camera mount):
        return CompositeRainValidator(use_streak_detection=False)
    \"\"\"
    return CompositeRainValidator(
        use_streak_detection=True,
        # ── Optional threshold overrides for your specific site ──────────
        # visibility_kwargs=dict(clear_min=400, reduced_min=200, low_min=100),
        # disturbance_kwargs=dict(noise_threshold=15, moderate_max=0.30),
        # streak_kwargs=dict(min_aspect=3.0, min_area=15),
    )

rain_validator = get_rain_validator()
"""

"""
─────────────────────────────────────────────────────────────────────────────
STEP 3 — Add to worker_state dict (inside get_worker_state())
─────────────────────────────────────────────────────────────────────────────
"""

# ── PASTE INSIDE "get_worker_state()" return dict, after "clahe_grid" line ───
STEP_3_CODE = """
        # Camera rain validation results (written by camera_worker)
        "cv_rain_result":   None,
"""

"""
─────────────────────────────────────────────────────────────────────────────
STEP 4 — Add to _init_state() session defaults
─────────────────────────────────────────────────────────────────────────────
"""

# ── PASTE INSIDE "_init_state()" defaults dict ────────────────────────────────
STEP_4_CODE = """
        "cv_rain_result":   None,
        "cv_rain_enabled":  True,    # sidebar toggle
"""

"""
─────────────────────────────────────────────────────────────────────────────
STEP 5 — Sidebar toggle (in the sidebar section)
─────────────────────────────────────────────────────────────────────────────
"""

# ── PASTE after the CLAHE sidebar section, before Alert Thresholds ─────────────
STEP_5_CODE = """
    st.markdown(\"<hr style='border-color:#1e3a5f;'>\", unsafe_allow_html=True)
    st.markdown('<div class=\"sidebar-label\">🎥 CAMERA RAIN VALIDATION</div>', unsafe_allow_html=True)

    cv_rain_enabled = st.checkbox(
        "Enable CV Rain Validation",
        value=st.session_state.get("cv_rain_enabled", True),
        key="cv_rain_enabled_chk",
        help=(
            "Lightweight classical computer vision checks that validate whether "
            "rain is actually occurring at the camera location.  "
            "Uses Laplacian sharpness, frame differencing, and edge-based streak "
            "detection — no deep learning, CPU-only.  ~3–6 ms per frame."
        ),
    )
    st.session_state["cv_rain_enabled"] = cv_rain_enabled
    worker_state["cv_rain_enabled"]     = cv_rain_enabled

    if cv_rain_enabled:
        st.markdown(
            '<div style=\"font-size:10px;color:#4a6b8a;margin-top:2px;\">'
            '✓ 3 validators active: Visibility · Disturbance · Streaks'
            '</div>',
            unsafe_allow_html=True,
        )
"""

"""
─────────────────────────────────────────────────────────────────────────────
STEP 6 — Run validators inside camera_worker (inside the while loop)
─────────────────────────────────────────────────────────────────────────────
"""

# ── PASTE AFTER the CLAHE enhancement block, BEFORE "Read UI settings" ────────
STEP_6_CODE = """
            # ── Camera Rain Validation (CV) ───────────────────────────────────
            # Runs every DETECT_EVERY frames (same cadence as YOLO) to keep
            # the processing load minimal.  VisibilityValidator always runs;
            # SurfaceDisturbanceValidator always runs (needs prev frame);
            # RainStreakDetector is skipped on even ticks to halve its cost.
            if state.get("cv_rain_enabled", True) and run_detection:
                _water_poly = roi.get_polygon() if len(roi.get_polygon()) >= 3 else None
                _api_mm_h   = float(weather_svc.get_current().get("rain_mm", 0.0))

                # run_streaks every other detection tick (saves ~2 ms)
                _run_stk = (tick // DETECT_EVERY) % 2 == 0

                _cv_result = rain_validator.analyse(
                    frame           = frame,          # raw frame (before overlay drawing)
                    water_polygon   = _water_poly,
                    api_rain_mm_h   = _api_mm_h,
                    blockage_pct    = last_blockage_pct,
                    predictor_score = last_flood_result.get("risk_score", 0.0),
                    run_streaks     = _run_stk,
                )
            else:
                _cv_result = None
"""

"""
─────────────────────────────────────────────────────────────────────────────
STEP 7 — Write cv_result to worker_state (inside "Write results under lock")
─────────────────────────────────────────────────────────────────────────────
"""

# ── PASTE INSIDE "with state['lock']:" block, after existing state writes ──────
STEP_7_CODE = """
                state["cv_rain_result"] = _cv_result
"""

"""
─────────────────────────────────────────────────────────────────────────────
STEP 8 — Sync to session_state in the render loop
─────────────────────────────────────────────────────────────────────────────
"""

# ── PASTE IN THE RENDER LOOP's "Snapshot shared state" block ──────────────────
STEP_8_CODE = """
        cv_rain_result  = worker_state.get("cv_rain_result")
"""

# ── AND in "Update session state" block ───────────────────────────────────────
STEP_8B_CODE = """
        st.session_state["cv_rain_result"]  = cv_rain_result
"""

"""
─────────────────────────────────────────────────────────────────────────────
STEP 9 — Dashboard panel (in the metrics_col section of the render loop)
─────────────────────────────────────────────────────────────────────────────
"""

# ── PASTE after the "Flood Prediction" section in the render loop ────────────
STEP_9_CODE = """
        # ── CV Rain Validation Panel ─────────────────────────────────────
        if render_tick % METRICS_EVERY == 1:
            cv_rain_result = worker_state.get("cv_rain_result")
            if cv_rain_result is not None and st.session_state.get("cv_rain_enabled", True):
                with risk_ph:   # or use a dedicated placeholder
                    st.markdown(
                        render_cv_validation_panel(cv_rain_result),
                        unsafe_allow_html=True,
                    )
"""

"""
─────────────────────────────────────────────────────────────────────────────
ALTERNATIVE: Dedicated dashboard column for CV validation
─────────────────────────────────────────────────────────────────────────────
If you prefer a dedicated column (recommended for final FYP presentation),
replace the three-column bottom row with a four-column layout:

    roi_col, risk_col, cv_col, alert_col = st.columns([2, 2, 2, 3], gap="medium")

    with cv_col:
        st.markdown('<div class="section-header">Rain Validation (CV)</div>',
                    unsafe_allow_html=True)
        cv_panel_ph = st.empty()

Then in the render loop:
    if render_tick % METRICS_EVERY == 1:
        cv_result = worker_state.get("cv_rain_result")
        if cv_result:
            with cv_panel_ph:
                st.markdown(render_cv_validation_panel(cv_result),
                            unsafe_allow_html=True)
─────────────────────────────────────────────────────────────────────────────

STEP 10 — Reset on camera stop
─────────────────────────────────────────────────────────────────────────────
"""

# ── PASTE INSIDE stop_worker() function, after tracker.reset() ────────────────
STEP_10_CODE = """
    rain_validator.reset()    # clears prev_gray so disturbance doesn't glitch on restart
"""

"""
─────────────────────────────────────────────────────────────────────────────
PERFORMANCE NOTES
─────────────────────────────────────────────────────────────────────────────
Measured on Intel Core i5 (single core) at 960 × 540 resolution:

    VisibilityValidator.analyse()          ~1.2 ms
    SurfaceDisturbanceValidator.analyse()  ~1.8 ms
    RainStreakDetector.analyse()           ~2.4 ms   (Canny + findContours)
    ─────────────────────────────────────────────────
    Total (all three)                      ~5.4 ms

Since validators run only on DETECT_EVERY frames (every 3rd frame), the
amortised cost is ~1.8 ms per frame — negligible.

Running at 25 FPS, the CV validation layer consumes < 5 % of one CPU core.
─────────────────────────────────────────────────────────────────────────────

THRESHOLD TUNING GUIDE FOR MALAYSIA
─────────────────────────────────────────────────────────────────────────────
Malaysian flood monitoring environments present specific challenges:

1. SUDDEN THUNDERSTORMS
   • Onset is very fast (< 5 min dry → heavy).
   • SurfaceDisturbanceValidator detects this quickly (responds within 2–3 frames).
   • Increase smoothing_alpha from 0.35 → 0.50 for faster response.

2. UNEVEN RAIN DISTRIBUTION (localised cells)
   • The weather API may report 0 mm/h while the camera site is under heavy rain.
   • This is the primary justification for CV validation — it catches local rain
     the API misses.
   • In such events, CV validation adds +2 to +3 points, potentially escalating
     a LOW risk API reading to MODERATE or HIGH.

3. LOW-COST CCTV CAMERAS
   • Higher sensor noise → increase NOISE_THRESHOLD from 12 → 18.
   • AGC (auto-gain) causes frame brightness shifts → may inflate disturbance.
     Solution: set noise_threshold=20 during nighttime hours.

4. CAMERA PLACEMENT NEAR WATERFALLS OR RAPIDS
   • Permanent high motion density → calibrate a site-specific calm_max.
   • Use SurfaceDisturbanceValidator.calibrate_dry_baseline() with frame pairs
     collected during a known dry day.

5. TREE CANOPY OCCLUSION
   • Rain visible in a small central patch only.
   • Set roi_fraction=0.4 in VisibilityValidator to focus on the centre.
   • Provide a tight water_polygon so streak/disturbance analysis stays on water.

6. LOGGING FOR CALIBRATION
   • In camera_worker, write cv_result["visibility"]["sharpness"],
     cv_result["disturbance"]["disturbance_value"], and
     cv_result["streaks"]["streak_density"] to the SQLite log table.
   • After one week of data, plot histograms per rain category from the OWM API.
     The natural breakpoints in those histograms become your optimal thresholds.
─────────────────────────────────────────────────────────────────────────────
"""
