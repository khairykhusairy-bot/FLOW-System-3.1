"""
patch_webrtc.py  —  run this ONCE from the Flow System V3.0 folder:

    python patch_webrtc.py

It injects the streamlit-webrtc branch into main.py and is safe to
re-run (it skips if the patch is already applied).
"""
import re, shutil, pathlib, sys

HERE = pathlib.Path(__file__).parent
MAIN = HERE / "main.py"

src = MAIN.read_text(encoding="utf-8")

# ── Guard: don't patch twice ─────────────────────────────────────────────────
if "WEBRTC_AVAILABLE" in src:
    print("main.py already patched — nothing to do.")
    sys.exit(0)

# ── Backup ────────────────────────────────────────────────────────────────────
shutil.copy(MAIN, HERE / "main.py.bak")
print("Backup saved → main.py.bak")

# ═════════════════════════════════════════════════════════════════════════════
# PATCH 1 — add import block after  "from water_level import WaterLevelMonitor"
# ═════════════════════════════════════════════════════════════════════════════
IMPORT_ANCHOR = "from water_level import WaterLevelMonitor"
IMPORT_ADDITION = """
# ── WebRTC smooth camera integration ─────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    from webrtc_processor import FLOWVideoProcessor
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
"""
src = src.replace(IMPORT_ANCHOR,
                  IMPORT_ANCHOR + IMPORT_ADDITION,
                  1)

# ═════════════════════════════════════════════════════════════════════════════
# PATCH 2 — add WebRTC toggle checkbox after the cam_source selectbox line
# ═════════════════════════════════════════════════════════════════════════════
SELECTBOX_ANCHOR = 'st.session_state.cam_source = cam_options[cam_choice]'
TOGGLE_ADDITION = """

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
        )"""
src = src.replace(SELECTBOX_ANCHOR,
                  SELECTBOX_ANCHOR + TOGGLE_ADDITION,
                  1)

# ═════════════════════════════════════════════════════════════════════════════
# PATCH 3 — inject WebRTC render branch before the legacy while loop
#
# We find the comment/header that precedes the while loop and replace
# everything from there through the while statement with:
#   if use_webrtc: ...  (WebRTC metrics panels)
#   else:
#       while ...:       (original loop, indented one level deeper)
# ═════════════════════════════════════════════════════════════════════════════

# The block we want to replace (comment + blank + while header)
OLD_WHILE_BLOCK = (
    "    # ── Lightweight Streamlit render loop"
    " ─────────────────────────────────────────\n"
    "    # This loop ONLY reads pre-computed results and renders them.\n"
    "    # All heavy work (detection, tracking, prediction) happens in the thread.\n"
    "    RENDER_INTERVAL = 0.04   # ~25 UI refreshes per second\n"
    "    METRICS_EVERY   = 5      # Update metrics panel every N render ticks\n"
    "    render_tick     = 0\n"
    "\n"
    "    while st.session_state.monitoring and worker_state[\"running\"]:"
)

# Fallback if comment text slightly differs:
if OLD_WHILE_BLOCK not in src:
    OLD_WHILE_BLOCK = (
        "    while st.session_state.monitoring and worker_state[\"running\"]:"
    )
    if OLD_WHILE_BLOCK not in src:
        print("ERROR: could not locate the while render loop in main.py. "
              "Please apply PATCH 3 manually — see instructions below.")
        MAIN.write_text(src, encoding="utf-8")
        sys.exit(1)

WEBRTC_BRANCH = """    # ══════════════════════════════════════════════════════════════════════════
    # WebRTC path — smooth 30 FPS stream in its own async/WebRTC thread
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.get("use_webrtc", False) and WEBRTC_AVAILABLE:
        stop_worker()   # no need for the OpenCV thread when WebRTC handles capture

        RTC_CONFIG = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        def _make_processor():
            return FLOWVideoProcessor(
                worker_state=worker_state,
                detector=detector,
                roi=roi,
                tracker=tracker,
                predictor=predictor,
                alert_mgr=alert_mgr,
                risk_engine=risk_engine,
                wl_monitor=wl_monitor,
                telegram=telegram,
                weather_svc=weather_svc,
                log_interval=LOG_INTERVAL,
            )

        webrtc_streamer(
            key="flow-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=_make_processor,
            media_stream_constraints={
                "video": {
                    "width":  {"ideal": 960},
                    "height": {"ideal": 540},
                    "frameRate": {"ideal": 30, "max": 30},
                },
                "audio": False,
            },
            async_processing=True,
        )

        # ── Pull latest results from the processor's shared state ─────────────
        with worker_state["lock"]:
            blockage_pct   = worker_state["blockage_pct"]
            roi_counts     = worker_state["roi_counts"]
            flood_result   = worker_state["flood_result"]
            alert_list     = worker_state["alert_list"]
            hist_blockage  = list(worker_state["history_blockage"])
            hist_risk      = list(worker_state["history_risk"])
            fps            = worker_state["fps"]
            frame_count    = worker_state["frame_count"]
            wl_level_cm    = worker_state.get("wl_level_cm")
            wl_trend       = worker_state.get("wl_trend", "Stable")
            wl_risk_status = worker_state.get("wl_risk_status", "Normal")
            wl_rise_rate   = worker_state.get("wl_rise_rate", 0.0)
            wl_history     = list(worker_state.get("wl_history", []))

        # sync session state so STOP → idle panels show last values
        st.session_state.frame_count      = frame_count
        st.session_state.blockage_pct     = blockage_pct
        st.session_state.roi_counts       = roi_counts
        st.session_state.flood_result     = flood_result
        st.session_state.alert_list       = alert_list
        st.session_state.history_blockage = hist_blockage
        st.session_state.history_risk     = hist_risk
        st.session_state["wl_level_cm"]   = wl_level_cm
        st.session_state["wl_trend"]      = wl_trend
        st.session_state["wl_risk_status"]= wl_risk_status
        st.session_state["wl_rise_rate"]  = wl_rise_rate
        st.session_state["wl_history"]    = wl_history

        fps_placeholder.markdown(
            f'<div style="font-size:11px;color:#4a6b8a;text-align:right;">'
            f'⚡ {fps:.1f} FPS (WebRTC) · Frame #{frame_count:,}</div>',
            unsafe_allow_html=True,
        )

        with metric_ph_blockage:
            render_blockage_bar(blockage_pct)
        with metric_ph_rain:
            if st.session_state.use_live_weather:
                try:
                    _wx = weather_svc.get_current()
                    _lrc = _wx.get("condition_label",
                                   rain_intensity_to_category(st.session_state.rain_intensity))
                except Exception:
                    _lrc = rain_intensity_to_category(st.session_state.rain_intensity)
            else:
                _lrc = rain_intensity_to_category(st.session_state.rain_intensity)
            render_rain_panel(
                st.session_state.rain_enabled,
                st.session_state.rain_intensity,
                rain_category=_lrc,
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

        if len(hist_blockage) > 2:
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

        with log_ph:
            recent = get_recent_logs(100)
            if recent:
                import pandas as pd
                df_log = pd.DataFrame(recent)
                cols_log = ["timestamp", "location", "total_roi_objects", "blockage_percentage",
                            "rain_intensity_value", "rain_intensity",
                            "temperature", "feels_like", "humidity", "wind_speed",
                            "water_level_cm", "water_level_trend",
                            "water_level_status", "water_rise_rate",
                            "flood_risk", "alert_triggered"]
                df_log = df_log[[c for c in cols_log if c in df_log.columns]]
                df_log = df_log.rename(columns={
                    "rain_intensity_value": "rain intensity (value)",
                    "rain_intensity":       "rain intensity (category)",
                    "water_level_cm":       "water level (cm)",
                    "water_level_trend":    "water trend",
                    "water_level_status":   "water status",
                    "water_rise_rate":      "rise rate (cm/min)",
                })
                st.dataframe(df_log, use_container_width=True, height=400)

    # ══════════════════════════════════════════════════════════════════════════
    # Legacy OpenCV thread path — original behaviour, fully unchanged
    # ══════════════════════════════════════════════════════════════════════════
    else:
        # ── Lightweight Streamlit render loop ─────────────────────────────────
        # This loop ONLY reads pre-computed results and renders them.
        # All heavy work (detection, tracking, prediction) happens in the thread.
        RENDER_INTERVAL = 0.04   # ~25 UI refreshes per second
        METRICS_EVERY   = 5      # Update metrics panel every N render ticks
        render_tick     = 0

        while st.session_state.monitoring and worker_state["running"]:"""

src = src.replace(OLD_WHILE_BLOCK, WEBRTC_BRANCH, 1)

# ── PATCH 3b — indent the body of the while loop by 4 extra spaces ───────────
# Everything from "render_tick += 1" until "    # User clicked STOP or thread died"
# needs one extra level of indentation to sit inside the new `else:` block.

lines = src.splitlines(keepends=True)
out_lines = []
in_old_while_body = False
stop_marker = "    # User clicked STOP or thread died"

for line in lines:
    stripped = line.rstrip('\n\r')

    if not in_old_while_body:
        # trigger: the line immediately after the while header we just renamed
        if stripped == "        while st.session_state.monitoring and worker_state[\"running\"]:":
            in_old_while_body = True
            out_lines.append(line)
            continue
        out_lines.append(line)
    else:
        # stop trigger: this line is at the old 4-space level (outside the while)
        if stripped == "    # User clicked STOP or thread died":
            # Add 4 more spaces (it now belongs inside else:)
            out_lines.append("    " + line)
            continue
        if stripped == "    stop_worker()" and in_old_while_body:
            # The stop_worker() call that follows the while loop
            out_lines.append("    " + line)
            in_old_while_body = False
            continue
        # still inside — add 4 spaces
        out_lines.append("    " + line)

src = "".join(out_lines)

MAIN.write_text(src, encoding="utf-8")
print("✅  main.py patched successfully!")
print()
print("Next steps:")
print("  1. pip install streamlit-webrtc   (if not already done)")
print("  2. streamlit run main.py")
print("  3. Tick '📡 Smooth stream (WebRTC)' in the sidebar, then click ▶ START")
print("  4. Allow camera access in your browser when prompted")
