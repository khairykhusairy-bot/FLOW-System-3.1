"""
FLOW — WebRTC Video Processor
webrtc_processor.py

Wraps the heavy per-frame pipeline (YOLOv8 detection, ROI masking,
tracking, rain overlay, HUD) inside a streamlit-webrtc
VideoProcessorBase so the camera runs in its own thread at full
frame-rate while the Streamlit UI refreshes independently.

The processor WRITES results into `worker_state` (the same shared
dict already used by the old camera_worker thread) so the rest of
main.py — metrics panels, DB logging, Telegram alerts — needs zero
changes.
"""

import time
import threading
import cv2
import numpy as np
from av import VideoFrame
from streamlit_webrtc import VideoProcessorBase

from utils import get_timestamp
from database import log_monitoring_data
from weather import rain_intensity_to_category


class FLOWVideoProcessor(VideoProcessorBase):
    """
    streamlit-webrtc callback class.

    Instantiated once per WebRTC session.  `recv()` is called for
    every incoming camera frame in its own asyncio thread.
    """

    def __init__(
        self,
        worker_state: dict,
        detector,
        roi,
        tracker,
        predictor,
        alert_mgr,
        risk_engine,
        wl_monitor,
        telegram,
        weather_svc,
        log_interval: int = 15,
    ):
        self._state       = worker_state
        self._detector    = detector
        self._roi         = roi
        self._tracker     = tracker
        self._predictor   = predictor
        self._alert_mgr   = alert_mgr
        self._risk_engine = risk_engine
        self._wl_monitor  = wl_monitor
        self._telegram    = telegram
        self._weather_svc = weather_svc
        self._log_interval = log_interval

        # FPS tracking
        self._fps_count = 0
        self._t_fps     = time.time()
        self._fps       = 0.0

        # DB log throttle
        self._last_log  = 0.0

        # Frame counter
        self._tick = 0

        # Rain animation state (per-processor)
        self._rain_drops = None

    # ------------------------------------------------------------------
    def recv(self, frame: VideoFrame) -> VideoFrame:
        """Called for every frame from the webcam."""
        img_bgr = frame.to_ndarray(format="bgr24")
        out_bgr = self._process(img_bgr)
        return VideoFrame.from_ndarray(out_bgr, format="bgr24")

    # ------------------------------------------------------------------
    def _process(self, frame: np.ndarray) -> np.ndarray:
        state = self._state
        self._tick += 1
        tick = self._tick

        # ── Read live settings from shared state ──────────────────────
        with state["lock"]:
            rain_enabled    = state.get("rain_enabled", False)
            rain_intensity  = state.get("rain_intensity", 0.0)
            use_live_weather= state.get("use_live_weather", True)
            show_labels     = state.get("show_labels", True)
            show_trails     = state.get("show_trails", True)
            blockage_warn   = state.get("blockage_warn_th", 50)
            roi_warn        = state.get("roi_warn_th", 10)
            wl_enabled      = state.get("wl_enabled", True)
            wl_gauge_roi    = state.get("wl_gauge_roi", [])
            tg_enabled      = state.get("tg_enabled", False)
            wl_thresh = {
                "normal":   state.get("wl_thresh_normal",   50),
                "warning":  state.get("wl_thresh_warning",  100),
                "danger":   state.get("wl_thresh_danger",   150),
                "critical": state.get("wl_thresh_critical", 180),
            }

        # Resize for speed
        display_frame = cv2.resize(frame, (960, 540))
        h, w = display_frame.shape[:2]

        # ── Detection ─────────────────────────────────────────────────
        conf = self._detector.confidence
        detections = self._detector.detect(display_frame)

        # ── ROI masking & blockage ────────────────────────────────────
        if self._roi.has_polygon():
            in_roi, blockage_pct, roi_counts = self._roi.filter_detections(
                detections, display_frame
            )
        else:
            in_roi      = detections
            blockage_pct = 0.0
            roi_counts  = {}
        total_roi = sum(roi_counts.values())

        # ── Tracking ──────────────────────────────────────────────────
        boxes      = [d["bbox"] for d in in_roi]
        track_ids, trails = self._tracker.update(boxes)

        # ── Draw detections ───────────────────────────────────────────
        display_frame = self._roi.draw_on_frame(
            display_frame, in_roi, track_ids,
            show_labels=show_labels,
            show_trails=show_trails,
            trails=trails,
        )

        # ── Weather / rain intensity ───────────────────────────────────
        if use_live_weather:
            try:
                wx = self._weather_svc.get_current()
                rain_intensity = wx.get("rain_intensity", 0.0)
            except Exception:
                pass

        # ── Flood risk prediction ─────────────────────────────────────
        wl_result = None
        wl_norm   = 0.0
        if wl_enabled and self._wl_monitor.enabled:
            if wl_gauge_roi:
                self._wl_monitor.set_gauge_roi(wl_gauge_roi)
            self._wl_monitor.thresholds = wl_thresh
            wl_result = self._wl_monitor.update(display_frame)
            if wl_result:
                wl_norm = wl_result.get("normalized", 0.0)
                display_frame = self._wl_monitor.draw_overlay(
                    display_frame, wl_result
                )

        wx_risk = self._risk_engine.get_weather_risk()
        flood_result = self._predictor.predict_fused(
            rain_intensity=rain_intensity,
            blockage_pct=blockage_pct / 100.0,
            water_level=wl_norm,
            weather_risk=wx_risk,
        )

        # ── Alerts ────────────────────────────────────────────────────
        new_alerts = self._alert_mgr.evaluate(
            blockage_pct=blockage_pct,
            roi_count=total_roi,
            flood_result=flood_result,
            blockage_warn_th=blockage_warn,
            roi_warn_th=roi_warn,
            wl_result=wl_result,
        )
        for a in new_alerts:
            from database import log_alert
            log_alert(a)

        # ── Telegram notifications ─────────────────────────────────────
        if tg_enabled:
            try:
                wx = self._weather_svc.get_current()
            except Exception:
                wx = {}
            self._telegram.evaluate(
                flood_result=flood_result,
                blockage_pct=blockage_pct,
                rain_intensity=rain_intensity,
                roi_count=total_roi,
                wl_result=wl_result,
                weather=wx,
                enabled=tg_enabled,
            )

        # ── Rain overlay ──────────────────────────────────────────────
        if rain_enabled and rain_intensity > 0.0:
            n_drops = int(rain_intensity * 400)
            if self._rain_drops is None or len(self._rain_drops) != n_drops:
                rng = np.random.default_rng(42)
                self._rain_drops = rng.integers(
                    low=[0, 0], high=[w, h],
                    size=(n_drops, 2), dtype=np.int32,
                )
            # animate vertically
            speed = int(4 + rain_intensity * 12)
            self._rain_drops[:, 1] = (self._rain_drops[:, 1] + speed) % h
            xs = self._rain_drops[:, 0]
            ys = self._rain_drops[:, 1]
            xe = (xs - 2).clip(0, w - 1)
            ye = (ys + 13).clip(0, h - 1)
            rain_layer = np.zeros((h, w), dtype=np.uint8)
            for t in (0.0, 0.33, 0.66, 1.0):
                px = (xs + t * (xe - xs)).astype(np.int32).clip(0, w - 1)
                py = (ys + t * (ye - ys)).astype(np.int32).clip(0, h - 1)
                rain_layer[py, px] = 255
            kernel = np.ones((2, 1), np.uint8)
            rain_layer = cv2.dilate(rain_layer, kernel, iterations=1)
            rain_color = np.zeros_like(display_frame)
            rain_color[:, :, 0] = rain_layer
            rain_color[:, :, 1] = rain_layer
            rain_color[:, :, 2] = rain_layer
            alpha = 0.35 + rain_intensity * 0.30
            cv2.addWeighted(rain_color, alpha, display_frame, 1.0, 0, display_frame)

        # ── HUD ───────────────────────────────────────────────────────
        cv2.putText(
            display_frame,
            f"Blockage: {blockage_pct:.1f}%  |  ROI: {total_roi}  |  Risk: {flood_result['risk']}",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA,
        )
        if self._alert_mgr.has_critical():
            cv2.rectangle(display_frame, (0, 0), (w, 36), (0, 0, 200), -1)
            cv2.putText(
                display_frame,
                "⚠ CRITICAL ALERT — CHECK ALERT CENTER",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # ── FPS ───────────────────────────────────────────────────────
        self._fps_count += 1
        elapsed = time.time() - self._t_fps
        if elapsed >= 1.0:
            self._fps = self._fps_count / elapsed
            self._fps_count = 0
            self._t_fps = time.time()

        # ── DB logging ────────────────────────────────────────────────
        now = time.time()
        if now - self._last_log >= self._log_interval:
            use_live = use_live_weather
            try:
                if use_live:
                    wx2 = self._weather_svc.get_current()
                    log_humidity    = wx2.get("humidity", None)
                    log_wind        = wx2.get("wind_speed", None)
                    log_temperature = wx2.get("temperature", None)
                    log_feels       = wx2.get("feels_like", None)
                    rain_category   = wx2.get("condition_label",
                                              rain_intensity_to_category(rain_intensity))
                else:
                    log_humidity = log_wind = log_temperature = log_feels = None
                    rain_category = rain_intensity_to_category(rain_intensity)
            except Exception:
                log_humidity = log_wind = log_temperature = log_feels = None
                rain_category = rain_intensity_to_category(rain_intensity)

            log_monitoring_data({
                "timestamp":            get_timestamp(),
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
                "water_level_cm":       wl_result.get("level_cm") if wl_result else None,
                "water_level_trend":    wl_result.get("trend") if wl_result else None,
                "water_level_status":   wl_result.get("risk_status") if wl_result else None,
                "water_rise_rate":      wl_result.get("rise_rate", 0.0) if wl_result else None,
                "location":             self._weather_svc.location_name,
            })
            self._last_log = now

        # ── Write results to shared state ─────────────────────────────
        history_max = 60
        with state["lock"]:
            state["frame_rgb"]        = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            state["fps"]              = round(self._fps, 1)
            state["frame_count"]      = tick
            state["blockage_pct"]     = blockage_pct
            state["roi_counts"]       = roi_counts
            state["total_roi"]        = total_roi
            state["flood_result"]     = flood_result
            state["alert_list"]       = self._alert_mgr.get_active_alerts()
            state["total_detections"] = state.get("total_detections", 0) + len(in_roi)
            state["history_blockage"] = (
                state["history_blockage"] + [blockage_pct]
            )[-history_max:]
            state["history_risk"] = (
                state["history_risk"] + [flood_result["risk_score"] * 100]
            )[-history_max:]
            if wl_result:
                state["wl_level_cm"]    = wl_result.get("level_cm")
                state["wl_rise_rate"]   = wl_result.get("rise_rate", 0.0)
                state["wl_trend"]       = wl_result.get("trend", "Stable")
                state["wl_risk_status"] = wl_result.get("risk_status", "Normal")
                state["wl_history"]     = wl_result.get("history_cm", [])

        return display_frame
