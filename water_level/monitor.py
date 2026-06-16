"""
FLOW — Water Level Module
monitor.py: WaterLevelMonitor — the single facade class used by main.py.

Usage in camera_worker (main.py):
--------------------------------------
    from water_level import WaterLevelMonitor

    wl_monitor = WaterLevelMonitor()
    wl_monitor.calibration.load()   # load saved calibration (if any)

    # Inside the frame loop:
    wl_result = wl_monitor.process(frame)
    display_frame = wl_monitor.draw(display_frame, wl_result)

    # Pass wl_result["level_cm"] to the alert / prediction systems:
    flood_result = predictor.predict(
        roi_count=total_roi,
        blockage_pct=blockage_pct,
        rain_intensity=rain_intensity,
        water_level=wl_result["level_norm"],   # already normalised 0-1
    )
--------------------------------------

The monitor is intentionally stateful so the smoother and trend tracker
accumulate data across frames without any external wiring.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from water_level.calibration import WaterLevelCalibration
from water_level.detector import WaterlineDetector
from water_level.smoothing import WaterLevelSmoother
from water_level.trend_analysis import WaterLevelTrend, STATUS_NORMAL
from water_level.visualization import draw_water_level_overlay
from water_level import config as _cfg


class WaterLevelMonitor:
    """
    Facade that orchestrates the full water-level estimation pipeline:

        Frame → Detect waterline → Calibrate (px→cm) → Smooth → Trend → Alert

    Parameters
    ----------
    gauge_roi_polygon  : (x,y) list defining the flood-gauge area in frame pixels.
    river_roi_polygon  : (x,y) list defining the river surface area (optional,
                          used for display only).
    enabled            : Set False to bypass processing (pass-through mode).
    """

    def __init__(
        self,
        gauge_roi_polygon: Optional[List[Tuple[int, int]]] = None,
        river_roi_polygon: Optional[List[Tuple[int, int]]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled

        # Sub-modules
        self.calibration = WaterLevelCalibration()
        self.detector    = WaterlineDetector(gauge_roi_polygon=gauge_roi_polygon)
        self.smoother    = WaterLevelSmoother()
        self.trend       = WaterLevelTrend()

        # ROI polygons (stored for visualization)
        self.gauge_roi: Optional[List[Tuple[int, int]]] = gauge_roi_polygon
        self.river_roi: Optional[List[Tuple[int, int]]] = river_roi_polygon

        # Live threshold values (cm) — updated from sidebar via main.py
        self.thresholds: Dict = {
            "normal":   _cfg.THRESHOLD_NORMAL,
            "warning":  _cfg.THRESHOLD_WARNING,
            "danger":   _cfg.THRESHOLD_DANGER,
            "critical": _cfg.THRESHOLD_CRITICAL,
        }

        # Alert state (simple cooldown to avoid log floods)
        self._alert_cooldown: Dict[str, float] = {}
        self._alert_cooldown_secs: float = 20.0

        # Last processed result (for access between frames)
        self._last_result: Dict = self._empty_result()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> Dict:
        """
        Run the full detection pipeline on one frame.

        Parameters
        ----------
        frame : BGR image from camera_worker.

        Returns
        -------
        dict with keys:
            waterline_y_px   : int | None  — detected pixel row
            level_cm         : float | None — smoothed real-world height in cm
            level_norm       : float        — level_cm normalised 0→1 (for predictor)
            rise_rate        : float        — cm/min (positive = rising)
            trend            : str
            risk_status      : str
            is_escalating    : bool
            is_critical_surge: bool
            history_cm       : list[float]
            alerts           : list[str]   — new alert messages (if any)
            detection_rate   : float
        """
        if not self.enabled:
            return self._last_result

        # ── Detect ────────────────────────────────────────────────────────────
        y_px: Optional[int] = self.detector.detect(frame)

        # ── Pixel → cm ────────────────────────────────────────────────────────
        raw_cm: Optional[float] = None
        if y_px is not None and self.calibration.is_calibrated:
            raw_cm = self.calibration.pixel_to_cm(y_px)

        # ── Smooth ────────────────────────────────────────────────────────────
        smoothed_cm: Optional[float] = None
        if raw_cm is not None:
            smoothed_cm = self.smoother.update(raw_cm)
        else:
            smoothed_cm = self.smoother.value   # keep last good estimate

        # ── Trend ─────────────────────────────────────────────────────────────
        trend_data: Dict = {}
        if smoothed_cm is not None:
            trend_data = self.trend.update(smoothed_cm)
        else:
            trend_data = self.trend.snapshot()

        # Override risk_status with live thresholds (trend_analysis uses config constants)
        if smoothed_cm is not None:
            t_n = self.thresholds.get("normal",   _cfg.THRESHOLD_NORMAL)
            t_w = self.thresholds.get("warning",  _cfg.THRESHOLD_WARNING)
            t_d = self.thresholds.get("danger",   _cfg.THRESHOLD_DANGER)
            t_c = self.thresholds.get("critical", _cfg.THRESHOLD_CRITICAL)
            from water_level.trend_analysis import (
                STATUS_NORMAL, STATUS_WARNING, STATUS_DANGER, STATUS_CRITICAL)
            if smoothed_cm >= t_c:
                live_risk = STATUS_CRITICAL
            elif smoothed_cm >= t_d:
                live_risk = STATUS_DANGER
            elif smoothed_cm >= t_w:
                live_risk = STATUS_WARNING
            else:
                live_risk = STATUS_NORMAL
            trend_data = dict(trend_data)
            trend_data["risk_status"] = live_risk

        # ── Normalise for flood predictor ─────────────────────────────────────
        level_norm = 0.5
        if smoothed_cm is not None:
            h_min = self.calibration.h_min_cm
            h_max = self.calibration.h_max_cm
            span  = max(h_max - h_min, 1.0)
            level_norm = float(max(0.0, min(1.0, (smoothed_cm - h_min) / span)))

        # ── Alerts ────────────────────────────────────────────────────────────
        alerts = self._evaluate_alerts(smoothed_cm, trend_data)

        result: Dict = {
            "waterline_y_px":   y_px,
            "level_cm":         smoothed_cm,
            "level_norm":       level_norm,
            "rise_rate":        trend_data.get("rise_rate_cm_per_min", 0.0),
            "trend":            trend_data.get("trend", "Stable"),
            "risk_status":      trend_data.get("risk_status", STATUS_NORMAL),
            "is_escalating":    trend_data.get("is_escalating", False),
            "is_critical_surge":trend_data.get("is_critical_surge", False),
            "history_cm":       trend_data.get("history_cm", []),
            "alerts":           alerts,
            "detection_rate":   round(self.detector.detection_rate, 3),
            "smoother_info":    self.smoother.diagnostics(),
        }
        self._last_result = result
        return result

    def draw(self, frame: np.ndarray, result: Optional[Dict] = None) -> np.ndarray:
        """
        Overlay water-level HUD elements onto ``frame``.

        If ``result`` is None the last processed result is used.
        """
        r = result or self._last_result
        return draw_water_level_overlay(
            frame,
            waterline_y_px = r.get("waterline_y_px"),
            level_cm       = r.get("level_cm"),
            rise_rate      = r.get("rise_rate", 0.0),
            trend          = r.get("trend", "Stable"),
            risk_status    = r.get("risk_status", STATUS_NORMAL),
            gauge_roi      = self.gauge_roi,
            river_roi      = self.river_roi,
            calibration    = self.calibration.to_dict(),
            show_gauge_bar = True,
            thresholds     = self.thresholds,
        )

    def set_gauge_roi(self, polygon: Optional[List[Tuple[int, int]]]):
        """
        Update gauge ROI polygon at runtime and re-anchor the calibration
        Y bounds to the polygon's vertical extent.

        After this call:
          • calibration.y_min_px = topmost Y of the polygon  (→ h_max_cm)
          • calibration.y_max_px = bottommost Y of the polygon (→ h_min_cm)

        This means 0 cm maps to the bottom of the drawn ROI and the
        maximum height maps to the top, spanning the full ROI end-to-end.
        """
        self.gauge_roi = polygon
        self.detector.set_gauge_roi(polygon)

        if polygon and len(polygon) >= 2:
            ys = [pt[1] for pt in polygon]
            roi_y_min = min(ys)   # top of polygon    → high water (h_max_cm)
            roi_y_max = max(ys)   # bottom of polygon → low water  (h_min_cm)
            if roi_y_min < roi_y_max:
                # Preserve the real-world height range; only re-anchor the pixels.
                self.calibration.y_min_px = roi_y_min
                self.calibration.y_max_px = roi_y_max

    def set_river_roi(self, polygon: Optional[List[Tuple[int, int]]]):
        """Update river surface ROI polygon (display only)."""
        self.river_roi = polygon

    def reset(self):
        """Reset smoother and trend buffers (e.g. after camera switch)."""
        self.smoother.reset()
        self.trend.reset()
        self.detector._last_y_px = None
        self._last_result = self._empty_result()

    # ── Alert Logic ────────────────────────────────────────────────────────────

    def _evaluate_alerts(
        self,
        level_cm: Optional[float],
        trend_data: Dict,
    ) -> List[str]:
        """Generate alert messages for the current frame."""
        alerts: List[str] = []
        now = time.time()

        def _throttled(key: str) -> bool:
            last = self._alert_cooldown.get(key, 0.0)
            if now - last >= self._alert_cooldown_secs:
                self._alert_cooldown[key] = now
                return False   # not throttled → fire alert
            return True        # still on cooldown → suppress

        if level_cm is None:
            return alerts

        t_normal   = self.thresholds.get("normal",   _cfg.THRESHOLD_NORMAL)
        t_warning  = self.thresholds.get("warning",  _cfg.THRESHOLD_WARNING)
        t_danger   = self.thresholds.get("danger",   _cfg.THRESHOLD_DANGER)
        t_critical = self.thresholds.get("critical", _cfg.THRESHOLD_CRITICAL)

        # Threshold alerts
        if level_cm >= t_critical:
            if not _throttled("critical"):
                alerts.append(
                    f"🚨 CRITICAL: Water level {level_cm:.1f} cm — "
                    f"exceeds critical threshold ({t_critical:.0f} cm)"
                )
        elif level_cm >= t_danger:
            if not _throttled("danger"):
                alerts.append(
                    f"⚠ DANGER: Water level {level_cm:.1f} cm "
                    f"(threshold: {t_danger:.0f} cm)"
                )
        elif level_cm >= t_warning:
            if not _throttled("warning"):
                alerts.append(
                    f"⚠ WARNING: Water level {level_cm:.1f} cm "
                    f"(threshold: {t_warning:.0f} cm)"
                )

        # Rise-rate alerts
        rise = trend_data.get("rise_rate_cm_per_min", 0.0)
        if rise >= _cfg.RISE_RATE_CRITICAL:
            if not _throttled("surge"):
                alerts.append(
                    f"🚨 CRITICAL SURGE: Rising at {rise:.2f} cm/min — immediate action required"
                )
        elif rise >= _cfg.RISE_RATE_WARNING:
            if not _throttled("rapid_rise"):
                alerts.append(
                    f"⚠ Rapid rise detected: {rise:.2f} cm/min"
                )

        return alerts

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "waterline_y_px":    None,
            "level_cm":          None,
            "level_norm":        0.5,
            "rise_rate":         0.0,
            "trend":             "Stable",
            "risk_status":       STATUS_NORMAL,
            "is_escalating":     False,
            "is_critical_surge": False,
            "history_cm":        [],
            "alerts":            [],
            "detection_rate":    0.0,
            "smoother_info":     {},
        }

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def last_result(self) -> Dict:
        return self._last_result

    @property
    def level_cm(self) -> Optional[float]:
        return self._last_result.get("level_cm")

    @property
    def risk_status(self) -> str:
        return self._last_result.get("risk_status", STATUS_NORMAL)

    @property
    def is_calibrated(self) -> bool:
        return self.calibration.is_calibrated
