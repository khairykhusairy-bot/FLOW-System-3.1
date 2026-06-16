"""
FLOW — Water Level Module
visualization.py: Overlays real-time water level information onto video frames.

Draws a premium HUD that matches FLOW's existing dark dashboard aesthetic:
  • Detected waterline (horizontal line across the frame)
  • Current water level in cm
  • Risk status badge (colour-coded)
  • Rise speed (cm/min)
  • Trend label
  • Gauge ROI and river ROI outlines
  • Timestamp
  • Calibration gauge bar + cm ruler on the left edge

All drawing uses cv2 primitives — no external fonts required.
"""

import cv2
import math
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple, Dict

from water_level.config import (
    COLOR_NORMAL,
    COLOR_WARNING,
    COLOR_DANGER,
    COLOR_CRITICAL,
    COLOR_WATERLINE,
    COLOR_GAUGE_ROI,
    COLOR_RIVER_ROI,
    THRESHOLD_NORMAL,
    THRESHOLD_WARNING,
    THRESHOLD_DANGER,
    THRESHOLD_CRITICAL,
)
from water_level.trend_analysis import (
    STATUS_NORMAL, STATUS_WARNING, STATUS_DANGER, STATUS_CRITICAL,
)

# ─── Helpers ───────────────────────────────────────────────────────────────────

def _status_color(status: str) -> Tuple[int, int, int]:
    return {
        STATUS_NORMAL:   COLOR_NORMAL,
        STATUS_WARNING:  COLOR_WARNING,
        STATUS_DANGER:   COLOR_DANGER,
        STATUS_CRITICAL: COLOR_CRITICAL,
    }.get(status, (200, 200, 200))


def _draw_text_bg(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.50,
    thickness: int = 1,
    text_color: Tuple = (255, 255, 255),
    bg_color: Tuple = (0, 0, 0),
    padding: int = 5,
    alpha: float = 0.65,
):
    """Draw text with a semi-transparent background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    x1, y1 = x - padding, y - th - padding
    x2, y2 = x + tw + padding, y + bl + padding

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


# ─── Main Drawing Function ─────────────────────────────────────────────────────

def draw_water_level_overlay(
    frame: np.ndarray,
    *,
    waterline_y_px: Optional[int],
    level_cm:       Optional[float],
    rise_rate:      float,
    trend:          str,
    risk_status:    str,
    gauge_roi:      Optional[List[Tuple[int, int]]] = None,
    river_roi:      Optional[List[Tuple[int, int]]] = None,
    calibration:    Optional[Dict] = None,
    show_gauge_bar: bool = True,
    timestamp:      Optional[str] = None,
    thresholds:     Optional[Dict] = None,
) -> np.ndarray:
    """
    Render all water-level HUD elements onto ``frame`` (in-place).

    Parameters
    ----------
    frame           : BGR image from camera_worker.
    waterline_y_px  : Detected waterline Y (full-frame coords).
    level_cm        : Smoothed water level in cm.
    rise_rate       : Rise rate in cm/min (from trend_analysis).
    trend           : Trend label string (from trend_analysis).
    risk_status     : Risk status string ('Normal'|'Warning'|'Danger'|'Critical').
    gauge_roi       : Polygon points of the gauge ROI (drawn in cyan).
    river_roi       : Polygon points of the river surface ROI (drawn in mint).
    calibration     : Dict with calibration data for drawing the gauge bar.
    show_gauge_bar  : Whether to draw the vertical gauge bar on the left.
    timestamp       : Override timestamp string; defaults to current time.

    Returns
    -------
    np.ndarray : The annotated frame (same object, modified in-place).
    """
    h, w = frame.shape[:2]
    status_color = _status_color(risk_status)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── ROI Polygons ──────────────────────────────────────────────────────────
    if gauge_roi and len(gauge_roi) >= 3:
        pts = np.array(gauge_roi, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], COLOR_GAUGE_ROI)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.polylines(frame, [pts], True, COLOR_GAUGE_ROI, 1, cv2.LINE_AA)

        # Centroid label
        cx = int(np.mean([p[0] for p in gauge_roi]))
        cy = int(np.mean([p[1] for p in gauge_roi]))
        cv2.putText(frame, "GAUGE ROI", (cx - 30, cy),
                    font, 0.38, COLOR_GAUGE_ROI, 1, cv2.LINE_AA)

        # ROI Y extent for bracket and ruler
        roi_ys    = [p[1] for p in gauge_roi]
        roi_y_min = min(roi_ys)
        roi_y_max = max(roi_ys)
        roi_x_max = max(p[0] for p in gauge_roi)

        # Vertical bracket on the right edge of the ROI
        bracket_x = min(roi_x_max + 6, w - 60)
        cv2.line(frame, (bracket_x, roi_y_min), (bracket_x, roi_y_max),
                 COLOR_GAUGE_ROI, 1, cv2.LINE_AA)
        cv2.line(frame, (bracket_x - 3, roi_y_min), (bracket_x + 3, roi_y_min),
                 COLOR_GAUGE_ROI, 1, cv2.LINE_AA)
        cv2.line(frame, (bracket_x - 3, roi_y_max), (bracket_x + 3, roi_y_max),
                 COLOR_GAUGE_ROI, 1, cv2.LINE_AA)

        # cm ruler alongside the bracket (replaces the old px span label)
        if calibration:
            _draw_roi_ruler(frame, bracket_x, roi_y_min, roi_y_max,
                            calibration, waterline_y_px, level_cm, status_color)

    if river_roi and len(river_roi) >= 3:
        pts = np.array(river_roi, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], COLOR_RIVER_ROI)
        cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)
        cv2.polylines(frame, [pts], True, COLOR_RIVER_ROI, 1, cv2.LINE_AA)

    # ── Waterline ─────────────────────────────────────────────────────────────
    if waterline_y_px is not None and 0 < waterline_y_px < h:
        # Dashed horizontal line
        dash_len, gap_len = 20, 8
        x = 0
        while x < w:
            x1 = min(x, w - 1)
            x2 = min(x + dash_len, w - 1)
            cv2.line(frame, (x1, waterline_y_px), (x2, waterline_y_px),
                     COLOR_WATERLINE, 2, cv2.LINE_AA)
            x += dash_len + gap_len

        # Arrow at left edge pointing to waterline (starts past the ruler label zone)
        cv2.arrowedLine(
            frame,
            (75, waterline_y_px),
            (8, waterline_y_px),
            COLOR_WATERLINE, 2, cv2.LINE_AA, tipLength=0.25,
        )

    # ── Vertical Gauge Bar ────────────────────────────────────────────────────
    if show_gauge_bar and calibration:
        _draw_gauge_bar(frame, level_cm, calibration, w, h, status_color, thresholds)

    # ── Water Level HUD Panel (top-right) ─────────────────────────────────────
    _draw_level_panel(frame, level_cm, rise_rate, trend, risk_status,
                      status_color, w, h)

    # ── Risk Status Badge (top-left, below FLOW header) ───────────────────────
    if risk_status != STATUS_NORMAL:
        badge_text = f"⚠ WATER LEVEL {risk_status.upper()}"
        _draw_text_bg(
            frame, badge_text,
            (10, 56), font_scale=0.50, thickness=1,
            text_color=(255, 255, 255), bg_color=status_color,
            padding=6, alpha=0.85,
        )

    # ── Timestamp ─────────────────────────────────────────────────────────────
    ts = timestamp or datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 80, h - 10), font, 0.38, (120, 160, 200), 1, cv2.LINE_AA)

    return frame


# ─── Sub-drawers ───────────────────────────────────────────────────────────────

def _draw_level_panel(
    frame: np.ndarray,
    level_cm: Optional[float],
    rise_rate: float,
    trend: str,
    risk_status: str,
    status_color: Tuple[int, int, int],
    w: int, h: int,
):
    """Draw the compact water level info panel in the top-right corner."""
    panel_x = w - 190
    panel_y = 10
    panel_w = 180
    panel_h = 90

    # Semi-transparent panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (10, 20, 35), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), status_color, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    tx = panel_x + 8
    ty = panel_y + 20

    # Title
    cv2.putText(frame, "WATER LEVEL", (tx, ty), font, 0.38, (100, 160, 220), 1, cv2.LINE_AA)
    ty += 22

    # Level value
    if level_cm is not None:
        lv_text = f"{level_cm:.1f} cm"
    else:
        lv_text = "--.- cm"
    cv2.putText(frame, lv_text, (tx, ty), font, 0.70, status_color, 2, cv2.LINE_AA)
    ty += 20

    # Rise rate
    rate_sign = "+" if rise_rate > 0 else ""
    rate_text = f"{rate_sign}{rise_rate:.2f} cm/min"
    rate_color = COLOR_CRITICAL if rise_rate >= 5.0 else (
        COLOR_DANGER if rise_rate >= 2.0 else (
        COLOR_WARNING if rise_rate > 0.3 else (150, 255, 150)))
    cv2.putText(frame, rate_text, (tx, ty), font, 0.38, rate_color, 1, cv2.LINE_AA)
    ty += 16

    # Trend
    cv2.putText(frame, trend, (tx, ty), font, 0.38, (200, 220, 255), 1, cv2.LINE_AA)


def _pick_ruler_steps(h_range: float, bar_h: int) -> Tuple[float, float]:
    """Return (major_step_cm, minor_step_cm) so major ticks are >= 20 px apart."""
    px_per_cm = bar_h / h_range if h_range > 0 else 1.0
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500):
        if step * px_per_cm >= 20:
            major_step = float(step)
            break
    else:
        major_step = float(h_range)
    minor_step = max(1.0, major_step / 5.0)
    # Skip minor ticks if they would be too dense (< 4 px apart)
    if minor_step * px_per_cm < 4.0:
        minor_step = major_step
    return major_step, minor_step


def _draw_roi_ruler(
    frame: np.ndarray,
    ruler_x: int,
    roi_y_min: int,
    roi_y_max: int,
    calibration: Dict,
    waterline_y_px: Optional[int],
    level_cm: Optional[float],
    status_color: Tuple[int, int, int],
) -> None:
    """Vertical cm ruler drawn on the right bracket of the Gauge ROI.

    The ruler is a zoomed window anchored at 0 cm and ending just above the
    current water level, so it fills the full ROI height with a compact,
    readable scale instead of the full calibrated range.

    Window:
        cm_lower = 0 cm  (always)
        cm_upper = next multiple of 5 above (level + 10 cm), minimum 20 cm

    Example: level = 16 cm  →  window 0–30 cm  (ticks at 0, 5, 10 … 30)

    The waterline indicator is placed at its proportional position within the
    zoomed window.  When the actual dashed line is outside the ROI bounds a
    directional arrow is added at the nearest boundary.
    """
    h_max_cm = float(calibration.get("h_max_cm", 200.0))

    roi_h = roi_y_max - roi_y_min
    if roi_h <= 0:
        return

    font    = cv2.FONT_HERSHEY_SIMPLEX
    label_x = ruler_x + 6

    # ── Dynamic zoom window ───────────────────────────────────────────────────
    if level_cm is not None:
        cm_lower = 0.0
        cm_upper = math.ceil((level_cm + 10.0) / 5.0) * 5.0   # next ×5 above level+10
        cm_upper = max(cm_upper, 20.0)                          # minimum 20 cm window
        cm_upper = min(cm_upper, h_max_cm)                      # cap at calibrated max
    else:
        # No level yet — fall back to full calibrated range so ruler is visible
        cm_lower = float(calibration.get("h_min_cm", 0.0))
        cm_upper = h_max_cm

    cm_span = cm_upper - cm_lower
    if cm_span <= 0:
        return

    def cm_to_ruler_y(cm_val: float) -> int:
        """Linear map: cm_lower → roi_y_max (bottom), cm_upper → roi_y_min (top)."""
        ratio = (cm_val - cm_lower) / cm_span
        return int(roi_y_max - ratio * roi_h)

    # ── Tick intervals based on the zoomed window ─────────────────────────────
    major_step, minor_step = _pick_ruler_steps(cm_span, roi_h)

    # ── Boundary labels at bracket end caps ───────────────────────────────────
    # Top cap: cm_upper value
    _draw_text_bg(
        frame, f"{int(cm_upper)}", (label_x, roi_y_min + 10),
        font_scale=0.28, thickness=1,
        text_color=COLOR_GAUGE_ROI, bg_color=(0, 0, 0),
        padding=2, alpha=0.55,
    )
    # Bottom cap: cm_lower value  (show "0" only; avoids cluttering near-zero tick)
    _draw_text_bg(
        frame, f"{int(cm_lower)}", (label_x, roi_y_max - 3),
        font_scale=0.28, thickness=1,
        text_color=COLOR_GAUGE_ROI, bg_color=(0, 0, 0),
        padding=2, alpha=0.55,
    )
    # "cm" unit above the top cap
    cv2.putText(frame, "cm", (label_x, max(roi_y_min - 3, 8)),
                font, 0.27, (130, 155, 185), 1, cv2.LINE_AA)

    # ── Minor ticks (no label) ────────────────────────────────────────────────
    cm_val = math.ceil(cm_lower / minor_step) * minor_step
    while cm_val <= cm_upper + 1e-6:
        is_major = (cm_val % major_step) < 1e-4 or (major_step - cm_val % major_step) < 1e-4
        if not is_major:
            ty = cm_to_ruler_y(cm_val)
            if roi_y_min <= ty <= roi_y_max:
                cv2.line(frame, (ruler_x, ty), (ruler_x + 3, ty), (90, 110, 130), 1)
        cm_val = round(cm_val + minor_step, 6)

    # ── Major ticks with numeric labels ───────────────────────────────────────
    cm_val = math.ceil(cm_lower / major_step) * major_step
    while cm_val <= cm_upper + 1e-6:
        ty = cm_to_ruler_y(cm_val)
        if roi_y_min <= ty <= roi_y_max:
            cv2.line(frame, (ruler_x - 4, ty), (ruler_x + 5, ty),
                     (155, 180, 210), 1, cv2.LINE_AA)
            _draw_text_bg(
                frame, str(int(round(cm_val))), (label_x, ty + 4),
                font_scale=0.30, thickness=1,
                text_color=(185, 210, 245), bg_color=(5, 12, 25),
                padding=2, alpha=0.60,
            )
        cm_val = round(cm_val + major_step, 6)

    # ── Waterline indicator ───────────────────────────────────────────────────
    if level_cm is not None:
        wl_label  = f"{level_cm:.1f}"
        # Position within the zoomed ruler (level always within [cm_lower, cm_upper])
        wl_ruler_y = cm_to_ruler_y(level_cm)
        wl_ruler_y = max(roi_y_min, min(roi_y_max, wl_ruler_y))

        # Wide coloured tick at the waterline cm position
        cv2.line(frame, (ruler_x - 7, wl_ruler_y), (ruler_x + 9, wl_ruler_y),
                 COLOR_WATERLINE, 2, cv2.LINE_AA)
        # Bold label — shift upward if too close to the bottom cap
        lbl_y = wl_ruler_y + 5 if wl_ruler_y < roi_y_max - 14 else wl_ruler_y - 7
        _draw_text_bg(
            frame, wl_label, (label_x, lbl_y),
            font_scale=0.38, thickness=1,
            text_color=COLOR_WATERLINE, bg_color=(10, 15, 30),
            padding=3, alpha=0.82,
        )

        # If the actual dashed line is outside the ROI, add a directional arrow
        if waterline_y_px is not None:
            if waterline_y_px < roi_y_min:
                cv2.arrowedLine(frame,
                                (ruler_x, roi_y_min + 14), (ruler_x, roi_y_min),
                                COLOR_WATERLINE, 2, cv2.LINE_AA, tipLength=0.45)
            elif waterline_y_px > roi_y_max:
                cv2.arrowedLine(frame,
                                (ruler_x, roi_y_max - 14), (ruler_x, roi_y_max),
                                COLOR_WATERLINE, 2, cv2.LINE_AA, tipLength=0.45)


def _draw_gauge_bar(
    frame: np.ndarray,
    level_cm: Optional[float],
    calibration: Dict,
    w: int, h: int,
    status_color: Tuple[int, int, int],
    thresholds: Optional[Dict] = None,
):
    """
    Draw a vertical flood gauge bar on the left edge with tick marks for
    Normal / Warning / Danger / Critical thresholds.

    ``thresholds`` should be a dict with keys:
        normal, warning, danger, critical  (all in cm)
    Falls back to the config constants when not provided.
    """
    bar_x     = 8
    bar_top   = 80
    bar_bot   = h - 40
    bar_h     = bar_bot - bar_top
    bar_w     = 14

    h_min = calibration.get("h_min_cm", 0.0)
    h_max = calibration.get("h_max_cm", 200.0)
    h_range = h_max - h_min
    if h_range <= 0:
        return

    # Resolve threshold values — prefer live sidebar values over config constants
    t_normal   = float(thresholds["normal"])   if thresholds and "normal"   in thresholds else THRESHOLD_NORMAL
    t_warning  = float(thresholds["warning"])  if thresholds and "warning"  in thresholds else THRESHOLD_WARNING
    t_danger   = float(thresholds["danger"])   if thresholds and "danger"   in thresholds else THRESHOLD_DANGER
    t_critical = float(thresholds["critical"]) if thresholds and "critical" in thresholds else THRESHOLD_CRITICAL

    def cm_to_bar_y(cm_val):
        """Map cm height → bar Y (bar_top = h_max, bar_bot = h_min)."""
        ratio = 1.0 - (cm_val - h_min) / h_range
        return int(bar_top + ratio * bar_h)

    # Bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (10, 20, 35), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (60, 80, 100), 1)

    # Colour bands between thresholds (drawn bottom-up)
    bands = [
        (h_min,     t_normal,   COLOR_NORMAL),
        (t_normal,  t_warning,  COLOR_WARNING),
        (t_warning, t_danger,   COLOR_DANGER),
        (t_danger,  t_critical, COLOR_CRITICAL),
        (t_critical, h_max,     COLOR_CRITICAL),
    ]
    for band_lo, band_hi, band_color in bands:
        clamped_lo = max(h_min, min(h_max, band_lo))
        clamped_hi = max(h_min, min(h_max, band_hi))
        if clamped_hi <= clamped_lo:
            continue
        y_top = cm_to_bar_y(clamped_hi)
        y_bot = cm_to_bar_y(clamped_lo)
        # Draw as a semi-transparent tint on the bar background
        band_overlay = frame.copy()
        cv2.rectangle(band_overlay, (bar_x + 1, y_top), (bar_x + bar_w - 1, y_bot),
                      band_color, -1)
        cv2.addWeighted(band_overlay, 0.25, frame, 0.75, 0, frame)

    # Fill up to current level (solid)
    if level_cm is not None:
        fill_y = cm_to_bar_y(level_cm)
        fill_y = max(bar_top, min(bar_bot, fill_y))
        cv2.rectangle(frame, (bar_x + 1, fill_y), (bar_x + bar_w - 1, bar_bot - 1),
                      status_color, -1)

    # ── Ruler: minor + major tick marks with cm labels ────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    major_step, minor_step = _pick_ruler_steps(h_range, bar_h)
    label_x = bar_x + bar_w + 6   # cm label left edge

    # "cm" unit header just above the topmost major tick
    first_major = math.ceil(h_min / major_step) * major_step
    last_major  = math.floor(h_max / major_step) * major_step
    top_label_y = cm_to_bar_y(max(last_major, h_min))
    if bar_top - 2 <= top_label_y <= bar_bot + 2:
        cv2.putText(frame, "cm", (label_x, max(bar_top - 1, top_label_y - 6)),
                    font, 0.28, (140, 165, 195), 1, cv2.LINE_AA)

    # Minor ticks (no label)
    cm_val = math.ceil(h_min / minor_step) * minor_step
    while cm_val <= h_max + 1e-6:
        is_major = (cm_val % major_step) < 1e-4 or (major_step - cm_val % major_step) < 1e-4
        if not is_major:
            ty = cm_to_bar_y(cm_val)
            if bar_top <= ty <= bar_bot:
                cv2.line(frame, (bar_x + bar_w, ty), (bar_x + bar_w + 3, ty),
                         (75, 95, 115), 1)
        cm_val = round(cm_val + minor_step, 6)

    # Major ticks with numeric cm labels
    cm_val = first_major
    while cm_val <= h_max + 1e-6:
        ty = cm_to_bar_y(cm_val)
        if bar_top - 1 <= ty <= bar_bot + 1:
            # Tick line spanning the full bar width
            cv2.line(frame, (bar_x - 4, ty), (bar_x + bar_w + 5, ty),
                     (155, 180, 210), 1, cv2.LINE_AA)
            label = str(int(round(cm_val)))
            _draw_text_bg(
                frame, label, (label_x, ty + 4),
                font_scale=0.30, thickness=1,
                text_color=(185, 210, 245), bg_color=(5, 12, 25),
                padding=2, alpha=0.60,
            )
        cm_val = round(cm_val + major_step, 6)

    # Threshold tick marks (coloured, drawn after ruler so they sit on top)
    # Letters moved right of the ruler labels to avoid overlap
    thresh_label_x = label_x + 34
    tick_marks = [
        (t_normal,   COLOR_NORMAL,   "N"),
        (t_warning,  COLOR_WARNING,  "W"),
        (t_danger,   COLOR_DANGER,   "D"),
        (t_critical, COLOR_CRITICAL, "C"),
    ]
    for th_cm, th_color, lbl in tick_marks:
        if h_min <= th_cm <= h_max:
            ty = cm_to_bar_y(th_cm)
            cv2.line(frame, (bar_x - 5, ty), (bar_x + bar_w + 5, ty),
                     th_color, 1, cv2.LINE_AA)
            cv2.putText(frame, lbl, (thresh_label_x, ty + 4),
                        font, 0.30, th_color, 1, cv2.LINE_AA)

    # Current level indicator circle
    if level_cm is not None:
        cur_y = cm_to_bar_y(level_cm)
        cur_y = max(bar_top, min(bar_bot, cur_y))
        cv2.circle(frame, (bar_x + bar_w // 2, cur_y), 5, status_color, -1)
        cv2.circle(frame, (bar_x + bar_w // 2, cur_y), 5, (255, 255, 255), 1)


def draw_calibration_overlay(
    frame: np.ndarray,
    y_min_px: int,
    y_max_px: int,
    h_min_cm: float,
    h_max_cm: float,
) -> np.ndarray:
    """
    Draw calibration reference lines on a frame during calibration setup.
    Used by the Streamlit calibration UI only.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Min level line (bottom of gauge range)
    cv2.line(frame, (0, y_max_px), (w, y_max_px), COLOR_NORMAL, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{h_min_cm:.0f} cm (min)",
                (8, y_max_px - 6), font, 0.45, COLOR_NORMAL, 1, cv2.LINE_AA)

    # Max level line (top of gauge range)
    cv2.line(frame, (0, y_min_px), (w, y_min_px), COLOR_CRITICAL, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{h_max_cm:.0f} cm (max)",
                (8, y_min_px - 6), font, 0.45, COLOR_CRITICAL, 1, cv2.LINE_AA)

    return frame
