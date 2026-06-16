"""
FLOW — Flood Level Observation Warning System
Rain Validation Module: Composite Rain Validator

PURPOSE
───────
This module is the **integration point** for all three lightweight camera-based
rain validation features:

    1. VisibilityValidator     — Laplacian sharpness (visibility degradation)
    2. SurfaceDisturbanceValidator — frame differencing (water surface motion)
    3. RainStreakDetector      — Canny + contour filter (rain streaks)

DESIGN PRINCIPLES (matching requirements)
──────────────────────────────────────────
• The weather API remains the PRIMARY rainfall source.
• Camera CV is SECONDARY validation only.
• No deep learning; no CNN; CPU-only; real-time capable.
• Integration is purely additive: CV signals add 0–3 points to a base
  weather-API score.  They can never decrease the score.
• The final risk label (LOW / MODERATE / HIGH / CRITICAL) uses the same
  thresholds as the rest of FLOW for consistency.

INTEGRATED FLOOD RISK SCORING FORMULA
──────────────────────────────────────
The composite score mirrors the logic requested in the specification:

    risk_score = 0

    # Primary: weather API rainfall (0–3 pts)
    if api_rain_mm_h > 5  : risk_score += 1
    if api_rain_mm_h > 15 : risk_score += 1   (cumulative, not separate)
    if api_rain_mm_h > 30 : risk_score += 1

    # Existing system: water level + debris (0–5 pts, from FloodPredictor)
    if water_level_rising  : risk_score += 3  (mapped from predictor score)
    if debris_detected     : risk_score += 2  (mapped from blockage %)

    # Camera validation layer (0–3 pts, from this module)
    if visibility_low      : risk_score += 1
    if disturbance_high    : risk_score += 1
    if streaks_detected    : risk_score += 1

    # Labels:
    0–3   → LOW
    4–6   → MODERATE
    7–9   → HIGH
    10–11 → CRITICAL

INTEGRATION WITH main.py / camera_worker
──────────────────────────────────────────
1. Instantiate CompositeRainValidator once (cache with @st.cache_resource).
2. In camera_worker, call:
       cv_result = rain_validator.analyse(frame, water_polygon, api_rain_mm_h,
                                          blockage_pct, predictor_risk_score)
3. Pass cv_result["final_risk"] to the dashboard.
4. The individual sub-results are available for the validation indicators panel.

See bottom of this file for the Streamlit dashboard render helper.
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from rain_validation.visibility         import VisibilityValidator
from rain_validation.surface_disturbance import SurfaceDisturbanceValidator
from rain_validation.rain_streaks        import RainStreakDetector


# ─── Risk Label Constants (shared with existing FLOW modules) ─────────────────
RISK_LOW      = "LOW"
RISK_MODERATE = "MODERATE"
RISK_HIGH     = "HIGH"
RISK_CRITICAL = "CRITICAL"

RISK_COLORS = {
    RISK_LOW:      "#2ecc71",
    RISK_MODERATE: "#f39c12",
    RISK_HIGH:     "#e74c3c",
    RISK_CRITICAL: "#9b59b6",
}

RISK_ICONS = {
    RISK_LOW:      "🟢",
    RISK_MODERATE: "🟡",
    RISK_HIGH:     "🔴",
    RISK_CRITICAL: "🟣",
}

# ─── Score Bands ──────────────────────────────────────────────────────────────
SCORE_LOW_MAX      = 3
SCORE_MODERATE_MAX = 6
SCORE_HIGH_MAX     = 9
# Above SCORE_HIGH_MAX → CRITICAL


class CompositeRainValidator:
    """
    Orchestrates all three CV validators and integrates their outputs with the
    existing FLOW risk-scoring pipeline.

    Thread safety: Not thread-safe. Create one instance per background thread
    (the camera_worker thread), not per Streamlit rerun.  Use st.cache_resource
    to persist the instance across reruns.
    """

    def __init__(
        self,
        use_streak_detection: bool = True,
        visibility_kwargs:     dict = None,
        disturbance_kwargs:    dict = None,
        streak_kwargs:         dict = None,
    ):
        """
        Parameters
        ──────────
        use_streak_detection : Set False if your camera mount is unstable (vibration
                               creates false streak-like artifacts) or if shutter
                               speed is too fast to capture streaks.
        visibility_kwargs    : Override defaults passed to VisibilityValidator(**kwargs).
        disturbance_kwargs   : Override defaults for SurfaceDisturbanceValidator(**kwargs).
        streak_kwargs        : Override defaults for RainStreakDetector(**kwargs).
        """
        self.use_streak_detection = use_streak_detection

        self._vis  = VisibilityValidator(**(visibility_kwargs or {}))
        self._dist = SurfaceDisturbanceValidator(**(disturbance_kwargs or {}))
        self._streak = RainStreakDetector(**(streak_kwargs or {})) if use_streak_detection else None

        # Caching: skip re-running full analysis if frame is very similar
        self._last_vis_result:    Optional[Dict] = None
        self._last_dist_result:   Optional[Dict] = None
        self._last_streak_result: Optional[Dict] = None
        self._frame_counter: int = 0

        # Performance stats
        self._last_analysis_ms: float = 0.0

    # ─── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        frame:            np.ndarray,
        water_polygon:    Optional[List[Tuple[int, int]]] = None,
        api_rain_mm_h:    float = 0.0,
        blockage_pct:     float = 0.0,
        predictor_score:  float = 0.0,
        run_streaks:      bool  = True,
    ) -> Dict:
        """
        Run all validation checks and return a composite risk result.

        Parameters
        ──────────
        frame            : Current BGR camera frame (OpenCV).
        water_polygon    : FLOW ROI polygon points — restricts CV analysis to water.
        api_rain_mm_h    : Live rainfall from OpenWeatherMap (mm/h).
        blockage_pct     : Current channel blockage % from DebrisDetector.
        predictor_score  : FloodPredictor.predict()["risk_score"] — the existing
                           ML model's 0–1 risk score.
        run_streaks      : Whether to run the streak detector this frame.
                           You can pass False every other frame to save CPU.

        Returns
        ───────
        {
            # ── Sub-results (display in indicator panel) ──────────────────
            "visibility"       : dict   # from VisibilityValidator.analyse()
            "disturbance"      : dict   # from SurfaceDisturbanceValidator.analyse()
            "streaks"          : dict   # from RainStreakDetector.analyse()  (or None)

            # ── Composite scoring breakdown ───────────────────────────────
            "api_rain_points"  : int    # 0–3  weather API contribution
            "cv_points"        : int    # 0–3  camera validation contribution
            "system_points"    : int    # 0–5  water level + debris
            "total_score"      : int    # 0–11

            # ── Final risk output ─────────────────────────────────────────
            "final_risk"       : str    # LOW | MODERATE | HIGH | CRITICAL
            "risk_color"       : str    # hex
            "risk_icon"        : str    # emoji

            # ── Meta ─────────────────────────────────────────────────────
            "analysis_ms"      : float  # processing time in milliseconds
            "cv_validation_summary": str  # one-line human-readable summary
        }
        """
        t0 = time.perf_counter()
        self._frame_counter += 1

        # ── Run CV validators ──────────────────────────────────────────────────
        vis_result  = self._vis.analyse(frame)
        dist_result = self._dist.analyse(frame, water_polygon)

        if self.use_streak_detection and run_streaks and self._streak is not None:
            streak_result = self._streak.analyse(frame, water_polygon)
        else:
            streak_result = self._last_streak_result or _null_streak_result()

        self._last_vis_result    = vis_result
        self._last_dist_result   = dist_result
        self._last_streak_result = streak_result

        # ── Camera validation points (0–3) ─────────────────────────────────────
        cv_points = 0
        if vis_result["visibility_low"]:
            cv_points += 1
        if dist_result["disturbance_high"]:
            cv_points += 1
        if streak_result.get("streaks_detected", False):
            cv_points += 1

        # ── Weather API points (0–3) ───────────────────────────────────────────
        # Tiered: each tier adds 1 point cumulatively.
        # Thresholds align with JMM Malaysia rainfall categories:
        #   Light <5 mm/h, Moderate 5–30, Heavy >30
        api_points = 0
        if api_rain_mm_h >= 5.0:
            api_points += 1
        if api_rain_mm_h >= 15.0:
            api_points += 1
        if api_rain_mm_h >= 30.0:
            api_points += 1

        # ── Existing system points (0–5) ───────────────────────────────────────
        # Map ML predictor score [0,1] → 0–3 pts (water level)
        # Map blockage_pct [0,100] → 0–2 pts (debris)
        water_pts   = int(min(predictor_score * 3, 3))  # 0→0, 0.5→1, 0.75→2, 1.0→3
        debris_pts  = 0
        if blockage_pct >= 50.0:
            debris_pts = 1
        if blockage_pct >= 75.0:
            debris_pts = 2
        system_points = water_pts + debris_pts

        # ── Total score → risk label ──────────────────────────────────────────
        total_score = api_points + cv_points + system_points

        if total_score <= SCORE_LOW_MAX:
            final_risk = RISK_LOW
        elif total_score <= SCORE_MODERATE_MAX:
            final_risk = RISK_MODERATE
        elif total_score <= SCORE_HIGH_MAX:
            final_risk = RISK_HIGH
        else:
            final_risk = RISK_CRITICAL

        # ── Human-readable CV summary ─────────────────────────────────────────
        flags = []
        if vis_result["visibility_low"]:
            flags.append(f"visibility {vis_result['status'].lower()}")
        if dist_result["disturbance_high"]:
            flags.append(f"surface {dist_result['level'].lower()}")
        if streak_result.get("streaks_detected"):
            flags.append(f"{streak_result['level'].lower()}")

        if flags:
            cv_summary = "CV confirms: " + " · ".join(flags) + f" (+{cv_points} pts)"
        else:
            cv_summary = f"CV: no rain signals detected (+{cv_points} pts)"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._last_analysis_ms = elapsed_ms

        return {
            # Sub-results
            "visibility":    vis_result,
            "disturbance":   dist_result,
            "streaks":       streak_result,
            # Score breakdown
            "api_rain_points": api_points,
            "cv_points":       cv_points,
            "system_points":   system_points,
            "total_score":     total_score,
            # Final output
            "final_risk":      final_risk,
            "risk_color":      RISK_COLORS[final_risk],
            "risk_icon":       RISK_ICONS[final_risk],
            # Meta
            "analysis_ms":     round(elapsed_ms, 2),
            "cv_validation_summary": cv_summary,
        }

    def reset(self):
        """Reset inter-frame state.  Call when the camera source changes."""
        self._dist.reset()
        self._last_vis_result    = None
        self._last_dist_result   = None
        self._last_streak_result = None

    @property
    def last_analysis_ms(self) -> float:
        return self._last_analysis_ms


# ─── Null result helper ───────────────────────────────────────────────────────

def _null_streak_result() -> Dict:
    return {
        "streak_count":     0,
        "streak_density":   0.0,
        "level":            "No Streaks",
        "streaks_detected": False,
        "color":            "#2ecc71",
        "detail":           "Streak detection disabled.",
    }


# ─── Streamlit Dashboard Render Helpers ───────────────────────────────────────
# Import these in ui.py or render inline in main.py dashboard section.

def render_cv_validation_panel(cv_result: Dict) -> str:
    """
    Return an HTML string for the camera validation indicators panel.
    Designed to match FLOW's existing dark-theme card style.

    Usage in main.py
    ────────────────
        from rain_validation.composite import render_cv_validation_panel
        st.markdown(render_cv_validation_panel(cv_result), unsafe_allow_html=True)
    """
    vis   = cv_result.get("visibility", {})
    dist  = cv_result.get("disturbance", {})
    stk   = cv_result.get("streaks", {})
    score = cv_result.get("total_score", 0)
    risk  = cv_result.get("final_risk", RISK_LOW)
    color = cv_result.get("risk_color", RISK_COLORS[RISK_LOW])
    icon  = cv_result.get("risk_icon", "🟢")
    ms    = cv_result.get("analysis_ms", 0.0)
    summary = cv_result.get("cv_validation_summary", "")

    api_pts    = cv_result.get("api_rain_points", 0)
    cv_pts     = cv_result.get("cv_points", 0)
    sys_pts    = cv_result.get("system_points", 0)

    def _indicator(label, value_str, status_color, detail):
        return (
            f'<div style="background:rgba(0,0,0,0.20);border-radius:6px;'
            f'padding:7px 10px;margin-bottom:6px;border-left:3px solid {status_color};">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-size:11px;color:#aaa;">{label}</span>'
            f'<span style="font-size:12px;font-weight:700;color:{status_color};">{value_str}</span>'
            f'</div>'
            f'<div style="font-size:10px;color:#888;margin-top:2px;">{detail}</div>'
            f'</div>'
        )

    vis_html = _indicator(
        "👁 Visibility",
        f"{vis.get('sharpness', 0):.0f}",
        vis.get("color", "#aaa"),
        vis.get("status", "—"),
    )
    dist_html = _indicator(
        "💧 Surface Disturbance",
        f"{dist.get('disturbance_value', 0):.3f}",
        dist.get("color", "#aaa"),
        dist.get("level", "—"),
    )
    stk_html = _indicator(
        "🌧 Rain Streaks",
        f"{stk.get('streak_count', 0)} streaks",
        stk.get("color", "#aaa"),
        stk.get("level", "—"),
    )

    score_bar = ""
    for i in range(11):
        filled = "background:#00d4ff;" if i < score else "background:#1e3a5f;"
        score_bar += (
            f'<div style="flex:1;height:6px;border-radius:2px;{filled}'
            f'margin:0 1px;"></div>'
        )

    html = f"""
<div style="background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.07);
     border-radius:10px;padding:10px 12px;margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <span style="font-size:10px;letter-spacing:1px;color:#4a6b8a;">🎥 CAMERA VALIDATION (CV)</span>
    <span style="font-size:14px;font-weight:800;color:{color};">{icon} {risk}</span>
  </div>
  {vis_html}
  {dist_html}
  {stk_html}
  <div style="font-size:10px;color:#888;margin:6px 0 4px;">Score: {score}/11
    &nbsp;·&nbsp; API:{api_pts}&nbsp;·&nbsp;CV:{cv_pts}&nbsp;·&nbsp;SYS:{sys_pts}
    &nbsp;·&nbsp; {ms:.1f} ms
  </div>
  <div style="display:flex;margin:4px 0 6px;">{score_bar}</div>
  <div style="font-size:10px;color:#5a8a6a;font-style:italic;">{summary}</div>
</div>
"""
    return html
