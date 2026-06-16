"""
FLOW — Water Level Module
trend_analysis.py: Tracks water level history and calculates rise rate / trend.

Stores timestamped readings in a rolling in-memory buffer and provides:
  • cm/min rise rate (over a configurable rolling window)
  • Human-readable trend label (stable / rising slowly / rising rapidly / critical surge)
  • Flood escalation flag (True when rate exceeds danger threshold)
  • Historical readings list (for charting)

All data lives in memory — no DB writes from this module.
Integration with database.py happens in main.py / camera_worker.
"""

import time
from collections import deque
from typing import Deque, List, Optional, Tuple, Dict

from water_level.config import (
    TREND_WINDOW_SECS,
    RISE_RATE_WARNING,
    RISE_RATE_CRITICAL,
    THRESHOLD_NORMAL,
    THRESHOLD_WARNING,
    THRESHOLD_DANGER,
    THRESHOLD_CRITICAL,
)


# ─── Trend Labels ──────────────────────────────────────────────────────────────
TREND_STABLE          = "Stable"
TREND_RISING_SLOW     = "Rising slowly"
TREND_RISING_FAST     = "Rising rapidly"
TREND_CRITICAL_SURGE  = "Critical surge"
TREND_FALLING         = "Falling"

# ─── Risk Status Labels ────────────────────────────────────────────────────────
STATUS_NORMAL   = "Normal"
STATUS_WARNING  = "Warning"
STATUS_DANGER   = "Danger"
STATUS_CRITICAL = "Critical"


class WaterLevelTrend:
    """
    Maintains a rolling history of (timestamp, level_cm) readings and
    derives rise rate and trend labels.

    Parameters
    ----------
    window_secs : Seconds of history used to compute the rise rate.
    max_history : Maximum number of readings stored in memory (for charting).
    """

    def __init__(
        self,
        window_secs: float = TREND_WINDOW_SECS,
        max_history: int   = 300,
    ):
        self._window: float = max(5.0, window_secs)
        # Deque stores (timestamp_float, level_cm) tuples
        self._readings: Deque[Tuple[float, float]] = deque(maxlen=max_history)
        self._rise_rate_cm_per_min: float = 0.0
        self._trend_label: str            = TREND_STABLE
        self._risk_status: str            = STATUS_NORMAL
        self._last_level_cm: Optional[float] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, level_cm: float) -> Dict:
        """
        Add a new smoothed reading and recompute all trend metrics.

        Parameters
        ----------
        level_cm : Smoothed water level in cm.

        Returns
        -------
        dict with keys:
            level_cm, rise_rate_cm_per_min, trend, risk_status,
            is_escalating, history_cm
        """
        now = time.time()
        self._readings.append((now, level_cm))
        self._last_level_cm = level_cm

        # ── Calculate rise rate over the window ───────────────────────────────
        self._rise_rate_cm_per_min = self._calc_rise_rate(now)

        # ── Classify trend ────────────────────────────────────────────────────
        self._trend_label = self._classify_trend(self._rise_rate_cm_per_min)

        # ── Classify risk status from absolute level ───────────────────────────
        self._risk_status = self._classify_risk(level_cm)

        return self.snapshot()

    def snapshot(self) -> Dict:
        """Return the latest computed trend data as a plain dict."""
        return {
            "level_cm":             self._last_level_cm,
            "rise_rate_cm_per_min": round(self._rise_rate_cm_per_min, 3),
            "trend":                self._trend_label,
            "risk_status":          self._risk_status,
            "is_escalating":        self._rise_rate_cm_per_min >= RISE_RATE_WARNING,
            "is_critical_surge":    self._rise_rate_cm_per_min >= RISE_RATE_CRITICAL,
            "history_cm":           self.history_cm(limit=120),
        }

    def history_cm(self, limit: int = 120) -> List[float]:
        """Return recent level readings (most recent last), up to `limit` points."""
        readings = list(self._readings)
        return [r[1] for r in readings[-limit:]]

    def history_timestamped(self, limit: int = 120) -> List[Tuple[float, float]]:
        """Return recent (timestamp, level_cm) tuples."""
        readings = list(self._readings)
        return readings[-limit:]

    def reset(self):
        self._readings.clear()
        self._rise_rate_cm_per_min = 0.0
        self._trend_label = TREND_STABLE
        self._risk_status = STATUS_NORMAL
        self._last_level_cm = None

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _calc_rise_rate(self, now: float) -> float:
        """
        Estimate rise rate (cm/min) using a least-squares slope over
        the most recent ``_window`` seconds of readings.

        Falls back to a simple Δ/Δt if there are fewer than 3 points.
        """
        cutoff = now - self._window
        recent = [(t, h) for t, h in self._readings if t >= cutoff]

        if len(recent) < 2:
            return 0.0

        if len(recent) == 2:
            dt_min = (recent[-1][0] - recent[0][0]) / 60.0
            if dt_min < 1e-6:
                return 0.0
            return (recent[-1][1] - recent[0][1]) / dt_min

        # Least-squares slope (time in minutes)
        t0 = recent[0][0]
        xs = [(t - t0) / 60.0 for t, _ in recent]
        ys = [h for _, h in recent]
        n  = len(xs)
        sum_x  = sum(xs)
        sum_y  = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_x2 = sum(x * x for x in xs)
        denom  = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-9:
            return 0.0
        slope = (n * sum_xy - sum_x * sum_y) / denom
        return slope

    @staticmethod
    def _classify_trend(rate: float) -> str:
        """Map rise rate (cm/min) to a human-readable trend label."""
        if rate >= RISE_RATE_CRITICAL:
            return TREND_CRITICAL_SURGE
        elif rate >= RISE_RATE_WARNING:
            return TREND_RISING_FAST
        elif rate > 0.3:
            return TREND_RISING_SLOW
        elif rate < -0.3:
            return TREND_FALLING
        else:
            return TREND_STABLE

    @staticmethod
    def _classify_risk(level_cm: float) -> str:
        """Map absolute water level to risk status label."""
        if level_cm >= THRESHOLD_CRITICAL:
            return STATUS_CRITICAL
        elif level_cm >= THRESHOLD_DANGER:
            return STATUS_DANGER
        elif level_cm >= THRESHOLD_WARNING:
            return STATUS_WARNING
        else:
            return STATUS_NORMAL

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def level_cm(self) -> Optional[float]:
        return self._last_level_cm

    @property
    def rise_rate(self) -> float:
        return self._rise_rate_cm_per_min

    @property
    def trend(self) -> str:
        return self._trend_label

    @property
    def risk_status(self) -> str:
        return self._risk_status

    @property
    def reading_count(self) -> int:
        return len(self._readings)
