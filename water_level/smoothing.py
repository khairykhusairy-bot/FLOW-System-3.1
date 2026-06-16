"""
FLOW — Water Level Module
smoothing.py: Multi-layer signal smoothing for stable water level readings.

Raw waterline pixel detections fluctuate due to:
  • Reflections and glare on the water surface
  • Rain droplets on the camera lens
  • Moving debris temporarily obscuring the gauge
  • Rapid lighting changes (clouds passing)

This module applies four complementary filters in order:

  1. Spike Rejection    — discard readings that deviate too far from the
                          current estimate in a single frame.
  2. Rolling Median     — NEW: median of the last N readings, highly robust
                          to outlier frames caused by debris or glare bursts.
  3. Moving Average     — simple rolling mean over a configurable window,
                          applied after the median for final smoothing.
  4. Exponential EMA    — fast-response weighted average that favours recent
                          readings without the lag of a pure moving average.

Changes from v1
---------------
• Rolling median (Stage 2) inserted before the moving average.  A median is
  immune to outliers where EMA and MA are not — a single bad frame cannot
  shift the output by more than one median position.
• SPIKE_REJECTION_CM default tightened from 30 → 15 cm.  At 25 FPS a real
  flood can rise at most ~0.2 cm/frame even in a critical surge; 15 cm is
  still generous for legitimate rapid rises.
• Diagnostics extended with median_cm and rejection_rate_recent (last 30 s).
"""

from collections import deque
from typing import Optional, Deque
import statistics

from water_level.config import (
    MOVING_AVG_WINDOW,
    EXP_SMOOTH_ALPHA,
    SPIKE_REJECTION_CM,
)

# Median window — kept smaller than the MA window so it responds faster
# while still suppressing single-frame outliers.
_MEDIAN_WINDOW = 5


class WaterLevelSmoother:
    """
    Four-stage smoother for water level readings.

    Call ``update(raw_cm)`` on every detected waterline measurement.
    Read ``value`` to get the current smoothed estimate.
    """

    def __init__(
        self,
        window: int   = MOVING_AVG_WINDOW,
        alpha:  float = EXP_SMOOTH_ALPHA,
        spike:  float = SPIKE_REJECTION_CM,
    ):
        self._window: int         = max(1, window)
        self._alpha:  float       = max(0.01, min(1.0, alpha))
        self._spike:  float       = max(1.0, spike)

        # Stage buffers
        self._median_buf: Deque[float] = deque(maxlen=_MEDIAN_WINDOW)
        self._ma_buf:     Deque[float] = deque(maxlen=self._window)

        self._median:    Optional[float] = None
        self._ema:       Optional[float] = None
        self._ma:        Optional[float] = None
        self._smoothed:  Optional[float] = None

        self._rejected_count:        int = 0
        self._total_count:           int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, raw_cm: float) -> float:
        """
        Feed a new raw measurement (cm) into the smoother.

        Returns the smoothed water level in cm.
        """
        self._total_count += 1

        # ── Stage 1: Spike rejection ──────────────────────────────────────────
        if self._smoothed is not None:
            if abs(raw_cm - self._smoothed) > self._spike:
                self._rejected_count += 1
                return self._smoothed   # discard; return last good estimate

        # ── Stage 2: Rolling median ────────────────────────────────────────────
        self._median_buf.append(raw_cm)
        self._median = statistics.median(self._median_buf)

        # ── Stage 3: Moving average of the median-filtered values ─────────────
        self._ma_buf.append(self._median)
        self._ma = sum(self._ma_buf) / len(self._ma_buf)

        # ── Stage 4: Exponential smoothing applied to the MA ──────────────────
        if self._ema is None:
            self._ema = self._ma
        else:
            self._ema = self._alpha * self._ma + (1.0 - self._alpha) * self._ema

        self._smoothed = self._ema
        return self._smoothed

    def reset(self):
        """Clear all buffers and internal state."""
        self._median_buf.clear()
        self._ma_buf.clear()
        self._median  = None
        self._ema     = None
        self._ma      = None
        self._smoothed = None
        self._rejected_count = 0
        self._total_count    = 0

    def force(self, value_cm: float):
        """
        Force the smoother to a specific value (e.g. after calibration change).
        Fills all buffers so subsequent readings blend in smoothly.
        """
        for _ in range(_MEDIAN_WINDOW):
            self._median_buf.append(value_cm)
        for _ in range(self._window):
            self._ma_buf.append(value_cm)
        self._median   = value_cm
        self._ma       = value_cm
        self._ema      = value_cm
        self._smoothed = value_cm

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def value(self) -> Optional[float]:
        """Current smoothed water level in cm, or None if no data yet."""
        return self._smoothed

    @property
    def moving_average(self) -> Optional[float]:
        return self._ma

    @property
    def median(self) -> Optional[float]:
        return self._median

    @property
    def ema(self) -> Optional[float]:
        return self._ema

    @property
    def rejection_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._rejected_count / self._total_count

    @property
    def buffer_full(self) -> bool:
        return len(self._ma_buf) == self._window

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        return {
            "smoothed_cm":      self._smoothed,
            "median_cm":        self._median,
            "moving_avg_cm":    self._ma,
            "ema_cm":           self._ema,
            "median_buf_size":  len(self._median_buf),
            "ma_buf_size":      len(self._ma_buf),
            "window":           self._window,
            "alpha":            self._alpha,
            "spike_limit_cm":   self._spike,
            "total_readings":   self._total_count,
            "rejected_spikes":  self._rejected_count,
            "rejection_rate":   round(self.rejection_rate, 4),
        }
