"""
FLOW — Flood Level Observation Warning System
Rain Validation Module: Surface Disturbance Validator

ALGORITHM EXPLANATION
─────────────────────
Rainfall striking a river or wet ground produces ripples, splashes, and
micro-turbulence that cause continuous small pixel-level changes between
frames.  We detect this using **frame differencing**:

    1. Convert consecutive frames to grayscale.
    2. Compute the absolute per-pixel difference between frames t and t-1.
    3. Apply a small Gaussian blur to suppress sensor noise.
    4. Threshold at a low value to isolate genuine motion pixels.
    5. Compute the **motion density** = fraction of pixels that changed.
    6. Classify the density into disturbance levels.

Why this works for rain validation
───────────────────────────────────
• Rain hitting the water surface creates a very high density of small,
  spatially distributed changes — unlike moving debris (large blobs) or
  lighting flicker (whole-frame shift).
• By restricting analysis to the water ROI polygon (if provided), we
  avoid false positives from wind-blown trees or passing vehicles.

THRESHOLD CALIBRATION FOR MALAYSIA
────────────────────────────────────
Under dry, slow-flowing conditions a river still has ~2–5 % moving pixels
from natural ripples and current.  Rainfall adds a dense overlay of small
inter-frame changes:

    Dry baseline     : 0.02 – 0.08  (2–8 % of ROI pixels differ)
    Light rain       : 0.08 – 0.15
    Moderate rain    : 0.15 – 0.30
    Heavy rain       : 0.30 – 0.55
    Very heavy rain  : > 0.55

Start with the defaults and log `disturbance_value` during known dry periods
to calibrate your specific site.  Outdoor cameras with rolling shutters tend
to produce higher baseline noise — increase `noise_threshold` if you see
false positives at night.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# ─── Default Thresholds ────────────────────────────────────────────────────────

# Pixel-wise absolute-difference value below which a change is treated as noise.
# Range 0–255.  10–15 suppresses CMOS sensor noise without masking rain ripples.
NOISE_THRESHOLD = 12

# Motion density thresholds (fraction of ROI pixels that "moved")
CALM_MAX      = 0.08   # ≤ 8 %  → Calm
MILD_MAX      = 0.18   # 8–18 % → Mild Disturbance
MODERATE_MAX  = 0.35   # 18–35% → Moderate Disturbance
# Above 35 %          → High Disturbance


class SurfaceDisturbanceValidator:
    """
    Estimates water-surface disturbance using inter-frame motion density.

    Usage
    ─────
        validator = SurfaceDisturbanceValidator()

        # In your camera loop — call once per frame:
        result = validator.analyse(current_frame, water_polygon)

    The `disturbance_high` flag maps directly into the composite risk score.
    """

    def __init__(
        self,
        noise_threshold: int   = NOISE_THRESHOLD,
        calm_max:        float = CALM_MAX,
        mild_max:        float = MILD_MAX,
        moderate_max:    float = MODERATE_MAX,
        smoothing_alpha: float = 0.35,
    ):
        """
        Parameters
        ──────────
        noise_threshold  : Minimum pixel difference to count as motion (0–255).
        calm_max         : Density below this → Calm.
        mild_max         : Density below this (above calm_max) → Mild Disturbance.
        moderate_max     : Density below this (above mild_max) → Moderate.
        smoothing_alpha  : Exponential moving-average factor for the density signal.
                           Lower = smoother but slower to respond (0.1–0.5 typical).
        """
        self.noise_threshold = int(np.clip(noise_threshold, 1, 254))
        self.calm_max    = calm_max
        self.mild_max    = mild_max
        self.moderate_max = moderate_max
        self.alpha       = float(np.clip(smoothing_alpha, 0.05, 1.0))

        self._prev_gray:  Optional[np.ndarray] = None
        self._smoothed_density: float = 0.0

    # ─── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        frame: np.ndarray,
        water_polygon: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict:
        """
        Compute surface disturbance for the current frame.

        Parameters
        ──────────
        frame          : Current BGR (or grayscale) frame.
        water_polygon  : Optional list of (x, y) points defining the water ROI.
                         When provided, only pixels inside this region are analysed.
                         If None, the full frame is used.

        Returns
        ───────
        {
            "disturbance_value"   : float  – smoothed motion density (0–1)
            "raw_density"         : float  – unsmoothed value this frame
            "level"               : str    – "Calm" | "Mild Disturbance"
                                             | "Moderate Disturbance" | "High Disturbance"
            "disturbance_high"    : bool   – True when level is Moderate or High
            "color"               : str    – hex colour for UI
            "detail"              : str    – human-readable description
        }

        On the very first call (no previous frame), returns zero-motion defaults.
        """
        gray = self._to_gray(frame)
        h, w = gray.shape

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            # First frame — can't compute difference yet
            self._prev_gray = gray
            return self._zero_result()

        # ── Step 1: Absolute frame difference ─────────────────────────────────
        diff = cv2.absdiff(gray, self._prev_gray)

        # ── Step 2: Gaussian blur to suppress CMOS/rolling-shutter noise ──────
        # A small 3×3 blur smears single-pixel noise while preserving rain ripple
        # patterns (which are spatially larger than 1 pixel).
        diff_smooth = cv2.GaussianBlur(diff, (3, 3), 0)

        # ── Step 3: Binary mask — pixels that genuinely changed ───────────────
        _, motion_mask = cv2.threshold(
            diff_smooth, self.noise_threshold, 255, cv2.THRESH_BINARY
        )

        # ── Step 4: Restrict to water ROI polygon (if provided) ───────────────
        if water_polygon and len(water_polygon) >= 3:
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(water_polygon, dtype=np.int32)
            cv2.fillPoly(roi_mask, [pts], 255)
            motion_mask = cv2.bitwise_and(motion_mask, roi_mask)
            total_pixels = int(np.count_nonzero(roi_mask))
        else:
            total_pixels = h * w

        # ── Step 5: Motion density ─────────────────────────────────────────────
        motion_pixels = int(np.count_nonzero(motion_mask))
        raw_density   = motion_pixels / max(total_pixels, 1)

        # ── Step 6: Exponential moving average (stabilises noisy readings) ────
        self._smoothed_density = (
            self.alpha * raw_density
            + (1 - self.alpha) * self._smoothed_density
        )

        # Update previous frame
        self._prev_gray = gray

        level, color, detail = self._classify(self._smoothed_density)
        disturbance_high = level in ("Moderate Disturbance", "High Disturbance")

        return {
            "disturbance_value": round(self._smoothed_density, 4),
            "raw_density":       round(raw_density, 4),
            "level":             level,
            "disturbance_high":  disturbance_high,
            "color":             color,
            "detail":            detail,
        }

    def reset(self):
        """Clear the stored previous frame.  Call if the camera source changes."""
        self._prev_gray = None
        self._smoothed_density = 0.0

    # ─── Internal Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame.copy()

    def _classify(self, density: float):
        if density <= self.calm_max:
            return (
                "Calm",
                "#2ecc71",
                f"Motion density {density:.3f} — calm surface, no rain disturbance.",
            )
        elif density <= self.mild_max:
            return (
                "Mild Disturbance",
                "#f1c40f",
                f"Motion density {density:.3f} — light ripples; consistent with light rain.",
            )
        elif density <= self.moderate_max:
            return (
                "Moderate Disturbance",
                "#e67e22",
                f"Motion density {density:.3f} — significant surface motion; indicative of moderate rain.",
            )
        else:
            return (
                "High Disturbance",
                "#e74c3c",
                f"Motion density {density:.3f} — high turbulence; heavy rainfall impact confirmed visually.",
            )

    @staticmethod
    def _zero_result() -> Dict:
        return {
            "disturbance_value": 0.0,
            "raw_density":       0.0,
            "level":             "Calm",
            "disturbance_high":  False,
            "color":             "#2ecc71",
            "detail":            "Initialising — awaiting second frame.",
        }

    # ─── Calibration Helper ────────────────────────────────────────────────────

    @staticmethod
    def calibrate_dry_baseline(
        frame_pairs: List[Tuple[np.ndarray, np.ndarray]],
        noise_threshold: int = NOISE_THRESHOLD,
    ) -> Dict:
        """
        Estimate baseline motion density from dry-period frame pairs.

        Usage
        ─────
            pairs = [(frame_a, frame_b), (frame_c, frame_d), ...]   # dry-day pairs
            baseline = SurfaceDisturbanceValidator.calibrate_dry_baseline(pairs)
            print(baseline)  # → {"calm_max": 0.09, "mild_max": 0.20, ...}

        The returned dict can be passed as **kwargs to __init__.
        """
        validator = SurfaceDisturbanceValidator(noise_threshold=noise_threshold)
        densities = []
        for a, b in frame_pairs:
            validator.reset()
            validator.analyse(a)         # sets _prev_gray
            result = validator.analyse(b)
            densities.append(result["raw_density"])
        if not densities:
            return {}
        arr = np.array(densities)
        p95 = float(np.percentile(arr, 95))
        return {
            "calm_max":    round(p95 * 1.2, 3),    # 20% above dry-day p95
            "mild_max":    round(p95 * 2.5, 3),
            "moderate_max": round(p95 * 5.0, 3),
        }
