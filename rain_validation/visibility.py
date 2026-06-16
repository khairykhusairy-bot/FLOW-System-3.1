"""
FLOW — Flood Level Observation Warning System
Rain Validation Module: Visibility Reduction Validator

ALGORITHM EXPLANATION
─────────────────────
Heavy rain and fog reduce scene sharpness.  We measure this using the
**Laplacian variance** method:

    1. Convert the frame to grayscale.
    2. Apply a Laplacian kernel (edge-detector) to the grayscale image.
       The kernel amplifies rapid changes in pixel intensity (= sharp edges).
    3. Compute the variance of the resulting values.
       • High variance → many strong edges → sharp, clear scene.
       • Low variance  → weak edges → blurry / foggy / rainy scene.

This is a classical, CPU-only technique with negligible compute cost (~1 ms
per 960-px frame).

THRESHOLD CALIBRATION FOR MALAYSIA
────────────────────────────────────
Malaysian cameras are typically low-cost CCTV units mounted outdoors.
Typical baseline sharpness under clear sky at a river site: 400–1 500.
During moderate to heavy rain: 80–300.
During thick fog or very heavy rain: < 80.

Recommended starting thresholds (tune by logging `sharpness` values over
several days and identifying natural breakpoints in your data):

    CLEAR_MIN   = 300   ← sharper than this → no visibility issue
    REDUCED_MIN = 150   ← in this band → some visibility degradation
    LOW_MIN     = 80    ← below this   → significantly reduced visibility

We also normalise sharpness to a 0–1 score so it integrates cleanly into
the composite risk engine.  The normalisation reference (`ref_sharpness`)
should be set to the median clear-sky sharpness observed at your site.
"""

import cv2
import numpy as np
from typing import Dict


# ─── Default Thresholds ────────────────────────────────────────────────────────
# All values are Laplacian variance units.
# Increase CLEAR_MIN if your camera is a sharper model.
# Decrease LOW_MIN for very cheap / low-res sensors.

CLEAR_MIN   = 300.0   # Above this: Clear / No visibility issue
REDUCED_MIN = 150.0   # 150–300:    Slightly Reduced
LOW_MIN     = 80.0    # 80–150:     Reduced
# Below LOW_MIN:         Significantly Reduced

# Reference sharpness for score normalisation.
# Set this to the 90th-percentile clear-sky sharpness at your site.
REF_SHARPNESS = 600.0


class VisibilityValidator:
    """
    Estimates visibility degradation from camera frames using Laplacian
    variance (sharpness).

    Usage
    ─────
        validator = VisibilityValidator()
        result    = validator.analyse(frame)

    The `result` dict is designed to slot directly into the composite rain
    risk score:  if result["visibility_low"] is True, add 1 to the score.
    """

    def __init__(
        self,
        clear_min:    float = CLEAR_MIN,
        reduced_min:  float = REDUCED_MIN,
        low_min:      float = LOW_MIN,
        ref_sharpness: float = REF_SHARPNESS,
        roi_fraction:  float = 0.6,
    ):
        """
        Parameters
        ──────────
        clear_min      : Laplacian variance above which scene is classified Clear.
        reduced_min    : Below clear_min → Slightly Reduced; below this → Reduced.
        low_min        : Below this → Significantly Reduced.
        ref_sharpness  : Normalisation reference (typical clear-sky value at site).
        roi_fraction   : Central crop fraction (0–1).  Using the central 60 % of the
                         frame avoids sky / bank edges that inflate sharpness scores.
        """
        self.clear_min    = clear_min
        self.reduced_min  = reduced_min
        self.low_min      = low_min
        self.ref_sharp    = max(ref_sharpness, 1.0)
        self.roi_fraction = max(0.2, min(1.0, roi_fraction))

        # Rolling baseline for adaptive thresholding.
        # Stores the last N clear-sky sharpness readings.
        self._baseline_buffer: list = []
        self._baseline_size   = 30    # 30 × ~1 s apart = 30-second rolling window

    # ─── Public API ────────────────────────────────────────────────────────────

    def analyse(self, frame: np.ndarray) -> Dict:
        """
        Compute sharpness and classify visibility status.

        Parameters
        ──────────
        frame : BGR or grayscale image from OpenCV.

        Returns
        ───────
        {
            "sharpness"        : float   – Laplacian variance (higher = sharper)
            "sharpness_score"  : float   – 0.0 (blurry) … 1.0 (sharp); 0–1 normalised
            "status"           : str     – "Clear" | "Slightly Reduced" | "Reduced"
                                           | "Significantly Reduced"
            "visibility_low"   : bool    – True when status is Reduced or worse
            "color"            : str     – hex colour for UI indicators
            "detail"           : str     – human-readable explanation
        }
        """
        sharpness = self._compute_sharpness(frame)
        status, color, detail = self._classify(sharpness)
        score = self._normalise(sharpness)

        # Update rolling baseline when scene is clear
        if status == "Clear":
            self._baseline_buffer.append(sharpness)
            if len(self._baseline_buffer) > self._baseline_size:
                self._baseline_buffer.pop(0)
            # Gradually adjust reference to the site's actual clear-sky sharpness
            if len(self._baseline_buffer) >= 5:
                self.ref_sharp = float(np.median(self._baseline_buffer)) * 0.9

        visibility_low = status in ("Reduced", "Significantly Reduced")

        return {
            "sharpness":       round(sharpness, 2),
            "sharpness_score": round(score, 4),
            "status":          status,
            "visibility_low":  visibility_low,
            "color":           color,
            "detail":          detail,
        }

    # ─── Internal Helpers ──────────────────────────────────────────────────────

    def _compute_sharpness(self, frame: np.ndarray) -> float:
        """
        Compute Laplacian variance on the central ROI of the frame.

        Step-by-step
        ────────────
        1. Convert to grayscale (if needed) — Laplacian is a single-channel op.
        2. Crop to central `roi_fraction` to exclude sky, banks, and frame edges
           which have fixed high-contrast borders that would bias the score.
        3. Apply cv2.Laplacian with kernel size 3 — a 3×3 second-derivative
           filter that highlights intensity discontinuities (= edges).
        4. Return var(laplacian).
           Variance rather than mean: positive and negative responses cancel in
           the mean but accumulate in the variance, making it robust to uniform
           shifts in intensity.
        """
        # Step 1 — grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Step 2 — centre crop
        h, w = gray.shape
        margin_y = int(h * (1 - self.roi_fraction) / 2)
        margin_x = int(w * (1 - self.roi_fraction) / 2)
        crop = gray[margin_y: h - margin_y, margin_x: w - margin_x]

        # Step 3 — Laplacian
        lap = cv2.Laplacian(crop, cv2.CV_64F, ksize=3)

        # Step 4 — variance
        return float(np.var(lap))

    def _normalise(self, sharpness: float) -> float:
        """
        Map sharpness to [0, 1] where 0 = very blurry and 1 = very sharp.
        Uses a soft-clamp so extreme values do not saturate the scale.
        """
        # tanh-based soft normalisation
        ratio = sharpness / self.ref_sharp
        return float(np.tanh(ratio))

    def _classify(self, sharpness: float):
        """Return (status_label, hex_color, detail_text) for a sharpness value."""
        if sharpness >= self.clear_min:
            return (
                "Clear",
                "#2ecc71",
                f"Scene sharpness {sharpness:.0f} — no visibility degradation detected.",
            )
        elif sharpness >= self.reduced_min:
            return (
                "Slightly Reduced",
                "#f1c40f",
                f"Scene sharpness {sharpness:.0f} — mild haze or light rain may be present.",
            )
        elif sharpness >= self.low_min:
            return (
                "Reduced",
                "#e67e22",
                f"Scene sharpness {sharpness:.0f} — moderate rain or fog detected visually.",
            )
        else:
            return (
                "Significantly Reduced",
                "#e74c3c",
                f"Scene sharpness {sharpness:.0f} — heavy rain or dense fog severely reduces visibility.",
            )

    # ─── Calibration Helper ────────────────────────────────────────────────────

    @staticmethod
    def calibrate_from_frames(frames: list, percentile: float = 90.0) -> dict:
        """
        Helper: pass a list of clear-sky frames to estimate good thresholds.

        Usage
        ─────
            import cv2
            caps = [cv2.imread(f"clear_{i}.jpg") for i in range(20)]
            thresholds = VisibilityValidator.calibrate_from_frames(caps)
            print(thresholds)

        Returns a dict of suggested threshold values you can pass to __init__.
        """
        validator = VisibilityValidator()
        sharpnesses = [validator._compute_sharpness(f) for f in frames if f is not None]
        if not sharpnesses:
            return {}
        arr = np.array(sharpnesses)
        p90 = float(np.percentile(arr, percentile))
        return {
            "clear_min":    round(p90 * 0.50, 1),   # 50% of clear-sky baseline
            "reduced_min":  round(p90 * 0.25, 1),   # 25%
            "low_min":      round(p90 * 0.13, 1),   # 13%
            "ref_sharpness": round(p90, 1),
        }
