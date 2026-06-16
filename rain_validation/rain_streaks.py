"""
FLOW — Flood Level Observation Warning System
Rain Validation Module: Rain Streak Detector

ALGORITHM EXPLANATION
─────────────────────
Falling rain appears as thin, near-vertical bright streaks in camera frames,
especially when there is a dark background (river water, vegetation).
We detect these using classical computer vision:

    1. Convert to grayscale.
    2. Apply Canny edge detection — identifies strong intensity gradients
       (= the edges of rain streaks, debris edges, foliage edges, etc.).
    3. Filter contours by geometry:
         • aspect ratio ≥ 2.5  (streaks are elongated, not boxy)
         • area ≥ 8 px²        (ignore single-pixel noise)
         • area ≤ 800 px²      (ignore large blobs = debris / person)
         • solidity ≥ 0.4      (streaks are fairly solid, not ring-like)
    4. Count surviving contours and divide by ROI area → streak density.

Limitations (important)
───────────────────────
• Rain streaks are only clearly visible when:
  - The camera shutter is slow enough (≥ 1/500 s captures ~3–6 cm streaks).
  - There is sufficient contrast (dark background, not heavy glare).
• This feature is explicitly labelled **supplementary** in FLOW.  It should
  NOT be the deciding factor — it is one point in a 10-point risk score.
• High-frequency vibration (e.g. wind on the mounting pole) can create
  false streak-like artifacts.  If your camera mount is unstable, set
  `use_streak_detection = False` in the composite validator.

THRESHOLD CALIBRATION FOR MALAYSIA
────────────────────────────────────
Malaysian outdoor cameras on river banks are often positioned below tree
canopy, which can cast leaf shadows resembling streaks.  Increase
`min_aspect` to 3.0 and `min_area` to 15 if you see false positives.

Typical streak density values (streaks per 1 000 ROI pixels):
    No rain    : 0.000 – 0.005
    Light rain : 0.005 – 0.020
    Moderate   : 0.020 – 0.060
    Heavy rain : > 0.060
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# ─── Default Canny Parameters ─────────────────────────────────────────────────
CANNY_LOW  = 50    # Lower hysteresis threshold
CANNY_HIGH = 130   # Upper hysteresis threshold

# ─── Contour Geometry Filters ─────────────────────────────────────────────────
MIN_ASPECT  = 2.5   # Min (height / width) for a bounding-rect aspect ratio
MIN_AREA    = 8.0   # px² — ignore sub-8-pixel blobs
MAX_AREA    = 800.0 # px² — ignore large objects
MIN_SOLIDITY = 0.40 # Contour area / convex hull area

# ─── Streak Density Thresholds (streaks per 1 000 ROI pixels) ─────────────────
NONE_MAX   = 0.005
LIGHT_MAX  = 0.025
MODERATE_MAX = 0.060
# Above MODERATE_MAX → Heavy Rain Streaks


class RainStreakDetector:
    """
    Detects visible rain streaks in camera frames using Canny edges and
    geometric contour filtering.

    Usage
    ─────
        detector = RainStreakDetector()
        result   = detector.analyse(frame, water_polygon)

    The `streaks_detected` flag is a supplementary binary input to the
    composite rain risk score (+1 point if True).
    """

    def __init__(
        self,
        canny_low:    int   = CANNY_LOW,
        canny_high:   int   = CANNY_HIGH,
        min_aspect:   float = MIN_ASPECT,
        min_area:     float = MIN_AREA,
        max_area:     float = MAX_AREA,
        min_solidity: float = MIN_SOLIDITY,
        none_max:     float = NONE_MAX,
        light_max:    float = LIGHT_MAX,
        moderate_max: float = MODERATE_MAX,
    ):
        self.canny_low    = canny_low
        self.canny_high   = canny_high
        self.min_aspect   = min_aspect
        self.min_area     = min_area
        self.max_area     = max_area
        self.min_solidity = min_solidity
        self.none_max     = none_max
        self.light_max    = light_max
        self.moderate_max = moderate_max

    # ─── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        frame: np.ndarray,
        water_polygon: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict:
        """
        Detect rain streaks in the frame.

        Parameters
        ──────────
        frame          : Current BGR (or grayscale) frame.
        water_polygon  : Optional water-area ROI polygon for masking.
                         Restricting to the water area reduces false positives
                         from vegetation or sky movement.

        Returns
        ───────
        {
            "streak_count"    : int    – number of streak-like contours found
            "streak_density"  : float  – streaks per 1 000 ROI pixels (×10⁻³)
            "level"           : str    – "No Streaks" | "Light Rain Streaks"
                                         | "Moderate Rain Streaks" | "Heavy Rain Streaks"
            "streaks_detected": bool   – True when level is not "No Streaks"
            "color"           : str    – hex colour for UI
            "detail"          : str    – human-readable description
        }
        """
        h, w = frame.shape[:2]

        # ── Step 1: Grayscale ─────────────────────────────────────────────────
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # ── Step 2: Canny edge detection ──────────────────────────────────────
        # A slight Gaussian blur first (3×3) removes fine pixel noise that
        # creates spurious single-pixel edges unrelated to rain streaks.
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges   = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # ── Step 3: Mask to ROI (if provided) ────────────────────────────────
        roi_area = h * w
        if water_polygon and len(water_polygon) >= 3:
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(water_polygon, dtype=np.int32)
            cv2.fillPoly(roi_mask, [pts], 255)
            edges    = cv2.bitwise_and(edges, roi_mask)
            roi_area = int(np.count_nonzero(roi_mask))

        # ── Step 4: Contour extraction ─────────────────────────────────────
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── Step 5: Geometric filtering ───────────────────────────────────────
        streak_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            # Bounding rectangle aspect ratio
            _, _, bw, bh = cv2.boundingRect(cnt)
            if bw == 0:
                continue
            aspect = bh / bw   # height-over-width → > 1 means taller than wide

            # Accept near-vertical streaks: aspect ≥ min_aspect
            # OR near-horizontal (when camera is tilted): aspect ≤ 1/min_aspect
            if aspect < self.min_aspect and (bw / max(bh, 1)) < self.min_aspect:
                continue

            # Solidity filter: convex-hull ratio
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < self.min_solidity:
                    continue

            streak_contours.append(cnt)

        streak_count = len(streak_contours)

        # ── Step 6: Streak density ─────────────────────────────────────────────
        # Normalise by ROI area in thousands of pixels for a stable metric
        # across different frame sizes.
        streak_density = streak_count / (max(roi_area, 1) / 1000.0)

        level, color, detail = self._classify(streak_density)
        streaks_detected = level != "No Streaks"

        return {
            "streak_count":     streak_count,
            "streak_density":   round(streak_density, 4),
            "level":            level,
            "streaks_detected": streaks_detected,
            "color":            color,
            "detail":           detail,
        }

    def _classify(self, density: float):
        if density <= self.none_max:
            return (
                "No Streaks",
                "#2ecc71",
                f"Streak density {density:.4f} — no rain streak pattern detected.",
            )
        elif density <= self.light_max:
            return (
                "Light Rain Streaks",
                "#f1c40f",
                f"Streak density {density:.4f} — sparse rain streaks visible in frame.",
            )
        elif density <= self.moderate_max:
            return (
                "Moderate Rain Streaks",
                "#e67e22",
                f"Streak density {density:.4f} — moderate rain streak pattern detected.",
            )
        else:
            return (
                "Heavy Rain Streaks",
                "#e74c3c",
                f"Streak density {density:.4f} — dense rain streak pattern; heavy rain visually confirmed.",
            )
