"""
FLOW — Water Level Module
config.py: Configuration constants and defaults for water level estimation.

All values here can be overridden at runtime via JSON (see calibration.py).
Edit this file to change permanent defaults; or use the Streamlit sidebar to
tune values without touching code.
"""

import os

# ─── Paths ─────────────────────────────────────────────────────────────────────
WATER_LEVEL_DIR     = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE    = os.path.join(WATER_LEVEL_DIR, "calibration.json")
ROI_FILE            = os.path.join(WATER_LEVEL_DIR, "water_roi.json")

# ─── Flood Alert Thresholds (cm) ───────────────────────────────────────────────
THRESHOLD_NORMAL   =  50.0
THRESHOLD_WARNING  = 100.0
THRESHOLD_DANGER   = 150.0
THRESHOLD_CRITICAL = 200.0

# Rise-rate thresholds (cm per minute)
RISE_RATE_WARNING  =  2.0
RISE_RATE_CRITICAL =  5.0

# ─── Smoothing ─────────────────────────────────────────────────────────────────
# Moving-average window (number of readings to average).
MOVING_AVG_WINDOW  = 10

# Exponential smoothing factor (0 < α ≤ 1).
# Lower → smoother, slower response.  Higher → noisier, faster response.
EXP_SMOOTH_ALPHA   = 0.20   # lowered from 0.25 — less lag amplification

# Temporal filter: reject readings that deviate more than this many cm from
# the current smoothed value in a single frame (spike suppression).
# Tightened from 30 → 15: a real flood rises at most ~0.2 cm/frame even in a
# critical surge at 25 FPS, so 15 cm is still very generous.
SPIKE_REJECTION_CM = 15.0

# ─── Trend Analysis ────────────────────────────────────────────────────────────
TREND_WINDOW_SECS  = 60

# ─── Detection Pipeline ────────────────────────────────────────────────────────
# Gaussian blur kernel size (must be odd).
# Kept for fallback only — main path now uses bilateral filter.
BLUR_KERNEL        = 7       # increased from 5 for better pre-smoothing

# Adaptive threshold block size (must be odd, >= 3).
ADAPTIVE_BLOCK     = 25      # reduced from 31 — smaller block = finer local contrast

# Adaptive threshold C constant.
# Reduced from 4 → 2: less aggressive subtraction means more edges preserved,
# particularly at the water/gauge boundary in overcast lighting.
ADAPTIVE_C         = 2

# Canny edge detection thresholds.
# Low threshold reduced to catch faint waterlines in low-contrast scenes.
# High threshold kept to reject weak noise edges.
CANNY_LOW          = 30      # reduced from 40
CANNY_HIGH         = 100     # reduced from 120

# Minimum contour area (px²) to consider as waterline candidate.
# Reduced from 200 → 120 — allows detection of the waterline in narrow gauge ROIs.
MIN_CONTOUR_AREA   = 120

# ─── Night / Low-Light ─────────────────────────────────────────────────────────
NIGHT_MODE_THRESHOLD = 60

# CLAHE clip limit and grid.
# Clip raised slightly to enhance contrast more aggressively in rainy/overcast scenes.
CLAHE_CLIP         = 3.5     # raised from 3.0
CLAHE_GRID         = (8, 8)

# ─── Visualization ─────────────────────────────────────────────────────────────
COLOR_NORMAL   = (0,  200,  80)
COLOR_WARNING  = (0,  165, 255)
COLOR_DANGER   = (0,   80, 255)
COLOR_CRITICAL = (0,    0, 220)

COLOR_WATERLINE = (255, 220,  60)
COLOR_GAUGE_ROI = (60,  220, 255)
COLOR_RIVER_ROI = (60,  255, 160)

# ─── Calibration Defaults ──────────────────────────────────────────────────────
DEFAULT_CALIB = {
    "y_min_px":   100,
    "y_max_px":   500,
    "h_min_cm":   0.0,
    "h_max_cm":   200.0,
    "camera_id":  "CAM-01",
    "location":   "River Gate A",
    "notes":      "Default calibration — please recalibrate for your site.",
}
