"""
FLOW — Flood Level Observation Warning System
config.py: Central configuration file

Edit values here, or let setup_polygon.py update ROI_POLYGON automatically.
"""

import os

# ─── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH  = os.path.abspath(__file__)   # This file's own path (used by setup_polygon.py)
MODEL_PATH   = "best.pt"                   # Custom YOLO weights; falls back to yolov8n if missing

# ─── Camera ────────────────────────────────────────────────────────────────────
WEBCAM_INDEX = 0                           # Camera index for setup_polygon.py and main.py

# ─── Polygon ROI ───────────────────────────────────────────────────────────────
# Set by setup_polygon.py — do not edit this line manually unless you know the coords.
# Format: list of (x, y) integer tuples. Needs >= 3 points.
ROI_POLYGON  = [(94, 334), (10, 534), (950, 533), (950, 338)]

# ─── Detection Defaults ────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE  = 0.35
DEFAULT_PRESET      = "center_river"       # Used only when ROI_POLYGON is empty

# ─── Alert Defaults ────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECS     = 12.0
BLOCKAGE_WARN_THRESHOLD = 50              # %
ROI_COUNT_WARN          = 10

# ─── Logging ───────────────────────────────────────────────────────────────────
DB_PATH      = "flow_monitoring.db"
LOG_INTERVAL = 5                           # Seconds between DB writes

# ─── Weather Location Presets ──────────────────────────────────────────────────
# Each entry: "Display Name" → (latitude, longitude)
# Used by the sidebar location picker in weather.py / main.py.
# Add or remove entries freely; the UI will reflect changes automatically.
WEATHER_LOCATIONS = {
    "📍 Custom Location":          None,                      # triggers lat/lon inputs
    "Kangar, Perlis":              (6.1248,  100.3673),
    "Alor Setar, Kedah":           (6.1248,  100.3673),
    "Sungai Petani, Kedah":        (5.6479,  100.4880),
    "Georgetown, Penang":          (5.4141,  100.3288),
    "Ipoh, Perak":                 (4.5975,  101.0901),
    "Kuala Lumpur":                (3.1390,  101.6869),
    "Putrajaya":                   (2.9264,  101.6964),
    "Shah Alam, Selangor":         (3.0733,  101.5185),
    "Seremban, Negeri Sembilan":   (2.7297,  101.9381),
    "Melaka":                      (2.1896,  102.2501),
    "Johor Bahru":                 (1.4927,  103.7414),
    "Kuantan, Pahang":             (3.8077,  103.3260),
    "Kota Bharu, Kelantan":        (6.1254,  102.2381),
    "Kuala Terengganu":            (5.3296,  103.1370),
    "Kuching, Sarawak":            (1.5535,  110.3593),
    "Kota Kinabalu, Sabah":        (5.9804,  116.0735),
}

# ─── Weather API Keys ────────────────────────────────────────────────────────
# Google Maps Platform Weather API
GOOGLE_WEATHER_API_KEY = "AIzaSyAlrDArY-Dy7Kh-HIf-_jnPcctACqsvIxE"

# OpenWeatherMap API (3-hour forecast intervals)
OWM_API_KEY = "a7bef62f4dc463f7af92b1256165e1c8"

# Default weather provider: "google" or "openweathermap"
WEATHER_PROVIDER = "google"
