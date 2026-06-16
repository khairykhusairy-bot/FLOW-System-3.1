"""
FLOW — Flood Level Observation Warning System
Flood Risk Engine: Weather-driven scoring system

Implements three layered risk calculations:

  Layer 1 — Rainfall Category (mm/h → Very Low … Critical)
            Works 24/7, even before START is clicked.

  Layer 2 — Weighted Flood Risk Score (0–100)
            Each input is first normalised to [0, 1] using physically motivated
            maximums, then weighted and scaled to 0–100:
              Score = 100 × (0.5 × norm_rain + 0.3 × norm_hours + 0.2 × norm_prev)
            Maps to: 0–25 Low | 26–50 Moderate | 51–75 High | >75 Severe

  Layer 3 — Integrated Flood Probability (0–1, only when monitoring is active)
            Rainfall is the primary driver; water level and blockage are amplifiers.
            P = (0.6 × RainfallRisk) + (0.2 × WaterLevelRisk) + (0.2 × BlockageRisk)
            Fused with the rule-based predictor output for the final "Flood Risk" label.
"""

import time
from collections import deque
from typing import Dict, List, Optional, Tuple


# ─── Constants ─────────────────────────────────────────────────────────────────

# Rainfall intensity thresholds (mm/h)
RAIN_THRESHOLDS = {
    "very_low":  5.0,    # < 5      → Very Low
    "low":      15.0,    # 5–15     → Low
    "moderate": 25.0,    # 15–25    → Moderate
    # > 25                           → High  (aligns with normalised scale max of 25 mm/h = 1.0)
}

# Continuous rain duration thresholds (hours)
CONTINUOUS_RAIN_BOOST_HOURS   = 3.0   # > 3 h  continuous → increase risk tier
CONTINUOUS_RAIN_SEVERE_HOURS  = 5.0   # > 5 h  continuous heavy → Very High

# Total accumulation threshold
ACCUMULATED_RAIN_CRITICAL_MM  = 80.0  # > 80 mm in 24 h → Critical

# Normalisation maximums for Layer 2 inputs (physically motivated)
_RAIN_INTENSITY_MAX  = 25.0   # mm/h — matches the normalised scale ceiling (25 mm/h = 1.0)
_CONTINUOUS_HOURS_MAX = 6.0   # hours — prolonged storm event
_PREV_RAINFALL_MAX   = 80.0   # mm — critical 24-h accumulation threshold

# Weighted score category thresholds (0–100 scale)
SCORE_LOW      = 25.0
SCORE_MODERATE = 50.0
SCORE_HIGH     = 75.0
# > 75 → Severe

# Human-readable labels
RAIN_CATEGORY_LABELS = {
    "very_low":  "Very Low",
    "low":       "Low",
    "moderate":  "Moderate",
    "high":      "High",
    "very_high": "Very High",
    "critical":  "Critical",
}

SCORE_RISK_LABELS = ["Low", "Moderate", "High", "Severe"]
SCORE_RISK_COLORS = {
    "Low":      "#2ecc71",
    "Moderate": "#f39c12",
    "High":     "#e74c3c",
    "Severe":   "#9b59b6",
}

# Colour palette for rainfall categories
RAIN_CATEGORY_COLORS = {
    "very_low":  "#2ecc71",
    "low":       "#27ae60",
    "moderate":  "#f39c12",
    "high":      "#e74c3c",
    "very_high": "#c0392b",
    "critical":  "#9b59b6",
}

# How often (seconds) to sample weather into the rolling window
_SAMPLE_INTERVAL = 60.0   # 1 minute


class RainfallTracker:
    """
    Tracks rolling rainfall metrics used by FloodRiskEngine:

      • current_mm_h       : latest reading (mm/h)
      • continuous_hours   : how many consecutive hours it has rained
      • accumulated_mm_24h : total mm fallen in the last 24 hours
      • is_raining         : True when current_mm_h > 0.5 mm/h
    """

    _24H_SECONDS = 24 * 3600

    def __init__(self):
        # Each entry: (timestamp, mm_h)
        self._samples: deque = deque()
        self._rain_start: Optional[float] = None   # epoch when continuous rain began
        self.current_mm_h: float = 0.0
        self.continuous_hours: float = 0.0
        self.accumulated_mm_24h: float = 0.0
        self.is_raining: bool = False
        self._last_sample_ts: float = 0.0

    def update(self, mm_h: float):
        """
        Call once per weather poll (every ~5 min) with the live mm/h value.
        Updates all rolling metrics.
        """
        now = time.time()
        self.current_mm_h = max(0.0, mm_h)
        self.is_raining = self.current_mm_h >= 0.5

        # Add a sample only if at least _SAMPLE_INTERVAL has passed
        if now - self._last_sample_ts >= _SAMPLE_INTERVAL:
            self._samples.append((now, self.current_mm_h))
            self._last_sample_ts = now

        # Prune samples older than 24 h
        cutoff = now - self._24H_SECONDS
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        # Accumulated rainfall: integrate mm/h × (interval / 3600)
        acc = 0.0
        prev_ts = None
        for ts, rate in self._samples:
            if prev_ts is not None:
                dt_h = (ts - prev_ts) / 3600.0
                acc += rate * dt_h
            prev_ts = ts
        self.accumulated_mm_24h = round(acc, 2)

        # Continuous rain hours
        if self.is_raining:
            if self._rain_start is None:
                self._rain_start = now
            self.continuous_hours = (now - self._rain_start) / 3600.0
        else:
            self._rain_start = None
            self.continuous_hours = 0.0

    def as_dict(self) -> Dict:
        return {
            "current_mm_h":        round(self.current_mm_h, 2),
            "continuous_hours":    round(self.continuous_hours, 2),
            "accumulated_mm_24h":  self.accumulated_mm_24h,
            "is_raining":          self.is_raining,
        }


def _mm_h_to_intensity_norm(mm_h: float) -> float:
    """
    Convert mm/h to FLOW's normalised 0-1 rain_intensity.
    Mirrors the formula in weather.py (_rain_to_intensity).
    Scale: 0 mm/h → 0.0, 25+ mm/h → 1.0.
    """
    return round(min(1.0, mm_h / 25.0), 4)


class FloodRiskEngine:
    """
    Weather-driven flood risk scoring engine.

    Usage
    -----
    Instantiate once (e.g. via st.cache_resource).
    Call `update_weather(mm_h)` every time fresh weather data arrives.
    Call `get_weather_risk()` at any time (idle or monitoring) to get the
    rainfall-only risk dict.
    Call `get_integrated_risk(water_level_norm, blockage_pct)` when
    monitoring is active to get the fully fused probability.
    """

    def __init__(self):
        self.tracker = RainfallTracker()
        self._last_weather_update: float = 0.0

        # Forecast accumulation estimate (set externally from OWM forecast)
        self._forecast_mm_next6h: float = 0.0

    # ─── Weather Update ───────────────────────────────────────────────────────

    def update_weather(self, mm_h: float, forecast_mm_next6h: float = 0.0):
        """
        Feed fresh live weather data into the engine.
        Call whenever WeatherService.get_current() returns a new value.

        Parameters
        ----------
        mm_h                : Current rainfall in mm/h (from OWM rain_mm).
        forecast_mm_next6h  : Projected total mm over next 6 h (from forecast).
                              Used to improve 24-h accumulation estimates.
                              Pass 0 if unavailable.
        """
        self.tracker.update(mm_h)
        self._forecast_mm_next6h = max(0.0, forecast_mm_next6h)
        self._last_weather_update = time.time()

    # ─── Layer 1: Rainfall Category ──────────────────────────────────────────

    def rainfall_category(self) -> Dict:
        """
        Map current mm/h + duration/accumulation to a risk category.

        Returns dict:
            {
              "key":        str   (e.g. "moderate"),
              "label":      str   (e.g. "Moderate"),
              "color":      str   (hex),
              "mm_h":       float,
              "continuous_hours": float,
              "accumulated_mm_24h": float,
              "upgrade_reason":  str | None,
            }
        """
        t = self.tracker
        mm_h = t.current_mm_h
        hours = t.continuous_hours
        acc = t.accumulated_mm_24h

        # Base category from instantaneous rate
        if mm_h < RAIN_THRESHOLDS["very_low"]:
            key = "very_low"
        elif mm_h < RAIN_THRESHOLDS["low"]:
            key = "low"
        elif mm_h < RAIN_THRESHOLDS["moderate"]:
            key = "moderate"
        else:
            key = "high"

        upgrade_reason: Optional[str] = None

        # Rule: continuous rain > 3 h → bump one tier
        if hours > CONTINUOUS_RAIN_BOOST_HOURS and key in ("very_low", "low", "moderate"):
            _tier_order = ["very_low", "low", "moderate", "high", "very_high", "critical"]
            idx = _tier_order.index(key)
            key = _tier_order[min(idx + 1, len(_tier_order) - 1)]
            upgrade_reason = f"Continuous rain >{CONTINUOUS_RAIN_BOOST_HOURS:.0f} h"

        # Rule: continuous heavy rain > 5 h → Very High
        if hours > CONTINUOUS_RAIN_SEVERE_HOURS and mm_h >= RAIN_THRESHOLDS["moderate"]:
            key = "very_high"
            upgrade_reason = f"Heavy rain >{CONTINUOUS_RAIN_SEVERE_HOURS:.0f} h continuous"

        # Rule: total accumulation > 80 mm/24 h → Critical
        if acc >= ACCUMULATED_RAIN_CRITICAL_MM:
            key = "critical"
            upgrade_reason = f"Total rainfall {acc:.1f} mm > 80 mm/24 h"

        return {
            "key":                 key,
            "label":               RAIN_CATEGORY_LABELS[key],
            "color":               RAIN_CATEGORY_COLORS[key],
            "mm_h":                round(mm_h, 2),
            "continuous_hours":    round(hours, 2),
            "accumulated_mm_24h":  round(acc, 2),
            "upgrade_reason":      upgrade_reason,
        }

    # ─── Layer 2: Weighted Flood Risk Score ──────────────────────────────────

    def flood_risk_score(self, previous_rainfall_mm: Optional[float] = None) -> Dict:
        """
        Compute the weighted flood risk score (0–100).

        Each input is normalised to [0, 1] before weighting so that all three
        terms are dimensionless and comparably scaled:
            norm_rain  = min(1, mm_h  / 25)    (25 mm/h = extreme tropical rain)
            norm_hours = min(1, hours / 6)     (6 h = prolonged event)
            norm_prev  = min(1, prev_mm / 80)  (80 mm = critical accumulation)
            Score = 100 × (0.5 × norm_rain + 0.3 × norm_hours + 0.2 × norm_prev)

        Parameters
        ----------
        previous_rainfall_mm : mm fallen in the period *before* the current event.
                               Defaults to the last-24h accumulation when None.

        Returns dict:
            {
              "score":     float  (0–100),
              "category":  str  ("Low" | "Moderate" | "High" | "Severe"),
              "color":     str,
              "breakdown": { "rainfall_term", "hours_term", "prev_rain_term" }
            }
        """
        t = self.tracker
        ri = t.current_mm_h
        ch = t.continuous_hours
        pr = previous_rainfall_mm if previous_rainfall_mm is not None else t.accumulated_mm_24h

        # Normalise each input to [0, 1] before applying weights
        rainfall_norm  = min(1.0, ri / _RAIN_INTENSITY_MAX)
        hours_norm     = min(1.0, ch / _CONTINUOUS_HOURS_MAX)
        prev_rain_norm = min(1.0, pr / _PREV_RAINFALL_MAX)

        rainfall_term  = 0.5 * rainfall_norm
        hours_term     = 0.3 * hours_norm
        prev_rain_term = 0.2 * prev_rain_norm
        score = 100.0 * (rainfall_term + hours_term + prev_rain_term)

        if score <= SCORE_LOW:
            cat = "Low"
        elif score <= SCORE_MODERATE:
            cat = "Moderate"
        elif score <= SCORE_HIGH:
            cat = "High"
        else:
            cat = "Severe"

        return {
            "score":    round(score, 2),
            "category": cat,
            "color":    SCORE_RISK_COLORS[cat],
            "breakdown": {
                "rainfall_norm":   round(rainfall_norm, 4),
                "hours_norm":      round(hours_norm, 4),
                "prev_rain_norm":  round(prev_rain_norm, 4),
                "rainfall_term":   round(rainfall_term, 4),
                "hours_term":      round(hours_term, 4),
                "prev_rain_term":  round(prev_rain_term, 4),
            },
        }

    # ─── Layer 3: Integrated Flood Probability (monitoring active) ───────────

    def integrated_flood_probability(
        self,
        water_level_norm: float,
        blockage_pct: float,
        camera_rain_norm: float = 0.0,
    ) -> Dict:
        """
        Compute the integrated flood probability when monitoring is active:
            P = (0.6 × RainfallRisk) + (0.2 × WaterLevelRisk) + (0.2 × BlockageRisk)

        Rainfall is the primary driver (60%) because it is the root cause of
        flooding.  Water level and blockage are amplifying factors (20% each).
        Giving water level equal weight to rainfall would double-count the same
        storm event (rainfall → water level rise), inflating the probability.

        All sub-risks are normalised to [0, 1].

        Parameters
        ----------
        water_level_norm : Water level as a 0-1 fraction of the calibrated max.
        blockage_pct     : Channel blockage percentage (0–100).
        camera_rain_norm : Camera-detected rain intensity (0–1). When OWM
                           underreports local rain, this prevents the score from
                           being anchored to the lower API reading.

        Returns dict:
            {
              "probability":       float  (0–1),
              "risk_label":        str    ("Low Risk" | "Medium Risk" | "High Risk"),
              "color":             str,
              "rainfall_risk":     float,
              "water_level_risk":  float,
              "blockage_risk":     float,
            }
        """
        # Rainfall risk: take the higher of OWM-derived score and camera detection.
        # OWM frequently underreports hyperlocal rain; using max() ensures the
        # engine reflects what the camera actually observes.
        score_dict = self.flood_risk_score()
        owm_rainfall_risk = min(1.0, score_dict["score"] / 100.0)
        rainfall_risk = max(owm_rainfall_risk, min(1.0, float(camera_rain_norm)))

        # Water level risk: already 0-1
        water_level_risk = max(0.0, min(1.0, float(water_level_norm)))

        # Blockage risk: 0–100 → 0–1
        blockage_risk = max(0.0, min(1.0, float(blockage_pct) / 100.0))

        probability = (
            0.6 * rainfall_risk
            + 0.2 * water_level_risk
            + 0.2 * blockage_risk
        )
        probability = round(min(1.0, probability), 4)

        # Map probability to risk label
        if probability < 0.30:
            label = "Low Risk"
            color = "#2ecc71"
        elif probability < 0.60:
            label = "Medium Risk"
            color = "#f39c12"
        else:
            label = "High Risk"
            color = "#e74c3c"

        return {
            "probability":      probability,
            "risk_label":       label,
            "color":            color,
            "rainfall_risk":    round(rainfall_risk, 4),
            "water_level_risk": round(water_level_risk, 4),
            "blockage_risk":    round(blockage_risk, 4),
        }

    # ─── Convenience: full weather-only risk summary (idle or active) ─────────

    def get_weather_risk(self) -> Dict:
        """
        Returns a consolidated weather-driven risk summary suitable for
        display in the idle state (before START is clicked).

        Keys:
            rainfall_category  : dict from rainfall_category()
            score_result       : dict from flood_risk_score()
            tracker_snapshot   : dict from RainfallTracker.as_dict()
            last_updated       : float epoch
        """
        return {
            "rainfall_category": self.rainfall_category(),
            "score_result":      self.flood_risk_score(),
            "tracker_snapshot":  self.tracker.as_dict(),
            "last_updated":      self._last_weather_update,
        }

    # ─── Forecast helper ──────────────────────────────────────────────────────

    @staticmethod
    def estimate_forecast_accumulation(forecast_entries: List[Dict]) -> float:
        """
        Sum the rain_mm values from OWM forecast entries (each is 3-h rain in mm)
        for approximately the next 6 h (2 entries × 3 h each).

        Call with WeatherService.get_forecast(hours=6).
        Returns total expected mm over the next 6 hours.
        """
        total = 0.0
        for entry in forecast_entries[:2]:   # first 2 × 3-h steps ≈ 6 h
            total += float(entry.get("rain_mm", 0.0))
        return round(total, 2)
