"""
FLOW — Water Level Module
calibration.py: Manages pixel↔real-world height mapping.

The calibration relationship is a simple linear interpolation:

    H = H_min + (Y - Y_min) / (Y_max - Y_min) * (H_max - H_min)

Where:
    Y       = detected waterline pixel row
    H       = calculated real-world water level (cm)
    Y_min   = pixel row corresponding to the lowest gauge mark
    Y_max   = pixel row corresponding to the highest gauge mark
    H_min   = real-world height at Y_min (usually 0 cm)
    H_max   = real-world height at Y_max (e.g. 200 cm)

Note: In image coordinates Y increases downward, so a *lower* physical water
level maps to a *higher* pixel Y value.  The formula handles this naturally as
long as Y_min < Y_max always and calibration points are set correctly.
"""

import json
import os
from typing import Optional, Dict, Tuple
from datetime import datetime

from water_level.config import CALIBRATION_FILE, DEFAULT_CALIB


class WaterLevelCalibration:
    """
    Stores and persists calibration data for one camera view.

    Usage
    -----
    cal = WaterLevelCalibration()
    cal.load()                              # load from JSON (or use defaults)
    cm  = cal.pixel_to_cm(detected_y_px)   # convert waterline pixel → height
    cal.save()                              # persist to JSON
    """

    def __init__(self):
        # Linear calibration points
        self.y_min_px: int   = DEFAULT_CALIB["y_min_px"]
        self.y_max_px: int   = DEFAULT_CALIB["y_max_px"]
        self.h_min_cm: float = DEFAULT_CALIB["h_min_cm"]
        self.h_max_cm: float = DEFAULT_CALIB["h_max_cm"]

        # Metadata
        self.camera_id: str  = DEFAULT_CALIB["camera_id"]
        self.location:  str  = DEFAULT_CALIB["location"]
        self.notes:     str  = DEFAULT_CALIB["notes"]
        self.last_saved: Optional[str] = None

        self._is_calibrated: bool = False   # True once user has confirmed calibration

    # ── Persistence ────────────────────────────────────────────────────────────

    def load(self, path: str = CALIBRATION_FILE) -> bool:
        """
        Load calibration from JSON.  Returns True on success, False on failure
        (the object falls back to DEFAULT_CALIB values).
        """
        if not os.path.exists(path):
            print(f"[WaterLevel] No calibration file found at '{path}' — using defaults.")
            return False
        try:
            with open(path, "r") as f:
                data: Dict = json.load(f)
            self.y_min_px    = int(data.get("y_min_px",   self.y_min_px))
            self.y_max_px    = int(data.get("y_max_px",   self.y_max_px))
            self.h_min_cm    = float(data.get("h_min_cm", self.h_min_cm))
            self.h_max_cm    = float(data.get("h_max_cm", self.h_max_cm))
            self.camera_id   = data.get("camera_id", self.camera_id)
            self.location    = data.get("location",  self.location)
            self.notes       = data.get("notes",     self.notes)
            self.last_saved  = data.get("saved_at",  None)
            self._is_calibrated = True
            print(f"[WaterLevel] Calibration loaded: Y=[{self.y_min_px},{self.y_max_px}] → "
                  f"H=[{self.h_min_cm},{self.h_max_cm}] cm")
            return True
        except Exception as e:
            print(f"[WaterLevel] Failed to load calibration: {e}")
            return False

    def save(self, path: str = CALIBRATION_FILE) -> bool:
        """Persist current calibration to JSON."""
        data = {
            "y_min_px":  self.y_min_px,
            "y_max_px":  self.y_max_px,
            "h_min_cm":  self.h_min_cm,
            "h_max_cm":  self.h_max_cm,
            "camera_id": self.camera_id,
            "location":  self.location,
            "notes":     self.notes,
            "saved_at":  datetime.now().isoformat(),
        }
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.last_saved = data["saved_at"]
            self._is_calibrated = True
            print(f"[WaterLevel] Calibration saved to '{path}'.")
            return True
        except Exception as e:
            print(f"[WaterLevel] Failed to save calibration: {e}")
            return False

    # ── Calibration Update ─────────────────────────────────────────────────────

    def set_points(
        self,
        y_min_px: int, h_min_cm: float,
        y_max_px: int, h_max_cm: float,
    ):
        """
        Define the two calibration points.

        Parameters
        ----------
        y_min_px : Pixel row of the *lowest* visible gauge mark (top of gauge).
        h_min_cm : Real-world height at y_min_px (e.g. 0 cm).
        y_max_px : Pixel row of the *highest* visible gauge mark (bottom of gauge).
        h_max_cm : Real-world height at y_max_px (e.g. 200 cm).
        """
        if y_min_px >= y_max_px:
            raise ValueError(
                f"y_min_px ({y_min_px}) must be less than y_max_px ({y_max_px}). "
                "Remember: image Y increases downward, so the *top* of the gauge "
                "has a smaller Y value."
            )
        self.y_min_px = int(y_min_px)
        self.y_max_px = int(y_max_px)
        self.h_min_cm = float(h_min_cm)
        self.h_max_cm = float(h_max_cm)
        self._is_calibrated = True

    # ── Conversion ─────────────────────────────────────────────────────────────

    def pixel_to_cm(self, y_px: float) -> float:
        """
        Convert a waterline pixel row to real-world height in cm.

        Uses linear interpolation between the two calibration points.
        Values outside the calibration range are extrapolated (clamped to
        ±20 % of the calibration range to avoid wild estimates).
        """
        span_px = self.y_max_px - self.y_min_px
        if span_px == 0:
            return self.h_min_cm

        ratio = (y_px - self.y_min_px) / span_px
        h_cm  = self.h_min_cm + ratio * (self.h_max_cm - self.h_min_cm)

        # Soft clamp — allow ±20 % outside the known range
        margin = abs(self.h_max_cm - self.h_min_cm) * 0.20
        lo = min(self.h_min_cm, self.h_max_cm) - margin
        hi = max(self.h_min_cm, self.h_max_cm) + margin
        return float(max(lo, min(hi, h_cm)))

    def cm_to_pixel(self, h_cm: float) -> int:
        """Inverse: convert a real-world height (cm) back to a pixel row."""
        span_cm = self.h_max_cm - self.h_min_cm
        if span_cm == 0:
            return self.y_min_px
        ratio = (h_cm - self.h_min_cm) / span_cm
        return int(self.y_min_px + ratio * (self.y_max_px - self.y_min_px))

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def summary(self) -> str:
        return (
            f"Y=[{self.y_min_px}px→{self.y_max_px}px] "
            f"H=[{self.h_min_cm:.0f}cm→{self.h_max_cm:.0f}cm] "
            f"loc='{self.location}'"
        )

    def to_dict(self) -> Dict:
        return {
            "y_min_px":     self.y_min_px,
            "y_max_px":     self.y_max_px,
            "h_min_cm":     self.h_min_cm,
            "h_max_cm":     self.h_max_cm,
            "camera_id":    self.camera_id,
            "location":     self.location,
            "notes":        self.notes,
            "is_calibrated": self._is_calibrated,
            "last_saved":   self.last_saved,
        }
