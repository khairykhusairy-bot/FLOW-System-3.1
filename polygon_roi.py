"""
FLOW — Flood Level Observation Warning System
Polygon ROI Module: Custom polygon region-of-interest management

Changes vs original:
  • __init__ now accepts load_from_config=True (default).
    If config.ROI_POLYGON is non-empty, it is used instead of the
    built-in DEFAULT_POLYGON, so the polygon set by setup_polygon.py
    is automatically honoured on every run of main.py.
  • Added classmethod PolygonROI.from_config() as an explicit factory.
  • Everything else is unchanged.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from utils import (
    overlay_polygon, point_in_polygon, polygon_area,
    bbox_intersection_with_polygon, bbox_center,
)


def _load_config_polygon() -> List[Tuple[int, int]]:
    """
    Try to import ROI_POLYGON from config.py.
    Returns the list if it is a valid polygon (>= 3 points), else [].
    """
    try:
        import config as _cfg
        poly = getattr(_cfg, "ROI_POLYGON", [])
        if isinstance(poly, (list, tuple)) and len(poly) >= 3:
            # Normalise to list-of-tuples
            return [tuple(int(v) for v in p) for p in poly]
    except ImportError:
        pass
    return []


class PolygonROI:
    """
    Manages a custom polygon region of interest for river monitoring.
    Handles polygon definition, object filtering, and blockage calculation.

    Initialisation priority (highest → lowest):
        1. Explicit `polygon` argument passed to __init__
        2. ROI_POLYGON from config.py  (when load_from_config=True, the default)
        3. Built-in DEFAULT_POLYGON
    """

    # Default polygon — covers centre region of a 960×540 frame
    DEFAULT_POLYGON: List[Tuple[int, int]] = [
        (180, 120), (780, 120),
        (860, 420), (100, 420),
    ]

    def __init__(
        self,
        polygon: Optional[List[Tuple[int, int]]] = None,
        load_from_config: bool = True,
    ):
        """
        Parameters
        ----------
        polygon          : Explicit polygon to use; takes highest priority.
        load_from_config : When True (default), fall back to config.ROI_POLYGON
                           if `polygon` is not supplied.
        """
        if polygon is not None and len(polygon) >= 3:
            chosen = polygon
            source = "explicit argument"
        elif load_from_config:
            cfg_poly = _load_config_polygon()
            if cfg_poly:
                chosen = cfg_poly
                source = "config.py"
            else:
                chosen = self.DEFAULT_POLYGON.copy()
                source = "built-in default"
        else:
            chosen = self.DEFAULT_POLYGON.copy()
            source = "built-in default"

        # When config polygon is empty and no explicit polygon given, start with NO polygon
        # (monitoring runs without ROI overlay until user draws one)
        if chosen is self.DEFAULT_POLYGON and source == "built-in default":
            chosen = []
            source = "none (draw via sidebar)"
        print(f"[PolygonROI] Polygon state: {source}.")
        self.polygon: List[Tuple[int, int]] = chosen
        self._area: float = polygon_area(self.polygon) if chosen else 1.0
        self.is_drawing: bool = False
        self._temp_polygon: List[Tuple[int, int]] = []

    # ─── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls) -> "PolygonROI":
        """Convenience factory: always reads polygon from config.py."""
        return cls(polygon=None, load_from_config=True)

    # ─── Polygon Management ───────────────────────────────────────────────────

    def set_polygon(self, polygon: List[Tuple[int, int]]):
        """Set a new polygon and recalculate area."""
        if len(polygon) >= 3:
            self.polygon = [tuple(int(v) for v in p) for p in polygon]
            self._area = polygon_area(self.polygon)

    def reload_from_config(self) -> bool:
        """
        Hot-reload the polygon from config.py at runtime (e.g. after
        setup_polygon.py has just saved a new one).

        Returns True if a valid polygon was found and applied.
        """
        cfg_poly = _load_config_polygon()
        if cfg_poly:
            self.set_polygon(cfg_poly)
            print(f"[PolygonROI] Reloaded {len(cfg_poly)}-point polygon from config.py.")
            return True
        print("[PolygonROI] No valid ROI_POLYGON found in config.py — polygon unchanged.")
        return False

    def reset_to_default(self):
        """Reset polygon to the built-in default river zone."""
        self.polygon = self.DEFAULT_POLYGON.copy()
        self._area = polygon_area(self.polygon)

    def get_area(self) -> float:
        """Return polygon area in pixels²."""
        return max(self._area, 1.0)

    def get_polygon(self) -> List[Tuple[int, int]]:
        return self.polygon.copy()

    def add_preset(self, preset_name: str, frame_w: int, frame_h: int):
        """Apply a named preset polygon scaled to frame dimensions."""
        presets = {
            "full_frame": [
                (10, 10), (frame_w - 10, 10),
                (frame_w - 10, frame_h - 10), (10, frame_h - 10),
            ],
            "center_river": [
                (int(frame_w * 0.15), int(frame_h * 0.20)),
                (int(frame_w * 0.85), int(frame_h * 0.20)),
                (int(frame_w * 0.92), int(frame_h * 0.82)),
                (int(frame_w * 0.08), int(frame_h * 0.82)),
            ],
            "narrow_channel": [
                (int(frame_w * 0.30), int(frame_h * 0.15)),
                (int(frame_w * 0.70), int(frame_h * 0.15)),
                (int(frame_w * 0.72), int(frame_h * 0.85)),
                (int(frame_w * 0.28), int(frame_h * 0.85)),
            ],
            "wide_river": [
                (int(frame_w * 0.05), int(frame_h * 0.25)),
                (int(frame_w * 0.95), int(frame_h * 0.25)),
                (int(frame_w * 0.95), int(frame_h * 0.75)),
                (int(frame_w * 0.05), int(frame_h * 0.75)),
            ],
        }
        if preset_name in presets:
            self.set_polygon(presets[preset_name])

    # ─── Object Classification ────────────────────────────────────────────────

    def has_polygon(self) -> bool:
        """Return True if a valid polygon (≥3 points) is set."""
        return len(self.polygon) >= 3

    def classify_detections(
        self,
        detections: List[Dict],
        frame_shape: Tuple,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split detections into inside-ROI and outside-ROI groups.
        When no polygon is set, all detections are treated as "inside".

        Returns:
            (inside_detections, outside_detections)
        """
        # No polygon set — treat everything as inside
        if not self.has_polygon():
            return [{**det, "in_roi": True} for det in detections], []

        inside, outside = [], []
        for det in detections:
            bbox   = det.get("bbox", (0, 0, 0, 0))
            center = bbox_center(bbox)
            if point_in_polygon(center, self.polygon):
                inside.append({**det, "in_roi": True})
            else:
                outside.append({**det, "in_roi": False})
        return inside, outside

    # ─── Blockage Calculation ─────────────────────────────────────────────────

    def calculate_blockage(
        self,
        inside_detections: List[Dict],
        frame_shape: Tuple,
    ) -> float:
        """
        Blockage % = (trash area inside ROI) / (polygon area) × 100.
        Uses sampling-based bbox/polygon intersection for accuracy.
        """
        total_trash_area = 0.0
        for det in inside_detections:
            bbox = det.get("bbox", (0, 0, 0, 0))
            total_trash_area += bbox_intersection_with_polygon(
                bbox, self.polygon, frame_shape
            )
        return round(min(100.0, (total_trash_area / self.get_area()) * 100.0), 2)

    # ─── Object Counting ──────────────────────────────────────────────────────

    def count_by_class(self, inside_detections: List[Dict]) -> Dict[str, int]:
        """Count objects by class label within the ROI."""
        counts: Dict[str, int] = {}
        for det in inside_detections:
            label = det.get("label", "unknown")
            counts[label] = counts.get(label, 0) + 1
        return counts

    # ─── Rendering ────────────────────────────────────────────────────────────

    def draw_on_frame(
        self,
        frame: np.ndarray,
        inside_detections: List[Dict],
        outside_detections: List[Dict],
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        Render polygon overlay and bounding boxes on the frame.
        Inside objects: bright, labelled.
        Outside objects: dimmed, still visible.
        """
        result = frame.copy()

        # No polygon — just draw detections without overlay
        if not self.has_polygon():
            all_dets = list(inside_detections) + list(outside_detections)
            for det in all_dets:
                x1, y1, x2, y2 = det["bbox"]
                color = self._label_color(det.get("label", ""))
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                if show_labels:
                    label = det.get("label", "")
                    conf  = det.get("confidence", 0)
                    text  = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(result, text, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            return result

        # Semi-transparent polygon fill + border
        result = overlay_polygon(
            result,
            self.polygon,
            fill_color=(0, 255, 255),
            border_color=(0, 255, 255),
            alpha=0.12,
            border_thickness=2,
        )

        # Outside detections (dimmed)
        for det in outside_detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (100, 100, 100), 1)
            if show_labels:
                label = det.get("label", "")
                conf  = det.get("confidence", 0)
                cv2.putText(
                    result, f"{label} {conf:.2f}",
                    (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (120, 120, 120), 1, cv2.LINE_AA,
                )

        # Inside detections (highlighted)
        for det in inside_detections:
            x1, y1, x2, y2 = det["bbox"]
            color = self._label_color(det.get("label", ""))
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                label = det.get("label", "")
                conf  = det.get("confidence", 0)
                text  = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                )
                cv2.rectangle(
                    result, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
                )
                cv2.putText(
                    result, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA,
                )

        # Corner indices
        for i, pt in enumerate(self.polygon):
            cv2.putText(
                result, str(i + 1), (pt[0] + 6, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA,
            )

        return result

    def _label_color(self, label: str) -> Tuple[int, int, int]:
        """Return distinct BGR colour for each debris class."""
        palette = {
            "bottle":        (0,  220, 120),
            "plastic_waste": (0,  180, 255),
            "log":           (60, 130, 255),
            "branch":        (0,  200, 200),
            "trash":         (0,  100, 255),
            "river_debris":  (180, 80, 255),
        }
        return palette.get(label.lower(), (0, 255, 180))
