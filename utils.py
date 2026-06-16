"""
FLOW — Flood Level Observation Warning System
Utilities Module: Helper functions and constants
"""

import cv2
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Dict

# Malaysia timezone (UTC+8)
MYT = timezone(timedelta(hours=8))


# ─── Dynamic YOLO Class Loader ────────────────────────────────────────────────
def load_model_classes(model_path: str = "best.pt") -> Dict[int, str]:
    """
    Read class names directly from a YOLO .pt model file.
    Returns a dict of {class_id: class_name} from the model.
    Falls back to an empty dict if the model cannot be loaded.
    This means adding new classes to best.pt automatically reflects
    everywhere in FLOW without any code changes.
    """
    import os
    if not os.path.exists(model_path):
        return {}
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        if hasattr(model, "names") and model.names:
            return dict(model.names)
    except Exception as e:
        print(f"[FLOW] load_model_classes: could not read {model_path}: {e}")
    return {}


def get_model_class_names(model_path: str = "best.pt") -> List[str]:
    """Return a sorted list of class name strings from best.pt."""
    classes = load_model_classes(model_path)
    return [classes[i] for i in sorted(classes.keys())]


# ─── YOLO Class Labels (dynamically populated from best.pt) ──────────────────
# This dict is populated at import time. Any new class added to best.pt is
# automatically included — no manual edits required.
DEBRIS_CLASSES: Dict[int, str] = load_model_classes("best.pt")

# Fallback COCO labels for when using standard YOLO weights
COCO_DEBRIS_MAP = {
    "bottle": "bottle",
    "cup": "plastic_waste",
    "vase": "trash",
    "sports ball": "trash",
    "frisbee": "plastic_waste",
    "handbag": "trash",
    "backpack": "trash",
    "umbrella": "trash",
    "suitcase": "trash",
    "chair": "log",
    "couch": "log",
    "potted plant": "branch",
    "dining table": "log",
    "book": "trash",
    "scissors": "trash",
    "teddy bear": "trash",
    "toothbrush": "plastic_waste",
    "fork": "trash",
    "knife": "trash",
    "spoon": "trash",
    "bowl": "plastic_waste",
}

# ─── Color Palette ─────────────────────────────────────────────────────────────
COLORS = {
    "roi_fill":    (0, 255, 255, 40),    # RGBA cyan fill
    "roi_border":  (0, 255, 255, 255),   # Cyan border
    "bbox_inside": (0, 255, 120),         # Green for inside ROI
    "bbox_outside":(128, 128, 128),       # Grey for outside ROI
    "text_bg":     (0, 0, 0),
    "alert_red":   (0, 60, 255),
    "alert_yellow":(0, 200, 255),
}

# ─── Risk Thresholds ───────────────────────────────────────────────────────────
BLOCKAGE_THRESHOLDS = {
    "low":    30.0,
    "medium": 60.0,
    "high":   80.0,
}

ALERT_THRESHOLDS = {
    "blockage_warning": 50.0,
    "blockage_critical": 75.0,
    "rain_warning": 0.6,
    "rain_critical": 0.85,
    "roi_count_warning": 10,
    "roi_count_critical": 20,
}


def get_timestamp() -> str:
    """Return current time in Malaysia timezone (UTC+8)."""
    return datetime.now(MYT).strftime("%Y-%m-%d %H:%M:%S")


def draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
    text_color: Tuple = (255, 255, 255),
    bg_color: Tuple = (0, 0, 0),
    padding: int = 4,
):
    """Draw text with a background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def resize_frame(frame: np.ndarray, width: int = 960) -> np.ndarray:
    """Resize frame maintaining aspect ratio."""
    h, w = frame.shape[:2]
    ratio = width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_LINEAR)


def overlay_polygon(
    frame: np.ndarray,
    polygon: List[Tuple[int, int]],
    fill_color: Tuple = (0, 255, 255),
    border_color: Tuple = (0, 255, 255),
    alpha: float = 0.15,
    border_thickness: int = 2,
) -> np.ndarray:
    """Draw a semi-transparent polygon overlay on the frame."""
    if len(polygon) < 3:
        return frame
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], fill_color)
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.polylines(result, [pts], True, border_color, border_thickness, cv2.LINE_AA)
    # Draw corner dots
    for pt in polygon:
        cv2.circle(result, pt, 5, border_color, -1)
        cv2.circle(result, pt, 7, (255, 255, 255), 1)
    return result


def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """Check if a point is inside a polygon using OpenCV."""
    if len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.float32)
    result = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return result >= 0


def polygon_area(polygon: List[Tuple[int, int]]) -> float:
    """Calculate polygon area using the Shoelace formula."""
    if len(polygon) < 3:
        return 1.0
    pts = np.array(polygon, dtype=np.float64)
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def bbox_intersection_with_polygon(
    bbox: Tuple[int, int, int, int],
    polygon: List[Tuple[int, int]],
    frame_shape: Tuple,
) -> float:
    x1, y1, x2, y2 = bbox
    if len(polygon) < 3 or x2 <= x1 or y2 <= y1:
        return 0.0

    fh = frame_shape[0] if len(frame_shape) >= 1 else (y2 + 1)
    fw = frame_shape[1] if len(frame_shape) >= 2 else (x2 + 1)

    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    shifted = np.array(
        [(p[0] - x1, p[1] - y1) for p in polygon], dtype=np.int32
    )
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted], 255)
    return float(np.count_nonzero(mask))


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def risk_level_to_color(risk: str) -> Tuple[int, int, int]:
    """Return BGR color for a risk level string."""
    mapping = {
        "Low Risk":    (80, 200, 80),
        "Medium Risk": (0, 165, 255),
        "High Risk":   (0, 60, 240),
    }
    return mapping.get(risk, (200, 200, 200))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to [0, 1]."""
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def format_percentage(value: float) -> str:
    return f"{value:.1f}%"


def clamp(value, lo, hi):
    return max(lo, min(hi, value))
