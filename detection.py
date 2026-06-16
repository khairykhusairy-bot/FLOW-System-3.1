"""
FLOW — Flood Level Observation Warning System
Detection Module: YOLO-based debris detection with OpenCV fallback
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os

# Try importing ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from utils import COCO_DEBRIS_MAP

# ─── Debris label sets ─────────────────────────────────────────────────────────
# NOTE: DEBRIS_KEYWORDS is only used for the COCO fallback model path.
# When using best.pt (custom model), all classes from the model are used directly.
DEBRIS_KEYWORDS = {
    "bottle", "plastic_waste", "log", "branch", "trash",
    "river_debris", "cup", "bag", "can", "wrapper"
}


class DebrisDetector:
    """
    Loads a YOLO model (best.pt or fallback to yolov8n) and runs inference
    on frames to detect river debris objects.

    Class names are read directly from whichever model is loaded.
    Adding new classes to best.pt is automatically reflected throughout
    FLOW with no further code changes required.
    """

    def __init__(self, model_path: str = "best.pt", confidence: float = 0.35):
        self.confidence = confidence
        self.model = None
        self.model_type = "none"
        self.class_names: Dict[int, str] = {}
        self.using_demo_mode = False
        self._demo_tick = 0

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Attempt to load YOLO model, with fallback chain."""
        if not ULTRALYTICS_AVAILABLE:
            print("[FLOW] ultralytics not installed — using demo mode.")
            self.using_demo_mode = True
            return

        # Try custom weights first
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.model_type = "custom"
                print(f"[FLOW] Loaded custom model: {model_path}")
                self._init_class_names()
                return
            except Exception as e:
                print(f"[FLOW] Failed to load {model_path}: {e}")

        # Fallback to yolov8n (downloads automatically)
        try:
            self.model = YOLO("yolov8n.pt")
            self.model_type = "coco"
            print("[FLOW] Loaded yolov8n (COCO) fallback model.")
            self._init_class_names()
            return
        except Exception as e:
            print(f"[FLOW] Could not load yolov8n: {e}")

        print("[FLOW] No model available — using demo simulation mode.")
        self.using_demo_mode = True

    def _init_class_names(self):
        """Extract class names from the loaded model."""
        if self.model and hasattr(self.model, "names"):
            self.class_names = dict(self.model.names)
        else:
            self.class_names = {i: f"class_{i}" for i in range(80)}

    def get_class_names(self) -> List[str]:
        """
        Return all class name strings known by the currently loaded model.
        This is the single source of truth for what FLOW can detect and log.
        """
        if self.using_demo_mode:
            return ["bottle", "plastic_waste", "log", "branch", "trash", "river_debris"]
        return [self.class_names[i] for i in sorted(self.class_names.keys())]

    def _map_label(self, raw_label: str) -> str:
        """
        Map a raw model label to its display name.

        - Custom model (best.pt): return the label exactly as the model defines it.
          No remapping — whatever class is in best.pt is what FLOW displays.
        - COCO fallback model: map standard COCO names to debris categories.
        """
        if self.model_type == "custom":
            # Pass through directly — trust the custom model's own class names
            return raw_label

        # COCO fallback path: try to map to known debris category
        label_lower = raw_label.lower().replace(" ", "_")
        mapped = COCO_DEBRIS_MAP.get(raw_label.lower(), None)
        if mapped:
            return mapped
        for kw in DEBRIS_KEYWORDS:
            if kw in label_lower:
                return kw
        return label_lower

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run inference on a frame and return detection results.

        Returns list of dicts:
            {
                "bbox": (x1, y1, x2, y2),
                "label": str,
                "confidence": float,
                "class_id": int,
            }
        """
        if self.using_demo_mode:
            return self._generate_demo_detections(frame)

        if self.model is None:
            return []

        try:
            results = self.model(
                frame,
                conf=self.confidence,
                verbose=False,
                stream=False,
            )
        except Exception as e:
            print(f"[FLOW] Inference error: {e}")
            return []

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                try:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    raw_label = self.class_names.get(cls_id, f"class_{cls_id}")
                    label = self._map_label(raw_label)

                    # Custom model: keep all detections (all classes are valid debris)
                    # COCO model: filter to only debris-mapped objects
                    if self.model_type == "coco":
                        if raw_label.lower() not in COCO_DEBRIS_MAP:
                            continue

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "label": label,
                        "confidence": conf,
                        "class_id": cls_id,
                    })
                except Exception:
                    continue

        return detections

    def _generate_demo_detections(self, frame: np.ndarray) -> List[Dict]:
        """
        Generate realistic simulated detections for demo / no-model mode.
        Uses whatever class names are defined in best.pt when available,
        otherwise uses a built-in fallback list.
        """
        h, w = frame.shape[:2]
        self._demo_tick += 1
        t = self._demo_tick

        np.random.seed(t // 8)  # Change every ~8 frames for smooth motion

        # Use real model class names if loaded, otherwise a safe fallback list
        if self.class_names:
            demo_classes = list(self.class_names.values())
        else:
            demo_classes = [
                "bottle", "plastic_waste", "log",
                "branch", "trash", "river_debris"
            ]

        detections = []
        n = np.random.randint(3, 10)

        for i in range(n):
            base_x = int((w * (i + 1)) / (n + 1))
            base_y = int(h * 0.35 + h * 0.3 * np.random.rand())
            dx = int(30 * np.sin(t * 0.05 + i * 1.2))
            dy = int(10 * np.cos(t * 0.08 + i * 0.8))

            x1 = max(0, base_x + dx - np.random.randint(20, 60))
            y1 = max(0, base_y + dy - np.random.randint(15, 40))
            x2 = min(w - 1, x1 + np.random.randint(40, 100))
            y2 = min(h - 1, y1 + np.random.randint(30, 70))

            label = demo_classes[i % len(demo_classes)]
            conf = round(0.55 + 0.40 * np.random.rand(), 3)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "confidence": conf,
                "class_id": i,
            })

        return detections

    def set_confidence(self, confidence: float):
        self.confidence = max(0.1, min(0.99, confidence))

    @property
    def status(self) -> str:
        if self.using_demo_mode:
            return "Demo Mode (No Model)"
        return f"YOLO ({self.model_type})"
