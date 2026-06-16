"""
FLOW — Flood Level Observation Warning System
Tracking Module: Simple centroid-based object tracker for debris
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import time


class CentroidTracker:
    """
    Lightweight centroid tracker that assigns stable IDs to detected objects.
    Uses Euclidean distance to match detections across frames.
    """

    def __init__(self, max_disappeared: int = 20, max_distance: float = 80.0):
        self.next_id = 0
        self.objects: OrderedDict[int, Tuple[int, int]] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.object_labels: Dict[int, str] = {}
        self.object_history: Dict[int, List[Tuple[int, int]]] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int], label: str):
        oid = self.next_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.object_labels[oid] = label
        self.object_history[oid] = [centroid]
        self.next_id += 1

    def deregister(self, oid: int):
        del self.objects[oid]
        del self.disappeared[oid]
        self.object_labels.pop(oid, None)
        self.object_history.pop(oid, None)

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with new detections.
        Returns: {object_id: {"centroid": (x,y), "label": str}}
        """
        if not detections:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self._export()

        # Compute input centroids
        input_centroids = []
        input_labels = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            input_centroids.append((cx, cy))
            input_labels.append(det.get("label", "unknown"))

        if not self.objects:
            for c, l in zip(input_centroids, input_labels):
                self.register(c, l)
            return self._export()

        # Match existing objects to input centroids
        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        D = np.zeros((len(obj_centroids), len(input_centroids)), dtype=np.float64)
        for i, oc in enumerate(obj_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        # Greedy matching: sort by distance
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = obj_ids[r]
            self.objects[oid] = input_centroids[c]
            self.object_labels[oid] = input_labels[c]
            self.disappeared[oid] = 0
            history = self.object_history.get(oid, [])
            history.append(input_centroids[c])
            if len(history) > 30:
                history = history[-30:]
            self.object_history[oid] = history
            used_rows.add(r)
            used_cols.add(c)

        # Handle unmatched existing objects
        for r in range(len(obj_centroids)):
            if r not in used_rows:
                oid = obj_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        # Register new detections
        for c in range(len(input_centroids)):
            if c not in used_cols:
                self.register(input_centroids[c], input_labels[c])

        return self._export()

    def _export(self) -> Dict[int, Dict]:
        return {
            oid: {
                "centroid": centroid,
                "label": self.object_labels.get(oid, "unknown"),
                "history": self.object_history.get(oid, []),
            }
            for oid, centroid in self.objects.items()
        }

    def draw_trails(self, frame, tracked_objects: Dict, color=(0, 255, 200), max_trail=15):
        """Draw motion trails for tracked objects."""
        import cv2
        for oid, info in tracked_objects.items():
            history = info.get("history", [])
            if len(history) < 2:
                continue
            trail = history[-max_trail:]
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                c = tuple(int(x * alpha) for x in color)
                cv2.line(frame, trail[i - 1], trail[i], c, 1, cv2.LINE_AA)
        return frame

    def reset(self):
        self.__init__(self.max_disappeared, self.max_distance)
