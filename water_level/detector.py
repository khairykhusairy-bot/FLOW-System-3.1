"""
FLOW — Water Level Module
detector.py: OpenCV-based waterline detection pipeline.

Detection strategy
------------------
The waterline is the boundary between the (darker, more uniform) water
surface and the flood gauge / river bank above it.  We find it by:

  1. Crop the frame to the gauge ROI (user-defined polygon or rectangle).
  2. Apply night-mode pre-processing when the scene is dark.
  3. Bilateral filter (edge-preserving) → CLAHE contrast enhancement.
  4. Multi-channel edge map: Canny on grayscale + Sobel_y on saturation channel.
  5. Adaptive-threshold binary map.
  6. Horizontal morphological close to bridge waterline gaps.
  7. Contour analysis with a composite score (width × horizontalness).
  8. Probabilistic Hough refinement on the winning candidate row ± a search band.
  9. Weighted-median of surviving Hough lines → sub-row Y estimate.
 10. Fallback: row-wise Sobel_y energy scan if contour step yields nothing.
 11. Return the Y coordinate in the *full frame* coordinate system so the
     calibration module can convert it to cm.

Key improvements over v1
------------------------
• Bilateral filter instead of pure Gaussian → preserves the sharp water/gauge
  edge while suppressing texture noise inside each region.
• HSV saturation channel used for a second Canny pass → murky water has very
  low saturation vs. painted gauge markings; the boundary is strongly visible
  in S even when grayscale contrast is low.
• Morphology kernel is now wide-horizontal (ROI_W//4 × 1) so it bridges gaps
  in a horizontal waterline without merging vertical gauge markings.
• Contour scoring = width_score × (1 - height/width) → prefers wide, flat
  contours over tall blobs (gauge digits, rust patches, etc.).
• Hough refinement: after the best contour row is found, a fast probabilistic
  Hough transform over a ±HOUGH_BAND_PX strip gives a sub-pixel-accurate slope
  + intercept from which the centre-column Y is read.
• Fallback uses Sobel_y (horizontal gradients) not Sobel_x — correct axis.
• Configurable runtime parameters so the Streamlit sidebar can tune them live.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict

from water_level.config import (
    BLUR_KERNEL,
    ADAPTIVE_BLOCK,
    ADAPTIVE_C,
    CANNY_LOW,
    CANNY_HIGH,
    MIN_CONTOUR_AREA,
    NIGHT_MODE_THRESHOLD,
    CLAHE_CLIP,
    CLAHE_GRID,
)

# ── Module-level tunables (can be overridden at runtime) ──────────────────────

# Minimum fraction of ROI width a contour must span to be a waterline candidate.
_MIN_WATERLINE_WIDTH_RATIO  = 0.15   # raised from 0.10 → fewer false short blobs

# Reject contours whose top edge falls in the top 5 % of the ROI.
_MIN_Y_RATIO = 0.05

# Reject contours whose top edge falls in the bottom 15 % of the ROI.
_MAX_Y_RATIO = 0.85

# Half-height of the search band (px) passed to Hough refinement.
_HOUGH_BAND_PX = 12

# Minimum Hough line votes to be considered reliable.
_HOUGH_MIN_VOTES = 20

# Weight of the saturation-channel edge map vs. the grayscale edge map.
# 0 = grayscale only, 1 = equal weight.
_SAT_EDGE_WEIGHT = 0.4


class WaterlineDetector:
    """
    Detects the waterline Y coordinate in each video frame.

    Parameters
    ----------
    gauge_roi_polygon : Optional list of (x,y) points defining the gauge area.
                        When None the full frame is used (not recommended).
    """

    def __init__(
        self,
        gauge_roi_polygon: Optional[List[Tuple[int, int]]] = None,
    ):
        self._gauge_roi: Optional[List[Tuple[int, int]]] = gauge_roi_polygon
        self._clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        self._last_y_px: Optional[int] = None
        self._frame_count: int = 0
        self._detection_count: int = 0

        # Runtime-tunable parameters (can be updated from the sidebar)
        self.canny_low:       int   = CANNY_LOW
        self.canny_high:      int   = CANNY_HIGH
        self.adaptive_block:  int   = ADAPTIVE_BLOCK
        self.adaptive_c:      int   = ADAPTIVE_C
        self.min_contour_area: int  = MIN_CONTOUR_AREA
        self.use_hough:       bool  = True
        self.use_sat_channel: bool  = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_gauge_roi(self, polygon: Optional[List[Tuple[int, int]]]):
        """Update the gauge ROI polygon at runtime."""
        self._gauge_roi = polygon

    def detect(self, frame: np.ndarray) -> Optional[int]:
        """
        Detect the waterline in ``frame``.

        Returns
        -------
        int or None : Y pixel coordinate of the detected waterline in the
                      *full-frame* coordinate system.
        """
        self._frame_count += 1

        roi_img, offset_x, offset_y = self._crop_to_roi(frame)
        if roi_img is None or roi_img.size == 0:
            return self._last_y_px

        roi_h, roi_w = roi_img.shape[:2]

        enhanced, sat_enhanced = self._preprocess(roi_img)
        combined = self._build_edge_map(enhanced, sat_enhanced, roi_w)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        waterline_y_local = self._find_waterline(contours, roi_w, roi_h)

        # Hough refinement: narrow-band pass around the candidate row
        if waterline_y_local is not None and self.use_hough:
            refined = self._hough_refine(combined, waterline_y_local, roi_h, roi_w)
            if refined is not None:
                waterline_y_local = refined

        if waterline_y_local is None:
            waterline_y_local = self._sobel_y_fallback(enhanced, roi_h)

        if waterline_y_local is None:
            return self._last_y_px

        y_full = waterline_y_local + offset_y
        self._last_y_px = y_full
        self._detection_count += 1
        return y_full

    def detect_with_debug(self, frame: np.ndarray) -> Tuple[Optional[int], Dict]:
        """Like detect() but returns intermediate images for Streamlit debug view."""
        roi_img, offset_x, offset_y = self._crop_to_roi(frame)
        debug: Dict = {
            "roi_img":           roi_img,
            "enhanced":          None,
            "sat_enhanced":      None,
            "edges":             None,
            "combined":          None,
            "waterline_y_local": None,
            "offset":            (offset_x, offset_y),
            "hough_refined":     False,
        }
        if roi_img is None or roi_img.size == 0:
            return self._last_y_px, debug

        roi_h, roi_w = roi_img.shape[:2]
        enhanced, sat_enhanced = self._preprocess(roi_img)
        combined = self._build_edge_map(enhanced, sat_enhanced, roi_w)

        debug["enhanced"]     = enhanced
        debug["sat_enhanced"] = sat_enhanced
        debug["combined"]     = combined

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        wl_local = self._find_waterline(contours, roi_w, roi_h)

        if wl_local is not None and self.use_hough:
            refined = self._hough_refine(combined, wl_local, roi_h, roi_w)
            if refined is not None:
                wl_local = refined
                debug["hough_refined"] = True

        if wl_local is None:
            wl_local = self._sobel_y_fallback(enhanced, roi_h)

        debug["waterline_y_local"] = wl_local
        if wl_local is None:
            return self._last_y_px, debug

        y_full = wl_local + offset_y
        self._last_y_px = y_full
        self._detection_count += 1
        return y_full, debug

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(
        self, roi_img: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Return (enhanced_gray, enhanced_saturation).

        Bilateral filter is used instead of a pure Gaussian: it smooths flat
        regions (water surface, concrete) while preserving the sharp waterline
        edge — reducing false edge detections from surface texture.
        """
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        # Night-mode boost
        if self._is_dark(gray):
            gray = self._night_enhance(gray)

        # Edge-preserving smoothing (bilateral: d=7, sigmaColor=50, sigmaSpace=50)
        # Falls back to Gaussian if ROI is too small for bilateral
        if gray.shape[0] > 15 and gray.shape[1] > 15:
            smoothed = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        else:
            k = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
            smoothed = cv2.GaussianBlur(gray, (k, k), 0)

        enhanced = self._clahe.apply(smoothed)

        # Saturation channel (murky water ≈ low S; painted gauge marks ≈ high S)
        sat_enhanced = None
        if self.use_sat_channel and roi_img.shape[0] > 5 and roi_img.shape[1] > 5:
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            if not self._is_dark(gray):  # skip if night (colours are unreliable)
                sat_smoothed  = cv2.bilateralFilter(sat, d=7,
                                                    sigmaColor=40, sigmaSpace=40)
                sat_enhanced  = self._clahe.apply(sat_smoothed)

        return enhanced, sat_enhanced

    def _build_edge_map(
        self,
        enhanced: np.ndarray,
        sat_enhanced: Optional[np.ndarray],
        roi_w: int,
    ) -> np.ndarray:
        """
        Build a combined binary edge map from:
          • Canny on enhanced grayscale
          • Canny on saturation channel (if available)
          • Adaptive threshold on grayscale
        Then apply a wide horizontal morphological close to bridge waterline gaps.
        """
        # Canny on grayscale
        edges_gray = cv2.Canny(enhanced, self.canny_low, self.canny_high,
                               apertureSize=3)

        # Canny on saturation (blended in)
        if sat_enhanced is not None:
            edges_sat = cv2.Canny(sat_enhanced,
                                  int(self.canny_low  * 0.8),
                                  int(self.canny_high * 0.8),
                                  apertureSize=3)
            # Weighted merge: grayscale dominates, saturation adds where grey misses
            alpha = int(255 * _SAT_EDGE_WEIGHT)
            edges_merged = cv2.addWeighted(edges_gray, 1.0, edges_sat,
                                           _SAT_EDGE_WEIGHT, 0)
            edges_merged = np.clip(edges_merged, 0, 255).astype(np.uint8)
        else:
            edges_merged = edges_gray

        # Adaptive threshold
        block = self.adaptive_block if self.adaptive_block % 2 == 1 else self.adaptive_block + 1
        block = max(block, 3)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block, self.adaptive_c,
        )

        combined = cv2.bitwise_or(edges_merged, thresh)

        # Wide horizontal close: bridges gaps along a horizontal waterline.
        # Kernel width = 25 % of ROI width (min 15 px), height = 1.
        h_close_w = max(15, roi_w // 4)
        h_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (h_close_w, 1))
        combined  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, h_kernel)

        # Small vertical open to remove thin vertical noise (gauge tick marks)
        v_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        combined  = cv2.morphologyEx(combined, cv2.MORPH_OPEN, v_kernel)

        return combined

    # ── Waterline Selection ────────────────────────────────────────────────────

    def _find_waterline(
        self,
        contours,
        roi_w: int,
        roi_h: int,
    ) -> Optional[int]:
        """
        Score and rank contours to find the best waterline candidate.

        Scoring
        -------
        A good waterline contour is:
          • Wide (spans a large fraction of the ROI horizontally)
          • Flat (bounding-box height much smaller than its width)

        score = width_fraction × (1 - height/width)

        The contour with the highest score whose top-edge Y is within
        [_MIN_Y_RATIO * roi_h, _MAX_Y_RATIO * roi_h] is selected.
        """
        min_width = max(5, int(roi_w * _MIN_WATERLINE_WIDTH_RATIO))
        min_y     = int(roi_h * _MIN_Y_RATIO)
        max_y     = int(roi_h * _MAX_Y_RATIO)

        best_score = -1.0
        best_y     = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)

            if bw < min_width:
                continue
            if by < min_y or by > max_y:
                continue

            # Flatness: 0 = square blob, →1 = perfectly horizontal line
            flatness     = 1.0 - min(bh / max(bw, 1), 1.0)
            width_frac   = min(bw / roi_w, 1.0)
            score        = width_frac * flatness

            if score > best_score:
                best_score = score
                best_y     = by

        return int(best_y) if best_y is not None else None

    # ── Hough Refinement ───────────────────────────────────────────────────────

    def _hough_refine(
        self,
        edge_map: np.ndarray,
        candidate_y: int,
        roi_h: int,
        roi_w: int,
    ) -> Optional[int]:
        """
        Run a probabilistic Hough transform in a narrow horizontal band
        around ``candidate_y`` and return the median Y of accepted lines.

        This sub-pixel-refines the waterline position and handles cases where
        the waterline is slightly sloped (e.g. camera not perfectly level).
        """
        band_top = max(0, candidate_y - _HOUGH_BAND_PX)
        band_bot = min(roi_h, candidate_y + _HOUGH_BAND_PX)
        if band_bot - band_top < 3:
            return None

        strip = edge_map[band_top:band_bot, :].copy()

        # Only look for nearly-horizontal lines (angle within ±15° of horizontal)
        lines = cv2.HoughLinesP(
            strip,
            rho=1,
            theta=np.pi / 180,
            threshold=_HOUGH_MIN_VOTES,
            minLineLength=max(10, roi_w // 6),
            maxLineGap=roi_w // 8,
        )

        if lines is None or len(lines) == 0:
            return None

        ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter: slope must be nearly horizontal (|dy/dx| < tan(15°) ≈ 0.27)
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx < 1:
                continue
            if dy / dx > 0.27:   # too steep — not a waterline
                continue
            # Weight longer lines more heavily
            length = np.sqrt(dx * dx + dy * dy)
            mid_y  = (y1 + y2) / 2.0
            ys.extend([mid_y] * max(1, int(length / 10)))

        if not ys:
            return None

        refined_local = float(np.median(ys))
        return int(round(band_top + refined_local))

    # ── Fallback ───────────────────────────────────────────────────────────────

    def _sobel_y_fallback(
        self, enhanced: np.ndarray, roi_h: int
    ) -> Optional[int]:
        """
        Fallback: scan rows top→bottom for the row with the highest
        *horizontal* gradient energy (Sobel in the Y direction).

        Sobel_y detects horizontal edges — the correct operator for a
        horizontal waterline. (v1 incorrectly used Sobel_x here.)
        """
        if roi_h < 10:
            return None

        # Sobel in Y direction (ksize=5 for better SNR vs. ksize=3)
        sobel_y    = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
        row_energy = np.abs(sobel_y).mean(axis=1)

        # Search in the middle 70 % of the ROI (avoid top/bottom edge artefacts)
        start_row = int(roi_h * 0.15)
        end_row   = int(roi_h * 0.90)
        search    = row_energy[start_row:end_row]
        if search.size == 0:
            return None

        best_local = int(np.argmax(search))
        return start_row + best_local

    # ── ROI Crop ───────────────────────────────────────────────────────────────

    def _crop_to_roi(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int, int]:
        """
        Crop frame to the gauge ROI polygon bounding box with polygon mask.
        Returns (cropped_image, x_offset, y_offset).
        """
        if not self._gauge_roi or len(self._gauge_roi) < 3:
            return frame, 0, 0

        pts = np.array(self._gauge_roi, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        fh, fw = frame.shape[:2]
        x  = max(0, x);      y  = max(0, y)
        x2 = min(fw, x + w); y2 = min(fh, y + h)
        if x2 <= x or y2 <= y:
            return frame, 0, 0

        crop = frame[y:y2, x:x2].copy()
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        shifted_pts = pts - np.array([x, y])
        cv2.fillPoly(mask, [shifted_pts], 255)
        crop = cv2.bitwise_and(crop, crop, mask=mask)

        return crop, x, y

    # ── Night helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_dark(gray: np.ndarray) -> bool:
        return float(np.mean(gray)) < NIGHT_MODE_THRESHOLD

    def _night_enhance(self, gray: np.ndarray) -> np.ndarray:
        """Blend histogram equalisation with CLAHE for low-light scenes."""
        eq           = cv2.equalizeHist(gray)
        clahe_result = self._clahe.apply(gray)
        return cv2.addWeighted(eq, 0.5, clahe_result, 0.5, 0)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def last_y_px(self) -> Optional[int]:
        return self._last_y_px

    @property
    def detection_rate(self) -> float:
        """Fraction of frames where a waterline was found."""
        if self._frame_count == 0:
            return 0.0
        return self._detection_count / self._frame_count
