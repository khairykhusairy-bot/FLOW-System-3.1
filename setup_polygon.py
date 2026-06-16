#!/usr/bin/env python3
"""
setup_polygon.py — Interactive polygon ROI setup tool for FLOW
==============================================================
Run ONCE (or any time you want to redefine the river ROI) before
starting main.py.  The saved polygon is written into config.py and
automatically picked up by PolygonROI / main.py on the next run.

Usage:
    python setup_polygon.py             # uses webcam defined in config.py
    python setup_polygon.py --demo      # generate a synthetic demo frame instead

Controls:
    Left-click       → Add a polygon point
    Right-click      → Remove last point
    ENTER / SPACE    → Confirm & save polygon to config.py
    R                → Reset all points
    S                → Toggle coordinate labels
    ESC              → Quit without saving
"""

import cv2
import numpy as np
import re
import sys
import os
import argparse

# ── Locate project root & import config ──────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

try:
    import config as _cfg
    WEBCAM_INDEX = _cfg.WEBCAM_INDEX
    CONFIG_PATH  = _cfg.CONFIG_PATH
except ImportError:
    print("[Setup] WARNING: config.py not found — using defaults.")
    WEBCAM_INDEX = 0
    CONFIG_PATH  = os.path.join(ROOT, "config.py")

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW = (
    "Polygon ROI Setup  |  "
    "Left-click=Add  Right-click=Undo  "
    "ENTER=Save  R=Reset  ESC=Quit"
)

# BGR colour palette
COL_POINT  = (0,  255, 180)
COL_LINE   = (0,  220, 255)
COL_FILL   = (0,  180, 255)
COL_LABEL  = (255, 255, 255)
COL_SAVED  = (0,  255, 80)
COL_BAR_BG = (18, 26,  42)

# ── State ─────────────────────────────────────────────────────────────────────
points      = []
show_coords = True
hover_pos   = [None]


# ─── Demo frame generator ─────────────────────────────────────────────────────
def generate_demo_frame(w: int = 960, h: int = 540) -> np.ndarray:
    """Produce a static river-scene background for use in demo mode."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky gradient
    for y in range(h // 3):
        b = int(40 + y * 1.2)
        g = int(20 + y * 0.6)
        r = int(10 + y * 0.3)
        frame[y] = (b, g, r)
    # River body
    frame[h // 3:, :] = (90, 60, 30)
    # Banks
    cv2.rectangle(frame, (0, h // 3 - 20), (w, h // 3 + 5), (50, 80, 30), -1)
    cv2.rectangle(frame, (0, h - 30),       (w, h),            (50, 80, 30), -1)
    # Ripples
    for i in range(0, w, 60):
        cv2.ellipse(frame, (i + 30, h // 3 + 25), (50, 8), 0, 0, 180, (100, 70, 40), 1)
    # Trees
    for i in range(0, w, 120):
        cv2.circle(frame, (i + 60, h // 3 - 30), 25, (30, 100, 20), -1)
        cv2.rectangle(frame, (i + 56, h // 3 - 15), (i + 64, h // 3 + 5), (60, 40, 20), -1)
    # Label
    cv2.putText(frame, "FLOW DEMO FRAME — draw your ROI polygon here",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    return frame


# ─── Overlay renderer ─────────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, pts: list, hover=None) -> np.ndarray:
    """Composite the instruction bar + polygon + points onto the frame."""
    h, w = frame.shape[:2]
    canvas = frame.copy()

    # ── Top instruction bar ───────────────────────────────────────────────────
    BAR_H = 40
    bar = np.full((BAR_H, w, 3), COL_BAR_BG, dtype=np.uint8)
    instructions = [
        ("Left-click: Add",   (0,  220, 255)),
        ("Right-click: Undo", (255, 180,  0)),
        ("ENTER: Save",       (0,  255, 120)),
        ("R: Reset",          (180, 180, 180)),
        ("S: Coords",         (160, 160, 255)),
        ("ESC: Quit",         (100, 100, 255)),
    ]
    x_off = 10
    for txt, col in instructions:
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(bar, txt, (x_off, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        x_off += tw + 20
        if x_off > w - 120:
            break
    canvas = np.vstack([bar, canvas])

    # All points are offset by BAR_H because of the bar
    off_pts = [(p[0], p[1] + BAR_H) for p in pts]

    # ── Filled polygon ────────────────────────────────────────────────────────
    if len(off_pts) >= 3:
        poly_arr = np.array(off_pts, dtype=np.int32)
        overlay  = canvas.copy()
        cv2.fillPoly(overlay, [poly_arr], COL_FILL)
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
        cv2.polylines(canvas, [poly_arr], True, COL_LINE, 2, cv2.LINE_AA)

    # ── Edge lines ────────────────────────────────────────────────────────────
    for i in range(len(off_pts) - 1):
        cv2.line(canvas, off_pts[i], off_pts[i + 1], COL_LINE, 2, cv2.LINE_AA)

    # ── Hover preview lines ───────────────────────────────────────────────────
    if off_pts and hover:
        hx, hy = hover[0], hover[1] + BAR_H
        cv2.line(canvas, off_pts[-1], (hx, hy), (120, 120, 120), 1, cv2.LINE_AA)
        if len(off_pts) >= 2:
            cv2.line(canvas, off_pts[0], (hx, hy), (80, 80, 80), 1, cv2.LINE_AA)

    # ── Corner dots + labels ──────────────────────────────────────────────────
    for i, (px, py) in enumerate(off_pts):
        cv2.circle(canvas, (px, py), 8, (0, 0, 0),   -1)
        cv2.circle(canvas, (px, py), 6, COL_POINT,    -1)
        cv2.circle(canvas, (px, py), 8, COL_POINT,     1)
        if show_coords:
            lbl = f"P{i+1}  ({pts[i][0]}, {pts[i][1]})"
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            lx, ly = px + 12, py - 8
            cv2.rectangle(canvas, (lx - 2, ly - lh - 2), (lx + lw + 2, ly + 4), (0, 0, 0), -1)
            cv2.putText(canvas, lbl, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_LABEL, 1, cv2.LINE_AA)

    # ── Status footer ─────────────────────────────────────────────────────────
    total_h = canvas.shape[0]
    if len(pts) >= 3:
        status = f"✓  {len(pts)} points defined — press ENTER to save"
        s_col  = COL_SAVED
    else:
        status = f"  {len(pts)} point(s) — need at least 3 to save"
        s_col  = (100, 100, 255)
    cv2.putText(canvas, status, (10, total_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, s_col, 1, cv2.LINE_AA)

    return canvas


# ─── config.py patcher ────────────────────────────────────────────────────────
def save_polygon_to_config(pts: list, config_path: str) -> bool:
    """
    Write (or update) the ROI_POLYGON entry in config.py.

    Strategy:
      • If the file exists and contains a ROI_POLYGON line → replace it.
      • If the file exists but has no ROI_POLYGON → append it after the
        WEBCAM_INDEX line, or at end-of-file.
      • If config.py does not exist → create a minimal one.

    Returns True on success, False on error.
    """
    formatted = repr(pts)   # e.g. [(10, 20), (300, 20), (300, 400), (10, 400)]
    new_line  = f"ROI_POLYGON  = {formatted}"

    # ── File doesn't exist → create minimal config ────────────────────────────
    if not os.path.exists(config_path):
        minimal = (
            '"""\nFLOW config — auto-created by setup_polygon.py\n"""\n\n'
            "import os\n\n"
            f"CONFIG_PATH  = os.path.abspath(__file__)\n"
            f"WEBCAM_INDEX = 0\n\n"
            f"# Polygon ROI — set by setup_polygon.py\n"
            f"{new_line}\n"
        )
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(minimal)
            print(f"[Setup] Created new config.py at: {config_path}")
            print(f"[Setup] Saved polygon: {pts}")
            return True
        except OSError as e:
            print(f"[Setup] ERROR writing {config_path}: {e}")
            return False

    # ── File exists → read and patch ─────────────────────────────────────────
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        print(f"[Setup] ERROR reading {config_path}: {e}")
        return False

    pattern = r"^ROI_POLYGON\s*=.*$"

    if re.search(pattern, content, re.MULTILINE):
        # Replace existing line
        content = re.sub(pattern, new_line, content, flags=re.MULTILINE)
        print("[Setup] Updated existing ROI_POLYGON in config.py.")
    else:
        # Append after WEBCAM_INDEX line if present, else at EOF
        anchor = "WEBCAM_INDEX"
        if anchor in content:
            idx = content.index(anchor)
            eol = content.index("\n", idx)
            content = (
                content[: eol + 1]
                + "\n# Polygon ROI — set by setup_polygon.py\n"
                + new_line + "\n"
                + content[eol + 1 :]
            )
        else:
            content += f"\n# Polygon ROI — set by setup_polygon.py\n{new_line}\n"
        print("[Setup] Added ROI_POLYGON to config.py.")

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        print(f"[Setup] ERROR writing {config_path}: {e}")
        return False

    print(f"[Setup] Polygon saved → {pts}")
    print(f"[Setup] Config path   → {config_path}")
    return True


# ─── Mouse callback ───────────────────────────────────────────────────────────
def on_mouse(event, x, y, flags, param):
    """Record clicks in original frame coordinates (subtract the top bar offset)."""
    global points
    BAR_H = 40
    fy = y - BAR_H
    if fy < 0:
        return

    hover_pos[0] = (x, fy)

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, fy))
        print(f"  + P{len(points)}: ({x}, {fy})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            removed = points.pop()
            print(f"  - Removed {removed}  ({len(points)} points left)")


# ─── Frame acquisition ────────────────────────────────────────────────────────
def acquire_reference_frame(demo: bool, webcam_index: int) -> np.ndarray:
    """
    Return a single reference frame — either from the webcam or a demo render.
    Exits the process cleanly if the webcam cannot be opened.
    """
    if demo:
        print("[Setup] Demo mode — using synthetic river frame.")
        return generate_demo_frame(960, 540)

    print(f"[Setup] Opening webcam {webcam_index} …")
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(
            f"[Setup] ERROR: Cannot open webcam {webcam_index}.\n"
            f"        • Check that the camera is connected.\n"
            f"        • Edit WEBCAM_INDEX in config.py if needed.\n"
            f"        • Run with --demo to use a synthetic frame instead."
        )
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[Setup] ERROR: Could not read a frame from the webcam.")
        sys.exit(1)

    actual_w, actual_h = frame.shape[1], frame.shape[0]
    print(f"[Setup] Captured frame: {actual_w}×{actual_h}")
    return frame


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    global show_coords, points

    parser = argparse.ArgumentParser(
        description="FLOW — Interactive Polygon ROI Setup"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use a synthetic demo frame instead of the webcam"
    )
    parser.add_argument(
        "--webcam", type=int, default=None,
        help="Override the webcam index from config.py"
    )
    args = parser.parse_args()

    webcam_idx = args.webcam if args.webcam is not None else WEBCAM_INDEX

    print(__doc__)
    reference_frame = acquire_reference_frame(args.demo, webcam_idx)

    fw, fh = reference_frame.shape[1], reference_frame.shape[0]
    print(f"[Setup] Working frame: {fw}×{fh}")
    print("[Setup] Click to add points. Press ENTER when done.\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, fw, fh + 40)   # +40 for the instruction bar
    cv2.setMouseCallback(WINDOW, on_mouse)

    saved = False

    while True:
        canvas = draw_overlay(reference_frame, points, hover=hover_pos[0])
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF

        # ── ENTER / SPACE → save ──────────────────────────────────────────────
        if key in (13, 32):
            if len(points) < 3:
                print("[Setup] Need at least 3 points to define a polygon.")
            else:
                ok = save_polygon_to_config(points, CONFIG_PATH)
                if ok:
                    saved = True
                    # Flash green confirmation for 2 s
                    confirm = canvas.copy()
                    cv2.putText(
                        confirm,
                        "  POLYGON SAVED TO config.py!  Run main.py to apply.",
                        (confirm.shape[1] // 2 - 280, confirm.shape[0] // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, COL_SAVED, 2, cv2.LINE_AA,
                    )
                    cv2.imshow(WINDOW, confirm)
                    cv2.waitKey(2500)
                    break
                else:
                    print("[Setup] Save failed — check file permissions.")

        # ── R → reset ─────────────────────────────────────────────────────────
        elif key == ord("r"):
            points = []
            print("[Setup] Points reset.")

        # ── S → toggle coordinate labels ──────────────────────────────────────
        elif key == ord("s"):
            show_coords = not show_coords

        # ── ESC → quit without saving ─────────────────────────────────────────
        elif key == 27:
            print("[Setup] Quit without saving.")
            break

    cv2.destroyAllWindows()

    if saved:
        print("\n[Setup] ✓ Done!  Your polygon has been saved to config.py.")
        print("[Setup]   Now run:  streamlit run main.py")
    else:
        print("\n[Setup] No polygon was saved.")
        print("[Setup]   Run setup_polygon.py again when ready.")


if __name__ == "__main__":
    main()
