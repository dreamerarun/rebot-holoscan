"""Colour perception for the scripted collector: yellow cube + orange box on the wooden table.

Measured on the C270 image (OpenCV HSV, H 0-180), 2026-11-23:
    cube    H 26-29  S ~207  V 210-225
    box     H 16-19  S 220-245
    gripper lime parts  H 37-39  S ~162   (must NOT be taken for the cube)
    table   H ~17    S ~47              (same hue as the box, far lower saturation)

Every detection carries `ok` + a reason, so the collector can refuse to move on anything doubtful.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

CUBE_HSV = ((23, 140, 100), (32, 255, 255))
BOX_HSV = ((10, 170, 80), (21, 255, 255))
CUBE_AREA = (250, 4000)      # px at 640x480 for a 4 cm cube 15-45 cm from the base (checked live)
BOX_AREA = (2000, 30000)


@dataclass
class Blob:
    ok: bool
    reason: str = ""
    centroid: tuple = (0.0, 0.0)     # px (x, y)
    area: float = 0.0
    bbox: tuple = (0, 0, 0, 0)       # x, y, w, h
    angle: float = 0.0               # deg, from minAreaRect (cube yaw in the image)
    contour: np.ndarray | None = field(default=None, repr=False)


def _mask(hsv, lo_hi):
    lo, hi = lo_hi
    m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _blobs(mask, min_area):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted([c for c in cs if cv2.contourArea(c) >= min_area], key=cv2.contourArea, reverse=True)


def _blob(c, ok=True, reason=""):
    m = cv2.moments(c)
    (_, _), (_, _), ang = cv2.minAreaRect(c)
    return Blob(ok, reason, (m["m10"] / m["m00"], m["m01"] / m["m00"]), cv2.contourArea(c),
                cv2.boundingRect(c), ang, c)


CUBE_MIN_Y_FRAC = 0.30       # cubes lie on the table: ignore yellow above 30 % of the image height
                             # (the arm's own yellow Seeed sticker shows up there in the folded pose)


CUBE_GLARE_HSV = ((23, 60, 245), (32, 255, 255))   # washed-out cube in the bright near-camera corner
                                                    # (2026-11-23: S 44-119 at V 255 at (0.28..0.33, +0.10))


def detect_cube(bgr, roi=None) -> Blob:
    """roi: optional uint8 mask (255 = look here). The collector passes the cube zone + the box area, so other
    yellow things (the arm's sticker, the power supply label) can never be taken for the cube.
    If the strict colour finds no cube at all, one retry also accepts near-white glare yellow (same checks)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b = _detect_cube(hsv, _mask(hsv, CUBE_HSV), roi)
    # glare retry only inside a search area (collector zone / calibration circle): on whole frames it can pick up
    # bright spots on the gripper
    if roi is None or b.ok or b.reason != "no yellow cube" and not b.reason.startswith("cube area"):
        return b
    mask = cv2.bitwise_or(_mask(hsv, CUBE_HSV), _mask(hsv, CUBE_GLARE_HSV))
    g = _detect_cube(hsv, mask, roi)
    if g.ok:
        g.reason = "glare"
        return g
    return b


def _detect_cube(hsv, mask, roi) -> Blob:
    mask[: int(CUBE_MIN_Y_FRAC * mask.shape[0])] = 0
    if roi is not None:
        mask = cv2.bitwise_and(mask, roi)
    blobs = _blobs(mask, 60)
    if not blobs:
        return Blob(False, "no yellow cube")
    big = [c for c in blobs if cv2.contourArea(c) >= CUBE_AREA[0]]
    if len(big) > 1:
        return _blob(big[0], False, f"{len(big)} yellow blobs (expected exactly one)")
    b = _blob(blobs[0])
    if not CUBE_AREA[0] <= b.area <= CUBE_AREA[1]:
        b.ok, b.reason = False, f"cube area {b.area:.0f} px outside {CUBE_AREA}"
        return b
    x, y, w, h = b.bbox
    fill = b.area / max(w * h, 1)
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > 2.0 or fill < 0.45:
        b.ok, b.reason = False, f"not cube-shaped (aspect {aspect:.1f}, fill {fill:.2f})"
    return b


def detect_box(bgr) -> Blob:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blobs = _blobs(_mask(hsv, BOX_HSV), 300)
    if not blobs:
        return Blob(False, "no orange box")
    b = _blob(blobs[0])
    if not BOX_AREA[0] <= b.area <= BOX_AREA[1]:
        b.ok, b.reason = False, f"box area {b.area:.0f} px outside {BOX_AREA}"
    return b


def draw(bgr, cube: Blob, box: Blob):
    out = bgr.copy()
    for b, col, name in ((box, (0, 140, 255), "box"), (cube, (0, 255, 255), "cube")):
        if b.contour is None:
            continue
        cv2.drawContours(out, [b.contour], -1, col if b.ok else (0, 0, 255), 2)
        cx, cy = int(b.centroid[0]), int(b.centroid[1])
        cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(out, f"{name} {b.area:.0f}px" + ("" if b.ok else " !"), (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def find_camera_index(name="C270"):
    """First /dev/videoN whose name contains `name` and that delivers frames (V4L2 index)."""
    import glob
    import re
    for path in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        try:
            if name.lower() not in open(path + "/name").read().lower():
                continue
        except OSError:
            continue
        idx = int(re.sub(r"\D", "", path.rsplit("/", 1)[-1]))
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            return idx
    raise RuntimeError(f"no working camera named {name!r} found")


def open_camera(index="auto", size=(640, 480), fps=30):
    idx = find_camera_index() if index == "auto" else int(index)
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap, idx


def load_calibration(path):
    import yaml
    c = yaml.safe_load(open(path))
    return np.array(c["H"], float), c


def pixel_to_table(H, px):
    """Image pixel (x, y) -> table point (x, y) in metres, robot base frame."""
    p = H @ np.array([px[0], px[1], 1.0])
    return p[:2] / p[2]
