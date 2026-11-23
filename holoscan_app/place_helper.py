"""Placement helper (camera only, motors untouched): where to put the box and the cube for the collector.

Shows: green outline = cube zone, orange circle = box target, live table coordinates of what the camera sees,
and whether each object is OK for the collector. q / Esc closes it.
"""
import math
import time
from pathlib import Path

import cv2
import numpy as np

from perception import detect_box, detect_cube, draw, load_calibration, open_camera, pixel_to_table

ROOT = Path(__file__).resolve().parents[1]
ZONE = {"x": (0.17, 0.33), "y": (-0.11, 0.11)}
BOX_TARGET = (0.20, -0.24)
BOX_CLEAR, REACH = 0.12, (0.15, 0.34)


def to_px(Hinv, x, y):
    p = Hinv @ np.array([x, y, 1.0])
    return int(p[0] / p[2]), int(p[1] / p[2])


def main():
    H, _ = load_calibration(ROOT / "config" / "camera_table.yaml")
    Hinv = np.linalg.inv(H)
    cap, _ = open_camera()
    zone = np.array([to_px(Hinv, x, y) for x, y in [(ZONE["x"][0], ZONE["y"][0]), (ZONE["x"][0], ZONE["y"][1]),
                                                     (ZONE["x"][1], ZONE["y"][1]), (ZONE["x"][1], ZONE["y"][0])]], np.int32)
    # box target circle: 5 cm radius drawn on the table
    circ = np.array([to_px(Hinv, BOX_TARGET[0] + 0.05 * math.cos(t), BOX_TARGET[1] + 0.05 * math.sin(t))
                     for t in np.linspace(0, 2 * math.pi, 40)], np.int32)
    win = "reBot placement helper - box into the orange circle, cube into the green zone (q closes)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)
    while True:
        ok, img = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        cube, box = detect_cube(img), detect_box(img)
        view = draw(img, cube, box)
        cv2.polylines(view, [zone], True, (0, 255, 0), 2)
        cv2.polylines(view, [circ], True, (0, 140, 255), 2)
        lines = []
        if box.ok:
            bx, by = pixel_to_table(H, box.centroid)
            dx, dy = BOX_TARGET[0] - bx, BOX_TARGET[1] - by
            r = math.hypot(bx, by)
            good = math.hypot(dx, dy) < 0.03
            hint = "OK" if good else ("move it " + ("further from" if dx > 0 else "closer to") + f" the arm {abs(dx) * 100:.0f} cm, "
                                      + ("more to the arm's right" if dy < 0 else "more to the arm's left") + f" {abs(dy) * 100:.0f} cm")
            lines.append((f"BOX at ({bx:.2f}, {by:+.2f}) m, {r * 100:.0f} cm from base: {hint}", (0, 255, 0) if good else (0, 200, 255)))
        else:
            lines.append((f"BOX: {box.reason}", (0, 0, 255)))
        if cube.ok:
            cx, cy = pixel_to_table(H, cube.centroid)
            why = None
            if not (ZONE["x"][0] <= cx <= ZONE["x"][1] and ZONE["y"][0] <= cy <= ZONE["y"][1]):
                why = "outside the green zone"
            elif box.ok and math.hypot(cx - bx, cy - by) < BOX_CLEAR:
                why = "too close to the box"
            lines.append((f"CUBE at ({cx:.2f}, {cy:+.2f}) m: {'OK' if why is None else why}", (0, 255, 0) if why is None else (0, 200, 255)))
        else:
            lines.append((f"CUBE: {cube.reason}", (0, 0, 255)))
        for i, (t, col) in enumerate(lines):
            cv2.putText(view, t, (8, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, t, (8, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        cv2.imshow(win, view)
        if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
