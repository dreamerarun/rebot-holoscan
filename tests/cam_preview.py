"""Live camera preview on the monitor (for placing the camera). q or Esc in the window closes it.
Saves logs/cam/preview.jpg every 2 s so the view can be checked remotely."""
import argparse
import time
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--size", default="640x480")
a = ap.parse_args()
w, h = (int(v) for v in a.size.split("x"))
out = Path(__file__).resolve().parents[1] / "logs" / "cam"
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(a.index, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, 30)
win = "reBot camera - placement preview (q to close)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 960, 720)
t_save, t0, n = 0.0, time.time(), 0
while True:
    ok, img = cap.read()
    if not ok:
        time.sleep(0.05)
        continue
    n += 1
    now = time.time()
    if now - t_save > 2.0:
        cv2.imwrite(str(out / "preview.jpg"), img)
        t_save = now
    view = img.copy()
    H, W = view.shape[:2]
    for k in (1, 2):  # rule-of-thirds guides
        cv2.line(view, (W * k // 3, 0), (W * k // 3, H), (255, 255, 255), 1)
        cv2.line(view, (0, H * k // 3), (W, H * k // 3), (255, 255, 255), 1)
    fps = n / max(now - t0, 1e-3)
    cv2.putText(view, f"{W}x{H}  {fps:4.1f} fps   aim: table + arm + gripper fill the frame",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(win, view)
    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
        break
cap.release()
cv2.destroyAllWindows()
