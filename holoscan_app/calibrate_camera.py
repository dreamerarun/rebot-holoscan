"""Camera -> table calibration for the scripted collector. MOTORS STAY OFF (read-only guard).

For each point (6-9, spread over the area you will use):
  1. cube on the table; move the real arm BY HAND so the open gripper straddles the cube -> press c
     (records the gripper position from the motors, forward kinematics)
  2. move the arm away by hand without touching the cube -> press c
     (records where the camera sees the cube)
Keys in the camera window:  c capture   u undo last point   s solve + save   q quit
Result: config/camera_table.yaml (homography image pixel -> table x, y in the robot base frame).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from kinematics import RebotKinematics
from perception import detect_box, detect_cube, draw, open_camera
from homography import solve  # noqa: F401 (kept importable here)
from real_arm import READ_ONLY, RealArmReader, load_real_cfg, measured_q7

assert READ_ONLY
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "camera_table.yaml"


def main():
    rcfg = load_real_cfg()
    reader = RealArmReader(rcfg)
    reader.open()
    kin = RebotKinematics()
    cap, idx = open_camera()
    print(f"[calib] camera index {idx}; motors are read only", flush=True)
    win = "reBot camera calibration (motors OFF) - c capture, u undo, s save, q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)
    pairs, pending, msg, msg_t = [], None, "", 0.0
    while True:
        ok, img = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        q = measured_q7(rcfg, reader.read())
        tcp = None if q is None else kin.fk(q[:6])
        cube, box = detect_cube(img), detect_box(img)
        view = draw(img, cube, box)
        for p in pairs:
            cv2.circle(view, (int(p["px"][0]), int(p["px"][1])), 5, (0, 255, 0), -1)
        step = "1/2: gripper around the cube, press c" if pending is None else "2/2: move the arm away, press c"
        lines = [f"points: {len(pairs)}   step {step}"]
        if tcp is not None:
            x, y, z = tcp.translation
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, -tcp.rotation[2, 0]))))
            lines.append(f"gripper x {x:+.3f} y {y:+.3f} z {z:+.3f} m   tilt from vertical {tilt:4.0f} deg")
        else:
            lines.append("no motor feedback")
        lines.append(f"cube: {'OK' if cube.ok else cube.reason}   box: {'OK' if box.ok else box.reason}")
        if msg and time.time() - msg_t < 5:
            lines.append(msg)
        for i, t in enumerate(lines):
            cv2.putText(view, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF

        def note(t):
            nonlocal msg, msg_t
            msg, msg_t = t, time.time()
            print(f"[calib] {t}", flush=True)

        if k == ord("c"):
            if pending is None:
                if tcp is None:
                    note("no motor feedback - not captured")
                elif tcp.translation[2] > 0.06:
                    note(f"gripper is {tcp.translation[2] * 100:.0f} cm high - lower it around the cube")
                else:
                    pending = {"xy": [float(tcp.translation[0]), float(tcp.translation[1])],
                               "z": float(tcp.translation[2]), "q_deg": np.degrees(q[:6]).round(2).tolist()}
                    note(f"gripper at ({pending['xy'][0]:+.3f}, {pending['xy'][1]:+.3f}) - now move the arm away")
            else:
                if not cube.ok:
                    note(f"cube not seen cleanly ({cube.reason}) - move the arm further away")
                elif tcp is not None and tcp.translation[2] < 0.10:
                    note("lift the arm higher / away first (gripper still near the table)")
                else:
                    pending["px"] = [float(cube.centroid[0]), float(cube.centroid[1])]
                    pairs.append(pending)
                    note(f"point {len(pairs)} saved: pixel ({pending['px'][0]:.0f}, {pending['px'][1]:.0f})")
                    pending = None
        elif k == ord("u") and (pairs or pending):
            if pending is not None:
                pending = None
            else:
                pairs.pop()
            note(f"undone - {len(pairs)} points")
        elif k == ord("s"):
            if len(pairs) < 5:
                note("need at least 5 points (6-9 recommended)")
            else:
                H, e_fit, e_loo = solve(pairs)
                OUT.write_text(yaml.safe_dump({
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "camera": "C270", "image_size": [640, 480],
                    "H": H.tolist(), "fit_error_mm_max": float(e_fit.max()),
                    "leave_one_out_mm_max": float(e_loo.max()) if len(e_loo) else None,
                    "leave_one_out_mm_mean": float(e_loo.mean()) if len(e_loo) else None,
                    "box_px": [float(v) for v in box.centroid] if box.ok else None,
                    "points": pairs}, sort_keys=False))
                note(f"saved {OUT.name}: fit max {e_fit.max():.1f} mm, unseen-point error "
                     f"{'n/a' if not len(e_loo) else f'mean {e_loo.mean():.1f} / max {e_loo.max():.1f} mm'}")
        elif k in (ord("q"), 27):
            break
    cap.release()
    reader.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
