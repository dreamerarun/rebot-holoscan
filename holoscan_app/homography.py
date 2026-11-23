"""Camera-to-table homography fit (pure math, no motor imports: safe for the live programs).

Moved out of calibrate_camera.py on 2026-11-23: importing that read-only tool installed the motorbridge
READ-ONLY GUARD process-wide and blocked every motor command after a live calibration saved its map.
"""
import cv2
import numpy as np


def solve(pairs):
    px = np.array([p["px"] for p in pairs], np.float64)
    xy = np.array([p["xy"] for p in pairs], np.float64)
    H, _ = cv2.findHomography(px, xy, 0)
    fit = np.array([(H @ np.r_[a, 1])[:2] / (H @ np.r_[a, 1])[2] for a in px])
    err_fit = np.linalg.norm(fit - xy, axis=1) * 1000
    loo = []                                   # leave-one-out: honest accuracy on unseen points
    if len(pairs) >= 6:
        for i in range(len(pairs)):
            k = [j for j in range(len(pairs)) if j != i]
            Hi, _ = cv2.findHomography(px[k], xy[k], 0)
            q = Hi @ np.r_[px[i], 1]
            loo.append(np.linalg.norm(q[:2] / q[2] - xy[i]) * 1000)
    return H, err_fit, np.array(loo)
