"""Per-point diagnosis of config/camera_table.yaml: residuals for homography and affine fits, leave-one-out."""
import os
import numpy as np, yaml, cv2
c = yaml.safe_load(open(os.path.expanduser("~/rebot-holoscan/config/camera_table.yaml")))
pts = c["points"]
px = np.array([p["px"] for p in pts], float); xy = np.array([p["xy"] for p in pts], float)
z = [p["z"] for p in pts]
def H_apply(H, a): q = H @ np.r_[a, 1]; return q[:2] / q[2]
def fit(kind, idx):
    if kind == "homography":
        H, _ = cv2.findHomography(px[idx], xy[idx], 0); return lambda a: H_apply(H, a)
    A = np.c_[px[idx], np.ones(len(idx))]; M, *_ = np.linalg.lstsq(A, xy[idx], rcond=None)
    return lambda a: np.r_[a, 1] @ M
for kind in ("homography", "affine"):
    f = fit(kind, list(range(len(pts))))
    res = [np.linalg.norm(f(px[i]) - xy[i]) * 1000 for i in range(len(pts))]
    loo = [np.linalg.norm(fit(kind, [j for j in range(len(pts)) if j != i])(px[i]) - xy[i]) * 1000 for i in range(len(pts))]
    print(f"{kind:10s} residual mm " + " ".join(f"{r:6.1f}" for r in res) + "   | leave-one-out mm " + " ".join(f"{r:6.1f}" for r in loo))
for i, p in enumerate(pts):
    print(f"point {i+1}: table ({p['xy'][0]:+.3f}, {p['xy'][1]:+.3f}) z {p['z']*100:4.1f} cm  pixel ({p['px'][0]:.0f}, {p['px'][1]:.0f})  joints {p['q_deg']}")

# --- tilt/height correction: follow the gripper's pointing axis down to the cube-centre plane
import sys, math
sys.path.insert(0, os.path.expanduser("~/rebot-holoscan/holoscan_app"))
from kinematics import RebotKinematics
k = RebotKinematics()
Zc = 0.02
xy_c = []
for p in pts:
    T = k.fk(np.radians(p["q_deg"]))
    d = T.rotation[:, 0]                       # end_link +x = pointing direction of the fingers
    t = T.translation
    tilt = math.degrees(math.acos(max(-1, min(1, -d[2]))))
    s = (t[2] - Zc) / -d[2] if d[2] < -0.2 else 0.0
    xy_c.append((t + s * d)[:2])
    print(f"  tilt {tilt:5.1f} deg, height {t[2]*100:4.1f} cm -> shift {np.linalg.norm(s*d[:2])*1000:5.1f} mm")
xy = np.array(xy_c)
for kind in ("homography", "affine"):
    f = fit(kind, list(range(len(pts))))
    res = [np.linalg.norm(f(px[i]) - xy[i]) * 1000 for i in range(len(pts))]
    loo = [np.linalg.norm(fit(kind, [j for j in range(len(pts)) if j != i])(px[i]) - xy[i]) * 1000 for i in range(len(pts))]
    print(f"CORRECTED {kind:10s} residual mm " + " ".join(f"{r:6.1f}" for r in res) + "   | leave-one-out mm " + " ".join(f"{r:6.1f}" for r in loo))
