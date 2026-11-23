"""Plan + validate the robot-driven calibration offline (no hardware). Run: hs_py tests/test_motion_plan.py"""
import math, sys
from pathlib import Path
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "holoscan_app"))
from motion_plan import CAL_GRID, HANDOVER, PHOTO_Q, MotionPlanner, RATE  # noqa: E402

cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config" / "teleop.yaml"))
limits = [tuple(cfg["sim_limits"][f"joint{i}"]) for i in range(1, 7)]
mp = MotionPlanner(limits)
rest = np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # exact folded zero (twin start; real arm rests here too)
segs = mp.calibration(rest, CAL_GRID, HANDOVER, PHOTO_Q)
rep = mp.validate(segs, rest)
tot = sum(r[1] for r in rep)
print(f"{len(segs)} segments, {sum(len(s.q) for s in segs)} samples, {tot / 60:.1f} min total, all validated")
for (label, secs, zmin), s in list(zip(rep, segs))[:14]:
    print(f"  {label:38s} {secs:5.1f} s   lowest TCP {zmin * 100:5.1f} cm  {s.event}")
print("  ...")
lows = [r[2] for r in rep]
print(f"lowest TCP over the whole plan: {min(lows) * 100:.1f} cm; photo pose TCP:",
      np.round(mp.k.fk(PHOTO_Q).translation, 3))
