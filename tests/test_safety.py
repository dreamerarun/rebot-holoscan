"""Offline tests for holoscan_app/safety.py (no hardware). Run: hs_py tests/test_safety.py"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "holoscan_app"))
from safety import SafetyFilter, SafetyParams  # noqa: E402

LIMITS = [(-2.8, 2.8), (-3.14, 0.0), (-3.14, 0.0), (-1.87, 1.57), (-1.57, 1.57), (-3.14, 3.14)]
DT = 1 / 60
D = math.radians
FOLDED = np.array([0.0, D(-0.3), D(-0.6), D(-1.8), D(2.8), 0.0, 0.0])  # measured rest pose (rad, + gripper m)
results = []


def check(name, cond, info=""):
    results.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   ({info})" if info else ""))


def new(**kw):
    fk = kw.pop("fk", None)
    return SafetyFilter(SafetyParams(limits=LIMITS, check_tracking=kw.pop("check_tracking", False), **kw), fk=fk)


def run(f, target_fn, secs, measured=FOLDED, t0=0.0, health=None, target_age=0.0, feedback_age=0.0):
    cmds, t = [], t0
    for _ in range(int(secs / DT)):
        t += DT
        tgt = target_fn(t) if callable(target_fn) else target_fn
        c = f.step(t, DT, tgt, target_age, measured, feedback_age, health)
        cmds.append(None if c is None else c.copy())
    return cmds, t


# 1 nothing to send without feedback
f = new()
check("no feedback -> nothing sent", f.step(0, DT, FOLDED, 0, None, 0) is None)

# 1b feedback but no leader yet -> hold the measured pose
f = new()
c = f.step(0, DT, None, 1e9, FOLDED, 0)
check("no leader yet -> hold measured pose", c is not None and np.allclose(c, FOLDED) and f.mode == "WAIT_ALIGN", f.reason)

# 2 leader far away -> hold at the measured pose, explain which joint
f = new()
far = FOLDED.copy(); far[1] = D(-60)
cmds, _ = run(f, far, 1.0)
check("leader far: stays WAIT_ALIGN", f.mode == "WAIT_ALIGN", f.reason)
check("leader far: arm does not move", np.allclose(cmds[-1], FOLDED))

# 3 folded rest pose + leader also folded: no start-up jump out of the limit margin
f = new()
lead = FOLDED.copy(); lead[1] = D(0.0); lead[2] = D(0.0)   # leader exactly on the 0 limit
cmds, _ = run(f, lead, 5.0)
jump = np.degrees(np.abs(np.array(cmds[-1])[:6] - FOLDED[:6])).max()
check("folded start: no jump", jump < 1.0, f"max move {jump:.2f} deg, mode {f.mode}")

# 4 blend speed <= 10 deg/s for 3 s, then follow <= 30 deg/s; acceleration limited
f = new()
start = FOLDED.copy(); start[1] = D(-30); start[2] = D(-40)
goal = start.copy(); goal[0] = D(60)                     # 60 deg base turn requested at once
cmds, _ = run(f, lambda t: start if t < 0.5 else goal, 8.0, measured=start)
v = np.degrees(np.abs(np.diff(np.array(cmds)[:, 0]))) / DT
blend_part = v[31:31 + 145]   # t 0.5 .. 2.9 s: still blending
check("blend speed <= 10 deg/s", blend_part.max() <= 10.0 + 1e-6, f"{blend_part.max():.2f}")
check("follow speed <= 30 deg/s", v.max() <= 30.0 + 1e-6, f"{v.max():.2f}")
dv = np.abs(np.diff(v))
big = np.where(dv / DT > 120.0 + 1e-3)[0]
# allowed: one final settling step onto the target (rounding), at most 3 deg/s of speed change
stop = int(np.argmax(v[60:] == 0.0)) + 60           # first standstill after the move
settle_ok = all(stop - 3 <= i < stop and dv[i] <= 3.0 for i in big)
check("acceleration <= 120 deg/s^2 (except the last 2 settle steps, <= 3 deg/s each)", settle_ok,
      f"max {dv.max() / DT:.1f} deg/s^2 over {len(big)} step(s), speed change {dv.max():.2f} deg/s")
check("reaches the goal", abs(math.degrees(cmds[-1][0]) - 60) < 0.5, f"{math.degrees(cmds[-1][0]):.1f}")

# 5 limits: target beyond the URDF limit is clamped to limit - 10 deg
f = new()
cmds, _ = run(f, lambda t: start if t < 0.5 else np.r_[start[:3], D(200), start[4:]], 15.0, measured=start)
check("joint4 clamped at 90-10 = 80 deg", abs(math.degrees(cmds[-1][3]) - (math.degrees(1.57) - 10)) < 0.2,
      f"{math.degrees(cmds[-1][3]):.1f}")

# 6 stale leader -> latched HOLD at the measured pose; stays held when the leader returns; rearm
f = new()
run(f, start, 4.0, measured=start)
moving = start.copy(); moving[0] = D(5)
c = f.step(10.0, DT, moving, 0.5, start, 0.0)              # leader 500 ms old
check("stale leader -> HOLD", f.mode == "HOLD", f.reason)
c2 = f.step(10.1, DT, moving, 0.0, start, 0.0)
check("HOLD is latched", f.mode == "HOLD" and np.allclose(c2, start))
f.rearm(10.2, start)
check("rearm -> WAIT_ALIGN", f.mode == "WAIT_ALIGN")

# 7 lost motor feedback -> HOLD
f = new()
run(f, start, 4.0, measured=start)
f.step(10.0, DT, start, 0.0, start, 0.2)
check("stale feedback -> HOLD", f.mode == "HOLD", f.reason)

# 8 temperature / torque / motor fault -> HOLD
for name, health in [("temperature", {"temp": [30, 30, 65, 30, 30, 30, 30]}),
                     ("torque", {"torque": [0, 22, 0, 0, 0, 0, 0]}),
                     ("fault", {"fault": "joint3: over-current"})]:
    f = new()
    run(f, start, 4.0, measured=start)
    f.step(10.0, DT, start, 0.0, start, 0.0, health)
    check(f"{name} -> HOLD", f.mode == "HOLD", f.reason)

# 9 tracking error (motors on): arm stuck while commanded away -> HOLD after 0.2 s
f = new(check_tracking=True)
cmds, t = run(f, lambda t: start if t < 0.5 else goal, 5.0, measured=start)   # measured never moves
check("arm stuck -> HOLD (collision)", f.mode == "HOLD", f.reason)

# 10 workspace: gripper below the table -> refuse to move further, no latch
f = new(fk=lambda q: (0.25, 0.0, 0.10 - 0.2 * (q[1] - start[1])))  # z falls as joint2 increases
down = start.copy(); down[1] = start[1] + D(60)
cmds, _ = run(f, lambda t: start if t < 0.5 else down, 10.0, measured=start)
z_end = 0.10 - 0.2 * (cmds[-1][1] - start[1])
check("never below 3 cm", z_end >= 0.03 - 1e-9, f"z={z_end * 100:.1f} cm, mode {f.mode}")

# 10b folded side of joint2/3: only 3 deg margin (arm can come down to rest)
f = new()
up = start.copy()
cmds, _ = run(f, lambda t: start if t < 0.5 else np.r_[start[0], 0.0, 0.0, start[3:]], 10.0, measured=start)
check("joint2/3 fold down to -3 deg (rest side margin)", abs(math.degrees(cmds[-1][1]) + 3) < 0.2 and abs(math.degrees(cmds[-1][2]) + 3) < 0.2,
      f"j2 {math.degrees(cmds[-1][1]):.1f}, j3 {math.degrees(cmds[-1][2]):.1f}")

# 10c already below the floor (drooped start): may move UP, never further down
f = new(fk=lambda q: (0.25, 0.0, 0.01 - 0.2 * (q[1] - start[1])))   # starts at z = 1 cm
upq = start.copy(); upq[1] = start[1] - D(20)                       # joint2 negative -> z rises
cmds, _ = run(f, lambda t: start if t < 0.5 else upq, 6.0, measured=start)
check("below the floor: moving up is allowed", abs(math.degrees(cmds[-1][1] - upq[1])) < 0.5, f"j2 {math.degrees(cmds[-1][1]):.1f} deg")
f = new(fk=lambda q: (0.25, 0.0, 0.01 - 0.2 * (q[1] - start[1])))
dn = start.copy(); dn[1] = start[1] + D(5)
cmds, _ = run(f, lambda t: start if t < 0.5 else dn, 6.0, measured=start)
check("below the floor: moving further down is refused", abs(cmds[-1][1] - start[1]) < 1e-6, f"j2 {math.degrees(cmds[-1][1]):.2f} deg")

# 11 gripper speed limit 3 cm/s
f = new()
g_open = start.copy(); g_open[6] = 0.05
cmds, _ = run(f, lambda t: start if t < 0.5 else g_open, 6.0, measured=start)
gv = np.abs(np.diff(np.array(cmds)[:, 6])) / DT
check("gripper <= 3 cm/s", gv.max() <= 0.03 + 1e-9, f"{gv.max() * 100:.2f} cm/s")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
