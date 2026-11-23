"""Offline: every recorded expert action of every staged episode must pass the policy app's action checks."""
import glob, math, sys
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/holoscan_app")
import numpy as np, yaml
import policy_app as pa
from rebot_sim_pipeline.conventions import action_to_command
from kinematics import RebotKinematics
cfg = yaml.safe_load(open(__file__.rsplit("/tests/", 1)[0] + "/config/teleop.yaml"))
limits = [tuple(cfg["sim_limits"][f"joint{i}"]) for i in range(1, 7)]

class Fake:                       # just the bits _check_action uses
    k, limits, next_t = RebotKinematics(), limits, None
bad, n, worst_jump = [], 0, 0.0
for ep in sorted(glob.glob(__file__.rsplit("/tests/", 1)[0] + "/data/staging/cube_to_box_scripted/episode_*")):
    acts = np.load(ep + "/actions.npy")
    f = Fake()
    for i, a in enumerate(acts):
        q6, g = action_to_command(a); q7 = np.append(q6, g)
        if f.next_t is not None:
            worst_jump = max(worst_jump, float(np.degrees(np.max(np.abs(q7[:6] - f.next_t[:6])))))
        why = pa.PolicyOp._check_action(f, q7.copy())
        n += 1
        if why:
            bad.append((ep.rsplit("/", 1)[1], i, why))
        f.next_t = q7
print(f"{n} expert actions checked, {len(bad)} would stop the policy; biggest expert step {worst_jump:.1f} deg (limit {pa.MAX_JUMP_DEG})")
for b in bad[:5]:
    print("  ", b)
assert not bad
