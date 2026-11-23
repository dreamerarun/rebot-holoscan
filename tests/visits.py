import json, sys
sys.path.insert(0, "holoscan_app")
import numpy as np
from kinematics import RebotKinematics
k = RebotKinematics()
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r.get("cmd") and r.get("measured")]
vis, cur, last = [], None, -99
for i, r in enumerate(rows):
    t = k.fk(np.array(r["cmd"][:6])).translation
    if t[2] < 0.032:
        if i - last > 30:
            cur = []; vis.append(cur)
        cur.append((i, r, t)); last = i
for v in vis[-4:]:
    for i, r, tc in (v[len(v)//2], v[-1]):
        tm = k.fk(np.array(r["measured"][:6])).translation
        e = np.round(np.degrees(np.array(r["cmd"][:6]) - np.array(r["measured"][:6])), 2)
        g = r["cmd"][6] * 1000
        print(f"cmd ({tc[0]:.3f},{tc[1]:+.3f},{tc[2]:.3f}) meas ({tm[0]:.3f},{tm[1]:+.3f},{tm[2]:.3f}) grip {g:.0f}mm err {e}")
    print("--")
