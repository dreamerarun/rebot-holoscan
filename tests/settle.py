import json, sys
sys.path.insert(0, "holoscan_app")
import numpy as np
from kinematics import RebotKinematics
k = RebotKinematics()
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r.get("cmd") and r.get("measured")]
for i, (a, b) in enumerate(zip(rows, rows[1:])):
    if a["cmd"][6] < 0.02 <= b["cmd"][6] and k.fk(np.array(a["cmd"][:6])).translation[2] < 0.06:
        tc = k.fk(np.array(a["cmd"][:6])).translation
        line = f"cmd ({tc[0]:.3f},{tc[2]*100:.1f}cm):"
        t0 = a["t"]
        for dt in (-1.0, -0.3, 0.0, 0.3, 0.6, 1.0, 1.5):
            r = min(rows[max(0, i - 200): i + 200], key=lambda r: abs(r["t"] - t0 - dt))
            tm = k.fk(np.array(r["measured"][:6])).translation
            line += f"  {dt:+.1f}s out {(np.hypot(*tm[:2]) - np.hypot(*tc[:2]))*1000:+.1f} z {(tm[2]-tc[2])*1000:+.1f}"
        print(line)
