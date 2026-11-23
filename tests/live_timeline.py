import json, sys
sys.path.insert(0, "holoscan_app")
import numpy as np
from kinematics import RebotKinematics
k = RebotKinematics()
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r.get("cmd") is not None and r.get("measured") is not None]
t_end = rows[-1]["t"]
sel = [r for r in rows if r["t"] > t_end - 75]
last_t = -9
for r in sel:
    if r["t"] - last_t < 2.5:
        continue
    last_t = r["t"]
    p = k.fk(np.array(r["measured"][:6])).translation
    print(f"t-{t_end - r['t']:5.1f}s  gripper xyz ({p[0]:.3f},{p[1]:+.3f},{p[2]*100:5.1f}cm)  grip cmd {r['cmd'][6]*1000:5.1f} mm  meas {r['measured'][6]*1000:5.1f} mm  {r.get('mode')}")
