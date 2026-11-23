import json, sys
sys.path.insert(0, "holoscan_app")
import numpy as np
from kinematics import RebotKinematics
k = RebotKinematics()
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r.get("cmd") and r.get("measured")]
# placements: gripper command switches from closed (<20 mm) to open while the TCP is low
for a, b in zip(rows, rows[1:]):
    if a["cmd"][6] < 0.02 <= b["cmd"][6]:
        tc = k.fk(np.array(a["cmd"][:6])).translation
        if tc[2] > 0.06:
            continue
        tm = k.fk(np.array(a["measured"][:6])).translation
        e = np.degrees(np.array(a["cmd"][:6]) - np.array(a["measured"][:6]))
        print(f"release at cmd ({tc[0]:.3f},{tc[1]:+.3f}) z {tc[2]*100:.1f} cm | measured ({tm[0]:.3f},{tm[1]:+.3f}) z {tm[2]*100:.1f} cm"
              f" | out {(np.hypot(*tm[:2]) - np.hypot(*tc[:2]))*1000:+.1f} mm | j2 {e[1]:+.2f} j3 {e[2]:+.2f} deg")
