import json, sys
sys.path.insert(0, "holoscan_app")
import numpy as np
from kinematics import RebotKinematics
k = RebotKinematics()
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r.get("cmd") and r.get("measured") and r.get("torque")]
    print("==", f.split("/")[-1])
    for i, (a, b) in enumerate(zip(rows, rows[1:])):
        if a["cmd"][6] > 0.02 >= b["cmd"][6]:            # gripper starts closing
            tc = k.fk(np.array(a["cmd"][:6])).translation
            if tc[2] > 0.05:
                continue
            tm = k.fk(np.array(a["measured"][:6])).translation
            e = np.degrees(np.array(a["cmd"][:6]) - np.array(a["measured"][:6]))
            up = rows[max(0, i - 400)]   # ~6 s earlier, high above
            print(f"  grasp cmd z {tc[2]*100:.1f} cm, measured z {tm[2]*100:.1f} cm | j2 err {e[1]:+.2f} j3 {e[2]:+.2f} j4 {e[3]:+.2f} deg"
                  f" | torque j2/j3/j4 {a['torque'][1]:+.2f}/{a['torque'][2]:+.2f}/{a['torque'][3]:+.2f} (high above: {up['torque'][1]:+.2f}/{up['torque'][2]:+.2f}/{up['torque'][3]:+.2f})")
