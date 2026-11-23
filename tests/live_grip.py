import json, sys
sys.path.insert(0, "holoscan_app")
import numpy as np
from kinematics import RebotKinematics
k = RebotKinematics()
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r.get("cmd") is not None and r.get("measured") is not None]
closed = [r for r in rows if r["cmd"][6] < 0.012]
print(f"rows {len(rows)}, gripper commanded closed in {len(closed)} rows ({len(closed)/60:.1f} s)")
if closed:
    g = np.array([r["measured"][6] for r in closed]) * 1000
    z = np.array([k.fk(np.array(r["measured"][:6])).translation[2] for r in closed]) * 100
    print(f"  finger opening while closed: median {np.median(g):.1f} mm (holding a cube ~8-9 mm, empty ~0.3 mm), min {g.min():.1f} max {g.max():.1f}")
    i0 = rows.index(closed[0])
    p = k.fk(np.array(rows[i0]["measured"][:6])).translation
    print(f"  closed at gripper ({p[0]:.3f}, {p[1]:+.3f}) z {p[2]*100:.1f} cm; highest lift while closed {z.max():.1f} cm")
    last = closed[-1]; p = k.fk(np.array(last["measured"][:6])).translation
    print(f"  opened again at ({p[0]:.3f}, {p[1]:+.3f}) z {p[2]*100:.1f} cm  (box is around (0.20, -0.23))")
tq = np.array([np.max(np.abs(r["torque"][:6])) for r in rows if r.get("torque")])
print(f"  max joint torque {tq.max():.1f} Nm; modes seen {sorted({r.get('mode') for r in rows})}")
