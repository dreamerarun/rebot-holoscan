"""What happened just before the last HOLD in the newest real_teleop log."""
import glob
import json
import math
import os

import numpy as np

f = max(glob.glob(os.path.expanduser("~/rebot-holoscan/logs/real_teleop_*.jsonl")), key=os.path.getmtime)
rows = [json.loads(l) for l in open(f)]
hold = next(i for i in range(len(rows)) if rows[i]["mode"] == "HOLD" and i > 0 and rows[i - 1]["mode"] == "FOLLOW"
            and rows[i]["t"] > rows[-1]["t"] - 600)
seg = rows[max(0, hold - 120): hold + 2]
t0 = seg[-2]["t"]
print(f"log {os.path.basename(f)}; HOLD at sample {hold}")
print("  t(s)  mode     cmd-measured deg  j1    j2    j3    j4    j5    j6  | torque j1..j6, grip")
for r in seg[::12] + seg[-3:]:
    if r["cmd"] and r["measured"]:
        d = np.degrees(np.array(r["cmd"][:6]) - np.array(r["measured"][:6]))
        tq = " ".join(f"{x:5.1f}" for x in (r["torque"] or []))
        print(f"{r['t'] - t0:6.2f}  {str(r['mode']):8s} {' '.join(f'{x:6.1f}' for x in d)} | {tq}")
r = seg[-2]
print("measured deg:", np.round(np.degrees(r["measured"][:7][:6]), 1))
print("command  deg:", np.round(np.degrees(r["cmd"][:6]), 1))
print("leader   deg:", np.round(np.degrees(r["target"][:6]), 1))
