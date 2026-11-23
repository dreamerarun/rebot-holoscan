"""Gripper command vs reading over the newest real_teleop log."""
import glob, json, os
f = max(glob.glob(os.path.expanduser("~/rebot-holoscan/logs/real_teleop_*.jsonl")), key=os.path.getmtime)
rows = [json.loads(l) for l in open(f)]
on = [r for r in rows if r["state"] == "ON" and r["cmd"] and r["measured"]]
print(f"{len(on)} samples while ON; gripper command vs reading (mm), every 0.5 s:")
t0, last = on[0]["t"], -1.0
for r in on:
    if r["t"] - last >= 0.5:
        last = r["t"]
        tq = r["torque"][6] if r["torque"] else None
        print(f"  t={r['t'] - t0:5.1f}s  mode {str(r['mode']):9s} cmd {r['cmd'][6] * 1000:5.1f}  reading {r['measured'][6] * 1000:5.1f}  grip torque {tq}")
