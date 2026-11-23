"""READ-ONLY power continuity test: poll all 7 motors at ~20 Hz, report any that stop answering."""
import math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "holoscan_app"))
from real_arm import READ_ONLY, RealArmReader, load_real_cfg
assert READ_ONLY
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 40
r = RealArmReader(load_real_cfg()); r.open()
t0, n, drops, j1min, j1max = time.time(), 0, {}, 999, -999
last_print = 0
while time.time() - t0 < secs:
    st = r.read(wait_s=0.03)
    n += 1
    for name, s in st.items():
        if s is None or s.status_code not in (0, 1):
            drops.setdefault(name, []).append(round(time.time() - t0, 2))
    if st.get("joint1") is not None:
        d = math.degrees(st["joint1"].pos) - 94.32
        j1min, j1max = min(j1min, d), max(j1max, d)
    if time.time() - last_print > 2:
        last_print = time.time()
        miss = [k for k, s in st.items() if s is None]
        print(f"{time.time() - t0:5.1f}s  base {math.degrees(st['joint1'].pos) - 94.32 if st.get('joint1') else float('nan'):6.1f} deg   "
              f"{'ALL 7 answering' if not miss else 'MISSING: ' + ', '.join(miss)}", flush=True)
    time.sleep(0.02)
r.close()
print(f"\n{n} polls, base turned from {j1min:.0f} to {j1max:.0f} deg")
print("RESULT: no dropouts" if not drops else "RESULT: DROPOUTS " + "; ".join(f"{k}: {len(v)}x (first at {v[0]}s)" for k, v in drops.items()))
