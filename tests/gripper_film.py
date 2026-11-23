"""READ ONLY: film the gripper with the camera while reading the gripper motor (2 frames/s, 30 s)."""
import sys, time, math
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "holoscan_app"))
from perception import open_camera
from real_arm import READ_ONLY, RealArmReader, load_real_cfg
assert READ_ONLY
out = Path(__file__).resolve().parents[1] / "logs" / "gripper_film"
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.jpg"):
    f.unlink()
r = RealArmReader(load_real_cfg()); r.open()
cap, _ = open_camera()
t0, k = time.time(), 0
while time.time() - t0 < float(sys.argv[1] if len(sys.argv) > 1 else 30):
    ok, img = cap.read()
    st = r.read()["gripper"]
    if ok and st is not None and time.time() - t0 >= k * 0.5:
        deg = math.degrees(st.pos)
        crop = img[60:300, 150:460]            # gripper region at the rest pose
        crop = cv2.resize(crop, None, fx=1.5, fy=1.5)
        cv2.putText(crop, f"{deg:7.1f} deg  t={time.time() - t0:4.1f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(out / f"g_{k:03d}.jpg"), crop)
        print(f"{time.time() - t0:5.1f}s  {deg:7.1f}", flush=True)
        k += 1
cap.release(); r.close()
