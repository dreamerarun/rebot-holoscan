"""Robot-driven camera calibration: the arm places the cube on a 3x3 grid, the camera photographs it.

    ScriptOp (60 Hz plan) -> arm  -> (twin)
       --sim : SimArmOp  = SafetyFilter + Isaac Sim twin (physics cube), no real motors
       --live: LiveArmOp = SafetyFilter + the real arm (same code as teleop) + twin shows the command

The whole plan is computed and validated before it starts (motion_plan.py). Any HOLD, failed grasp
or key x aborts: the arm then just holds where it is (park with p).
Keys (terminal or `echo X >> ~/rebot-holoscan/keys`):
    g  go on (after placing the cube under the gripper)      x  abort
    live only: e enable  s hold  a re-arm  p park  q park + quit
Result (live): config/camera_table.yaml (the old file is kept as camera_table.prev.yaml)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec

import real_teleop as rt
from common import SIM_JOINTS, load_cfg
from motion_plan import CAL_GRID, HANDOVER, PHOTO_Q, RATE, Z_MIN_NEAR, MotionPlanner
from safety import SafetyFilter, SafetyParams

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_KEYS = []
CAM = {"frame": None, "t": 0.0, "fps": 0.0}          # latest camera frame (live view owns the camera)
STATUS = {"step": "waiting for the arm", "points": 0, "last": ""}
GRASP_RANGE = (0.004, 0.025)     # finger opening (m) that means "holding the 4 cm cube"


def say(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------- script
class ScriptOp(Operator):
    def __init__(self, fragment, *args, arm_op, limits, sim, **kw):
        self.arm_op, self.limits, self.sim = arm_op, limits, sim
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.output("target")

    def start(self):
        self.planner = MotionPlanner(self.limits)
        self.state, self.segs, self.si, self.i = "WAIT_ARM", None, 0, 0
        self.hold_q = None
        self.pairs, self.photo = [], None
        self.t_wait = None
        say("[cal] waiting for the arm" + (" (sim twin)" if self.sim else " - type e to enable it"))

    def _abort(self, why):
        STATUS["step"], STATUS["last"] = "ABORTED", why
        if self.state not in ("ABORT", "DONE"):
            say(f"[cal] ABORTED: {why}. The arm holds where it is" + ("" if self.sim else " - type p to park"))
            self.state = "ABORT"

    def _emit(self, op_output, q7):
        op_output.emit({"t": time.perf_counter(), "q": np.asarray(q7, float), "ok": True}, "target")

    def compute(self, op_input, op_output, context):
        last = getattr(self.arm_op, "last", None)
        meas = None if last is None else last.get("measured")
        for k in SCRIPT_KEYS[:]:
            SCRIPT_KEYS.remove(k)
            if k == "x":
                self._abort("abort key")
            elif k == "g" and self.state == "WAIT_KEY":
                say("[cal] go")
                self._next(None)

        if self.state == "WAIT_ARM":
            if last is not None and last.get("mode") == "FOLLOW" and meas is not None:
                # plan BEFORE emitting: Holoscan delivers a message when compute() returns, so anything emitted
                # before the ~1.4 s of planning would arrive 1.4 s old and the SafetyFilter would (rightly) HOLD
                try:
                    self.segs = self.planner.calibration(meas, CAL_GRID, HANDOVER, PHOTO_Q)
                    rep = self.planner.validate(self.segs, meas)
                except ValueError as exc:
                    self._abort(f"plan rejected: {exc} (start pose deg {np.degrees(meas[:6]).round(1).tolist()})")
                    return
                say(f"[cal] plan: {len(self.segs)} moves, {sum(r[1] for r in rep) / 60:.1f} min, "
                    f"lowest planned gripper point {min(r[2] for r in rep[1:]) * 100:.1f} cm - validated, starting")
                self.state, self.si, self.i = "RUN", 0, 0
                say(f"[cal] > {self.segs[0].label}")
                self._emit(op_output, meas)          # fresh timestamp, after planning
                return
            if meas is not None:
                self._emit(op_output, meas)
            else:   # keep the arm operator ticking until it has its first reading
                op_output.emit({"t": time.perf_counter(), "q": None, "ok": False}, "target")
            return

        if self.state in ("ABORT", "DONE"):
            if self.state == "DONE" and self.hold_q is not None:
                self._emit(op_output, self.hold_q)
            elif meas is not None:
                self._emit(op_output, meas)
            return

        if last is None or last.get("mode") not in ("FOLLOW", "BLEND"):
            self._abort(f"arm left FOLLOW ({None if last is None else last.get('mode')})")
            return

        seg = self.segs[self.si]
        if self.state == "WAIT_KEY":
            self._emit(op_output, seg.q[-1])
            if self.sim and time.perf_counter() - self.t_wait > 1.5:
                say("[cal] (sim) cube is under the gripper - go")
                self._next(None)
            return
        if self.state == "PHOTO":
            self._emit(op_output, seg.q[-1])
            if self.photo is None or self.photo.is_alive():
                return
            self._photo_done(seg)
            self._next(op_output)
            return

        if self.state == "ARRIVE":                 # plan step done: has the real/sim arm arrived?
            self._emit(op_output, seg.q[-1])
            err = 999.0 if meas is None else float(np.degrees(np.max(np.abs(np.asarray(meas)[:6] - seg.q[-1][:6]))))
            if err < 3.0:
                self.state = "RUN"
                self._event(seg, meas)
            elif time.perf_counter() - self.t_arrive > 4.0:
                self._abort(f"arm did not reach the end of '{seg.label}' ({err:.1f} deg off after 4 s)")
            return

        q7 = seg.q[self.i]
        self.hold_q = q7
        self._emit(op_output, q7)
        self.i += 1
        if self.i >= len(seg.q):
            self.state, self.t_arrive = "ARRIVE", time.perf_counter()

    def _event(self, seg, meas):
        ev = seg.event
        if ev.startswith("wait_key"):
            self.state, self.t_wait = "WAIT_KEY", time.perf_counter()
            say("[cal] put the cube on the table UNDER the open gripper, step back, then type g")
            return
        if ev == "check_grasp":
            g = None if meas is None else float(meas[6])
            if g is None or not GRASP_RANGE[0] <= g <= GRASP_RANGE[1]:
                self._abort(f"grasp check failed (finger opening {('?' if g is None else f'{g * 1000:.1f} mm')}, "
                            f"expected {GRASP_RANGE[0] * 1000:.0f}-{GRASP_RANGE[1] * 1000:.0f} mm)")
                return
            say(f"[cal]   holding the cube (finger opening {g * 1000:.1f} mm)")
        if ev.startswith("photo"):
            if self.sim:
                say(f"[cal]   photo {ev.split(':')[1]} (sim: skipped, camera is the real one)")
            else:
                self.state = "PHOTO"
                self.photo = threading.Thread(target=self._take_photo, args=(seg,), daemon=True)
                self.photo.start()
                return
        if ev == "done":
            self.state = "DONE"
            self._finish()
            return
        self._next(None)

    def _next(self, op_output):
        self.si += 1
        self.i = 0
        self.state = "RUN"
        if self.si < len(self.segs):
            STATUS["step"] = self.segs[self.si].label
            say(f"[cal] > {self.segs[self.si].label}")

    # ------------------------------------------------------------------ photos (live)
    def _take_photo(self, seg):
        """Use frames from the live camera stream taken AFTER the arm stopped: median of 5 clean detections."""
        from perception import detect_cube
        t_req, seen, pts, last_img = time.time(), 0.0, [], None
        while time.time() - t_req < 3.0 and len(pts) < 5:
            if CAM["frame"] is not None and CAM["t"] > max(t_req + 0.3, seen):
                seen, img = CAM["t"], CAM["frame"]
                last_img = img
                c = detect_cube(img)
                if c.ok:
                    pts.append(c.centroid)
            time.sleep(0.02)
        if len(pts) >= 3:
            seg.info["result"] = (tuple(np.median(np.array(pts), axis=0)), last_img)
        else:
            seg.info["result"] = None
            if last_img is not None:
                cv2.imwrite(str(ROOT / "logs" / "cam" / f"cal_miss_{seg.event.split(':')[1]}.jpg"), last_img)

    def _photo_done(self, seg):
        res = seg.info.get("result")
        x, y = seg.info["xy"]
        if res is None:
            STATUS["last"] = f"point at ({x:.2f}, {y:+.2f}): cube NOT seen - skipped"
            say(f"[cal]   photo: cube NOT detected at ({x:.2f}, {y:+.2f}) - point skipped (frame saved)")
            return
        (px, py), img = res
        self.pairs.append({"px": [float(px), float(py)], "xy": [x, y]})
        cv2.imwrite(str(ROOT / "logs" / "cam" / f"cal_{len(self.pairs):02d}.jpg"), img)
        STATUS["points"], STATUS["last"] = len(self.pairs), f"photo OK: pixel ({px:.0f}, {py:.0f}) = table ({x:.2f}, {y:+.2f})"
        say(f"[cal]   photo: cube at pixel ({px:.0f}, {py:.0f}) <-> table ({x:.2f}, {y:+.2f})  [{len(self.pairs)} points]")

    def _finish(self):
        say("[cal] sequence finished")
        if self.sim or len(self.pairs) < 6:
            if not self.sim:
                say(f"[cal] only {len(self.pairs)} points - calibration NOT saved")
            return
        from calibrate_camera import solve
        H, e_fit, e_loo = solve(self.pairs)
        out = ROOT / "config" / "camera_table.yaml"
        if out.exists():
            shutil.copy(out, ROOT / "config" / "camera_table.prev.yaml")
        out.write_text(yaml.safe_dump({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "method": "robot-placed cube, 3x3 grid",
            "camera": "C270", "image_size": [640, 480], "H": H.tolist(),
            "fit_error_mm_max": float(e_fit.max()), "leave_one_out_mm_mean": float(e_loo.mean()),
            "leave_one_out_mm_max": float(e_loo.max()), "points": self.pairs}, sort_keys=False))
        say(f"[cal] SAVED camera_table.yaml: fit max {e_fit.max():.1f} mm, unseen points mean "
            f"{e_loo.mean():.1f} / max {e_loo.max():.1f} mm")


# ---------------------------------------------------------------------------- live camera view
class CamViewOp(Operator):
    """Owns the camera: live Holoviz window with the cube/box outlines and the script status; keeps the latest
    frame in CAM for the photos."""

    def setup(self, spec: OperatorSpec):
        spec.output("tensors")
        spec.output("specs")

    def start(self):
        from perception import open_camera
        self.cap, idx = open_camera()
        self.n, self.t0 = 0, time.time()
        say(f"[cam] live view from camera index {idx}")

    def stop(self):
        self.cap.release()

    def compute(self, op_input, op_output, context):
        import cupy as cp
        from holoscan.operators import HolovizOp
        from perception import detect_box, detect_cube, draw
        ok, bgr = self.cap.read()
        if not ok:
            time.sleep(0.05)
            return
        now = time.time()
        CAM["frame"], CAM["t"] = bgr, now
        self.n += 1
        if now - self.t0 > 2.0:
            CAM["fps"], self.n, self.t0 = self.n / (now - self.t0), 0, now
        view = draw(bgr, detect_cube(bgr), detect_box(bgr))
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        lines = [f"step: {STATUS['step']}", f"calibration points: {STATUS['points']} / 9   camera {CAM['fps']:4.1f} fps"]
        if STATUS["last"]:
            lines.append(STATUS["last"])
        lines.append("keys in the terminal:  g go   x abort   s hold   p park   q quit")
        spec = HolovizOp.InputSpec("txt", "text")
        spec.text, spec.color, spec.priority = lines, [1.0, 1.0, 0.3, 1.0], 10
        op_output.emit({"video": cp.asarray(rgb),
                        "txt": np.asarray([(0.015, 0.03 + 0.05 * i, 0.04) for i in range(len(lines))], dtype=np.float32)},
                       "tensors")
        op_output.emit([spec], "specs")


# ---------------------------------------------------------------------------- sim arm
class SimArmOp(Operator):
    """SafetyFilter + Isaac Sim twin: same interface (.last) as LiveArmOp, no real motors."""

    def __init__(self, fragment, *args, cfg, limits, **kw):
        self.cfg, self.limits = cfg, limits
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.input("target", policy=IOSpec.QueuePolicy.POP)

    def start(self):
        from common import SimLink
        self.link = SimLink(self.cfg)
        self.filter = None
        self.last = None
        self.t_target, self.target = None, None
        self.reset_sent = False

    def stop(self):
        self.link.close()

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("target")
        now = time.perf_counter()
        if msg and msg.get("q") is not None and (self.t_target is None or msg["t"] > self.t_target):
            self.target, self.t_target = np.asarray(msg["q"], float), msg["t"]
        st = self.link.poll_state()
        if not st or time.time() - st.get("t", 0) > 1.0:
            return
        q = dict(zip(st["names"], st["q"]))
        meas = np.array([q[f"joint{i}"] for i in range(1, 7)] + [q["finger_left"]])
        cube = st.get("cube")
        if not self.reset_sent and cube is not None:   # cube onto the handover point (resend until it is there)
            if math.hypot(cube[0] - HANDOVER[0], cube[1] - HANDOVER[1]) < 0.01:
                self.reset_sent = True
            elif now - getattr(self, "t_reset", 0.0) > 0.5:
                self.link.pub.send_string(json.dumps({"reset": True, "cube_xy": list(HANDOVER), "cube_yaw": 0.0}))
                self.t_reset = now
        if self.filter is None:
            self.filter = SafetyFilter(SafetyParams(limits=self.limits, vmax_deg_s=15.0, check_tracking=True,
                                                    tcp_min_z=Z_MIN_NEAR),
                                       fk=lambda qa: self.k.fk(qa).translation)
        cmd = self.filter.step(now, 1 / RATE, self.target, 0.0 if self.target is not None else 1e9, meas, 0.0, None)
        if cmd is not None:
            self.link.send(list(cmd[:6]) + [float(cmd[6]), -float(cmd[6])])
        self.last = {"t": now, "state": "ON", "mode": self.filter.mode, "cmd": cmd, "measured": meas}

    @property
    def k(self):
        if not hasattr(self, "_k"):
            from kinematics import RebotKinematics
            self._k = RebotKinematics()
        return self._k


# ---------------------------------------------------------------------------- app
class CalApp(Application):
    def __init__(self, a, cfg):
        self.a, self.cfg = a, cfg
        super().__init__()

    def compose(self):
        limits = [tuple(self.cfg["sim_limits"][f"joint{i}"]) for i in range(1, 7)]
        if self.a.sim:
            arm = SimArmOp(self, cfg=self.cfg, limits=limits, name="sim_arm")
        else:
            from real_common import load_real_cfg
            rt.LiveArmOp.tcp_min_z = Z_MIN_NEAR
            arm = rt.LiveArmOp(self, rcfg=load_real_cfg(), limits=limits, vmax=15.0, dry=False,
                               with_gripper=True, grip_force=0.10, name="live_arm")
            self.add_flow(arm, rt.SimOp(self, cfg=self.cfg, name="sim"))
            from holoscan.operators import HolovizOp
            cam = CamViewOp(self, PeriodicCondition(self, recess_period=timedelta(milliseconds=2)), name="camera")
            viz = HolovizOp(self, name="view", width=960, height=720, window_title="reBot calibration - live camera",
                            tensors=[dict(name="video", type="color")])
            self.add_flow(cam, viz, {("tensors", "receivers"), ("specs", "input_specs")})
        script = ScriptOp(self, PeriodicCondition(self, recess_period=timedelta(seconds=1 / RATE)),
                          arm_op=arm, limits=limits, sim=self.a.sim, name="script")
        self.add_flow(script, arm)


def route(k):
    if k in ("g", "x"):
        SCRIPT_KEYS.append(k)
    elif k in ("e", "a", "s", "p", "q", "o"):
        rt.KEYS.append(k)


def key_sources(keys_file):
    def stdin_loop():
        for line in sys.stdin:
            route(line.strip().lower()[:1])

    def file_loop():
        import os
        open(keys_file, "a").close()
        with open(keys_file) as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                k = line.strip().lower()[:1]
                if k:
                    say(f"[cal] key '{k}'")
                    route(k)
    if sys.stdin.isatty():
        threading.Thread(target=stdin_loop, daemon=True).start()
    if keys_file:
        threading.Thread(target=file_loop, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--sim", action="store_true", help="Isaac Sim twin only (start ./run_twin_gui.sh --vision)")
    m.add_argument("--live", action="store_true", help="the real arm")
    ap.add_argument("--keys-file", default=str(ROOT / "keys"))
    a = ap.parse_args()
    key_sources(a.keys_file)
    from holoscan.schedulers import MultiThreadScheduler
    app = CalApp(a, load_cfg())
    app.scheduler(MultiThreadScheduler(app, name="mt", worker_thread_number=6, stop_on_deadlock=False))
    app.run()


if __name__ == "__main__":
    main()
