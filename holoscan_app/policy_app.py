"""Trained ACT policy run as a Holoscan pipeline, shown live in Holoviz.

    source --frame--> policy --target--> arm (SafetyFilter inside) [--> Isaac twin mirror]
                         \\--viz--> view --> Holoviz window (camera + planned path + timings)

  source  : --live  C270 camera (30 fps)             | --replay N  a recorded episode (15 fps, no hardware)
  policy  : 60 Hz loop; at 15 Hz sends camera + arm state to the ACT policy server (GPU) and gets the next action;
            interpolates between actions; checks every action (finite, reach box, no jumps, run time) -> else STOP
  arm     : --live  LiveArmOp: real arm via the SafetyFilter (15 deg/s, floor 3.2 cm, torque, tracking, temps)
            --replay SimArmOp: the Isaac twin executes the policy's actions (real arm untouched)

Flow: e (live) / automatic (replay) -> arm goes to the observe pose -> g starts the policy -> x stops (holds),
r rises and returns to the observe pose, then g again. Live arm keys: s hold, p park, q park + quit.
Replay mode compares every action with what the scripted expert did in that recording (open-loop check).
Needs the policy server running:  ~/rebot_lerobot/.venv/bin/python holoscan_app/policy_server.py --ckpt <...>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import zmq
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, ConditionType, IOSpec, Operator, OperatorSpec

import real_teleop as rt
import robot_calibrate as rc
from common import load_cfg
from motion_plan import PHOTO_Q, RATE, READY_Q, Z_MIN_NEAR, MotionPlanner
from perception import load_calibration
from rebot_sim_pipeline.conventions import action_to_command, arm_state

ROOT = Path(__file__).resolve().parents[1]
KEYS = []
POLICY_HZ = 15                    # the dataset's rate: one policy step per 4 ticks of the 60 Hz loop
STEP = int(round(RATE / POLICY_HZ))
REACH_BOX = {"x": (0.08, 0.40), "y": (-0.32, 0.20), "z": (0.0, 0.40)}   # gripper must stay in here (cube zone + box)
MAX_JUMP_DEG = 20.0               # one policy step may not move any joint more than this
LIMIT_STOP = math.radians(10.0)   # an action more than this beyond a joint limit stops the policy
MAX_RUN_S = 120.0
SERVER_TIMEOUT_MS = 500
STATS = {}                        # per-operator compute times for the overlay


def say(m):
    print(m, flush=True)


def timed(name):
    def deco(fn):
        def wrap(self, *a, **k):
            t0 = time.perf_counter()
            try:
                return fn(self, *a, **k)
            finally:
                ms = (time.perf_counter() - t0) * 1000
                STATS[name] = 0.9 * STATS.get(name, ms) + 0.1 * ms
        return wrap
    return deco


# ---------------------------------------------------------------------------- sources
class CamSourceOp(Operator):
    """Live C270 camera -> 'frame' (RGB + BGR). Also keeps rc.CAM up to date (shared camera conventions)."""

    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def start(self):
        from perception import open_camera
        self.cap, idx = open_camera()
        self.n, self.t0 = 0, time.time()
        say(f"[cam] camera index {idx}")

    def stop(self):
        self.cap.release()

    @timed("camera")
    def compute(self, op_input, op_output, context):
        ok, bgr = self.cap.read()
        if not ok:
            time.sleep(0.02)
            return
        now = time.time()
        rc.CAM["frame"], rc.CAM["t"] = bgr, now
        self.n += 1
        if now - self.t0 > 2.0:
            rc.CAM["fps"], self.n, self.t0 = self.n / (now - self.t0), 0, now
        op_output.emit({"bgr": bgr, "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "t": now}, "frame")


class ReplaySourceOp(Operator):
    """A recorded episode at 15 fps: frame + the recorded state and expert action. Advances only while the policy
    runs (RUNNING flag), otherwise shows frame 0."""
    RUNNING = {"on": False, "done": False}

    def __init__(self, fragment, *args, episode_dir, **kw):
        self.dir = Path(episode_dir)
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def start(self):
        self.files = sorted((self.dir / "images" / "front").glob("frame_*.jpg"))
        self.states, self.actions = np.load(self.dir / "states.npy"), np.load(self.dir / "actions.npy")
        self.i = 0
        say(f"[replay] {self.dir.name}: {len(self.files)} frames")

    @timed("replay")
    def compute(self, op_input, op_output, context):
        if self.RUNNING["on"] and self.i < len(self.files) - 1:
            self.i += 1
        elif self.RUNNING["on"]:
            self.RUNNING["done"] = True
        elif not self.RUNNING["on"]:
            self.i = 0
        bgr = cv2.imread(str(self.files[self.i]))
        op_output.emit({"bgr": bgr, "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "t": time.time(),
                        "state": self.states[self.i], "gt": self.actions[self.i], "i": self.i,
                        "n": len(self.files)}, "frame")


# ---------------------------------------------------------------------------- policy client
class PolicyClient:
    def __init__(self, port=5610):
        self.ctx, self.port = zmq.Context.instance(), port
        self._new()

    def _new(self):
        self.s = self.ctx.socket(zmq.REQ)
        self.s.setsockopt(zmq.LINGER, 0)
        self.s.setsockopt(zmq.RCVTIMEO, SERVER_TIMEOUT_MS)
        self.s.connect(f"tcp://127.0.0.1:{self.port}")

    def call(self, header, img=None):
        try:
            self.s.send_multipart([json.dumps(header).encode()] + ([img.tobytes()] if img is not None else []))
            return self.s.recv_json()
        except zmq.error.Again:
            self.s.close()
            self._new()                   # a REQ socket is stuck after a timeout
            return {"error": f"no reply from the policy server within {SERVER_TIMEOUT_MS} ms"}


# ---------------------------------------------------------------------------- policy loop
class PolicyOp(Operator):
    def __init__(self, fragment, *args, arm_op, limits, replay, **kw):
        self.arm_op, self.limits, self.replay = arm_op, limits, replay
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.input("frame", policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        spec.output("target")
        spec.output("viz")

    def start(self):
        from kinematics import RebotKinematics
        self.k, self.mp, self.client = RebotKinematics(), MotionPlanner(self.limits), PolicyClient()
        self.state, self.why = "WAIT_ARM", ""
        self.segs, self.si, self.i, self.after = [], 0, 0, "READY"
        self.hold_q = None
        self.frame = None
        self.tick, self.steps, self.t_run = 0, 0, 0.0
        self.prev_t, self.next_t, self.interp = None, None, 0
        self.last_reply, self.rt_ms, self.gt_err = {}, 0.0, None
        self.t_viz = 0.0
        say("[pol] waiting for the arm" + ("" if self.replay else " - type e to enable it"))

    def _emit(self, op_output, q7):
        op_output.emit({"t": time.perf_counter(), "q": np.asarray(q7, float), "ok": True}, "target")

    def _emit_or_none(self, op_output, q7):
        # the arm operators only tick when a message arrives: EVERY policy tick must send one, even "no target"
        if q7 is not None:
            self._emit(op_output, q7)
        else:
            op_output.emit({"t": time.perf_counter(), "q": None, "ok": False}, "target")

    def _plan(self, segs, meas, label, after):
        try:
            self.mp.validate(segs, meas)
        except ValueError as exc:
            self._stop(f"plan rejected: {exc}")
            return
        self.segs, self.si, self.i, self.state, self.after = segs, 0, 0, "MOVE", after
        say(f"[pol] > {label}: {sum(len(s.q) for s in segs) / RATE:.0f} s, validated")

    def _stop(self, why):
        if self.state == "RUN":
            say(f"[pol] policy STOPPED after {self.steps} steps: {why}. Holding - r returns to the observe pose")
        elif self.state != "STOPPED":
            say(f"[pol] STOPPED: {why}")
        self.state, self.why = "STOPPED", why
        ReplaySourceOp.RUNNING["on"] = False

    def _check_action(self, q7):
        if not np.all(np.isfinite(q7)):
            return "action is not finite"
        for j, (lo, hi) in enumerate(self.limits):
            # small overshoots happen at the observe pose (joint2/3 sit 3 deg from their 0 limit): clip them (the
            # SafetyFilter clamps too); far outside means a broken output -> stop
            if not lo - LIMIT_STOP <= q7[j] <= hi + LIMIT_STOP:
                return f"joint{j + 1} {math.degrees(q7[j]):.0f} deg far outside its limits"
            q7[j] = min(max(q7[j], lo), hi)
        p = self.k.fk(q7[:6]).translation
        for ax, v in zip("xyz", p):
            lo, hi = REACH_BOX[ax]
            if not lo <= v <= hi:
                return f"gripper {ax} = {v:.2f} m outside the allowed box {REACH_BOX[ax]}"
        if self.next_t is not None:
            jump = float(np.degrees(np.max(np.abs(q7[:6] - self.next_t[:6]))))
            if jump > MAX_JUMP_DEG:
                return f"policy jumped {jump:.0f} deg in one step"
        return None

    @timed("policy")
    def compute(self, op_input, op_output, context):
        f = op_input.receive("frame")
        if f is not None:
            self.frame = f
        last = getattr(self.arm_op, "last", None)
        meas = None if last is None else last.get("measured")
        self.tick += 1
        for key in KEYS[:]:
            KEYS.remove(key)
            if key == "x" and self.state in ("RUN", "MOVE"):
                self._stop("stop key")
            elif key == "g" and self.state == "READY":
                r = self.client.call({"op": "reset"})
                if "error" in r:
                    self._stop(r["error"])
                else:
                    self.state, self.steps, self.t_run = "RUN", 0, time.perf_counter()
                    self.prev_t = self.next_t = np.asarray(self.hold_q, float)
                    self.interp = 0
                    ReplaySourceOp.RUNNING.update(on=True, done=False)
                    say("[pol] policy RUNNING - x stops")
            elif key == "r" and self.state in ("STOPPED", "READY") and meas is not None:
                rise = self.mp.park(meas, float(meas[6]))[:-1]          # straight up only
                q_up = rise[-1].q[-1][:6] if rise else meas[:6]
                back = self.mp.joint_move("back to the observe pose", q_up, PHOTO_Q, float(meas[6]))
                back.z_min = 0.12
                self._plan(rise + [back], meas, "rise, back to the observe pose", "READY")

        try:
            self._tick(op_output, last, meas)
        finally:
            if time.perf_counter() - self.t_viz > 1.0 / POLICY_HZ and self.frame is not None:
                self.t_viz = time.perf_counter()
                op_output.emit({"frame": self.frame, "state": self.state, "why": self.why, "steps": self.steps,
                                "reply": self.last_reply, "rt_ms": self.rt_ms, "gt_err": self.gt_err,
                                "arm": last, "target": self.hold_q, "replay": self.replay}, "viz")

    def _tick(self, op_output, last, meas):
        if self.state == "WAIT_ARM":
            if last is not None and last.get("mode") == "FOLLOW" and meas is not None:
                segs = [self.mp.joint_move("to the ready pose", meas[:6], READY_Q, float(meas[6])),
                        self.mp.joint_move("to the observe pose", READY_Q, PHOTO_Q, float(meas[6]))]
                self.hold_q = np.asarray(meas, float)
                self._plan(segs, meas, "go to the observe pose", "READY")
            self._emit_or_none(op_output, meas)
            return
        if last is None or last.get("mode") not in ("FOLLOW", "BLEND"):
            if self.state != "STOPPED":
                self._stop(f"arm left FOLLOW ({None if last is None else last.get('mode')})")
            self._emit_or_none(op_output, meas)     # keep the arm operator ticking (its keys: p park, q quit)
            return

        if self.state == "MOVE":
            seg = self.segs[self.si]
            if self.i < len(seg.q):
                self.hold_q = seg.q[self.i]
                self.i += 1
            elif self.si < len(self.segs) - 1:
                self.si, self.i = self.si + 1, 0
            elif meas is not None and np.degrees(np.max(np.abs(meas[:6] - seg.q[-1][:6]))) < 3.0:
                self.state = self.after
                if self.state == "READY":
                    say("[pol] at the observe pose - put the cube in the green zone, step back, then g")
            self._emit(op_output, self.hold_q)
            return

        if self.state in ("READY", "STOPPED"):
            self._emit_or_none(op_output, self.hold_q if self.hold_q is not None else meas)
            return

        # ---------------- RUN
        if time.perf_counter() - self.t_run > MAX_RUN_S:
            self._stop(f"run time over {MAX_RUN_S:.0f} s")
        elif self.replay and ReplaySourceOp.RUNNING["done"]:
            self._stop("end of the recorded episode")
        elif self.tick % (STEP) == 0:
            self._policy_step(meas)
        if self.state == "RUN":                  # interpolate prev -> next over one policy period
            self.interp = min(self.interp + 1, STEP)
            a = self.interp / (STEP)
            self.hold_q = (1 - a) * self.prev_t + a * self.next_t
        self._emit(op_output, self.hold_q)

    def _policy_step(self, meas):
        f = self.frame
        if f is None or (not self.replay and time.time() - f["t"] > 0.3):
            self._stop("no fresh camera frame")
            return
        state = f["state"] if self.replay else arm_state(meas[:6], meas[6])
        t0 = time.perf_counter()
        r = self.client.call({"op": "act", "state": np.asarray(state, float).tolist(), "shape": list(f["rgb"].shape)},
                             np.ascontiguousarray(f["rgb"]))
        self.rt_ms = (time.perf_counter() - t0) * 1000
        if "error" in r:
            self._stop(r["error"])
            return
        q6, grip = action_to_command(np.asarray(r["action"], float))
        q7 = np.append(q6, grip)
        bad = self._check_action(q7)
        if bad:
            self._stop(bad)
            return
        self.last_reply = r
        if self.replay:
            self.gt_err = np.abs(np.asarray(r["action"]) - np.asarray(f["gt"]))
        self.prev_t, self.next_t, self.interp = np.asarray(self.hold_q, float), q7, 0
        self.steps += 1


# ---------------------------------------------------------------------------- view
class ViewOp(Operator):
    """Camera frame + the policy's planned gripper path (projected onto the table) + status and timings."""

    def setup(self, spec: OperatorSpec):
        spec.input("viz", policy=IOSpec.QueuePolicy.POP)
        spec.output("tensors")
        spec.output("specs")

    def start(self):
        from kinematics import RebotKinematics
        self.k = RebotKinematics()
        H, _ = load_calibration(ROOT / "config" / "camera_table.yaml")
        self.Hi = np.linalg.inv(H)

    def _px(self, x, y):
        p = self.Hi @ np.array([x, y, 1.0])
        return int(p[0] / p[2]), int(p[1] / p[2])

    @timed("view")
    def compute(self, op_input, op_output, context):
        import cupy as cp
        from holoscan.operators import HolovizOp
        v = op_input.receive("viz")
        if v is None:
            return
        img = v["frame"]["bgr"].copy()
        r = v["reply"] or {}
        pts = []
        for a in r.get("plan", []):
            q6, g = action_to_command(np.asarray(a, float))
            p = self.k.fk(q6).translation
            pts.append((self._px(p[0], p[1]), g))
        for (p0, _), (p1, g1) in zip(pts, pts[1:]):
            cv2.line(img, p0, p1, (0, 200, 0) if g1 > 0.02 else (0, 0, 255), 2)
        if v["target"] is not None and v["state"] in ("RUN", "MOVE", "READY", "STOPPED"):
            p = self.k.fk(np.asarray(v["target"])[:6]).translation
            cv2.circle(img, self._px(p[0], p[1]), 7, (255, 0, 255), 2)
        arm = v["arm"] or {}
        lines = [f"HOLOSCAN  {'replay (sim twin executes)' if v['replay'] else 'LIVE real arm'}   "
                 f"policy {v['state']}" + (f": {v['why']}" if v["state"] == "STOPPED" and v["why"] else ""),
                 f"ACT steps {v['steps']}   GPU inference {r.get('infer_ms', 0):5.1f} ms   round trip {v['rt_ms']:5.1f} ms"
                 + ("   NEW CHUNK" if r.get("refill") else ""),
                 "operator time ms: " + "  ".join(f"{k} {STATS[k]:.1f}" for k in ("camera", "replay", "policy",
                                                                                     "view") if k in STATS),
                 f"safety filter: {arm.get('mode', '-')}"]
        if v["replay"]:
            f = v["frame"]
            lines.append(f"recording frame {f['i']}/{f['n']}")
            if v["gt_err"] is not None:
                e = v["gt_err"]
                lines.append(f"policy vs expert: joints max {e[:6].max():4.1f} deg   gripper {e[6]:4.1f} (of 45)")
        lines.append("path: green = gripper open, red = closing   magenta = current target")
        lines.append("keys: g run   x stop   r back to observe" + ("" if v["replay"] else "   s hold   p park   q quit"))
        spec = HolovizOp.InputSpec("txt", "text")
        spec.text, spec.color, spec.priority = lines, [1.0, 1.0, 0.3, 1.0], 10
        op_output.emit({"video": cp.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                        "txt": np.asarray([(0.012, 0.03 + 0.045 * i, 0.035) for i in range(len(lines))],
                                          dtype=np.float32)}, "tensors")
        op_output.emit([spec], "specs")


# ---------------------------------------------------------------------------- app
class PolicyApp(Application):
    def __init__(self, a, cfg):
        self.a, self.cfg = a, cfg
        super().__init__()

    def compose(self):
        from holoscan.operators import HolovizOp
        limits = [tuple(self.cfg["sim_limits"][f"joint{i}"]) for i in range(1, 7)]
        if self.a.replay is not None:
            arm = rc.SimArmOp(self, cfg=self.cfg, limits=limits, initial_cube=None, name="sim_arm")
            src = ReplaySourceOp(self, PeriodicCondition(self, recess_period=timedelta(seconds=1 / POLICY_HZ)),
                                 episode_dir=self.a.episode_dir, name="replay")
            title = f"reBot ACT policy on Holoscan - replay {Path(self.a.episode_dir).name}"
        else:
            from real_common import load_real_cfg
            rt.LiveArmOp.tcp_min_z = Z_MIN_NEAR
            arm = rt.LiveArmOp(self, rcfg=load_real_cfg(), limits=limits, vmax=15.0, dry=False,
                               with_gripper=True, grip_force=0.10, name="live_arm")
            self.add_flow(arm, rt.SimOp(self, cfg=self.cfg, name="sim"))
            src = CamSourceOp(self, PeriodicCondition(self, recess_period=timedelta(milliseconds=2)), name="camera")
            title = "reBot ACT policy on Holoscan - LIVE"
        pol = PolicyOp(self, PeriodicCondition(self, recess_period=timedelta(seconds=1 / RATE)), arm_op=arm,
                       limits=limits, replay=self.a.replay is not None, name="policy")
        view = ViewOp(self, name="view")
        viz = HolovizOp(self, name="holoviz", width=960, height=720, window_title=title,
                        tensors=[dict(name="video", type="color")])
        self.add_flow(src, pol, {("frame", "frame")})
        self.add_flow(pol, arm, {("target", "target")})
        self.add_flow(pol, view, {("viz", "viz")})
        self.add_flow(view, viz, {("tensors", "receivers"), ("specs", "input_specs")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--live", action="store_true")
    m.add_argument("--replay", type=int, help="staged episode number (1 = oldest) to replay into the sim twin")
    ap.add_argument("--task", default="cube_to_box_scripted")
    ap.add_argument("--keys-file", default=str(ROOT / "keys"))
    a = ap.parse_args()
    if a.replay is not None:
        eps = sorted((ROOT / "data" / "staging" / a.task).glob("episode_*"))
        a.episode_dir = eps[a.replay - 1]

    def route(k):
        if k in ("g", "x", "r"):
            KEYS.append(k)
        elif k in ("e", "a", "s", "p", "o") and a.replay is None:
            rt.KEYS.append(k)
        elif k == "q":
            if a.replay is None:
                rt.KEYS.append(k)              # live: park, then quit
            else:
                import os
                import signal
                os.kill(os.getpid(), signal.SIGINT)

    def file_loop():
        import os
        open(a.keys_file, "a").close()
        with open(a.keys_file) as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                k = line.strip().lower()[:1]
                if k:
                    say(f"[pol] key '{k}'")
                    route(k)

    def stdin_loop():
        for line in sys.stdin:
            route(line.strip().lower()[:1])
    if sys.stdin.isatty():
        threading.Thread(target=stdin_loop, daemon=True).start()
    threading.Thread(target=file_loop, daemon=True).start()
    from holoscan.schedulers import MultiThreadScheduler
    app = PolicyApp(a, load_cfg())
    app.scheduler(MultiThreadScheduler(app, name="mt", worker_thread_number=6, stop_on_deadlock=False))
    app.run()


if __name__ == "__main__":
    main()
