"""Scripted data collector: the robot picks the cube and puts it in the box by itself; each episode is recorded
as a LeRobot staging episode (same format as the teleop recorder).

    CollectorOp (60 Hz) -> arm (--sim: SimArmOp + Isaac twin | --live: LiveArmOp + real arm) [+ live camera window]

Per episode: observe pose -> find cube + box -> CHECK (one cube, inside the calibrated zone, 15-34 cm reach,
>= 12 cm from the box, box not moved, two detections agree) -> validated plan -> RECORDED pick & place
(grasp checked) -> back to observe -> success check (cube in the box) -> reset (live: you put the cube somewhere
new + g; sim: the twin moves it) -> next.
Any HOLD / fault aborts the run (the arm holds; p parks). Missed grasp: the arm rises and returns, episode dropped.
Keys:  g go (live: after placing the cube)   x abort   + live arm keys e s a p q
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec

import real_teleop as rt
import robot_calibrate as rc
from common import FrameSub, load_cfg
from motion_plan import PHOTO_Q, RATE, READY_Q, Z_MIN_NEAR, MotionPlanner
from perception import detect_box, detect_cube, load_calibration, pixel_to_table
from rebot_sim_pipeline.conventions import JOINT_NAMES, arm_action, arm_state
from rebot_sim_pipeline.staging import EpisodeStager, staged_episodes

ROOT = Path(__file__).resolve().parents[1]
KEYS = []
ZONE = {"x": (0.17, 0.33), "y": (-0.11, 0.11)}     # calibrated grid +- margin
BOX_CLEAR = 0.12
REACH = (0.15, 0.34)
REC_FPS = 15
GRASP_RANGE = (0.004, 0.025)
REACH_BIAS = 0.0     # m (0.015 with the 7-point calibration; re-tune after the 12-point one): aim the grasp this much further out along the reach direction - the real arm (gravity
                     # sag + calibration) landed ~1 cm short towards the base (user, 2026-11-23, twice)


def say(m):
    print(m, flush=True)


class CollectorOp(Operator):
    def __init__(self, fragment, *args, arm_op, limits, sim, task, episodes, **kw):
        self.arm_op, self.limits, self.sim, self.task, self.max_eps = arm_op, limits, sim, task, episodes
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.output("target")

    # ------------------------------------------------------------------ setup
    def start(self):
        self.mp = MotionPlanner(self.limits)
        self.state, self.segs, self.si, self.i = "WAIT_ARM", None, 0, 0
        self.hold_q, self.box_ref, self.prev_det = None, None, None
        self.rng = np.random.default_rng(11)
        self.stager = EpisodeStager(ROOT / "data" / "staging" / self.task, ["front"])
        self.saved = len(staged_episodes(ROOT / "data" / "staging" / self.task))
        self.ep, self.ok_eps, self.rec_next = 0, 0, 0.0
        self.recording, self.t_ep = False, 0.0
        self.frames = FrameSub() if self.sim else None
        self.last_sim_frame = None
        if not self.sim:
            self.H, cal = load_calibration(ROOT / "config" / "camera_table.yaml")
            self.Hinv = np.linalg.inv(self.H)
            say(f"[col] calibration: {len(cal['points'])} points, unseen-point error mean {cal['leave_one_out_mm_mean']:.1f} mm")
            self._zone_overlay()
        say(f"[col] staging -> data/staging/{self.task} ({self.saved} episodes there); "
            + ("sim twin" if self.sim else "LIVE arm - type e to enable"))
        self._status("waiting for the arm")

    def _status(self, step, last=None):
        if (step, last) != getattr(self, "_last_status", None):     # also print changes in the terminal
            self._last_status = (step, last)
            say(f"[col] status: {step}" + (f" - {last}" if last else ""))
        rc.STATUS["step"] = step
        rc.STATUS["points"] = f"episodes saved {self.saved}, this run {self.ok_eps}/{self.ep}"
        if last is not None:
            rc.STATUS["last"] = last

    def _roi(self, shape):
        """Image mask: the cube zone + 4 cm margin, plus a 9 cm circle around the box (for the success check)."""
        if getattr(self, "_roi_mask", None) is not None:
            return self._roi_mask
        m = np.zeros(shape[:2], np.uint8)
        xs, ys, d = ZONE["x"], ZONE["y"], 0.04
        poly = [self._px(x, y) for x, y in [(xs[0] - d, ys[0] - d), (xs[0] - d, ys[1] + d),
                                             (xs[1] + d, ys[1] + d), (xs[1] + d, ys[0] - d)]]
        cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        if self.box_ref is not None:
            circ = [self._px(self.box_ref[0] + 0.09 * math.cos(t), self.box_ref[1] + 0.09 * math.sin(t))
                    for t in np.linspace(0, 2 * math.pi, 36)]
            cv2.fillPoly(m, [np.array(circ, np.int32)], 255)
            self._roi_mask = m          # final once the box is known
        return m

    def _px(self, x, y):
        p = self.Hinv @ np.array([x, y, 1.0])
        return [int(p[0] / p[2]), int(p[1] / p[2])]

    def _zone_overlay(self):
        xs, ys = ZONE["x"], ZONE["y"]
        poly = []
        for x, y in [(xs[0], ys[0]), (xs[0], ys[1]), (xs[1], ys[1]), (xs[1], ys[0])]:
            p = self.Hinv @ np.array([x, y, 1.0])
            poly.append([p[0] / p[2], p[1] / p[2]])
        rc.CAM["zone_px"] = np.array(poly, np.int32)

    # ------------------------------------------------------------------ perception
    def _perceive(self):
        """-> (cube_xy, cube_yaw, box_xy, reason). reason None = valid."""
        if self.sim:
            st = self.arm_op.link.state or {}
            if "cube" not in st:
                return None, None, None, "no twin state"
            c = st["cube"]
            w, x, y, z = c[3:7]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            return (c[0], c[1]), yaw, tuple(st["bin"]), None
        img = rc.CAM["frame"]
        if img is None:
            return None, None, None, "no camera frame"
        box = detect_box(img)
        if not box.ok:
            return None, None, None, f"box: {box.reason}"
        if self.box_ref is None:
            self.box_ref = tuple(pixel_to_table(self.H, box.centroid))
        cube = detect_cube(img, self._roi(img.shape))
        if not cube.ok:
            return None, None, None, f"cube: {cube.reason}"
        cxy = pixel_to_table(self.H, cube.centroid)
        bxy = pixel_to_table(self.H, box.centroid)
        # cube yaw on the table: map two corners of the min-area rectangle through the homography
        rect = cv2.boxPoints(cv2.minAreaRect(cube.contour))
        p0, p1 = (pixel_to_table(self.H, rect[0]), pixel_to_table(self.H, rect[1]))
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        return tuple(cxy), yaw, tuple(bxy), None

    def _check(self, cxy, bxy):
        x, y = cxy
        if not (ZONE["x"][0] <= x <= ZONE["x"][1] and ZONE["y"][0] <= y <= ZONE["y"][1]):
            return f"cube at ({x:.2f}, {y:+.2f}) is outside the calibrated zone (green outline)"
        r = math.hypot(x, y)
        if not REACH[0] <= r <= REACH[1]:
            return f"cube {r * 100:.0f} cm from the base (allowed {REACH[0] * 100:.0f}-{REACH[1] * 100:.0f})"
        if math.hypot(x - bxy[0], y - bxy[1]) < BOX_CLEAR:
            return "cube too close to the box"
        if self.box_ref is not None and math.hypot(bxy[0] - self.box_ref[0], bxy[1] - self.box_ref[1]) > 0.02:
            return "the box has moved since the start - put it back or restart"
        return None

    # ------------------------------------------------------------------ helpers
    def _emit(self, op_output, q7):
        op_output.emit({"t": time.perf_counter(), "q": np.asarray(q7, float), "ok": True}, "target")

    def _abort(self, why):
        if self.recording:
            self.stager.discard()
            self.recording = False
        if self.state != "ABORT":
            say(f"[col] ABORTED: {why}. The arm holds where it is" + ("" if self.sim else " - type p to park"))
            self._status("ABORTED", why)
            self.state = "ABORT"

    def _start_plan(self, segs, meas, label):
        try:
            self.mp.validate(segs, meas)
        except ValueError as exc:
            return str(exc)
        self.segs, self.si, self.i, self.state = segs, 0, 0, "RUN"
        say(f"[col] > {label}: {sum(len(s.q) for s in segs) / RATE:.0f} s, validated")
        return None

    def _record_frame(self, last):
        now = time.perf_counter()
        if now < self.rec_next:
            return
        self.rec_next = max(self.rec_next + 1.0 / REC_FPS, now - 0.5 / REC_FPS)
        if self.sim:
            f = self.frames.latest()
            if f is not None:
                self.last_sim_frame = f[1]
            img = self.last_sim_frame
        else:
            img = None if rc.CAM["frame"] is None else cv2.cvtColor(rc.CAM["frame"], cv2.COLOR_BGR2RGB)
        if img is None or last.get("cmd") is None or last.get("measured") is None:
            return
        cmd, meas = last["cmd"], last["measured"]
        self.stager.add({"front": img}, action=arm_action(cmd[:6], cmd[6]), state=arm_state(meas[:6], meas[6]),
                        raw_action=cmd.astype(np.float32), raw_state=meas.astype(np.float32))

    # ------------------------------------------------------------------ main tick
    def compute(self, op_input, op_output, context):
        last = getattr(self.arm_op, "last", None)
        meas = None if last is None else last.get("measured")
        for k in KEYS[:]:
            KEYS.remove(k)
            if k == "x":
                self._abort("abort key")
            elif k == "g" and self.state == "WAIT_USER":
                self.state, self.t_wait, self.prev_det = "SEE", time.time(), None

        if self.state == "WAIT_ARM":
            if last is not None and last.get("mode") == "FOLLOW" and meas is not None:
                segs = [self.mp.joint_move("to the ready pose", meas[:6], READY_Q, float(meas[6])),
                        self.mp.joint_move("to the observe pose", READY_Q, PHOTO_Q, float(meas[6]))]
                err = self._start_plan(segs, meas, "go to the observe pose")
                if err:
                    self._abort(f"plan rejected: {err}")
                self.after = "SEE"
                self._emit(op_output, meas)
                return
            if meas is not None:
                self._emit(op_output, meas)
            else:
                op_output.emit({"t": time.perf_counter(), "q": None, "ok": False}, "target")
            return

        if self.state == "ABORT" or self.state == "FINISHED":
            if self.hold_q is not None and self.state == "FINISHED":
                self._emit(op_output, self.hold_q)
            elif meas is not None:
                self._emit(op_output, meas)
            return
        if last is None or last.get("mode") not in ("FOLLOW", "BLEND"):
            self._abort(f"arm left FOLLOW ({None if last is None else last.get('mode')})")
            return
        # ---------------- waiting / perceiving at the observe pose
        if self.state in ("SEE", "WAIT_USER"):
            # perceive + plan FIRST and emit at the very end: Holoscan delivers a message when compute() returns,
            # so a hold emitted before ~0.2 s of planning would arrive stale (2026-11-23: HOLD "leader stale 216 ms")
            hold = self.hold_q if self.hold_q is not None else meas
            try:
                self._see(meas)
            finally:
                if self.state == "RUN" and self.segs:
                    self._emit(op_output, self.segs[0].q[0])
                elif self.state not in ("ABORT",):
                    self._emit(op_output, hold)
            return

        # ---------------- executing a plan
        self._run(op_output, last, meas)

    def _see(self, meas):
        if True:
            if self.state == "WAIT_USER":
                return
            if self.ep >= self.max_eps:
                self.state = "FINISHED"
                say(f"[col] done: {self.ok_eps}/{self.ep} episodes succeeded and were saved")
                self._status("finished - type p to park", f"{self.ok_eps}/{self.ep} episodes saved")
                return
            if time.time() - getattr(self, "t_see", 0.0) < 0.5:
                return
            self.t_see = time.time()
            cxy, yaw, bxy, why = self._perceive()
            if why is None:
                if self.box_ref is None:
                    self.box_ref = bxy
                why = self._check(cxy, bxy)
            if why is None and not self.sim:     # two detections 0.5 s apart must agree within 1 cm
                if self.prev_det is None or math.hypot(cxy[0] - self.prev_det[0], cxy[1] - self.prev_det[1]) > 0.010:
                    jump = None if self.prev_det is None else math.hypot(cxy[0] - self.prev_det[0], cxy[1] - self.prev_det[1])
                    self.prev_det = cxy
                    self._status("checking the cube position", "two detections must agree within 1 cm"
                                 + ("" if jump is None else f" (last jump {jump * 1000:.0f} mm - is something moving?)"))
                    return
            if why is not None:
                self._status("waiting for a valid cube", why)
                if self.sim:                     # the twin puts the cube somewhere valid
                    self._sim_reset()
                return
            if not self.sim and REACH_BIAS:
                say(f"[col]   detected cube at ({cxy[0]:.3f}, {cxy[1]:+.3f}), aiming {REACH_BIAS * 100:.0f} cm further out")
                r = math.hypot(*cxy)
                cxy = (cxy[0] * (r + REACH_BIAS) / r, cxy[1] * (r + REACH_BIAS) / r)
            try:
                segs = self.mp.pick_place(meas, cxy, bxy, PHOTO_Q, cube_yaw=yaw)
            except ValueError as exc:
                self._status("waiting for a valid cube", str(exc))
                if self.sim:
                    self._sim_reset()
                return
            self.ep += 1
            err = self._start_plan(segs, meas, f"episode {self.ep}: cube ({cxy[0]:.2f}, {cxy[1]:+.2f}) -> box")
            if err:
                self.ep -= 1
                self._status("waiting for a valid cube", f"plan rejected: {err}")
                return
            self.after = "CHECK"
            self.stager.start(self.saved + 1)
            self.recording, self.t_ep, self.rec_next = True, time.time(), time.perf_counter()
            self._status(f"episode {self.ep}: RECORDING", f"cube at ({cxy[0]:.2f}, {cxy[1]:+.2f})")
            return

    def _run(self, op_output, last, meas):
        seg = self.segs[self.si]
        if self.state == "ARRIVE":
            self._emit(op_output, seg.q[-1])
            if self.recording:
                self._record_frame(last)
            err = 999.0 if meas is None else float(np.degrees(np.max(np.abs(np.asarray(meas)[:6] - seg.q[-1][:6]))))
            # at the grasp the arm must really be there: 1 deg (3 deg after 2 s, the elbow sags under gravity)
            near_cube = seg.label in ("descend to the cube", "settle at the cube")
            tol = 1.0 if (near_cube and time.perf_counter() - self.t_arrive < 2.0) else 3.0
            if err < tol:
                self._event(seg, meas)
            elif time.perf_counter() - self.t_arrive > 4.0:
                self._abort(f"arm did not reach the end of '{seg.label}' ({err:.1f} deg off)")
            return
        q7 = seg.q[self.i]
        self.hold_q = q7
        self._emit(op_output, q7)
        if self.recording:
            self._record_frame(last)
        # film the grasp (camera frames every 0.25 s) so every landing can be checked afterwards
        if not self.sim and seg.label in ("descend to the cube", "settle at the cube", "grasp") and \
                rc.CAM["frame"] is not None and time.time() - getattr(self, "t_film", 0.0) > 0.25:
            self.t_film = time.time()
            d = ROOT / "logs" / "cam" / "grasps" / f"ep{self.saved + 1:03d}_{self.ep:02d}"
            d.mkdir(parents=True, exist_ok=True)
            self.film_n = getattr(self, "film_n", 0) + 1
            cv2.imwrite(str(d / f"f{self.film_n:05d}.jpg"), rc.CAM["frame"])
        self.i += 1
        if self.i >= len(seg.q):
            self.state, self.t_arrive = "ARRIVE", time.perf_counter()

    def _event(self, seg, meas):
        if seg.event == "check_grasp":
            g = float(meas[6])
            if not GRASP_RANGE[0] <= g <= GRASP_RANGE[1]:
                say(f"[col]   missed the grasp (finger {g * 1000:.1f} mm) - rising and returning, episode dropped")
                self.stager.discard()
                self.recording = False
                rise = self.mp.park(meas, 0.03)[:-1]           # rise straight up only
                q_up = rise[-1].q[-1][:6] if rise else meas[:6]
                segs = rise + [self.mp.joint_move("back to the observe pose", q_up, PHOTO_Q, 0.03)]
                err = self._start_plan(segs, meas, "recover after a missed grasp")
                if err:
                    self._abort(f"recovery plan rejected: {err}")
                self.after = "RESET"
                self._status("missed grasp - returning", f"finger {g * 1000:.1f} mm")
                return
            say(f"[col]   holding the cube ({g * 1000:.1f} mm)")
        self.si += 1
        self.i = 0
        self.state = "RUN"
        if self.si < len(self.segs):
            return
        # plan finished
        if self.after == "CHECK":
            self._finish_episode()
        elif self.after == "RESET":
            self._reset()
        else:
            self.state = "SEE"

    def _finish_episode(self):
        cxy, _, bxy, why = self._perceive()
        if self.sim:
            inside = cxy is not None and math.hypot(cxy[0] - bxy[0], cxy[1] - bxy[1]) < 0.05
        else:   # live: in the box = not visible on the table any more, or seen within 6 cm of the box centre
            inside = (why is not None and why.startswith("cube")) or \
                     (cxy is not None and math.hypot(cxy[0] - bxy[0], cxy[1] - bxy[1]) < 0.06)
        n = self.stager.frames
        if inside and n >= REC_FPS:
            path = self.stager.finish({
                "task": "Pick up the yellow cube and place it in the orange box", "fps": REC_FPS,
                "resolution": [640, 480] if self.last_sim_frame is None else
                              [int(self.last_sim_frame.shape[1]), int(self.last_sim_frame.shape[0])],
                "robot_type": "seeed_b601_dm_follower", "feature_names": [f"{j}.pos" for j in JOINT_NAMES],
                "camera_keys": {"front": "front"}, "source": "sim-scripted" if self.sim else "real-scripted",
                "recorder": "rebot-holoscan collector.py", "success": True,
                "real_fps": round(n / max(time.time() - self.t_ep, 1e-3), 2)})
            self.saved += 1
            self.ok_eps += 1
            say(f"[col]   SUCCESS - episode saved ({n} frames) -> {path.name}   [{self.ok_eps}/{self.ep}]")
            self._status("episode saved", f"SUCCESS: {n} frames")
        else:
            self.stager.discard()
            say(f"[col]   cube NOT in the box - episode dropped   [{self.ok_eps}/{self.ep}]")
            self._status("episode dropped", "cube not in the box")
        self.recording = False
        self._reset()

    def _reset(self):
        if self.sim:
            self._sim_reset()
            self.state = "SEE"
        else:
            self.state = "WAIT_USER"
            self._status("RESET: take the cube out of the box, put it in the green zone, step back, then g")
            say("[col] take the cube out of the box, put it somewhere new inside the green zone, step back, then type g")

    def _sim_reset(self):
        import json
        x, y = float(self.rng.uniform(0.19, 0.31)), float(self.rng.uniform(-0.05, 0.10))
        self.arm_op.link.pub.send_string(json.dumps({"reset": True, "cube_xy": [x, y],
                                                     "cube_yaw": float(self.rng.uniform(-0.7, 0.7))}))
        self.t_see = time.time() + 1.0      # let it settle before looking


class CollectorApp(Application):
    def __init__(self, a, cfg):
        self.a, self.cfg = a, cfg
        super().__init__()

    def compose(self):
        limits = [tuple(self.cfg["sim_limits"][f"joint{i}"]) for i in range(1, 7)]
        if self.a.sim:
            arm = rc.SimArmOp(self, cfg=self.cfg, limits=limits, initial_cube=None, name="sim_arm")
        else:
            from holoscan.operators import HolovizOp
            from real_common import load_real_cfg
            rt.LiveArmOp.tcp_min_z = Z_MIN_NEAR
            arm = rt.LiveArmOp(self, rcfg=load_real_cfg(), limits=limits, vmax=15.0, dry=False,
                               with_gripper=True, grip_force=0.10, name="live_arm")
            self.add_flow(arm, rt.SimOp(self, cfg=self.cfg, name="sim"))
            cam = rc.CamViewOp(self, PeriodicCondition(self, recess_period=timedelta(milliseconds=2)), name="camera")
            viz = HolovizOp(self, name="view", width=960, height=720, window_title="reBot collector - live camera",
                            tensors=[dict(name="video", type="color")])
            self.add_flow(cam, viz, {("tensors", "receivers"), ("specs", "input_specs")})
        col = CollectorOp(self, PeriodicCondition(self, recess_period=timedelta(seconds=1 / RATE)), arm_op=arm,
                          limits=limits, sim=self.a.sim, task=self.a.task, episodes=self.a.episodes, name="collector")
        self.add_flow(col, arm)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--sim", action="store_true")
    m.add_argument("--live", action="store_true")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--task", default=None, help="staging folder (default cube_to_box_scripted[_sim])")
    ap.add_argument("--keys-file", default=str(ROOT / "keys"))
    a = ap.parse_args()
    a.task = a.task or ("cube_to_box_scripted_sim" if a.sim else "cube_to_box_scripted")

    def route(k):
        if k in ("g", "x"):
            KEYS.append(k)
        elif k in ("e", "a", "s", "p", "q", "o"):
            rt.KEYS.append(k)

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
                    say(f"[col] key '{k}'")
                    route(k)

    def stdin_loop():
        for line in sys.stdin:
            route(line.strip().lower()[:1])
    if sys.stdin.isatty():
        threading.Thread(target=stdin_loop, daemon=True).start()
    threading.Thread(target=file_loop, daemon=True).start()
    from holoscan.schedulers import MultiThreadScheduler
    app = CollectorApp(a, load_cfg())
    app.scheduler(MultiThreadScheduler(app, name="mt", worker_thread_number=6, stop_on_deadlock=False))
    app.run()


if __name__ == "__main__":
    main()
