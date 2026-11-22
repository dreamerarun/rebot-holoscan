"""Record real reBot demonstrations through Holoscan (leader teleop + front camera -> LeRobot staging).

    control lane  (60 Hz): LeaderOp -> LiveArmOp (SafetyFilter -> real arm + gripper)
    camera lane   (30 Hz): CameraOp -> RecorderOp -> EpisodeStager (JPEG + state/action .npy)
                                         \\-> Holoviz window: live camera + status

Keys (type + Enter in this terminal, or `echo X >> ~/rebot-holoscan/keys`):
    n  start recording an episode / stop and SAVE it
    x  discard the episode being recorded
    e  enable the arm   s  hold   a  re-arm   p  park + off   q  park, off, quit

Recorded convention = rebot-sim-pipeline/conventions.py (same as Seeed's leader/follower datasets):
    action            = the safety-filtered command sent to the arm (leader convention, gripper 0..45)
    observation.state = the measured arm (follower convention, gripper 0..270)
Convert to a LeRobot dataset afterwards with ./convert_real_dataset.sh

    ./run_real_record.sh                         # task "cube_to_box", camera index 1, 15 deg/s
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import cupy as cp
import cv2
import numpy as np
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.schedulers import MultiThreadScheduler

import real_teleop as rt
from common import find_leader_port, load_cfg
from real_common import load_real_cfg
from rebot_sim_pipeline.conventions import JOINT_NAMES, arm_action, arm_state
from rebot_sim_pipeline.staging import EpisodeStager, staged_episodes

ROOT = Path(__file__).resolve().parents[1]
REC_KEYS = []
FPS = 30


class CameraOp(Operator):
    def __init__(self, fragment, *args, index, size, **kw):
        self.index, self.size = index, size
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def start(self):
        w, h = self.size
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera index {self.index} cannot be opened")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

    def stop(self):
        self.cap.release()

    def compute(self, op_input, op_output, context):
        ok, bgr = self.cap.read()            # blocks until the next camera frame (~33 ms)
        if ok:
            op_output.emit({"t": time.perf_counter(), "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)}, "frame")


class RecorderOp(Operator):
    def __init__(self, fragment, *args, arm_op, staging, task, **kw):
        self.arm_op, self.staging, self.task = arm_op, staging, task
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.input("frame")
        spec.output("tensors")
        spec.output("specs")

    def start(self):
        self.stager = EpisodeStager(self.staging, ["front"])
        self.saved = len(staged_episodes(self.staging))
        self.t_start = None
        self.msg, self.msg_t = "", 0.0
        self.max_s = 90.0
        print(f"[rec] staging -> {self.staging} ({self.saved} episodes already there)", flush=True)

    def stop(self):
        if self.stager.active:
            self.stager.discard()
            print("[rec] unsaved episode discarded on exit", flush=True)

    def _note(self, text):
        self.msg, self.msg_t = text, time.perf_counter()
        print(f"[rec] {text}", flush=True)

    def _save(self):
        n = self.stager.frames
        if n < FPS:  # under 1 s: not a demonstration
            self.stager.discard()
            self._note(f"episode too short ({n} frames) - discarded")
        else:
            h, w = self.last_shape
            path = self.stager.finish({
                "task": self.task, "fps": FPS, "resolution": [w, h], "robot_type": "seeed_b601_dm_follower",
                "feature_names": [f"{j}.pos" for j in JOINT_NAMES], "camera_keys": {"front": "front"},
                "source": "real", "recorder": "rebot-holoscan real_record.py", "success": True})
            self.saved += 1
            self._note(f"SAVED episode {self.saved} ({n} frames, {n / FPS:.1f} s) -> {path.name}")
        self.t_start = None

    def compute(self, op_input, op_output, context):
        f = op_input.receive("frame")
        rgb = f["rgb"]
        self.last_shape = rgb.shape[:2]
        last = getattr(self.arm_op, "last", None)
        arm_ready = last is not None and last["state"] == "ON" and last["mode"] == "FOLLOW" \
            and last["cmd"] is not None and last["measured"] is not None

        for k in REC_KEYS[:]:
            REC_KEYS.remove(k)
            if k == "n":
                if self.stager.active:
                    self._save()
                elif not arm_ready:
                    self._note("cannot record: arm is not following yet (enable with e, align the leader)")
                else:
                    self.stager.start(self.saved + 1)
                    self.t_start = time.perf_counter()
                    self._note(f"RECORDING episode {self.saved + 1}")
            elif k == "x" and self.stager.active:
                self.stager.discard()
                self.t_start = None
                self._note("episode discarded")

        if self.stager.active:
            if not arm_ready:
                self.stager.discard()
                self.t_start = None
                self._note("arm left FOLLOW (hold/park) - episode discarded")
            else:
                cmd, meas = last["cmd"], last["measured"]
                self.stager.add({"front": rgb},
                                action=arm_action(cmd[:6], cmd[6]), state=arm_state(meas[:6], meas[6]),
                                raw_action=cmd.astype(np.float32), raw_state=meas.astype(np.float32))
                if time.perf_counter() - self.t_start > self.max_s:
                    self._note(f"{self.max_s:.0f} s limit reached")
                    self._save()

        # ------------------------------------------------ live view
        if self.stager.active:
            el = time.perf_counter() - self.t_start
            head, col = f"REC  episode {self.saved + 1}   {int(el // 60):02d}:{el % 60:04.1f}", [1.0, 0.2, 0.2, 1.0]
        elif last is None or last["state"] == "OFF":
            head, col = "ARM OFF - type e to enable", [1.0, 0.8, 0.2, 1.0]
        elif last["mode"] != "FOLLOW":
            head, col = f"ARM {last['mode']} - {'type a to re-arm' if last['mode'] == 'HOLD' else 'wait'}", [1.0, 0.8, 0.2, 1.0]
        else:
            head, col = "READY - place the cube, then type n to record", [0.3, 1.0, 0.4, 1.0]
        info = [f"saved: {self.saved} episodes   task: {self.task}",
                "n start/stop+save   x discard   s hold   a re-arm   p park   q quit"]
        if self.msg and time.perf_counter() - self.msg_t < 4.0:
            info.insert(0, self.msg)
        out = {"video": cp.asarray(rgb),
               "head": np.asarray([(0.02, 0.03, 0.06)], dtype=np.float32),
               "info": np.asarray([(0.02, 0.86 - 0.05 * (len(info) - 1) + 0.05 * i, 0.04) for i in range(len(info))],
                                  dtype=np.float32)}
        s1 = HolovizOp.InputSpec("head", "text")
        s1.text, s1.color, s1.priority = [head], col, 10
        s2 = HolovizOp.InputSpec("info", "text")
        s2.text, s2.color, s2.priority = info, [1.0, 1.0, 1.0, 1.0], 9
        op_output.emit(out, "tensors")
        op_output.emit([s1, s2], "specs")


class NullOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        op_input.receive("in")


class RecordApp(Application):
    def __init__(self, args, cfg, rcfg, port):
        self.a, self.cfg, self.rcfg, self.port = args, cfg, rcfg, port
        super().__init__()

    def compose(self):
        a = self.a
        limits = [tuple(self.cfg["sim_limits"][f"joint{i}"]) for i in range(1, 7)]
        leader = rt.LeaderOp(self, PeriodicCondition(self, recess_period=timedelta(seconds=1 / rt.RATE)),
                             cfg=self.cfg, port=self.port, name="leader")
        arm = rt.LiveArmOp(self, rcfg=self.rcfg, limits=limits, vmax=a.vmax, dry=a.dry,
                           with_gripper=True, grip_force=a.grip_force, name="live_arm")
        self.add_flow(leader, arm)
        if a.sim:
            self.add_flow(arm, rt.SimOp(self, cfg=self.cfg, name="sim"))
        else:
            self.add_flow(arm, NullOp(self, name="null"), {("show", "in")})

        w, h = (int(v) for v in a.size.split("x"))
        cam = CameraOp(self, PeriodicCondition(self, recess_period=timedelta(milliseconds=2)),
                       index=a.cam_index, size=(w, h), name="camera")
        rec = RecorderOp(self, arm_op=arm, staging=ROOT / "data" / "staging" / a.task,
                         task=a.task_text, name="recorder")
        viz = HolovizOp(self, name="view", width=960, height=720, window_title="reBot recording - Holoscan",
                        tensors=[dict(name="video", type="color")])
        self.add_flow(cam, rec)
        self.add_flow(rec, viz, {("tensors", "receivers"), ("specs", "input_specs")})


def route(k):
    if k in ("n", "x"):
        REC_KEYS.append(k)
    elif k in ("e", "a", "s", "p", "q", "o"):
        rt.KEYS.append(k)


def keyboard():
    for line in sys.stdin:
        route(line.strip().lower()[:1])


def key_file(path):
    import os
    open(path, "a").close()
    with open(path) as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue
            k = line.strip().lower()[:1]
            if k:
                print(f"[rec] key '{k}' from {path}", flush=True)
                route(k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="cube_to_box", help="staging folder name")
    ap.add_argument("--task-text", default="Pick up the yellow cube and place it in the orange box")
    ap.add_argument("--cam-index", type=int, default=1)
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--vmax", type=float, default=15.0)
    ap.add_argument("--grip-force", type=float, default=0.10)
    ap.add_argument("--sim", action="store_true", help="also drive the Isaac Sim twin")
    ap.add_argument("--dry", action="store_true", help="never send anything to the motors")
    ap.add_argument("--leader-port", default="/dev/ttyUSB0")
    ap.add_argument("--keys-file")
    a = ap.parse_args()
    if not 1.0 <= a.vmax <= 45.0 or not 0.03 <= a.grip_force <= 0.2:
        sys.exit("--vmax 1..45, --grip-force 0.03..0.2")
    if not sys.stdin.isatty() and not a.dry and not a.keys_file:
        sys.exit("run this in a terminal (or with --keys-file)")
    if sys.stdin.isatty():
        threading.Thread(target=keyboard, daemon=True).start()
    if a.keys_file:
        threading.Thread(target=key_file, args=(a.keys_file,), daemon=True).start()
    app = RecordApp(a, load_cfg(), load_real_cfg(), find_leader_port(a.leader_port))
    app.scheduler(MultiThreadScheduler(app, name="mt", worker_thread_number=6, stop_on_deadlock=False))
    app.run()


if __name__ == "__main__":
    main()
