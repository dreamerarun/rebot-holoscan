"""Scripted pick-and-place for the reBot B601-DM twin, on Holoscan.

    ScriptedPickOp (60 Hz) -> SimOp -> Isaac Sim
      reads the cube pose from the twin, plans with Pinocchio IK:
      pre-grasp -> descend -> close -> lift -> carry over the bin -> open -> home
      then checks success, moves the cube to a new random spot and repeats.

The same SimOp/command format as the leader teleop: later the real-arm stage plugs
in next to it. Needs the twin started with --vision (cube + bin).

    ./run_scripted_pick.sh                 # 5 cycles, random cube placement
    ./run_scripted_pick.sh --cycles 0      # forever
    ./run_scripted_pick.sh --fixed         # always the default cube spot
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta

import numpy as np
import pinocchio as pin
import zmq
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec

from common import SIM_JOINTS, load_cfg
from kinematics import ARM, RebotKinematics, cube_yaw, top_down, wrap

RATE = 60.0
GRIP_OPEN = 0.04     # finger joint (m); pad gap ~ 2*q + 2.1 cm
GRIP_CLOSE = 0.0     # force-limited by the finger drive (8 N), stops on the cube
PRE_Z = 0.10         # pre-grasp height above the grasp point
GRASP_DZ = 0.0       # grasp height offset from the cube centre (-6 mm made the tips hit the table)
CARRY_Z = 0.16       # TCP height when carrying over the bin


def smooth(s):
    return s * s * (3 - 2 * s)


class Planner:
    """Builds the per-tick (q_arm, grip) list for one pick-and-place."""

    def __init__(self, kin: RebotKinematics):
        self.k = kin

    def joint_move(self, q_a, q_b, g_a, g_b, secs):
        n = max(int(secs * RATE), 1)
        return [((1 - smooth(i / n)) * q_a + smooth(i / n) * q_b, g_a + (g_b - g_a) * smooth(i / n), )
                for i in range(1, n + 1)]

    def cart_move(self, T_a, T_b, q_start, grip, secs):
        n = max(int(secs * RATE), 1)
        d = pin.log6(T_a.actInv(T_b))
        out, q = [], q_start
        for i in range(1, n + 1):
            T = T_a * pin.exp6(d * smooth(i / n))
            q, pe, re = self.k.ik(T, q, iters=60)
            if pe > 3e-3 or re > 3e-2:
                raise RuntimeError(f"IK failed along the path (pos err {pe * 1000:.1f} mm)")
            out.append((q.copy(), grip))
        return out

    def hold(self, q, grip, secs):
        return [(q.copy(), grip)] * max(int(secs * RATE), 1)

    def plan(self, q_now, cube_pose, bin_xy):
        cx, cy, cz = cube_pose[:3]
        radial = np.arctan2(cy, cx) + np.pi / 2  # fingers across the reach direction
        yaw = cube_yaw(cube_pose[3:7])
        cands = [yaw + k * np.pi / 2 for k in range(4)]
        a = min(cands, key=lambda c: abs(wrap(c - radial)))  # cube face closest to that
        a = radial + wrap(a - radial)
        T_grasp = top_down((cx, cy, cz + GRASP_DZ), a)
        T_pre = top_down((cx, cy, cz + GRASP_DZ + PRE_Z), a)
        T_lift = top_down((cx, cy, CARRY_Z), a)
        b = np.arctan2(bin_xy[1], bin_xy[0]) + np.pi / 2
        T_bin = top_down((bin_xy[0], bin_xy[1], CARRY_Z), b)

        q_pre = self.k.ik_multi(T_pre, q_now)
        if q_pre is None:
            raise RuntimeError(f"cube at ({cx:.2f}, {cy:.2f}) is out of reach")
        steps, phases = [], []

        def add(name, seg):
            phases.append((len(steps), name))
            steps.extend(seg)

        add("move to pre-grasp", self.joint_move(q_now[:ARM], q_pre, GRIP_OPEN, GRIP_OPEN, 2.5))
        add("descend", self.cart_move(T_pre, T_grasp, q_pre, GRIP_OPEN, 1.6))
        q_g = steps[-1][0]
        add("settle at cube", self.hold(q_g, GRIP_OPEN, 0.4))
        add("close gripper", [(q_g, GRIP_OPEN + (GRIP_CLOSE - GRIP_OPEN) * smooth(i / 48)) for i in range(1, 49)])
        add("hold", self.hold(q_g, GRIP_CLOSE, 0.5))
        add("lift", self.cart_move(T_grasp, T_lift, q_g, GRIP_CLOSE, 1.5))
        add("carry to bin", self.cart_move(T_lift, T_bin, steps[-1][0], GRIP_CLOSE, 2.5))
        q_b = steps[-1][0]
        add("release", [(q_b, GRIP_CLOSE + (GRIP_OPEN - GRIP_CLOSE) * smooth(i / 30)) for i in range(1, 31)])
        add("settle", self.hold(q_b, GRIP_OPEN, 0.8))
        add("return home", self.joint_move(q_b, np.zeros(ARM), GRIP_OPEN, 0.0, 2.5))
        return steps, phases


class ScriptedPickOp(Operator):
    def __init__(self, fragment, *args, cfg, cycles, fixed, at=None, **kw):
        self.cfg, self.cycles, self.fixed, self.at = cfg, cycles, fixed, at
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.output("cmd")

    def start(self):
        self.sub = zmq.Context.instance().socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.connect(self.cfg["sim_link"]["state_endpoint"])
        self.planner = Planner(RebotKinematics())
        self.rng = np.random.default_rng(7)
        self.state = None
        self.steps, self.phases, self.i = None, [], 0
        self.cycle, self.ok = 0, 0
        self.mode = "settle"          # settle -> run -> check -> reset -> settle ...
        self.t_mode = time.perf_counter()
        self.last_q = None
        self.requested = None
        self.pending_reset = ({"cube_xy": list(at[:2]), "cube_yaw": float(np.radians(at[2]))} if (at := self.at)
                              else {"cube_xy": None, "cube_yaw": 0.0})
        print("[pick] scripted expert ready; waiting for the twin (start it with --vision)", flush=True)

    def stop(self):
        self.sub.close(0)

    def _poll(self):
        try:
            while True:
                self.state = json.loads(self.sub.recv_string(zmq.NOBLOCK))
        except zmq.Again:
            pass

    def _goto(self, mode):
        self.mode, self.t_mode = mode, time.perf_counter()

    def compute(self, op_input, op_output, context):
        self._poll()
        st = self.state
        if st is None or "cube" not in st:
            return
        since = time.perf_counter() - self.t_mode
        cmd = {}
        if self.pending_reset is not None:
            cmd["reset"] = self.pending_reset
            self.requested = self.pending_reset
            self.pending_reset = None
            self._goto("settle")

        if self.mode == "settle" and since > 1.5 and self.requested is not None:
            want = self.requested.get("cube_xy") or [0.26, 0.06]
            if np.hypot(st["cube"][0] - want[0], st["cube"][1] - want[1]) > 0.01:
                self.pending_reset = self.requested  # reset message was lost: send again
                return
        if self.mode == "settle" and since > 1.5:
            if self.cycles and self.cycle >= self.cycles:
                print(f"[pick] done: {self.ok}/{self.cycle} successful", flush=True)
                try:
                    self.fragment.stop_execution()
                except AttributeError:
                    import os
                    os._exit(0)
                return
            self.cycle += 1
            q_now = np.array(st["q"][:ARM]) if self.last_q is None else self.last_q
            try:
                self.steps, self.phases = self.planner.plan(q_now, st["cube"], st["bin"])
                self.i = 0
                c = st["cube"]
                print(f"[pick] cycle {self.cycle}: cube at x={c[0]:.3f} y={c[1]:.3f} "
                      f"yaw={np.degrees(cube_yaw(c[3:7])):.0f} deg, {len(self.steps) / RATE:.1f} s plan", flush=True)
                self._goto("run")
            except RuntimeError as exc:
                print(f"[pick] cycle {self.cycle}: plan failed: {exc}", flush=True)
                self._goto("reset")

        if self.mode == "run":
            for start, name in self.phases:
                if start == self.i:
                    if name == "close gripper":  # how well does the sim arm track the plan at the grasp?
                        k = self.planner.k
                        want = k.fk(np.asarray(self.steps[self.i][0])).translation
                        have = k.fk(np.array(st["q"][:ARM])).translation
                        err = (have - want) * 1000
                        print(f"[pick]   TCP tracking at grasp: dx={err[0]:+.1f} dy={err[1]:+.1f} dz={err[2]:+.1f} mm",
                              flush=True)
                    print(f"[pick]   {name}", flush=True)
            q_arm, g = self.steps[self.i]
            self.last_q = np.asarray(q_arm)
            cmd["q"] = list(map(float, q_arm)) + [float(g), float(-g)]
            self.i += 1
            if self.i >= len(self.steps):
                self._goto("check")

        if self.mode == "check" and since > 0.8:
            c, bx, by = st["cube"], *st["bin"]
            inside = abs(c[0] - bx) < 0.055 and abs(c[1] - by) < 0.055 and c[2] < 0.07
            self.ok += inside
            print(f"[pick] cycle {self.cycle}: {'SUCCESS - cube in bin' if inside else 'MISS'} "
                  f"(cube at {c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})  total {self.ok}/{self.cycle}", flush=True)
            self._goto("reset")

        if self.mode == "reset":
            if self.at:
                self.pending_reset = {"cube_xy": list(self.at[:2]), "cube_yaw": float(np.radians(self.at[2]))}
            elif self.fixed:
                self.pending_reset = {"cube_xy": None, "cube_yaw": 0.0}
            else:
                self.pending_reset = {"cube_xy": [float(self.rng.uniform(0.20, 0.32)), float(self.rng.uniform(-0.02, 0.14))],
                                      "cube_yaw": float(self.rng.uniform(-0.8, 0.8))}
            self._goto("wait_reset")

        if cmd:
            op_output.emit(cmd, "cmd")


class SimOp(Operator):
    def __init__(self, fragment, *args, cfg, **kw):
        self.cfg = cfg
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.input("cmd")

    def start(self):
        self.pub = zmq.Context.instance().socket(zmq.PUB)
        self.pub.setsockopt(zmq.SNDHWM, 4)
        self.pub.bind(self.cfg["sim_link"]["cmd_endpoint"])
        self.seq = 0

    def stop(self):
        self.pub.close(0)

    def compute(self, op_input, op_output, context):
        c = op_input.receive("cmd")
        self.seq += 1
        msg = {"seq": self.seq, "t": time.time(), "names": SIM_JOINTS}
        if "q" in c:
            msg["q"] = c["q"]
        if "reset" in c:
            msg["reset"] = True
            msg.update({k: v for k, v in c["reset"].items() if v is not None})
        self.pub.send_string(json.dumps(msg))


class ScriptedPickApp(Application):
    def __init__(self, cfg, cycles, fixed, at=None):
        self.cfg, self.cycles, self.fixed, self.at = cfg, cycles, fixed, at
        super().__init__()

    def compose(self):
        expert = ScriptedPickOp(self, PeriodicCondition(self, recess_period=timedelta(seconds=1 / RATE)),
                                cfg=self.cfg, cycles=self.cycles, fixed=self.fixed, at=self.at, name="expert")
        sim = SimOp(self, cfg=self.cfg, name="sim")
        self.add_flow(expert, sim)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=5, help="0 = run forever")
    ap.add_argument("--fixed", action="store_true", help="always reset the cube to its default spot")
    ap.add_argument("--at", help="always put the cube at x,y,yaw_deg (e.g. 0.275,0.124,25)")
    a = ap.parse_args()
    at = tuple(float(v) for v in a.at.split(",")) if a.at else None
    ScriptedPickApp(load_cfg(), a.cycles, a.fixed, at).run()


if __name__ == "__main__":
    main()
