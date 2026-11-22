"""Stage 1 SHADOW MODE: leader -> safety filter -> (would-be) real-arm commands. Motors stay OFF.

    LeaderOp (60 Hz) -> MapOp -> ShadowArmOp -> SimOp
                                  reads the real arm (read-only guard active),
                                  runs SafetyFilter, logs the motor commands it WOULD send,
                                  shows the filtered command on the Isaac Sim arm.

The Isaac Sim arm therefore moves exactly like the real arm would: slowly, inside the limits,
only after the leader is aligned with the real arm.

Keys (type + Enter in this terminal):  a = re-arm after a HOLD,  s = soft stop (HOLD),  q = quit

    ./run_real_shadow.sh                  # with the 102 Leader on /dev/ttyUSB0
    ./run_real_shadow.sh --mock           # no leader: slow motion around the arm's current pose
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from datetime import timedelta

import numpy as np
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec

from common import SIM_JOINTS, LeaderDevice, Mapper, SimLink, find_leader_port, load_cfg
from kinematics import RebotKinematics
from real_arm import READ_ONLY, STATUS, RealArmReader, load_real_cfg, motor_to_sim
from safety import SafetyFilter, SafetyParams

assert READ_ONLY, "shadow mode must run with the read-only guard"
RATE = 60.0
SHARED = {"measured": None, "keys": []}


class LeaderOp(Operator):
    """Real leader -> sim targets, or a slow mock motion around the measured arm pose."""

    def __init__(self, fragment, *args, cfg, port, mock, **kw):
        self.cfg, self.port, self.mock = cfg, port, mock
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.output("target")

    def start(self):
        self.mapper = Mapper(self.cfg)
        self.t0 = time.perf_counter()
        self.q0 = None
        if not self.mock:
            self.dev = LeaderDevice(self.cfg, self.port)
            self.dev.open()

    def stop(self):
        if not self.mock:
            self.dev.close()

    def compute(self, op_input, op_output, context):
        now = time.perf_counter()
        if self.mock:
            m = SHARED["measured"]
            if m is None:  # keep the arm op ticking so it can take the first reading
                op_output.emit({"t": now, "q": None, "ok": False}, "target")
                return
            if self.q0 is None:
                self.q0 = np.array(m, float)
            t = now - self.t0
            q = self.q0.copy()
            q[0] += math.radians(25) * math.sin(0.25 * t)               # base +-25 deg
            q[1] += -math.radians(20) * (1 - math.cos(0.2 * t)) / 2      # shoulder lifts up to 20 deg
            q[6] = 0.025 * (1 - math.cos(0.3 * t)) / 2                   # gripper half open
            op_output.emit({"t": now, "q": q, "ok": True}, "target")
            return
        ok, deg = self.dev.read()
        if deg:
            q8 = self.mapper(deg)
            op_output.emit({"t": now, "q": np.r_[q8[:6], q8[6]], "ok": ok}, "target")


class ShadowArmOp(Operator):
    def __init__(self, fragment, *args, rcfg, params, **kw):
        self.rcfg, self.params = rcfg, params
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.input("target")
        spec.output("cmd")

    def start(self):
        self.reader = RealArmReader(self.rcfg)
        self.reader.open()
        kin = RebotKinematics()
        self.safety = SafetyFilter(self.params, fk=lambda q: kin.fk(q).translation)
        self.t_target = None
        self.t_feedback = None
        self.target = None
        self.t_print = 0.0
        self.log = open(f"../logs/real_shadow_{time.strftime('%Y%m%d_%H%M%S')}.jsonl", "w")
        print("[shadow] real arm connected READ-ONLY; motors stay OFF. Commands are only logged.", flush=True)

    def stop(self):
        self.reader.close()
        self.log.close()

    def _measured(self):
        try:
            states = self.reader.read()
        except Exception as exc:
            print(f"[shadow] feedback read failed: {exc}", flush=True)
            return None, None
        if any(s is None for s in states.values()):
            return None, states
        sim = motor_to_sim(self.rcfg, states)
        q = np.array([sim[f"joint{i}"] for i in range(1, 7)] + [sim.get("finger_left") or 0.0])
        self.t_feedback = time.perf_counter()
        return q, states

    def _motor_cmd(self, cmd):
        """sim-space command -> the motor positions (deg) that WOULD be sent."""
        out = {}
        for i, j in enumerate(self.rcfg["joints"]):
            out[j["name"]] = math.degrees(cmd[i] / j["direction"]) + j["zero_offset_deg"]
        g = self.rcfg["gripper"]
        out["gripper"] = g["closed_deg"] + (cmd[6] / g["sim_open_m"]) * (g["open_deg"] - g["closed_deg"])
        return out

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("target")
        now = time.perf_counter()
        if msg and msg.get("ok"):
            self.target, self.t_target = np.asarray(msg["q"], float), msg["t"]
        measured, states = self._measured()
        SHARED["measured"] = measured
        health = None
        if states and all(s is not None for s in states.values()):
            faults = [f"{n}: {STATUS.get(s.status_code, s.status_code)}" for n, s in states.items()
                      if s.status_code not in (0, 1)]
            health = {"temp": [max(s.t_mos, s.t_rotor) for s in states.values()],
                      "torque": [s.torq for s in states.values()],
                      "fault": ", ".join(faults) if faults else None}
        for k in SHARED["keys"][:]:
            SHARED["keys"].remove(k)
            if k == "a":
                self.safety.rearm(now, measured)
                print("[shadow] re-armed", flush=True)
            elif k == "s":
                self.safety.soft_stop(now, measured)
        t_age = 1e9 if self.t_target is None else now - self.t_target
        f_age = 1e9 if self.t_feedback is None else now - self.t_feedback
        cmd = self.safety.step(now, 1 / RATE, self.target, t_age, measured, f_age, health)
        if cmd is None:
            return
        would_send = self._motor_cmd(cmd)
        self.log.write(json.dumps({
            "t": time.time(), "mode": self.safety.mode, "reason": self.safety.reason, "notes": self.safety.notes,
            "target": None if self.target is None else self.target.tolist(), "cmd": cmd.tolist(),
            "measured": None if measured is None else measured.tolist(), "would_send_motor_deg": would_send}) + "\n")
        op_output.emit(cmd, "cmd")
        if now - self.t_print > 1.0:
            self.t_print = now
            gap = "" if self.target is None else \
                f" | cmd-target max {np.degrees(np.abs(cmd[:6] - self.target[:6])).max():5.1f} deg"
            notes = f" | {'; '.join(self.safety.notes)}" if self.safety.notes else ""
            print(f"[shadow] {self.safety.mode:10s} {self.safety.reason}{gap}{notes}", flush=True)
            print("         would send (motor deg): " +
                  " ".join(f"{k}={v:7.1f}" for k, v in would_send.items()), flush=True)


class SimOp(Operator):
    def __init__(self, fragment, *args, cfg, **kw):
        self.cfg = cfg
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.input("cmd")

    def start(self):
        self.link = SimLink(self.cfg)

    def stop(self):
        self.link.close()

    def compute(self, op_input, op_output, context):
        c = op_input.receive("cmd")
        self.link.send(list(c[:6]) + [float(c[6]), -float(c[6])])


class ShadowApp(Application):
    def __init__(self, cfg, rcfg, port, mock, params):
        self.cfg, self.rcfg, self.port, self.mock, self.params = cfg, rcfg, port, mock, params
        super().__init__()

    def compose(self):
        leader = LeaderOp(self, PeriodicCondition(self, recess_period=timedelta(seconds=1 / RATE)),
                          cfg=self.cfg, port=self.port, mock=self.mock, name="leader")
        arm = ShadowArmOp(self, rcfg=self.rcfg, params=self.params, name="shadow_arm")
        sim = SimOp(self, cfg=self.cfg, name="sim")
        self.add_flow(leader, arm)
        self.add_flow(arm, sim)


def keyboard():
    for line in sys.stdin:
        k = line.strip().lower()[:1]
        if k == "q":
            import os
            os._exit(0)
        if k in ("a", "s"):
            SHARED["keys"].append(k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="no leader: slow motion around the current arm pose")
    ap.add_argument("--leader-port")
    ap.add_argument("--vmax", type=float, default=30.0, help="deg/s (start at 30)")
    a = ap.parse_args()
    cfg, rcfg = load_cfg(), load_real_cfg()
    limits = [tuple(cfg["sim_limits"][f"joint{i}"]) for i in range(1, 7)]
    params = SafetyParams(limits=limits, vmax_deg_s=a.vmax, check_tracking=False)  # motors off: arm cannot follow
    port = None if a.mock else find_leader_port(a.leader_port or "/dev/ttyUSB0")
    if sys.stdin.isatty():
        threading.Thread(target=keyboard, daemon=True).start()
    ShadowApp(cfg, rcfg, port, a.mock, params).run()


if __name__ == "__main__":
    main()
