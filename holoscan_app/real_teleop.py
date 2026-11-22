"""Stage 2: the 102 Leader drives the REAL reBot B601-DM through Holoscan (+ the Isaac Sim twin).

    LeaderOp (60 Hz) -> LiveArmOp -> SimOp
      102 Leader        SafetyFilter -> Damiao POS_VEL (motor-side speed cap)
                        real feedback -> HOLD on any fault

Starts with the motors OFF. Nothing moves until you type  e + Enter.
Keys (type + Enter in this terminal):
    e  enable the arm (only if the leader is aligned with the arm and no motor reports a fault)
    a  re-arm after a HOLD (then align the leader again)
    s  soft stop: HOLD where it is
    p  park: fold slowly to the rest pose, then switch the motors off
    q  park, switch off and quit            (Ctrl-C does the same)
Gripper: FORCE_POS with a current cap (--grip-force, default 0.10 = Seeed default); --no-gripper leaves it off.

    ./run_real_teleop.sh                  # first session: 20 deg/s
    ./run_real_teleop.sh --vmax 30
    ./run_real_teleop.sh --dry            # everything except sending to the motors
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

from common import LeaderDevice, Mapper, SimLink, find_leader_port, load_cfg
from kinematics import RebotKinematics
from real_common import health_of, load_real_cfg, measured_q7
from real_live import RealArmLive, rest_pose
from safety import SafetyFilter, SafetyParams

RATE = 60.0
KEYS = []


def say(msg):
    print(msg, flush=True)


class LeaderOp(Operator):
    def __init__(self, fragment, *args, cfg, port, **kw):
        self.cfg, self.port = cfg, port
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.output("target")

    def start(self):
        self.mapper = Mapper(self.cfg)
        self.dev = LeaderDevice(self.cfg, self.port)
        self.dev.open()

    def stop(self):
        self.dev.close()

    def compute(self, op_input, op_output, context):
        ok, deg = self.dev.read()
        now = time.perf_counter()
        if deg:
            q8 = self.mapper(deg)
            op_output.emit({"t": now, "q": np.r_[q8[:6], q8[6]], "ok": ok}, "target")
        else:
            op_output.emit({"t": now, "q": None, "ok": False}, "target")


class LiveArmOp(Operator):
    def __init__(self, fragment, *args, rcfg, limits, vmax, dry, with_gripper=True, grip_force=0.10, **kw):
        self.rcfg, self.limits, self.vmax, self.dry = rcfg, limits, vmax, dry
        self.with_gripper, self.grip_force = with_gripper, grip_force
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.input("target")
        spec.output("show")

    # ------------------------------------------------------------ lifecycle
    def start(self):
        self.arm = RealArmLive(self.rcfg, dry=self.dry)
        self.arm.vlim = math.radians(self.vmax) * 1.2          # motor-side cap just above the software cap
        self.arm.grip_ratio = self.grip_force
        self.arm.open()
        bad = self.arm.check_modes()
        if bad:
            raise RuntimeError(f"motors not in POS_VEL mode: {bad} (refusing to run)")
        kin = RebotKinematics()
        self.fk = lambda q: kin.fk(q).translation
        self.state = "OFF"                                     # OFF / ON / PARKING
        self.filter = None
        self.park = None
        self.target, self.t_target = None, None
        self.t_print = 0.0
        self.parked_since = None
        self.quit_after_park = False
        self.last = None
        self.log = open(f"../logs/real_teleop_{time.strftime('%Y%m%d_%H%M%S')}.jsonl", "w")
        st0 = self.arm.read()
        on = [n for n, s in st0.items() if s is not None and s.status_code == 1 and n != "gripper"]
        m0 = measured_q7(self.rcfg, st0)
        if on and m0 is not None:
            self.arm.enabled = True
            self.filter = self._new_filter(m0, self.vmax)
            self.filter.soft_stop(time.perf_counter(), m0)
            self.state = "ON"
            say(f"[live] motors already ON ({', '.join(on)}): taking over in HOLD at the current pose. "
                "Type p to park, or a to re-arm.")
        say(f"[live] connected ({'DRY RUN - nothing is sent to the motors' if self.dry else 'LIVE'}), "
            f"speed cap {self.vmax:.0f} deg/s, motor cap {math.degrees(self.arm.vlim):.0f} deg/s")
        if self.state == "OFF":
            say("[live] motors are OFF. Align the leader with the real arm, then type  e + Enter  to enable.")

    def stop(self):
        if self.arm.enabled and self.state != "OFF" and not self.dry:
            m = measured_q7(self.rcfg, self.arm.read())
            if m is not None:
                self.arm.send(m)          # hold exactly where it is (CAN timeout is off: it keeps holding)
            say("[live] EMERGENCY EXIT: motors left ON and HOLDING their position. Support the arm, then either\n"
                "       restart this program and type p to park, or switch the arm power off while holding it.")
        self.arm.close()
        self.log.close()

    # ------------------------------------------------------------ helpers
    def _new_filter(self, measured, vmax, park=False):
        p = SafetyParams(limits=self.limits, vmax_deg_s=vmax, check_tracking=not self.dry)
        if park:
            p.blend_s, p.blend_vmax_deg_s, p.align_tol_deg, p.leader_timeout_s = 0.0, vmax, 360.0, 1e9
        f = SafetyFilter(p, fk=self.fk)
        f.cmd = np.array(measured, float)
        return f

    def _quit(self):
        import os
        import signal
        say("[live] quitting")
        os.kill(os.getpid(), signal.SIGINT)

    def _aligned(self, measured):
        if self.target is None:
            return False, "no leader data"
        d = np.degrees(np.abs(self.target[:6] - measured[:6]))
        j = int(d.argmax())
        return d.max() <= 10.0, f"joint{j + 1} is {d[j]:.0f} deg away"

    def _at_rest(self, measured):
        return np.degrees(np.abs(measured[:6] - rest_pose()[:6])).max() < 4.0

    # ------------------------------------------------------------ tick
    def compute(self, op_input, op_output, context):
        msg = op_input.receive("target")
        now = time.perf_counter()
        if msg and msg.get("ok") and msg.get("q") is not None:
            self.target, self.t_target = np.asarray(msg["q"], float), msg["t"]
        states = self.arm.read()
        measured = measured_q7(self.rcfg, states)
        health = health_of(states)
        t_age = 1e9 if self.t_target is None else now - self.t_target
        f_age = 0.0 if measured is not None else 1e9

        for k in KEYS[:]:
            KEYS.remove(k)
            self._key(k, measured, health, now)

        cmd = None
        if self.state == "ON":
            cmd = self.filter.step(now, 1 / RATE, self.target, t_age, measured, f_age, health)
            if cmd is not None:
                self.arm.send(cmd)
        elif self.state == "PARKING":
            rest = rest_pose()
            rest[6] = self.park.cmd[6]          # park does not close/open the gripper
            cmd = self.park.step(now, 1 / RATE, rest, 0.0, measured, f_age, health)
            if cmd is not None:
                self.arm.send(cmd)
            if self.park.mode == "HOLD":
                say(f"[live] PARK stopped: {self.park.reason}. HOLDING. Type p to try again or support the arm.")
                self.filter, self.state = self.park, "ON"
                self.quit_after_park = False
            elif measured is not None and self._at_rest(measured):
                self.parked_since = self.parked_since or now
                if now - self.parked_since > 0.5:
                    self.arm.disable_arm(say)
                    self.state, self.parked_since = "OFF", None
                    say("[live] parked and switched OFF. Type e to enable again, q to quit.")
                    if self.quit_after_park:
                        self._quit()

        # latest snapshot for other operators (e.g. the recorder)
        self.last = {"t": now, "state": self.state, "mode": None if self.filter is None else self.filter.mode,
                     "cmd": None if cmd is None else cmd.copy(),
                     "measured": None if measured is None else measured.copy(), "vmax": self.vmax}
        show = cmd if cmd is not None else measured
        if show is not None:
            op_output.emit(show, "show")
        self.log.write(json.dumps({
            "t": time.time(), "state": self.state,
            "mode": None if self.filter is None else self.filter.mode,
            "target": None if self.target is None else self.target.tolist(),
            "cmd": None if cmd is None else cmd.tolist(),
            "measured": None if measured is None else measured.tolist(),
            "torque": None if health is None else health["torque"]}) + "\n")
        if now - self.t_print > 1.0:
            self.t_print = now
            self._status(measured, cmd, health)

    def _key(self, k, measured, health, now):
        if k == "e":
            if self.state != "OFF":
                say("[live] already enabled")
                return
            if measured is None:
                say("[live] cannot enable: no feedback from all motors")
                return
            if health and health["fault"]:
                say(f"[live] cannot enable: motor fault {health['fault']}")
                return
            ok, why = self._aligned(measured)
            if not ok:
                say(f"[live] cannot enable: leader not aligned ({why}, need < 10)")
                return
            ok, msg = self.arm.enable_arm(say)
            say(f"[live] {msg}")
            if ok and self.with_gripper:
                gok, gmsg = self.arm.enable_gripper(say)
                say(f"[live] {gmsg}")
                ok = ok and gok
            m2 = measured_q7(self.rcfg, self.arm.read())          # start the filter exactly where the arm is now
            self.filter = self._new_filter(m2 if m2 is not None else measured, self.vmax)
            self.state = "ON"
            if not ok:
                self.filter.soft_stop(now, measured)
        elif k == "a" and self.filter is not None and self.state == "ON":
            self.filter.rearm(now, measured)
            say("[live] re-armed: align the leader with the arm")
        elif k == "s" and self.filter is not None and self.state == "ON":
            self.filter.soft_stop(now, measured)
            say("[live] soft stop: HOLDING")
        elif k == "q":
            if self.state == "OFF":
                self._quit()
                return
            self.quit_after_park = True
            self._key("p", measured, health, now)
        elif k == "p":
            if self.state == "OFF" or measured is None:
                say("[live] nothing to park (motors off)" if self.state == "OFF" else "[live] no feedback")
                return
            self.park = self._new_filter(self.filter.cmd if self.filter is not None else measured, 15.0, park=True)
            self.state = "PARKING"
            say("[live] parking slowly (15 deg/s)...")

    def _status(self, measured, cmd, health):
        if self.state == "OFF":
            ok, why = (False, "no feedback") if measured is None else self._aligned(measured)
            line = f"[live] OFF        leader {'ALIGNED - type e to enable' if ok else 'not aligned: ' + why}"
        elif self.state == "PARKING":
            line = "[live] PARKING    folding to the rest pose"
        else:
            line = f"[live] {self.filter.mode:10s} {self.filter.reason}"
            if self.filter.notes:
                line += " | " + "; ".join(self.filter.notes)
        if measured is not None and cmd is not None and self.state != "OFF":
            line += f" | arm lag {np.degrees(np.abs(cmd[:6] - measured[:6])).max():4.1f} deg"
        if health:
            line += f" | max torque {max(abs(t) for t in health['torque'][:6]):4.1f} Nm, max temp {max(health['temp']):.0f} C"
        say(line)


class SimOp(Operator):
    def __init__(self, fragment, *args, cfg, **kw):
        self.cfg = cfg
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.input("show")

    def start(self):
        self.link = SimLink(self.cfg)

    def stop(self):
        self.link.close()

    def compute(self, op_input, op_output, context):
        c = op_input.receive("show")
        self.link.send(list(c[:6]) + [float(c[6]), -float(c[6])])


class RealTeleopApp(Application):
    def __init__(self, cfg, rcfg, port, vmax, dry, with_gripper=True, grip_force=0.10):
        self.cfg, self.rcfg, self.port, self.vmax, self.dry = cfg, rcfg, port, vmax, dry
        self.with_gripper, self.grip_force = with_gripper, grip_force
        super().__init__()

    def compose(self):
        limits = [tuple(self.cfg["sim_limits"][f"joint{i}"]) for i in range(1, 7)]
        leader = LeaderOp(self, PeriodicCondition(self, recess_period=timedelta(seconds=1 / RATE)),
                          cfg=self.cfg, port=self.port, name="leader")
        arm = LiveArmOp(self, rcfg=self.rcfg, limits=limits, vmax=self.vmax, dry=self.dry,
                        with_gripper=self.with_gripper, grip_force=self.grip_force, name="live_arm")
        sim = SimOp(self, cfg=self.cfg, name="sim")
        self.add_flow(leader, arm)
        self.add_flow(arm, sim)


def key_file(path):
    """Keys appended to a file (echo s >> keys) - lets a second shell control the arm."""
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
            if k in ("e", "a", "s", "p", "q"):
                say(f"[live] key '{k}' from {path}")
                KEYS.append(k)


def keyboard():
    for line in sys.stdin:
        k = line.strip().lower()[:1]
        if k in ("e", "a", "s", "p", "q"):
            KEYS.append(k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vmax", type=float, default=20.0, help="deg/s software speed cap (max 45)")
    ap.add_argument("--dry", action="store_true", help="never send anything to the motors")
    ap.add_argument("--leader-port", default="/dev/ttyUSB0")
    ap.add_argument("--no-gripper", action="store_true", help="leave the gripper motor off")
    ap.add_argument("--grip-force", type=float, default=0.10, help="gripper FORCE_POS current cap 0.03-0.2 (Seeed default 0.1)")
    ap.add_argument("--keys-file", help="also accept keys appended to this file (echo s >> FILE)")
    a = ap.parse_args()
    if not 1.0 <= a.vmax <= 45.0:
        sys.exit("--vmax must be between 1 and 45 deg/s")
    if not sys.stdin.isatty() and not a.dry and not a.keys_file:
        sys.exit("run this in a terminal (or with --keys-file): the keys e/a/s/p/q control the arm")
    if sys.stdin.isatty():
        threading.Thread(target=keyboard, daemon=True).start()
    if a.keys_file:
        threading.Thread(target=key_file, args=(a.keys_file,), daemon=True).start()
    if not 0.03 <= a.grip_force <= 0.2:
        sys.exit("--grip-force must be between 0.03 and 0.2")
    RealTeleopApp(load_cfg(), load_real_cfg(), find_leader_port(a.leader_port), a.vmax, a.dry,
                  not a.no_gripper, a.grip_force).run()


if __name__ == "__main__":
    main()
