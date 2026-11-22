"""READ-ONLY digital twin: the Isaac Sim arm mirrors the REAL reBot (motors stay OFF).

    RealReadOp (30 Hz) -> MirrorOp -> Isaac Sim
      Damiao feedback      offsets/directions from config/real_arm.yaml

Move the real arm slowly BY HAND and watch the sim follow. Used to verify every joint's
direction and zero offset before any motor is ever enabled. Prints a line per second and
the min/max seen per joint (to calibrate the gripper range).

    ./run_real_mirror.sh          (start the twin first: ./run_twin_gui.sh)
"""
from __future__ import annotations

import json
import math
import time
from datetime import timedelta

from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec

from common import SIM_JOINTS, SimLink, load_cfg
from real_arm import READ_ONLY, STATUS, RealArmReader, load_real_cfg, motor_to_sim

assert READ_ONLY


class RealReadOp(Operator):
    def __init__(self, fragment, *args, rcfg, **kw):
        self.rcfg = rcfg
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.output("states")

    def start(self):
        self.reader = RealArmReader(self.rcfg)
        self.reader.open()
        print("[real] connected READ-ONLY (motors are never enabled)", flush=True)

    def stop(self):
        self.reader.close()

    def compute(self, op_input, op_output, context):
        try:
            op_output.emit(self.reader.read(), "states")
            self.down_since = None
        except Exception as exc:  # USB-CAN unplugged/re-enumerated: reconnect, never crash
            now = time.time()
            if getattr(self, "down_since", None) is None:
                self.down_since = now
                print(f"[real] lost the USB-CAN bridge ({exc}); reconnecting...", flush=True)
            if now - getattr(self, "last_try", 0) > 1.0:
                self.last_try = now
                try:
                    self.reader.close()
                    self.reader = RealArmReader(self.rcfg)
                    self.reader.open()
                    print("[real] reconnected", flush=True)
                except Exception:
                    pass


class MirrorOp(Operator):
    def __init__(self, fragment, *args, rcfg, cfg, **kw):
        self.rcfg, self.cfg = rcfg, cfg
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.input("states")

    def start(self):
        self.link = SimLink(self.cfg)
        self.last_q = [0.0] * len(SIM_JOINTS)
        self.t_print = 0.0
        self.minmax = {}
        self.log = open("../logs/real_mirror.jsonl", "a")

    def stop(self):
        self.link.close()
        self.log.close()

    def compute(self, op_input, op_output, context):
        states = op_input.receive("states")
        sim = motor_to_sim(self.rcfg, states)
        for i, n in enumerate(SIM_JOINTS):
            if sim.get(n) is not None:
                self.last_q[i] = sim[n]
        self.link.send(self.last_q)
        raw = {n: (math.degrees(s.pos) if s else None) for n, s in states.items()}
        for n, v in raw.items():
            if v is not None:
                lo, hi = self.minmax.get(n, (v, v))
                self.minmax[n] = (min(lo, v), max(hi, v))
        self.log.write(json.dumps({"t": time.time(), "raw_deg": raw}) + "\n")
        now = time.time()
        if now - self.t_print >= 1.0:
            self.t_print = now
            bad = [f"{n}:{STATUS.get(s.status_code, s.status_code)}" for n, s in states.items() if s and s.status_code != 0]
            missing = [n for n, s in states.items() if s is None]
            line = "[mirror] sim deg " + " ".join(f"{math.degrees(self.last_q[i]):7.1f}" for i in range(6))
            line += f" | gripper raw {raw.get('gripper') if raw.get('gripper') is None else round(raw['gripper'], 1)}"
            line += " | range " + " ".join(f"{n}[{lo:.0f},{hi:.0f}]" for n, (lo, hi) in self.minmax.items())
            if bad:
                line += f" | STATUS {bad}"
            if missing:
                line += f" | NO REPLY {missing}"
            print(line, flush=True)


class RealMirrorApp(Application):
    def compose(self):
        rcfg, cfg = load_real_cfg(), load_cfg()
        rd = RealReadOp(self, PeriodicCondition(self, recess_period=timedelta(seconds=1 / rcfg["read_hz"])),
                        rcfg=rcfg, name="real_read")
        self.add_flow(rd, MirrorOp(self, rcfg=rcfg, cfg=cfg, name="mirror"))


if __name__ == "__main__":
    RealMirrorApp().run()
