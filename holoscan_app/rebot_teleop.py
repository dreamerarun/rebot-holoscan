"""reBot teleop pipeline on NVIDIA Holoscan.

    LeaderOp (60 Hz) -> MapOp -> SimLinkOp -> StatusOp
      102 Leader         deg->rad/m      ZMQ <-> Isaac Sim twin
      (FashionStar)      clamp, rate     (targets out, state back)
                         limit, watchdog

Step 1 drives only the Isaac Sim twin. The real B601-DM follower will be one more
operator fed by MapOp (not part of this file yet).

Run:
    source ~/rebot-holoscan/env.sh
    hs_py holoscan_app/rebot_teleop.py            # real leader on USB
    hs_py holoscan_app/rebot_teleop.py --mock     # synthetic leader motion, no hardware
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import yaml
import zmq
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec

ROOT = Path(__file__).resolve().parents[1]
LEADER_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                 "wrist_yaw", "wrist_roll", "gripper"]
SIM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
              "finger_left", "finger_right"]


# --------------------------------------------------------------------------- leader
def find_leader_port(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    # Prefer the stable by-id name of the leader's CH340 (USB vendor 1a86): it survives re-plugs,
    # while ttyUSB0 may come back as ttyUSB1.
    ch340 = sorted(glob.glob("/dev/serial/by-id/*1a86*"))
    if len(ch340) == 1:
        return ch340[0]
    cands = sorted(glob.glob("/dev/serial/by-id/*")) or sorted(
        glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise RuntimeError("No serial device found for the 102 Leader (plug it in, then `ls /dev/ttyUSB*`).")
    raise RuntimeError(f"Several serial devices {cands}; pass --leader-port.")


def unwrap(value: float, lo: float, hi: float) -> float:
    """Bring a multi-turn reading into the 360 deg window centred on the joint range."""
    centre = 0.5 * (lo + hi)
    return value - round((value - centre) / 360.0) * 360.0


class LeaderOp(Operator):
    """Reads the 7 leader servos (degrees). The leader is passive: torque is released."""

    def __init__(self, fragment, *args, cfg, port, mock=False, **kwargs):
        self.cfg, self.port, self.mock = cfg, port, mock
        self.bus = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("leader")

    def start(self):
        self.t0 = time.perf_counter()
        self.last = None
        if self.mock:
            print("[leader] MOCK leader: synthetic motion, no hardware", flush=True)
            return
        from motorbridge_smart_servo import FashionStarServo

        ids = self.cfg["leader"]["ids"]
        self.bus = FashionStarServo(self.port, baudrate=int(self.cfg["leader"]["baudrate"]))
        missing = [n for n, i in ids.items() if not self.bus.ping(i)]
        if missing:
            raise RuntimeError(f"Leader servos not answering: {missing}")
        for i in ids.values():          # same as the Seeed plugin's configure()
            self.bus.unlock(i)          # leader arm stays limp (hand-guided)
            time.sleep(0.05)
        for i in ids.values():
            self.bus.reset_multi_turn(i)
        print(f"[leader] connected on {self.port}: 7/7 servos answer", flush=True)

    def stop(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None

    def _mock(self, t):
        s = math.sin
        return {"shoulder_pan": 35 * s(0.5 * t), "shoulder_lift": 60 + 25 * s(0.4 * t),
                "elbow_flex": -70 + 25 * s(0.45 * t + 1.0), "wrist_flex": 25 * s(0.7 * t),
                "wrist_yaw": 30 * s(0.6 * t + 0.5), "wrist_roll": 60 * s(0.3 * t),
                "gripper": 22.5 + 22.5 * s(0.8 * t)}

    def compute(self, op_input, op_output, context):
        now = time.perf_counter()
        if self.mock:
            deg, ok = self._mock(now - self.t0), True
        else:
            ok = True
            try:
                mon = self.bus.sync_monitor(list(self.cfg["leader"]["ids"].values()))
                deg = {}
                for name, sid in self.cfg["leader"]["ids"].items():
                    m = mon.get(sid)
                    if m is None:
                        raise RuntimeError(f"no reply from {name} (id {sid})")
                    lo, hi = self.cfg["leader"]["ranges"][name]
                    deg[name] = float(np.clip(unwrap(m.angle_deg, lo, hi), lo, hi))
            except Exception as exc:  # keep running; MapOp's watchdog holds the last target
                ok = False
                deg = self.last or {}
                if int(now) != int(getattr(self, "_last_err", 0)):
                    print(f"[leader] read failed: {exc}", flush=True)
                    self._last_err = now
        if ok:
            self.last = deg
        op_output.emit({"t": now, "ok": ok, "deg": deg}, "leader")


# --------------------------------------------------------------------------- mapping
class MapOp(Operator):
    """Leader degrees -> sim targets (rad, m), clamped, rate limited, with watchdog."""

    def __init__(self, fragment, *args, cfg, **kwargs):
        self.cfg = cfg
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("leader")
        spec.output("target")

    def start(self):
        self.prev = None
        self.last_fresh = None

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("leader")
        f = self.cfg["filter"]
        hold = False
        if msg["ok"] and msg["deg"]:
            self.last_fresh = msg["t"]
            deg = msg["deg"]
            q = []
            for m in self.cfg["arm_mapping"]:
                sim_deg = deg[m["leader"]] * m["scale"] + m["offset_deg"]
                lo, hi = self.cfg["sim_limits"][m["sim"]]
                q.append(float(np.clip(math.radians(sim_deg), lo, hi)))
            g = self.cfg["gripper_mapping"]
            span = g["leader_open_deg"] - g["leader_closed_deg"]
            openness = float(np.clip((deg[g["leader"]] - g["leader_closed_deg"]) / span, 0.0, 1.0))
            grip = g["sim_closed_m"] + openness * (g["sim_open_m"] - g["sim_closed_m"])
            q = np.array(q + [grip, -grip])
            if self.prev is not None:
                step = math.radians(f["max_step_deg"])
                d = q[:6] - self.prev[:6]
                d[np.abs(d) < math.radians(f["deadband_deg"])] = 0.0
                q[:6] = self.prev[:6] + np.clip(d, -step, step)
            self.prev = q
        else:
            hold = True
        if self.prev is None:
            return  # nothing valid yet
        stale = self.last_fresh is None or (time.perf_counter() - self.last_fresh) > f["watchdog_s"]
        op_output.emit({"t": time.perf_counter(), "q": self.prev.tolist(),
                        "hold": hold or stale, "leader": msg["deg"]}, "target")


# --------------------------------------------------------------------------- sim link
class SimLinkOp(Operator):
    """Publishes targets to the Isaac Sim twin and reads its joint state back."""

    def __init__(self, fragment, *args, cfg, **kwargs):
        self.cfg = cfg
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("target")
        spec.output("status")

    def start(self):
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.SNDHWM, 2)
        self.pub.bind(self.cfg["sim_link"]["cmd_endpoint"])
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.connect(self.cfg["sim_link"]["state_endpoint"])
        self.state = None
        self.seq = 0

    def stop(self):
        self.pub.close(0)
        self.sub.close(0)

    def compute(self, op_input, op_output, context):
        tgt = op_input.receive("target")
        self.seq += 1
        self.pub.send_string(json.dumps({"seq": self.seq, "t": time.time(), "q": tgt["q"],
                                         "names": SIM_JOINTS, "hold": tgt["hold"]}))
        try:
            while True:
                self.state = json.loads(self.sub.recv_string(zmq.NOBLOCK))
        except zmq.Again:
            pass
        op_output.emit({"target": tgt, "sim": self.state}, "status")


class StatusOp(Operator):
    """Once a second: leader degrees, sim target vs sim actual, loop rate."""

    def setup(self, spec: OperatorSpec):
        spec.input("status")

    def start(self):
        self.n = 0
        self.t_last = time.perf_counter()

    def compute(self, op_input, op_output, context):
        s = op_input.receive("status")
        self.n += 1
        now = time.perf_counter()
        if now - self.t_last < 1.0:
            return
        hz = self.n / (now - self.t_last)
        self.n, self.t_last = 0, now
        tgt = np.array(s["target"]["q"])
        lead = s["target"]["leader"] or {}
        line = (f"[status] {hz:5.1f} Hz  leader(deg) "
                + " ".join(f"{lead.get(j, float('nan')):7.1f}" for j in LEADER_JOINTS)
                + "  | target(deg) " + " ".join(f"{math.degrees(v):7.1f}" for v in tgt[:6])
                + f" grip {tgt[6] * 1000:4.1f}mm")
        sim = s["sim"]
        if sim and time.time() - sim.get("t", 0) < 1.0:
            act = np.array(sim["q"])
            err = np.degrees(np.abs(act[:6] - tgt[:6])).max()
            line += f"  | sim err max {err:4.1f} deg"
        else:
            line += "  | sim: not connected"
        if s["target"]["hold"]:
            line += "  [HOLD: no fresh leader data]"
        print(line, flush=True)


# --------------------------------------------------------------------------- app
class RebotTeleopApp(Application):
    def __init__(self, cfg, port, mock):
        self.cfg, self.port, self.mock = cfg, port, mock
        super().__init__()

    def compose(self):
        period = timedelta(seconds=1.0 / float(self.cfg["rate_hz"]))
        leader = LeaderOp(self, PeriodicCondition(self, recess_period=period),
                          cfg=self.cfg, port=self.port, mock=self.mock, name="leader")
        mapper = MapOp(self, cfg=self.cfg, name="map")
        sim = SimLinkOp(self, cfg=self.cfg, name="sim_link")
        status = StatusOp(self, name="status")
        self.add_flow(leader, mapper, {("leader", "leader")})
        self.add_flow(mapper, sim, {("target", "target")})
        self.add_flow(sim, status, {("status", "status")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "teleop.yaml"))
    ap.add_argument("--leader-port", default=None)
    ap.add_argument("--mock", action="store_true", help="synthetic leader motion, no hardware")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    port = None if a.mock else find_leader_port(a.leader_port or cfg["leader"]["port"])
    RebotTeleopApp(cfg, port, a.mock).run()


if __name__ == "__main__":
    main()
