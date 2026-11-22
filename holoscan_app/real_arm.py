"""Real reBot B601-DM access.

Stage 0/1 (now): READ ONLY. Importing this module installs a guard on motorbridge that
blocks every call able to enable, move, zero, change mode or write a motor. The guard is
process-wide and cannot be switched off from here; the later command path will live in a
separate module that the user runs explicitly.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import yaml
from motorbridge import Controller, Motor

ROOT = Path(__file__).resolve().parents[1]
STATUS = {0: "disabled", 1: "ENABLED", 8: "over-voltage", 9: "under-voltage", 10: "over-current",
          11: "MOS over-temp", 12: "rotor over-temp", 13: "comm lost", 14: "overload"}

_FORBIDDEN_MOTOR = ["enable", "disable", "ensure_mode", "send_mit", "send_pos_vel", "send_vel", "send_force_pos",
                    "set_zero_position", "store_parameters", "write_register_f32", "write_register_u32",
                    "damiao_write_param_f32", "damiao_write_param_u32", "clear_error", "set_can_timeout_ms"]
_FORBIDDEN_CTRL = ["enable_all", "disable_all", "shutdown"]


def _blocked(name):
    def f(*a, **k):
        raise RuntimeError(f"READ-ONLY GUARD: {name}() is blocked")
    return f


for _n in _FORBIDDEN_MOTOR:
    setattr(Motor, _n, _blocked(f"Motor.{_n}"))
for _n in _FORBIDDEN_CTRL:
    setattr(Controller, _n, _blocked(f"Controller.{_n}"))
READ_ONLY = True


def load_real_cfg(path=None):
    return yaml.safe_load(Path(path or ROOT / "config" / "real_arm.yaml").read_text())


class RealArmReader:
    """Reads the 7 Damiao motors with feedback requests only (motors stay disabled)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ctrl = None
        self.motors = []   # (name, Motor)

    def open(self):
        c = self.cfg
        self.ctrl = Controller.from_dm_serial(c["port"], int(c["baud"]))
        for j in c["joints"]:
            self.motors.append((j["name"], self.ctrl.add_damiao_motor(int(j["motor_id"]), int(j["feedback_id"]), j["model"])))
        g = c["gripper"]
        self.motors.append(("gripper", self.ctrl.add_damiao_motor(int(g["motor_id"]), int(g["feedback_id"]), g["model"])))

    def close(self):
        if self.ctrl is not None:
            self.ctrl.close()
            self.ctrl = None

    def read(self, wait_s=0.012):
        """Returns {name: MotorState or None}."""
        for _, m in self.motors:
            m.request_feedback()
        t_end = time.perf_counter() + wait_s
        while time.perf_counter() < t_end:
            try:
                self.ctrl.poll_feedback_once()
            except Exception:
                pass
            time.sleep(0.001)
        return {n: m.get_state() for n, m in self.motors}


def motor_to_sim(cfg, states):
    """Motor readings -> sim joint targets (rad) + gripper finger (m). None where a motor did not answer."""
    out = {}
    for j in cfg["joints"]:
        st = states.get(j["name"])
        if st is None:
            out[j["sim"]] = None
            continue
        out[j["sim"]] = j["direction"] * (st.pos - math.radians(j["zero_offset_deg"]))
    g, st = cfg["gripper"], states.get("gripper")
    if st is None or g.get("open_deg") is None:
        out["finger_left"] = None
    else:
        lo, hi = math.radians(g["closed_deg"]), math.radians(g["open_deg"])
        out["finger_left"] = float(np.clip((st.pos - lo) / (hi - lo), 0, 1)) * g["sim_open_m"]
    if out.get("finger_left") is not None:
        out["finger_right"] = -out["finger_left"]
    return out
