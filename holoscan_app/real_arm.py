"""Real reBot B601-DM, READ-ONLY access.

Importing this module installs a guard on motorbridge that blocks every call able to enable,
move, zero, change mode or write a motor (process-wide). Used by the mirror and shadow tools.
The live command path is real_live.py, which never imports this module.
"""
from __future__ import annotations

import time

from motorbridge import Controller, Motor

from real_common import STATUS, load_real_cfg, measured_q7, motor_to_sim  # noqa: F401 (re-exported)

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
