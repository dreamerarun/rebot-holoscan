"""Pure helpers shared by the read-only tools and the live command path (no motor I/O here)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
STATUS = {0: "disabled", 1: "ENABLED", 8: "over-voltage", 9: "under-voltage", 10: "over-current",
          11: "MOS over-temp", 12: "rotor over-temp", 13: "comm lost", 14: "overload"}


def load_real_cfg(path=None):
    return yaml.safe_load(Path(path or ROOT / "config" / "real_arm.yaml").read_text())


def motor_to_sim(cfg, states):
    """Motor readings -> sim joint values (rad) + gripper finger (m). None where a motor did not answer."""
    out = {}
    for j in cfg["joints"]:
        st = states.get(j["name"])
        out[j["sim"]] = None if st is None else j["direction"] * (st.pos - math.radians(j["zero_offset_deg"]))
    g, st = cfg["gripper"], states.get("gripper")
    if st is None or g.get("open_deg") is None:
        out["finger_left"] = None
    else:
        lo, hi = math.radians(g["closed_deg"]), math.radians(g["open_deg"])
        out["finger_left"] = float(np.clip((st.pos - lo) / (hi - lo), 0, 1)) * g["sim_open_m"]
    if out.get("finger_left") is not None:
        out["finger_right"] = -out["finger_left"]
    return out


def sim_to_motor_rad(cfg, q7):
    """sim command (6 arm rad + finger m) -> motor positions (rad), dict by motor name."""
    out = {}
    for i, j in enumerate(cfg["joints"]):
        out[j["name"]] = q7[i] / j["direction"] + math.radians(j["zero_offset_deg"])
    g = cfg["gripper"]
    out["gripper"] = math.radians(g["closed_deg"] + (q7[6] / g["sim_open_m"]) * (g["open_deg"] - g["closed_deg"]))
    return out


def measured_q7(cfg, states):
    """All 7 readings in sim space, or None if any motor is missing."""
    if not states or any(s is None for s in states.values()):
        return None
    sim = motor_to_sim(cfg, states)
    return np.array([sim[f"joint{i}"] for i in range(1, 7)] + [sim.get("finger_left") or 0.0])


def health_of(states):
    if not states or any(s is None for s in states.values()):
        return None
    faults = [f"{n}: {STATUS.get(s.status_code, s.status_code)}" for n, s in states.items() if s.status_code not in (0, 1)]
    return {"temp": [max(s.t_mos, s.t_rotor) for s in states.values()],
            "torque": [s.torq for s in states.values()],
            "fault": ", ".join(faults) if faults else None}
