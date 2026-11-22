"""READ-ONLY check of the reBot B601-DM motors (Stage 0 of the safety plan).

Never enables, moves, zeroes, changes mode or writes any motor: every such call is
replaced by a guard that raises before anything is sent.

    hs_py tests/read_real_arm.py                # one snapshot + motor config registers
    hs_py tests/read_real_arm.py --watch 60     # live angles for 60 s (move the arm BY HAND, motors off)
"""
from __future__ import annotations

import argparse
import math
import time

import motorbridge
import yaml
from motorbridge import Controller, Motor

CFG = "/home/dexstro/ReBot_Arm_DigitalTwin_DM/reBotArm_ros2_DM/third_party/reBotArm_control_py/config/rebotarm_dm.yaml"
STATUS = {0: "disabled", 1: "ENABLED", 8: "over-voltage", 9: "under-voltage", 10: "over-current",
          11: "MOS over-temp", 12: "rotor over-temp", 13: "comm lost", 14: "overload"}

# ---------------------------------------------------------------- read-only guard
FORBIDDEN_MOTOR = ["enable", "disable", "ensure_mode", "send_mit", "send_pos_vel", "send_vel", "send_force_pos",
                   "set_zero_position", "store_parameters", "write_register_f32", "write_register_u32",
                   "damiao_write_param_f32", "damiao_write_param_u32", "clear_error", "set_can_timeout_ms"]
FORBIDDEN_CTRL = ["enable_all", "disable_all", "shutdown"]


def _blocked(name):
    def f(*a, **k):
        raise RuntimeError(f"READ-ONLY GUARD: {name}() is blocked in this script")
    return f


for n in FORBIDDEN_MOTOR:
    setattr(Motor, n, _blocked(f"Motor.{n}"))
for n in FORBIDDEN_CTRL:
    setattr(Controller, n, _blocked(f"Controller.{n}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--watch", type=float, default=0.0, help="seconds of live readout at 5 Hz")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(CFG))
    print(f"motorbridge {motorbridge.VERSION if hasattr(motorbridge, 'VERSION') else ''} | {a.port} @ {a.baud}")
    ctrl = Controller.from_dm_serial(a.port, a.baud)
    motors = []
    for j in cfg["joints"]:
        m = ctrl.add_damiao_motor(int(j["motor_id"]), int(j["feedback_id"]), str(j["model"]))
        motors.append((j["name"], j["model"], int(j["motor_id"]), m))

    def snapshot():
        for _, _, _, m in motors:
            m.request_feedback()
        t_end = time.time() + 0.15
        while time.time() < t_end:
            try:
                ctrl.poll_feedback_once()
            except Exception:
                pass
            time.sleep(0.005)
        return [(name, model, mid, m.get_state()) for name, model, mid, m in motors]

    rows = snapshot()
    print(f"\n{'joint':8s} {'model':6s} {'id':>4s}  {'status':14s} {'pos deg':>9s} {'vel':>7s} {'torque':>7s} {'T_mos':>6s} {'T_rotor':>7s}")
    answered = 0
    for name, model, mid, st in rows:
        if st is None:
            print(f"{name:8s} {model:6s} 0x{mid:02X}  NO REPLY")
            continue
        answered += 1
        print(f"{name:8s} {model:6s} 0x{mid:02X}  {STATUS.get(st.status_code, f'code {st.status_code}'):14s} "
              f"{math.degrees(st.pos):9.2f} {st.vel:7.3f} {st.torq:7.3f} {st.t_mos:6.1f} {st.t_rotor:7.1f}")
    print(f"\n{answered}/{len(rows)} motors answered")

    print("\nmotor config registers (read only):")
    regs = [("PMAX", motorbridge.RID_PMAX), ("VMAX", motorbridge.RID_VMAX), ("TMAX", motorbridge.RID_TMAX)]
    for name, model, mid, m in motors:
        vals = []
        for rn, rid in regs:
            try:
                vals.append(f"{rn}={m.get_register_f32(rid, 300):.2f}")
            except Exception as exc:
                vals.append(f"{rn}=? ({type(exc).__name__})")
        try:
            mode = m.get_register_u32(motorbridge.RID_CTRL_MODE, 300)
            vals.append(f"mode={ {1: 'MIT', 2: 'POS_VEL', 3: 'VEL', 4: 'FORCE_POS'}.get(mode, mode)}")
        except Exception as exc:
            vals.append(f"mode=? ({type(exc).__name__})")
        print(f"  {name:8s} " + "  ".join(vals))

    if a.watch > 0:
        print(f"\nLIVE for {a.watch:.0f} s (degrees). Motors stay OFF: move the arm slowly by hand.")
        print("  " + " ".join(f"{n:>8s}" for n, *_ in motors))
        t_end = time.time() + a.watch
        while time.time() < t_end:
            rows = snapshot()
            print("  " + " ".join(f"{math.degrees(st.pos):8.1f}" if st else f"{'--':>8s}" for *_, st in rows), flush=True)
            time.sleep(0.05)

    ctrl.close()
    print("\ndone (nothing was enabled or commanded)")


if __name__ == "__main__":
    main()
