"""LIVE access to the real reBot B601-DM (joint1-6 in POS_VEL, gripper in force-limited FORCE_POS).

Only used by real_teleop.py, which the user starts and controls with explicit keys.
Damiao POS_VEL mode: each target is sent with a per-motor speed cap (vlim), so the motor itself
refuses to move faster than vlim even if the software above it misbehaves.
"""
from __future__ import annotations

import math
import time

import numpy as np
from motorbridge import MODE_FORCE_POS, RID_CTRL_MODE, Controller

from real_common import sim_to_motor_rad

POS_VEL = 2


class RealArmLive:
    def __init__(self, cfg, dry=False):
        self.cfg, self.dry = cfg, dry
        self.ctrl = None
        self.arm = []            # (name, Motor) joint1..6
        self.grip = None
        self.enabled = False
        self.vlim = 0.4          # rad/s motor-side cap, set by the caller
        self.grip_enabled = False
        self.grip_ratio = 0.10   # FORCE_POS current cap (fraction of max), Seeed default
        self.grip_vlim = 2.6     # rad/s at the gripper motor (~3 cm/s at the fingers)

    # ------------------------------------------------------------------ connection
    def open(self):
        c = self.cfg
        self.ctrl = Controller.from_dm_serial(c["port"], int(c["baud"]))
        for j in c["joints"]:
            self.arm.append((j["name"], self.ctrl.add_damiao_motor(int(j["motor_id"]), int(j["feedback_id"]), j["model"])))
        g = c["gripper"]
        self.grip = ("gripper", self.ctrl.add_damiao_motor(int(g["motor_id"]), int(g["feedback_id"]), g["model"]))

    def close(self):
        if self.ctrl is not None:
            self.ctrl.close()
            self.ctrl = None

    def _poll(self, wait_s):
        t_end = time.perf_counter() + wait_s
        while time.perf_counter() < t_end:
            try:
                self.ctrl.poll_feedback_once()
            except Exception:
                pass
            time.sleep(0.001)

    def read(self, wait_s=0.010):
        """Feedback requests (motors reply with their state). {name: MotorState or None}."""
        for _, m in self.arm + [self.grip]:
            m.request_feedback()
        self._poll(wait_s)
        return {n: m.get_state() for n, m in self.arm + [self.grip]}

    def check_modes(self):
        """All arm motors must already be in POS_VEL (read-only check; we never change modes)."""
        bad = []
        for n, m in self.arm:
            try:
                if m.get_register_u32(RID_CTRL_MODE, 300) != POS_VEL:
                    bad.append(n)
            except Exception:
                bad.append(n)
        return bad

    # ------------------------------------------------------------------ motion
    def enable_arm(self, log=print):
        """Enable joint6 -> joint1 one at a time, each immediately told to hold its measured position.
        Returns (ok, message). If any joint moves > 3 deg the remaining joints are not enabled and
        the already-enabled ones keep holding (POS_VEL keeps its last target; CAN timeout is off)."""
        if self.dry:
            log("[live] DRY RUN: would enable joint6..joint1 now (nothing sent)")
            self.enabled = True
            return True, "dry run"
        for name, m in reversed(self.arm):
            before = self.read()[name]
            if before is None:
                return False, f"{name} did not answer before enable"
            m.enable()
            m.send_pos_vel(float(before.pos), self.vlim)           # hold exactly where it is
            time.sleep(0.15)
            after = self.read()[name]
            moved = 999.0 if after is None else abs(math.degrees(after.pos - before.pos))
            log(f"[live] {name} enabled, holding (moved {moved:.2f} deg)")
            self.enabled = True
            if moved > 3.0:
                return False, f"{name} moved {moved:.1f} deg on enable: stopped, holding"
        return True, "all 6 arm joints enabled and holding"

    def enable_gripper(self, log=print):
        """Gripper: FORCE_POS (position with a current cap), enabled last, told to hold its opening."""
        if self.dry:
            log(f"[live] DRY RUN: would enable the gripper in FORCE_POS, current cap {self.grip_ratio:.2f}")
            self.grip_enabled = True
            return True, "dry run"
        name, m = self.grip
        before = self.read()[name]
        if before is None:
            return False, "gripper did not answer"
        m.ensure_mode(MODE_FORCE_POS, 1000)          # RAM only (not stored): power cycle restores POS_VEL
        m.enable()
        m.send_force_pos(float(before.pos), self.grip_vlim, self.grip_ratio)
        time.sleep(0.15)
        after = self.read()[name]
        moved = 999.0 if after is None else abs(math.degrees(after.pos - before.pos))
        self.grip_enabled = True
        log(f"[live] gripper enabled (FORCE_POS, current cap {self.grip_ratio:.2f}), holding (moved {moved:.2f} deg)")
        if moved > 5.0:
            return False, f"gripper moved {moved:.1f} deg on enable"
        return True, "gripper enabled"

    def send(self, q7):
        """Send a filtered sim-space command to joint1..6 (+ gripper if enabled)."""
        if self.dry or not self.enabled:
            return
        tgt = sim_to_motor_rad(self.cfg, q7)
        for name, m in self.arm:
            m.send_pos_vel(float(tgt[name]), self.vlim)
        if self.grip_enabled:
            g = self.cfg["gripper"]
            lo, hi = sorted((math.radians(g["open_deg"]), math.radians(g["closed_deg"])))
            self.grip[1].send_force_pos(float(np.clip(tgt["gripper"], lo - 0.03, hi + 0.03)),
                                        self.grip_vlim, self.grip_ratio)

    def disable_arm(self, log=print):
        if self.dry:
            log("[live] DRY RUN: would disable the arm now")
            self.enabled = False
            self.grip_enabled = False
            return
        for _, m in self.arm + ([self.grip] if self.grip_enabled else []):
            try:
                m.disable()
            except Exception as exc:
                log(f"[live] disable error: {exc}")
        self.enabled = False
        self.grip_enabled = False


def rest_pose():
    """Folded rest pose in sim space, joint2/3 kept 3 deg above their mechanical stop."""
    q = np.zeros(7)
    q[1] = math.radians(-3.0)
    q[2] = math.radians(-3.0)
    return q
