"""SafetyFilter: the single checkpoint every real-arm command passes.

Pure Python (no I/O), so it can be unit-tested offline. Works in sim joint space
(rad for joint1-6, metres for the gripper finger).

Modes
    WAIT_ALIGN  holding the measured pose until the target is within align_tol of it on every joint
    BLEND       following at a reduced speed (blend_vmax) for blend_s seconds
    FOLLOW      following at vmax with an acceleration limit
    HOLD        latched: command frozen at the measured pose; only rearm() leaves it

Every fault latches HOLD (never "disable": the Damiao arm would drop).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

ARM = 6


@dataclass
class SafetyParams:
    limits: list                       # [(lo, hi)] rad for joint1..6 (URDF)
    margin_deg: float = 10.0
    # joint2/3 upper limit (0) is the folded rest stop: only 3 deg margin there, so a folded leader
    # brings the arm down to rest instead of hovering 10 deg above it (elbow carrying the forearm)
    margin_hi_deg: dict = field(default_factory=lambda: {1: 3.0, 2: 3.0})
    vmax_deg_s: float = 30.0
    amax_deg_s2: float = 120.0
    blend_vmax_deg_s: float = 10.0
    blend_s: float = 3.0
    align_tol_deg: float = 10.0
    grip_max_m: float = 0.05
    grip_vmax_m_s: float = 0.03
    tcp_min_z: float = 0.03            # m above the base plane (table)
    base_keepout_r: float = 0.08       # m: TCP may not enter this radius around the base axis...
    base_keepout_z: float = 0.20       # ...below this height
    leader_timeout_s: float = 0.10
    feedback_timeout_s: float = 0.05
    track_err_deg: float = 15.0
    track_err_s: float = 0.20
    temp_max_c: float = 60.0
    # Nm, j1..j6 + gripper. joint2 (shoulder, 4340P, peak ~27 Nm) carries the whole arm: 9-15 Nm at 0.4 m reach,
    # so 20 Nm; heat is covered by temp_max_c.
    # gripper: its FORCE_POS current cap already limits it to ~2 Nm, which it reaches just driving its screw
    # open/closed; 3 Nm here is only the backup check
    # joint4/5 (4310, peak ~10 Nm): 4 Nm was reached just carrying the cube with the wrist angled -> 6 Nm
    torque_max: list = field(default_factory=lambda: [12.0, 20.0, 14.0, 6.0, 6.0, 3.0, 3.0])
    check_tracking: bool = True        # False in shadow mode (motors off, the arm cannot follow)


class SafetyFilter:
    def __init__(self, p: SafetyParams, fk=None):
        self.p = p
        self.fk = fk                       # q_arm -> TCP xyz (for workspace checks); None = skip
        m = math.radians(p.margin_deg)
        self.lo = np.array([lo + m for lo, _ in p.limits])
        self.hi = np.array([hi - math.radians(p.margin_hi_deg.get(i, p.margin_deg)) for i, (_, hi) in enumerate(p.limits)])
        self.mode = "WAIT_ALIGN"
        self.reason = "waiting for the leader to match the arm"
        self.cmd = None                    # last command (7: 6 arm + gripper finger)
        self.vel = np.zeros(7)
        self.t_mode = 0.0
        self.t_track_bad = None
        self.notes = []

    # -------------------------------------------------------------- helpers
    def _hold(self, t, measured, reason):
        if self.mode != "HOLD":
            self.mode, self.reason, self.t_mode = "HOLD", reason, t
            if measured is not None:
                self.cmd = np.array(measured, float)
            self.vel[:] = 0.0

    def rearm(self, t, measured):
        """User re-arms after a HOLD: back to alignment, starting from the measured pose."""
        self.mode, self.reason, self.t_mode = "WAIT_ALIGN", "re-armed: waiting for alignment", t
        if measured is not None:
            self.cmd = np.array(measured, float)
        self.vel[:] = 0.0
        self.t_track_bad = None

    def soft_stop(self, t, measured):
        self._hold(t, measured, "soft stop by user")

    def _workspace_ok(self, q_arm, q_now=None):
        """Refuse a move that takes the gripper below the floor or into the base zone - but a gripper that is
        already there (e.g. a drooped start pose) may always move up / out of it."""
        if self.fk is None:
            return True, ""
        x, y, z = self.fk(q_arm)
        x0, y0, z0 = self.fk(q_now) if q_now is not None else (x, y, 1e9)
        if z < self.p.tcp_min_z and z < z0 - 1e-5:
            return False, f"gripper would go below {self.p.tcp_min_z * 100:.0f} cm (z={z * 100:.1f} cm)"
        r, r0 = math.hypot(x, y), math.hypot(x0, y0)
        if r < self.p.base_keepout_r and z < self.p.base_keepout_z and (r < r0 - 1e-5 or z < z0 - 1e-5):
            return False, "gripper would enter the zone around the base"
        return True, ""

    # -------------------------------------------------------------- main step
    def step(self, t, dt, target, target_age, measured, feedback_age, health=None):
        """target/measured: arrays of 7 (6 arm rad + gripper m) or None.
        health: optional dict with lists 'temp' (C), 'torque' (Nm), 'fault' (str or None).
        Returns the command to send (7) or None if there is nothing safe to send yet."""
        p = self.p
        self.notes = []
        if self.cmd is None:
            if measured is None:
                self.reason = "no feedback from the arm yet"
                return None
            self.cmd = np.array(measured, float)

        # ---- faults that latch HOLD
        if measured is None or feedback_age > p.feedback_timeout_s:
            self._hold(t, None, f"no fresh motor feedback ({feedback_age * 1000:.0f} ms)")
            return self.cmd.copy()
        if health:
            if health.get("fault"):
                self._hold(t, measured, f"motor fault: {health['fault']}")
            temps = health.get("temp") or []
            if temps and max(temps) > p.temp_max_c:
                self._hold(t, measured, f"motor temperature {max(temps):.0f} C")
            torq = health.get("torque") or []
            for i, (tq, lim) in enumerate(zip(torq, p.torque_max)):
                if abs(tq) > lim:
                    self._hold(t, measured, f"torque on {'gripper' if i == 6 else f'joint{i + 1}'} {tq:.1f} Nm")
        if self.mode in ("BLEND", "FOLLOW") and (target is None or target_age > p.leader_timeout_s):
            self._hold(t, measured, f"leader data stale ({target_age * 1000:.0f} ms)")
        if p.check_tracking and self.mode in ("BLEND", "FOLLOW"):
            err = np.degrees(np.abs(self.cmd[:ARM] - np.asarray(measured)[:ARM])).max()
            if err > p.track_err_deg:
                self.t_track_bad = self.t_track_bad or t
                if t - self.t_track_bad > p.track_err_s:
                    self._hold(t, measured, f"arm {err:.0f} deg behind its command (collision?)")
            else:
                self.t_track_bad = None
        if self.mode == "HOLD":
            return self.cmd.copy()
        if target is None:  # no leader data yet (WAIT_ALIGN): hold where the arm is
            self.reason = "waiting for leader data"
            return self.cmd.copy()

        # ---- clamp the target to limits (minus margin) and the gripper range
        tgt = np.array(target, float)
        # A joint already inside the margin zone (e.g. the folded rest pose sits on joint2/3's 0 limit)
        # may stay there or move inward, never further out: no start-up jump out of the zone.
        hard_lo = np.array([lo for lo, _ in self.p.limits])
        hard_hi = np.array([hi for _, hi in self.p.limits])
        lo_eff = np.maximum(np.minimum(self.lo, self.cmd[:ARM]), hard_lo)
        hi_eff = np.minimum(np.maximum(self.hi, self.cmd[:ARM]), hard_hi)
        clamped = np.clip(tgt[:ARM], lo_eff, hi_eff)
        if np.any(np.abs(clamped - tgt[:ARM]) > 1e-6):
            self.notes.append("target clamped to joint limits")
        tgt[:ARM] = clamped
        tgt[ARM] = float(np.clip(tgt[ARM], 0.0, p.grip_max_m))

        # ---- alignment gate
        if self.mode == "WAIT_ALIGN":
            diff = np.degrees(np.abs(tgt[:ARM] - np.asarray(measured)[:ARM]))
            if diff.max() <= p.align_tol_deg:
                self.mode, self.reason, self.t_mode = "BLEND", "aligned: blending in slowly", t
            else:
                j = int(diff.argmax())
                self.reason = f"move the leader to the arm: joint{j + 1} is {diff[j]:.0f} deg away (need < {p.align_tol_deg:.0f})"
                return self.cmd.copy()
        if self.mode == "BLEND" and t - self.t_mode >= p.blend_s:
            self.mode, self.reason, self.t_mode = "FOLLOW", "following", t

        # ---- speed + acceleration limit
        vmax = math.radians(p.blend_vmax_deg_s if self.mode == "BLEND" else p.vmax_deg_s)
        amax = math.radians(p.amax_deg_s2)
        v_des = np.zeros(7)
        err = tgt[:ARM] - self.cmd[:ARM]
        # brake in time (discrete): with n ticks left, decelerating by amax*dt per tick covers
        # amax*dt^2*n(n+1)/2, so the fastest speed that still stops on the target is n*amax*dt
        n_left = 0.5 * (np.sqrt(1.0 + 8.0 * np.abs(err) / (amax * dt * dt)) - 1.0)
        v_stop = n_left * amax * dt
        v_des[:ARM] = np.sign(err) * np.minimum.reduce([np.full(ARM, vmax), v_stop, np.abs(err) / dt])
        v_des[ARM] = np.clip((tgt[ARM] - self.cmd[ARM]) / dt, -p.grip_vmax_m_s, p.grip_vmax_m_s)
        dv = np.clip(v_des[:ARM] - self.vel[:ARM], -amax * dt, amax * dt)
        self.vel[:ARM] += dv
        self.vel[ARM] = v_des[ARM]
        # never overshoot the target
        nxt = self.cmd + self.vel * dt
        over = np.sign(tgt - self.cmd) != np.sign(tgt - nxt)
        nxt[over] = tgt[over]
        self.vel[over] = 0.0

        # ---- workspace check on the next command
        ok, why = self._workspace_ok(nxt[:ARM], self.cmd[:ARM])
        if not ok:
            self.notes.append(why)
            self.vel[:] = 0.0
            return self.cmd.copy()  # stay put (not a latch): the user steers back
        self.cmd = nxt
        return self.cmd.copy()
