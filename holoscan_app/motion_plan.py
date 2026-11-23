"""Scripted motion planning for the real reBot (calibration now, the data collector later).

A plan is a list of Segments, each a list of 60 Hz sim-space commands q7 = (joint1..6 rad, finger m),
plus an optional event the executor handles when the segment ends (wait for a key, photo, grasp check).
Everything is planned and VALIDATED before the first command is sent:
  - IK converges to < 2 mm / 2 deg on every Cartesian sample
  - every joint stays inside the SafetyFilter limits (URDF minus margins)
  - the TCP never goes below the segment's minimum height (transit 5 cm, near the table 2.2 cm)
  - no joint moves faster than the plan speed (no jumps between samples)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from kinematics import ARM, RebotKinematics, top_down

RATE = 60.0
DT = 1.0 / RATE

# heights of the TCP (between the finger pads, 15 mm behind the tips), metres above the base plane
Z_PRE = 0.10        # above an object before descending
Z_GRASP = 0.025     # cube (4 cm) centre ~0.02: pads cover 1-5 cm, fingertips ~1 cm above the table
Z_PLACE = 0.030     # release height: the cube drops ~5 mm
Z_RETREAT = 0.12
Z_MIN_NEAR = 0.022  # hard floor while near the table
Z_MIN_TRANSIT = 0.05
GRIP_OPEN = 0.030   # finger joint (m)
GRIP_HANDOVER = 0.040   # wider while the user slides the cube in
Z_HANDOVER = 0.07   # wait here for the cube: fingertips ~5.5 cm up, 1.5 cm above the cube's top
GRIP_CLOSE = 0.0    # force-limited on the real gripper (FORCE_POS cap), stops on the cube

V_JOINT_DEG = 12.0  # plan speed for joint-space moves (below the 15 deg/s SafetyFilter cap)
V_DOWN = 0.02       # m/s vertical descent
V_UP = 0.03         # m/s vertical ascent


@dataclass
class Segment:
    label: str
    q: list                      # list of np.ndarray (7,)
    z_min: float = Z_MIN_TRANSIT
    event: str = ""              # "", "wait_key:g", "check_grasp", "photo:<i>", "check_open"
    info: dict = field(default_factory=dict)


def smooth(s):
    return s * s * (3 - 2 * s)


class MotionPlanner:
    def __init__(self, limits, margin_deg=10.0, margin_rest_deg=3.0):
        self.k = RebotKinematics()
        self.hard_lo = np.array([lo for lo, _ in limits])
        self.hard_hi = np.array([hi for _, hi in limits])
        m = math.radians(margin_deg)
        self.lo = np.array([lo + m for lo, _ in limits])
        self.hi = np.array([hi - m for _, hi in limits])
        self.hi[1] = limits[1][1] - math.radians(margin_rest_deg)    # rest side of joint2/3
        self.hi[2] = limits[2][1] - math.radians(margin_rest_deg)

    # ------------------------------------------------------------ primitives
    def pose(self, x, y, z):
        """Top-down TCP pose with the fingers closing across the reach direction."""
        return top_down((x, y, z), math.atan2(y, x) + math.pi / 2)

    def ik(self, T, seed):
        q = self.k.ik_multi(T, seed)
        if q is None:
            raise ValueError(f"no IK solution for {np.round(T.translation, 3)}")
        return q

    def joint_move(self, label, q_a, q_b, g, v_deg=V_JOINT_DEG, event=""):
        d = np.degrees(np.max(np.abs(np.asarray(q_b) - np.asarray(q_a))))
        secs = max(1.0, 1.5 * d / v_deg)          # smoothstep peak speed = 1.5 x average
        n = int(secs * RATE)
        out = [np.r_[(1 - smooth(i / n)) * np.asarray(q_a) + smooth(i / n) * np.asarray(q_b), g] for i in range(1, n + 1)]
        return Segment(label, out, Z_MIN_TRANSIT, event)

    def vertical(self, label, x, y, z_a, z_b, q_seed, g, speed, z_min, event=""):
        T_a, T_b = self.pose(x, y, z_a), self.pose(x, y, z_b)
        n = max(int(abs(z_b - z_a) / speed * RATE), 2)
        d = pin.log6(T_a.actInv(T_b))
        out, q = [], np.asarray(q_seed)[:ARM]
        for i in range(1, n + 1):
            q, pe, re = self.k.ik(T_a * pin.exp6(d * smooth(i / n)), q, iters=80)
            if pe > 2e-3 or re > 2e-2:
                raise ValueError(f"{label}: IK error {pe * 1000:.1f} mm at sample {i}")
            out.append(np.r_[q, g])
        return Segment(label, out, z_min, event)

    def gripper(self, label, q, g_a, g_b, secs, event=""):
        n = int(secs * RATE)
        return Segment(label, [np.r_[np.asarray(q)[:ARM], g_a + (g_b - g_a) * smooth(i / n)] for i in range(1, n + 1)],
                       Z_MIN_NEAR, event)

    def hold(self, label, q, g, secs, z_min=Z_MIN_TRANSIT, event=""):
        return Segment(label, [np.r_[np.asarray(q)[:ARM], g]] * max(int(secs * RATE), 1), z_min, event)

    # ------------------------------------------------------------ validation
    def validate(self, segs, q_start):
        prev = np.asarray(q_start)[:ARM]
        prev_z = self.k.fk(prev).translation[2]
        vmax = math.radians(V_JOINT_DEG * 1.6) * DT      # per-sample bound (smoothstep peak + margin)
        report = []
        for s in segs:
            zs = []
            for i, q7 in enumerate(s.q):
                qa = q7[:ARM]
                # same rule as the SafetyFilter: a joint already inside the margin zone (e.g. the folded rest
                # pose on joint2/3's 0 limit) may stay there or move inward, never further out
                out_now = np.maximum(self.lo - qa, qa - self.hi)
                out_prev = np.maximum(self.lo - prev, prev - self.hi)
                # (a joint resting slightly past its URDF limit on a mechanical stop is fine as long as it
                #  only moves back inward)
                bad = (out_now > 1e-6) & (out_now > out_prev + 1e-9)
                if np.any(bad):
                    j = int(np.argmax(np.where(bad, out_now, -1)))
                    raise ValueError(f"{s.label}: joint{j + 1} {math.degrees(qa[j]):.1f} deg outside the safe range")
                step = np.max(np.abs(qa - prev))
                if step > vmax:
                    raise ValueError(f"{s.label}: joint jump {math.degrees(step):.2f} deg in one sample")
                z = self.k.fk(qa).translation[2]
                zs.append(z)
                if z < s.z_min - 1e-4 and z < prev_z - 1e-5:     # below the floor AND going lower
                    raise ValueError(f"{s.label}: TCP {z * 100:.1f} cm below its floor {s.z_min * 100:.1f} cm")
                prev, prev_z = qa, z
            report.append((s.label, len(s.q) / RATE, min(zs) if zs else None))
        return report

    def park(self, q_now, grip, z_up=0.12):
        """Rise straight up (keeping the gripper orientation) to z_up if lower, then fold to the ready pose."""
        q0 = np.asarray(q_now)[:ARM]
        segs = []
        T = self.k.fk(q0)
        if T.translation[2] < z_up - 0.005:
            T_up = pin.SE3(T.rotation, T.translation + np.array([0.0, 0.0, z_up - T.translation[2]]))
            n = max(int((z_up - T.translation[2]) / V_UP * RATE), 2)
            d = pin.log6(T.actInv(T_up))
            out, q = [], q0
            for i in range(1, n + 1):
                q, pe, re = self.k.ik(T * pin.exp6(d * smooth(i / n)), q, iters=80)
                if pe > 2e-3 or re > 2e-2:
                    raise ValueError(f"rise: IK error {pe * 1000:.1f} mm")
                out.append(np.r_[q, grip])
            segs.append(Segment("rise straight up", out, Z_MIN_NEAR))
            q0 = out[-1][:ARM]
        segs.append(self.joint_move("fold to the rest pose", q0, READY_Q, grip))
        return segs

    # ------------------------------------------------------------ the calibration plan
    def calibration(self, q_start, grid, handover, photo_q):  # noqa: C901
        segs = []
        q0 = np.asarray(q_start)[:ARM]
        hx, hy = handover
        # always via the folded ready pose first: from a drooped/odd start a direct joint move can sweep low
        segs.append(self.joint_move("to the ready pose", q0, READY_Q, 0.0))
        q_pre_h = self.ik(self.pose(hx, hy, Z_PRE), READY_Q)
        segs.append(self.joint_move("to the handover point", READY_Q, q_pre_h, 0.0))
        segs.append(self.gripper("open the gripper", q_pre_h, 0.0, GRIP_HANDOVER, 1.3))
        segs.append(self.vertical("lower to the handover height", hx, hy, Z_PRE, Z_HANDOVER, q_pre_h, GRIP_HANDOVER,
                                  V_DOWN, Z_MIN_NEAR))
        q = segs[-1].q[-1][:ARM]
        segs.append(self.hold("wait: slide the cube between the fingers, then g", q, GRIP_HANDOVER, 0.2,
                              Z_MIN_NEAR, event="wait_key:g"))
        segs.append(self.vertical("descend to the cube", hx, hy, Z_HANDOVER, Z_GRASP, q, GRIP_HANDOVER, V_DOWN, Z_MIN_NEAR))
        q = segs[-1].q[-1][:ARM]
        segs.append(self.gripper("close on the cube", q, GRIP_HANDOVER, GRIP_CLOSE, 1.4))
        segs.append(self.hold("check the grasp", q, GRIP_CLOSE, 0.5, Z_MIN_NEAR, event="check_grasp"))
        segs.append(self.vertical("lift", hx, hy, Z_GRASP, Z_PRE, q, GRIP_CLOSE, V_UP, Z_MIN_NEAR))
        q = segs[-1].q[-1][:ARM]
        for i, (x, y) in enumerate(grid, 1):
            q_pre = self.ik(self.pose(x, y, Z_PRE), q)
            segs.append(self.joint_move(f"point {i}: carry to ({x:.2f}, {y:+.2f})", q, q_pre, GRIP_CLOSE))
            segs.append(self.vertical(f"point {i}: lower", x, y, Z_PRE, Z_PLACE, q_pre, GRIP_CLOSE, V_DOWN, Z_MIN_NEAR))
            q = segs[-1].q[-1][:ARM]
            segs.append(self.gripper(f"point {i}: release", q, GRIP_CLOSE, GRIP_OPEN, 0.8))
            segs.append(self.hold(f"point {i}: settle", q, GRIP_OPEN, 0.4, Z_MIN_NEAR))
            segs.append(self.vertical(f"point {i}: rise", x, y, Z_PLACE, Z_RETREAT, q, GRIP_OPEN, V_UP, Z_MIN_NEAR))
            q = segs[-1].q[-1][:ARM]
            segs.append(self.joint_move(f"point {i}: out of the camera view", q, photo_q, GRIP_OPEN))
            photo = self.hold(f"point {i}: photo", photo_q, GRIP_OPEN, 1.0, event=f"photo:{i}")
            photo.info["xy"] = (x, y)
            segs.append(photo)
            q_pre = self.ik(self.pose(x, y, Z_PRE), photo_q)
            segs.append(self.joint_move(f"point {i}: back above the cube", photo_q, q_pre, GRIP_OPEN))
            segs.append(self.vertical(f"point {i}: descend", x, y, Z_PRE, Z_GRASP, q_pre, GRIP_OPEN, V_DOWN, Z_MIN_NEAR))
            q = segs[-1].q[-1][:ARM]
            segs.append(self.gripper(f"point {i}: grasp", q, GRIP_OPEN, GRIP_CLOSE, 1.2))
            segs.append(self.hold(f"point {i}: check the grasp", q, GRIP_CLOSE, 0.5, Z_MIN_NEAR, event="check_grasp"))
            segs.append(self.vertical(f"point {i}: lift", x, y, Z_GRASP, Z_PRE, q, GRIP_CLOSE, V_UP, Z_MIN_NEAR))
            q = segs[-1].q[-1][:ARM]
        # put the cube back at the handover point and go home
        q_pre_h = self.ik(self.pose(hx, hy, Z_PRE), q)
        segs.append(self.joint_move("back to the handover point", q, q_pre_h, GRIP_CLOSE))
        segs.append(self.vertical("lower", hx, hy, Z_PRE, Z_PLACE, q_pre_h, GRIP_CLOSE, V_DOWN, Z_MIN_NEAR))
        q = segs[-1].q[-1][:ARM]
        segs.append(self.gripper("release", q, GRIP_CLOSE, GRIP_OPEN, 0.8))
        segs.append(self.vertical("rise", hx, hy, Z_PLACE, Z_RETREAT, q, GRIP_OPEN, V_UP, Z_MIN_NEAR))
        q = segs[-1].q[-1][:ARM]
        segs.append(self.joint_move("out of the camera view", q, photo_q, GRIP_OPEN))
        segs.append(self.hold("done", photo_q, GRIP_OPEN, 0.5, event="done"))
        return segs


CAL_GRID = [(0.18, -0.10), (0.18, 0.00), (0.18, 0.10),
            (0.25, 0.10), (0.25, 0.00), (0.25, -0.10),
            (0.32, -0.10), (0.32, 0.00), (0.32, 0.10)]
HANDOVER = (0.25, 0.0)
PHOTO_Q = np.radians([60.0, -3.0, -3.0, 0.0, 0.0, 0.0])   # folded, base turned 60 deg to +y (was 80: less cable flex)
READY_Q = np.radians([0.0, -3.0, -3.0, 0.0, 0.0, 0.0])    # folded rest, wrist level
