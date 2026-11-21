"""reBot B601-DM kinematics (Pinocchio) for scripted motions.

Frames (from the URDF): end_link's origin is at the fingertips, its +x axis points
out of the fingers, the fingers close along its y axis. The TCP used here sits
between the finger pads, TCP_OFFSET metres back from the tips.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets" / "rebot_b601_dm.urdf"
ARM = 6                 # joint1..joint6; q[6:8] are the fingers
TCP_OFFSET = 0.015      # m behind the fingertips, along -x of end_link


class RebotKinematics:
    def __init__(self, urdf=URDF):
        self.model = pin.buildModelFromUrdf(str(urdf))
        ee = self.model.getFrameId("end_link")
        f = self.model.frames[ee]
        tcp = pin.Frame("tcp", f.parentJoint, ee, f.placement * pin.SE3(np.eye(3), np.array([-TCP_OFFSET, 0, 0])),
                        pin.FrameType.OP_FRAME)
        self.tcp = self.model.addFrame(tcp)
        self.data = self.model.createData()
        self.lo = self.model.lowerPositionLimit[:ARM].copy()
        self.hi = self.model.upperPositionLimit[:ARM].copy()

    def _full(self, q_arm):
        q = pin.neutral(self.model)
        q[:ARM] = q_arm
        return q

    def fk(self, q_arm) -> pin.SE3:
        q = self._full(q_arm)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.tcp)
        return self.data.oMf[self.tcp].copy()

    def ik(self, target: pin.SE3, q0, iters=200, tol=1e-4, damping=1e-3, step=0.8):
        """Damped least squares on the 6 arm joints. Returns (q_arm, pos_err_m, rot_err_rad)."""
        q_arm = np.clip(np.asarray(q0, float)[:ARM].copy(), self.lo, self.hi)
        for _ in range(iters):
            q = self._full(q_arm)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.tcp)
            err = pin.log6(self.data.oMf[self.tcp].actInv(target)).vector  # in TCP frame
            if np.linalg.norm(err) < tol:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.tcp, pin.LOCAL)[:, :ARM]
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q_arm = np.clip(q_arm + step * dq, self.lo, self.hi)
        cur = self.fk(q_arm)
        return q_arm, float(np.linalg.norm(cur.translation - target.translation)), \
            float(np.linalg.norm(pin.log3(cur.rotation.T @ target.rotation)))

    def ik_multi(self, target, q_ref, seeds=40, rng_seed=0):
        """Several starts; keeps the converged solution closest to q_ref."""
        rng = np.random.default_rng(rng_seed)
        starts = [np.asarray(q_ref, float)[:ARM]] + [rng.uniform(self.lo, self.hi) for _ in range(seeds)]
        best = None
        for s in starts:
            q, pe, re = self.ik(target, s)
            if pe < 2e-3 and re < 2e-2:
                d = np.linalg.norm(q - np.asarray(q_ref)[:ARM])
                if best is None or d < best[1]:
                    best = (q, d)
        return None if best is None else best[0]


def top_down(pos, finger_axis_angle):
    """TCP pose: fingers pointing straight down, closing along (cos a, sin a, 0)."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(finger_axis_angle), np.sin(finger_axis_angle), 0.0])
    z = np.cross(x, y)
    return pin.SE3(np.column_stack([x, y, z]), np.asarray(pos, float))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cube_yaw(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
