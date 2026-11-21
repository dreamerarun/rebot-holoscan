"""Headless Isaac Sim test: load the reBot B601-DM USD, drive joints to targets, read back."""
import os
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from pxr import Gf, UsdPhysics  # noqa: E402

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.objects import GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.join(ROOT, "assets", "usd", "rebot_b601_dm", "rebot_b601_dm.usda")

stage = stage_utils.create_new_stage()
scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
GroundPlane("/World/ground")
stage_utils.add_reference_to_stage(USD, "/World/rebot")
# The importer puts physics in a "Physics" variant set with no default selection.
stage.GetPrimAtPath("/World/rebot").GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
app.update()

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(5):
    app.update()

arm = Articulation("/World/rebot")
names = arm.dof_names
print("DOFS", arm.num_dofs, names, flush=True)
lo, hi = (np.asarray(x.numpy() if hasattr(x, "numpy") else x).reshape(-1) for x in arm.get_dof_limits())
print("LIMITS", np.round(lo, 3).tolist(), np.round(hi, 3).tolist(), flush=True)

# A safe pose well inside the limits: small bend on every joint, gripper half open.
target = np.zeros(arm.num_dofs, dtype=np.float32)
pose = {"joint1": 0.4, "joint2": -0.6, "joint3": -0.8, "joint4": 0.3, "joint5": 0.3,
        "joint6": 0.5, "finger_left": 0.025, "finger_right": -0.025}
for i, n in enumerate(names):
    target[i] = pose.get(n, 0.0)
arm.set_dof_position_targets(target.reshape(1, -1))

t0 = time.time()
steps = 0
while steps < 240:  # ~4 s of sim time at 60 Hz
    app.update()
    steps += 1
wall = time.time() - t0
q = arm.get_dof_positions()
q = np.asarray(q.numpy() if hasattr(q, "numpy") else q).reshape(-1)
err = np.abs(q - target)
print("TARGET", np.round(target, 3).tolist(), flush=True)
print("REACHED", np.round(q, 3).tolist(), flush=True)
print(f"MAX_ERR {err.max():.4f} rad/m  ({steps} updates in {wall:.1f}s = {steps / wall:.0f} Hz)", flush=True)
print("SIM_SMOKE", "OK" if err.max() < 0.05 else "CHECK", flush=True)
timeline.stop()
app.close()
