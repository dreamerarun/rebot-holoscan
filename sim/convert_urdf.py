"""Convert the reBot B601-DM URDF to USD for Isaac Sim 6 (headless).

Run:  OMNI_KIT_ACCEPT_EULA=YES .venv-isaacsim/bin/python sim/convert_urdf.py
"""
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

for ext in ("isaacsim.asset.importer.urdf", "isaacsim.robot.schema"):
    enable_extension(ext)
app.update()

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
urdf = os.path.join(root, "assets", "rebot_b601_dm.urdf")
out_dir = os.path.join(root, "assets", "usd")

config = URDFImporterConfig(
    urdf_path=urdf,
    usd_path=out_dir,
    merge_fixed_joints=False,      # keep end_link frame for IK / camera mounts
    fix_base=True,                 # table-mounted arm
    collision_from_visuals=False,  # URDF has its own collision meshes
    joint_drive_type="force",
    joint_target_type="position",
    # Position-servo feel similar to the Damiao POS_VEL mode.
    override_joint_stiffness={"joint[1-3]": 600.0, "joint[4-6]": 150.0, "finger_.*": 3000.0},
    override_joint_damping={"joint[1-3]": 40.0, "joint[4-6]": 10.0, "finger_.*": 100.0},
    robot_type="Manipulator",
)
out = URDFImporter(config).import_urdf()
print("CONVERT_OK", out, flush=True)
app.close()
sys.exit(0)
