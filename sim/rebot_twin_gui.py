"""Isaac Sim 6 twin of the reBot B601-DM, driven by the Holoscan teleop app over ZMQ.

Run (on dexstro's monitor):
    source ~/rebot-holoscan/env.sh
    DISPLAY=:1 sim_py sim/rebot_twin_gui.py            # GUI
    sim_py sim/rebot_twin_gui.py --headless            # no window
"""
import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true")
ap.add_argument("--cmd", default="tcp://127.0.0.1:5605")
ap.add_argument("--state", default="tcp://127.0.0.1:5606")
ap.add_argument("--physics-hz", type=float, default=120.0)
ap.add_argument("--vision", action="store_true", help="add cube, bin and a scene camera streamed over ZMQ")
ap.add_argument("--frames", default="tcp://127.0.0.1:5607")
ap.add_argument("--cam-size", default="640x480")
ap.add_argument("--cam-hz", type=float, default=30.0)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1920, "height": 1080,
                     "window_width": 1920, "window_height": 1080})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import zmq  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.objects import GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.join(ROOT, "assets", "usd", "rebot_b601_dm", "rebot_b601_dm.usda")

# ---------------------------------------------------------------- scene
stage = stage_utils.create_new_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr(int(args.physics_hz))
GroundPlane("/World/ground")
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(800.0)
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(2500.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))

stage_utils.add_reference_to_stage(USD, "/World/rebot")
stage.GetPrimAtPath("/World/rebot").GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")


def make_material(path, rgb, rough=0.5):
    from pxr import Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def make_box(path, size, pos, mat, dynamic=False, mass=0.05):
    from pxr import UsdShade
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(cube.GetPrim())
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(mass)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    return cube


CUBE_START = (0.26, 0.06, 0.021)
BIN_XY = (0.22, -0.22)


def grip_material():
    """High-friction physics material for fingers and cube (rubber pad on plastic)."""
    from pxr import UsdShade
    mat = UsdShade.Material.Define(stage, "/World/Physics/grip")
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr(1.5)
    m.CreateDynamicFrictionAttr(1.2)
    m.CreateRestitutionAttr(0.0)
    return mat


def bind_physics_material(prim, mat):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")


if args.vision:
    red = make_material("/World/Looks/red", (0.85, 0.05, 0.05), 0.35)
    green = make_material("/World/Looks/green", (0.05, 0.6, 0.15), 0.6)
    grey = make_material("/World/Looks/table", (0.55, 0.55, 0.58), 0.8)
    # low table slab under the workspace (top at z=0), cube and an open bin on it
    make_box("/World/table", (0.9, 0.9, 0.02), (0.2, 0.0, -0.011), grey)
    cube_geom = make_box("/World/cube", (0.04, 0.04, 0.04), CUBE_START, red, dynamic=True, mass=0.03)
    grip = grip_material()
    bind_physics_material(cube_geom.GetPrim(), grip)
    n_fingers = 0
    for prim in stage.Traverse():  # finger links of the arm
        if prim.GetName() in ("finger_left_link", "finger_right_link"):
            bind_physics_material(prim, grip)
            n_fingers += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            pa = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            pa.CreateSolverPositionIterationCountAttr(32)
            pa.CreateSolverVelocityIterationCountAttr(4)
    print(f"[twin] grip friction material on cube + {n_fingers} finger links", flush=True)
    bx, by = BIN_XY
    wall, h, side = 0.006, 0.06, 0.12
    make_box("/World/bin/floor", (side, side, wall), (bx, by, wall / 2), green)
    for i, (dx, dy, sx, sy) in enumerate([(side / 2, 0, wall, side), (-side / 2, 0, wall, side),
                                           (0, side / 2, side, wall), (0, -side / 2, side, wall)]):
        make_box(f"/World/bin/wall{i}", (sx, sy, h), (bx + dx, by + dy, h / 2), green)
    # fixed scene camera looking at the workspace (USD cameras look down -Z)
    cam = UsdGeom.Camera.Define(stage, "/World/scene_cam")
    cam.CreateFocalLengthAttr(16.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.80, 0.55, 0.60), Gf.Vec3d(0.18, -0.06, 0.10), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(view.GetInverse())
app.update()

frame_pub = annot = None
if args.vision:
    import struct

    import omni.replicator.core as rep
    cw, ch = (int(v) for v in args.cam_size.split("x"))
    rp = rep.create.render_product("/World/scene_cam", (cw, ch))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    frame_pub = zmq.Context.instance().socket(zmq.PUB)
    frame_pub.setsockopt(zmq.SNDHWM, 2)
    frame_pub.bind(args.frames)
    print(f"[twin] scene camera {cw}x{ch} streaming on {args.frames}", flush=True)

if not args.headless:
    try:
        from isaacsim.core.rendering_manager import ViewportManager
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[1.25, -1.15, 0.85], target=[0.0, 0.0, 0.28])
        print("[twin] camera view set", flush=True)
    except Exception as exc:  # camera placement is cosmetic
        print(f"[twin] camera view not set: {exc}", flush=True)

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(5):
    app.update()

arm = Articulation("/World/rebot")
names = list(arm.dof_names)
print(f"[twin] articulation ready: {names}", flush=True)


def to_np(x):
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float32).reshape(-1)


target = to_np(arm.get_dof_positions())
arm.set_dof_position_targets(target.reshape(1, -1))

# ---------------------------------------------------------------- link
ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.CONFLATE, 1)
sub.setsockopt_string(zmq.SUBSCRIBE, "")
sub.connect(args.cmd)
pub = ctx.socket(zmq.PUB)
pub.setsockopt(zmq.SNDHWM, 2)
pub.bind(args.state)
print(f"[twin] listening for targets on {args.cmd}, state on {args.state}", flush=True)

cube = None
if args.vision:
    from isaacsim.core.experimental.prims import RigidPrim
    cube = RigidPrim("/World/cube")


def reset_cube(xy=None, yaw=0.0):
    """Put the cube back on the table: default start pose, or at (x, y) with a yaw (rad)."""
    if cube is None:
        return
    x, y = xy if xy else CUBE_START[:2]
    quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]  # w, x, y, z
    cube.set_world_poses(positions=np.array([[x, y, CUBE_START[2]]]), orientations=np.array([quat]))
    cube.set_velocities(linear_velocities=np.zeros((1, 3)), angular_velocities=np.zeros((1, 3)))
    print(f"[twin] cube reset to x={x:.3f} y={y:.3f} yaw={np.degrees(yaw):.0f} deg", flush=True)


last_cmd = None
n_cmd = 0
t_report = time.time()
frames = 0
n_frames_sent = 0
t_next_frame = 0.0
while app.is_running():
    try:
        msg = json.loads(sub.recv_string(zmq.NOBLOCK))
        if msg.get("reset"):
            reset_cube(msg.get("cube_xy"), float(msg.get("cube_yaw", 0.0)))
        if "q" in msg:
            q = dict(zip(msg["names"], msg["q"]))
            target = np.array([q.get(n, target[i]) for i, n in enumerate(names)], dtype=np.float32)
            arm.set_dof_position_targets(target.reshape(1, -1))
            last_cmd = time.time()
            n_cmd += 1
    except zmq.Again:
        pass  # no new command: keep holding the last target

    app.update()
    frames += 1
    if not timeline.is_playing():
        continue
    pos = to_np(arm.get_dof_positions())
    state = {"t": time.time(), "names": names, "q": pos.tolist()}
    if cube is not None:  # ground-truth object poses for the scripted expert
        cp_, cq_ = cube.get_world_poses()
        state["cube"] = to_np(cp_).tolist() + to_np(cq_).tolist()  # x y z  qw qx qy qz
        state["bin"] = list(BIN_XY)
    pub.send_string(json.dumps(state))

    now = time.time()
    if annot is not None and now >= t_next_frame:
        t_next_frame = now + 1.0 / args.cam_hz
        img = annot.get_data()
        if img is not None and getattr(img, "size", 0):
            img = np.asarray(img)
            h, w = img.shape[:2]
            frame_pub.send(struct.pack("<dHH", now, w, h) + np.ascontiguousarray(img[..., :3]).tobytes(), copy=False)
            n_frames_sent += 1
    if cube is not None and frames % 30 == 0:
        p = cube.get_world_poses()[0]
        if float(to_np(p)[2]) < -0.2:
            reset_cube()

    if now - t_report > 5.0:
        age = "none yet" if last_cmd is None else f"{now - last_cmd:.2f}s ago"
        print(f"[twin] {frames / (now - t_report):5.1f} fps, {n_cmd} cmds, last cmd {age}, "
              f"{n_frames_sent} camera frames", flush=True)
        frames, n_cmd, n_frames_sent, t_report = 0, 0, 0, now

sub.close(0)
pub.close(0)
timeline.stop()
app.close()
