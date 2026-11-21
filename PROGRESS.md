# reBot B601-DM + Isaac Sim twin on NVIDIA Holoscan (dexstro / GB10)

Status as of 2026-11-22. Next session: connect the real arm.

## What works

| Step | Status | How to run |
|---|---|---|
| Holoscan 4.6 (CUDA 13) + Isaac Sim 6.2 installed natively on the GB10 | done | `source ~/rebot-holoscan/env.sh` (`hs_py`, `sim_py`) |
| B601-DM URDF -> USD, physics checked | done | `assets/usd/rebot_b601_dm/` |
| 102 Leader -> Holoscan -> Isaac Sim twin (60 Hz) | done, directions verified | `./run_twin_gui.sh` then `./run_teleop.sh` |
| Vision demo: Holoscan two lanes vs plain loop | done | `./run_twin_gui.sh --vision` then `./run_vision_demo.sh holoscan` or `loop` (add `--mock` without leader) |
| Scripted pick-and-place (Pinocchio IK) | done, 8/10 random spots | `./run_twin_gui.sh --vision` then `./run_scripted_pick.sh --cycles 5` |
| Real B601-DM follower | **next** | - |

Only one of teleop / vision demo / scripted pick at a time (they share the sim command port 5605).
Stop everything: close the Isaac Sim window, Ctrl-C the Holoscan app.

## Hardware

- 102 Leader: CH340 USB-serial, `/dev/ttyUSB0`, FashionStar servos ids 0-6 at 1 Mbaud (read with `motorbridge-smart-servo`).
- B601-DM follower (not connected yet): USB-CAN, expected `/dev/ttyACM0`, Damiao motors 0x01-0x07 (feedback 0x11-0x17),
  config `~/ReBot_Arm_DigitalTwin_DM/reBotArm_ros2_DM/third_party/reBotArm_control_py/config/rebotarm_dm.yaml`.

## Measured

- Leader teleop: 60 Hz, sim tracks within 1-3 deg.
- Vision demo (Faster R-CNN ~120 ms/frame on the GB10): Holoscan control lane 60 Hz (worst gap 21-26 ms) vs plain loop 8 Hz (gap 125-140 ms).
  Isaac Sim GUI drops to ~15 fps while the detector runs (shared GPU).
- Scripted pick: 8/10. Misses at far reach (~30 cm, y >= 0.12): the sim arm stops ~6 mm short, the cube sits
  off-centre in the 1.2 cm finger pads and slips on lift.

## Fixes made to the sim asset

- Finger colliders: convexHull filled the gap between the L-shaped finger pads (cube could never get between them).
  `payloads/instances.usda` lines 743 and 852 -> `convexDecomposition`. Backup: `instances.usda.bak-convexhull`.
  **Re-running `sim/convert_urdf.py` undoes this.**
- Twin binds a high-friction material (1.5 / 1.2) to the cube and both finger links; articulation solver 32 iterations.

## Layout

```
config/teleop.yaml            leader -> sim joint map, limits, rate limit, watchdog, ZMQ ports
holoscan_app/rebot_teleop.py  LeaderOp -> MapOp -> SimLinkOp -> StatusOp
holoscan_app/common.py        shared leader / mapper / sim link / frame / detector classes
holoscan_app/vision_demo_holoscan.py, vision_demo_loop.py
holoscan_app/kinematics.py    Pinocchio IK (TCP 15 mm behind the fingertips)
holoscan_app/scripted_pick.py ScriptedPickOp -> SimOp
sim/rebot_twin_gui.py         Isaac Sim twin (--vision: cube, bin, scene camera on :5607)
sim/convert_urdf.py           URDF -> USD (see warning above)
tests/                        smoke tests, grab_frames.py
logs/                         run logs
```

Gotchas: Isaac Sim on arm64 needs `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1` (in the run scripts);
the USD "Physics" variant must be set to `physx`; Holoscan 5.0 is a different API (we use 4.6);
windows bigger than 1024x768 get squashed by GNOME on the 1080p monitor.

## Next steps (real arm)

1. Plug the B601-DM USB-CAN into dexstro, power the arm, workspace clear.
2. Read-only check: all 7 Damiao motors answer, read positions (no enable).
3. Compare real joint angles with the sim at the same pose (directions, zero offsets).
4. Add `RealArmOp` next to `SimOp`: same limits, lower speed cap, slow start-up move to the leader pose.
5. First real run is started by the user, low speed, hand on the power switch. Scripted pick on the real arm only after that.
