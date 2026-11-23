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

## Real-arm safety rules (user requirement 2026-11-22: "ensure safer robot movements, I do not want to damage the robot")

Nothing moves until every stage below has passed. The user enables motors and starts every motion run; Claude only reads.

**Stage 0 - read only (motors never enabled)**
- All 7 Damiao motors answer; read position, velocity, torque, temperature, bus voltage; no errors.
- User moves the arm slowly by hand (motors off). Compare real joint angles with the sim/FK at the same pose:
  directions, zero offsets, limits. Fix the mapping before anything is enabled.

**Stage 1 - shadow mode (motors still off)**
- Full pipeline runs: leader -> MapOp -> SafetyOp -> RealArmOp, but RealArmOp only logs what it WOULD send.
- Check the logged commands against the real arm position for a few minutes of leader motion.

**SafetyOp: the single checkpoint every real-arm command passes**
- Joint limits = URDF limits minus a 10 deg margin.
- Workspace: FK of the TCP must stay above the table (z >= 3 cm) and outside a cylinder around the base.
- Speed: max 30 deg/s per joint at first (leader teleop in sim allows 720 deg/s), plus an acceleration ramp.
  Raised in steps (30 -> 60 -> 90 deg/s) only after a clean session at each step.
- Motor-side cap as well: POS_VEL mode with a low vlim, so the motor itself refuses to go faster
  even if our software misbehaves.
- Gripper: low MIT gain and a torque limit, so it cannot crush or overload the 4310.

**Start-up (the most dangerous moment)**
- On enable, the first target is the CURRENT measured position of the arm (no jump).
- Alignment gate: the follower only starts following once the leader is within 10 deg of the follower on every joint;
  until then it holds. Then it moves to the leader over >= 3 s.
- First enable: user supports the arm by hand, power switch within reach.

**While running: faults HOLD, they do not drop**
- Disabling Damiao motors makes the arm fall under gravity, so a fault means "hold the current position", not "disable".
- Hold triggers: leader data older than 100 ms; motor feedback older than 50 ms or a bus error;
  tracking error > 15 deg for > 0.2 s (likely collision); torque above a per-joint threshold;
  motor temperature > 60 C; any limit/workspace violation.
- After a hold the user must re-arm explicitly (key press); it never resumes by itself.
- Arm/disarm and a soft stop key in the terminal; the physical power switch stays the real emergency stop.

**Shutdown**
- "Park": slow move (<= 20 deg/s) to the rest pose, check it is resting, only then disable.

**Rules**
- Real arm runs with the control lane only; no DNN/vision in the same process until teleop is proven.
- Every command and state is logged (logs/real_*.jsonl) for review.
- Scripted pick on the real arm only after leader teleop is clean, at reduced speed, with the cube in a known safe spot.

## Real arm status (2026-11-22, end of day)

Done and verified on the real B601-DM:
- Stage 0 read-only: 7/7 Damiao motors, zero offsets j1 94.32 / j6 -16.43 (software only, motors untouched),
  directions verified by hand, gripper range closed 0 / open -248 deg. `./run_real_mirror.sh` = sim mirrors real (motors off).
- Stage 1 shadow: `./run_real_shadow.sh` (motors off, sim shows the filtered command).
- Stage 2 live: `./run_real_teleop.sh` - leader drives the real arm + gripper through Holoscan.
  Starts OFF; keys e / a / s / p / q (terminal, or `--keys-file ~/rebot-holoscan/keys` and `echo X >> keys`).
  First sessions at 15 deg/s: every joint enabled with 0.00 deg movement, no HOLDs, temps 33-39 C.
- Gripper: FORCE_POS, current cap 0.10 (Seeed default), enabled after the arm; park keeps its opening.
- SafetyFilter: 21/21 offline tests (`hs_py tests/test_safety.py`).

Learned on the real arm:
- Elbow (joint3) carries up to ~10 Nm and sags up to ~10 deg with the forearm held out (no gravity compensation;
  Damiao POS_VEL has no torque feed-forward).
- The rest side of joint2/3 needs only a 3 deg margin; with 10 deg the arm hovered above rest and park never finished.
- Ctrl-C = emergency exit: motors stay ON holding (CAN timeout register is 0). Restart the program -> it takes over
  in HOLD -> press p.
- NEVER run Seeed's example/4_pos_vel_control.py on this arm: it commands motor zero right after enable
  (base would swing ~94 deg).

Next options:
1. Speed steps: 15 -> 20 -> 30 deg/s after clean sessions.
2. Elbow sag: gravity compensation (MIT mode + SDK gravity model, or stiffer POS_VEL gains) - needs care, test in shadow first.
3. Data collection on the real arm through Holoscan: leader teleop + camera (Dell webcam / USB camera) -> LeRobot dataset.
4. Policy on the GB10 (ACT / SmolVLA) through the same SafetyFilter: camera lane + inference lane + control lane in Holoscan.
5. Scripted pick on the real arm (known cube spot), slow, after 1-2.

## Data collection status (2026-11-22, night) - resume here

Ready, not started yet (0 episodes kept; the first one had a wrong time base and was deleted).
- Camera: Logitech C270, OpenCV index 1, raised and looking down at the table. Do not move it.
- Task: yellow cube -> orange box (box in one fixed spot). Goal: 30 episodes, train, then add 10-20 where it fails.
- Start: `~/rebot-holoscan/open_recorder_terminal.sh` (opens a terminal on the desktop + the Holoviz camera window).
  Keys in that terminal (or `echo X >> ~/rebot-holoscan/keys`): e enable, n start/stop+save, x discard,
  s hold, a re-arm, p park, o off (only when resting folded), q park + quit.
- Fold the leader to its rest pose first: e is refused until every joint is within 10 deg of the arm.
- Recording runs at a fixed 15 fps; each save prints the real fps and warns if it is off by > 5 %.
- Afterwards: `./convert_real_dataset.sh cube_to_box` -> data/lerobot/rebot_real_cube_to_box (LeRobot v3).

Fixed tonight (all committed):
- stale-leader HOLD right after enable: accept only leader messages newer than the last one
  (Holoscan does not tick the leader while the arm operator is busy enabling).
- joints 2/3 fell > 15 deg behind on big lifts / park -> motor-side vlim now 2x the software cap.
- start-up with motors ON while the arm rests folded -> switches them off cleanly; `o` key does the same.
- C270 delivered ~20 fps -> recorder samples on a fixed 15 Hz clock and checks the real rate.

## 2026-11-23 evening - stopped here (arm folded upright, base ~59 deg, ALL motors OFF)
Done today:
- Approach from 15 cm, joint swings floored at 12 cm (fingers brushed the cube).
- Settle 1 s + 1 deg arrival before every grasp/release (arm trails its descent ~5 mm out / 3 mm up).
- Near-table heights +1 cm: Z_GRASP 0.035, Z_PLACE 0.040, Z_MIN_NEAR 0.032 (real fingertips touched the table).
- Calibration: 4x3 grid, handover waits at grasp height, photo search 9 cm, glare fallback in detect_cube (ROI only).
- New 12-point calibration SAVED for the current (re-fixed) camera position: fit max 10.9 mm, leave-one-out mean 9.8 / max 16.3 mm.
  camera_table.prev.yaml = old 7-point map, INVALID now (camera fell and was moved).
- Collector REACH_BIAS = 0 (re-tune only if grasps still land off with the new map).
- Failed motor send = lost bridge (HOLD + reconnect), tests/test_send_failure.py.
BUG FOUND (fixed): robot_calibrate._finish imported calibrate_camera -> real_arm, which installs the READ-ONLY GUARD
  process-wide: after every calibration save all motor commands were blocked (this, not a USB glitch, made the earlier park
  "lag 17 deg"). solve() moved to homography.py; robot_calibrate imports that.
TODO next session (before any live run):
1. DONE: real_live.read() fixed name list (tests/test_reopen_fail.py)
2. DONE: tests/test_live_imports.py passes
3. Joint-2 power connector came loose twice (arm dropped once): keep it tied down; check before enabling.
4. Then: collector with the new map, watch where the fingers land, 10 episodes.

## 2026-11-23 21:05 - 15 real scripted episodes collected (data/staging/cube_to_box_scripted, 675 MB)
- 12-point map + REACH_BIAS 5 mm: last runs 11 grasps of 13 attempts; the 2 misses were both ~1 cm short toward the base.
- Map scatter about +-1 cm (cube lands a few mm differently each release) - for more accuracy: repeat each calibration point 2-3x and average.
- Joint-2 power cable re-fixed after its 3rd drop (always in the reaching pose) - watch it.
- Arm parked + OFF. Next: convert to LeRobot (convert_real_dataset.sh), then train a policy (ACT/SmolVLA) on the GB10.

## 2026-11-23 22:35 - ACT policy trained + Holoscan policy app
- Dataset: data/lerobot/rebot_real_cube_to_box_scripted (15 ep, 12628 frames, 15 fps) via convert_real_dataset.sh cube_to_box_scripted.
- ACT: train/act_real_v1 (chunk 50, 25k steps, batch 8, 1h14m; 7.6 step/s with the twin closed, 4.6 with it). Final l1 0.035.
- Holoscan app: holoscan_app/policy_app.py (--replay N sim twin | --live real arm), server holoscan_app/policy_server.py (LeRobot venv, pyzmq added).
  Launch: ./run_policy_server.sh <ckpt>/pretrained_model ; ./open_policy_terminal.sh --replay 13 ; keys g run, x stop, r back to observe.
- Replay ep13 (training data, open loop): final ckpt gripper vs expert mean 1.3 cm, 90% 2.3, max 3.0 (ckpt 5000: 2.3/4.0/5.3).
- NOT yet run on the real arm (needs the user go-ahead + joint-2 cable check). No held-out episodes yet.

## 2026-11-24 01:45 - SmolVLA trained on zeux; Holoscan vs plain-loop videos
- SmolVLA: fine-tuned lerobot/smolvla_base on zeux (RTX PRO 6000, ~/smolvla-train, batch 32, 20k steps, GPU guard 88/82 C).
  Copied to train/smolvla_zeux/{010000,020000}. dexstro had overheated (SoC 96-97 C) and powered off at 23:12 during the first attempt.
- Replay ep13 (training data): SmolVLA 20k mean 0.6-0.7 cm (ACT 1.3 cm). SmolVLA chunk inference ~580 ms on the GB10.
- Videos in videos/{act_final,smolvla_final}: compare_twin.mp4 (twin camera), compare_overlay.mp4 (camera + plan + timings),
  compare_screen.mp4; left = Holoscan pipeline, right = plain Python loop; wall-clock (stalls show as frozen frames).
  60 Hz arm loop largest gap: ACT 19 ms vs 83 ms; SmolVLA 18 ms vs 586 ms (17 freezes per episode).
- record_all_videos.sh re-records both. train/ + videos/ now git-ignored (old commits still hold ACT checkpoints, .git 3.8 GB).
- Not run live yet: SmolVLA on the real arm (needs the user + joint-2 cable check).

## 2026-11-24 ~02:15 - SmolVLA LIVE on the real arm (through the Holoscan pipeline)
- 5 attempts. User observation: only attempt 1 picked properly (cube at the usual spot ~(0.28,+0.01)); the rest failed.
  Motor log: attempts 2 and 5 reported ~8.7 mm finger gap (something caught, likely an edge - not a proper carry),
  3 and 4 closed on nothing; the policy tends to go to the familiar spot first (15 episodes, positions clustered).
- Holoscan: 60 Hz arm loop largest gap 26-31 ms in every run despite ~580 ms SmolVLA chunk inference; no faults.
- Next: collect ~50+ scripted episodes spread over the whole green zone (edges included), retrain SmolVLA on zeux (~1 h).
- Display: after the 23:12 power-off dexstro came up at 1024x768 (monitor EDID not read); fixed with xrandr --output HDMI-0 --mode 1920x1080.
