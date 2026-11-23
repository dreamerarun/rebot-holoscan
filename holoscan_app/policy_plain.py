"""The same policy run WITHOUT Holoscan, for the comparison video: one plain Python loop doing everything in turn.

    loop @ 60 Hz:  read twin state -> (15 Hz) BLOCKING policy request -> SafetyFilter -> twin command -> draw window

Same replay episode, policy server, action checks, SafetyFilter and Isaac twin as policy_app.py --replay; the only
difference is the architecture. While the GPU computes a new action chunk, this loop waits, so the arm gets no new
command (the twin stops), and the replay keeps running in wall-clock time. SIM TWIN ONLY - never the real arm.

usage: policy_plain.py --replay 13 [--record out.mp4] [--server-timeout-ms 2000]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

from common import SimLink, load_cfg
from kinematics import RebotKinematics
from motion_plan import PHOTO_Q, RATE, READY_Q, Z_MIN_NEAR, MotionPlanner
from perception import load_calibration
from rebot_sim_pipeline.conventions import action_to_command
from safety import SafetyFilter, SafetyParams
from videorec import VideoRec, put_lines

ROOT = Path(__file__).resolve().parents[1]
POLICY_HZ = 15
REACH_BOX = {"x": (0.08, 0.40), "y": (-0.32, 0.26), "z": (0.0, 0.40)}
MAX_JUMP_DEG, LIMIT_STOP = 20.0, math.radians(10.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", type=int, required=True)
    ap.add_argument("--task", default="cube_to_box_scripted")
    ap.add_argument("--record", default=None)
    ap.add_argument("--server-timeout-ms", type=int, default=500)
    a = ap.parse_args()

    cfg = load_cfg()
    limits = [tuple(cfg["sim_limits"][f"joint{i}"]) for i in range(1, 7)]
    ep = sorted((ROOT / "data" / "staging" / a.task).glob("episode_*"))[a.replay - 1]
    files = sorted((ep / "images" / "front").glob("frame_*.jpg"))
    states, actions = np.load(ep / "states.npy"), np.load(ep / "actions.npy")
    k, mp, link = RebotKinematics(), MotionPlanner(limits), SimLink(cfg)
    flt = SafetyFilter(SafetyParams(limits=limits, vmax_deg_s=15.0, check_tracking=True, tcp_min_z=Z_MIN_NEAR),
                       fk=lambda q: k.fk(q).translation)
    H, _ = load_calibration(ROOT / "config" / "camera_table.yaml")
    Hi = np.linalg.inv(H)
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, a.server_timeout_ms)
    sock.connect("tcp://127.0.0.1:5610")
    rec = VideoRec(a.record) if a.record else None
    win = f"reBot policy WITHOUT Holoscan - plain Python loop - replay {ep.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)
    print(f"[plain] {ep.name}: {len(files)} frames", flush=True)

    def px(x, y):
        p = Hi @ np.array([x, y, 1.0])
        return int(p[0] / p[2]), int(p[1] / p[2])

    def meas_now():
        st = link.poll_state()
        if not st or time.time() - st.get("t", 0) > 1.0:
            return None
        q = dict(zip(st["names"], st["q"]))
        return np.array([q[f"joint{i}"] for i in range(1, 7)] + [q["finger_left"]])

    def command(target, meas, now, dt):
        cmd = flt.step(now, dt, target, 0.0, meas, 0.0, None)
        if cmd is not None:
            link.send(list(cmd[:6]) + [float(cmd[6]), -float(cmd[6])])
        return cmd

    # ---- 1) to the observe pose (same planned move as the Holoscan app)
    meas = None
    while meas is None:
        meas = meas_now()
        time.sleep(0.05)
    t_prev = time.perf_counter()
    while flt.mode != "FOLLOW":                  # let the filter align on the current pose
        now = time.perf_counter()
        m = meas_now()
        meas = m if m is not None else meas
        command(meas, meas, now, now - t_prev)
        t_prev = now
        time.sleep(1 / RATE)
    segs = [mp.joint_move("ready", meas[:6], READY_Q, float(meas[6])),
            mp.joint_move("observe", READY_Q, PHOTO_Q, float(meas[6]))]
    mp.validate(segs, meas)
    for q in [q for s in segs for q in s.q]:
        now = time.perf_counter()
        m = meas_now()
        meas = m if m is not None else meas
        command(q, meas, now, now - t_prev)
        t_prev = now
        time.sleep(max(0.0, 1 / RATE - (time.perf_counter() - now)))
    hold = np.asarray(segs[-1].q[-1], float)
    t_end = time.perf_counter() + 2.0
    while time.perf_counter() < t_end:
        now = time.perf_counter()
        m = meas_now()
        meas = m if m is not None else meas
        command(hold, meas, now, now - t_prev)
        t_prev = now
        time.sleep(1 / RATE)
    print("[plain] at the observe pose - running the policy", flush=True)

    # ---- 2) the policy, everything in one loop
    sock.send_multipart([json.dumps({"op": "reset"}).encode()])
    sock.recv_json()
    prev_t = next_t = hold.copy()
    interp, steps, dists, gap_max, stalls, wait_ms, infer_ms = 0, 0, [], 0.0, 0, 0.0, 0.0
    plan, why = [], "end of the recorded episode"
    t0 = t_prev = time.perf_counter()
    next_due, t_draw = t0, 0.0
    while True:
        now = time.perf_counter()
        gap = (now - t_prev) * 1000
        if steps:
            gap_max = max(gap_max, gap)
            stalls += gap > 50
        dt, t_prev = now - t_prev, now
        i = int((now - t0) * POLICY_HZ)
        if i >= len(files) - 1:
            break
        m = meas_now()
        meas = m if m is not None else meas
        bgr = cv2.imread(str(files[i]))
        if now >= next_due:                      # BLOCKING request: the whole loop waits for the GPU here
            next_due = max(next_due + 1.0 / POLICY_HZ, now - 0.5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tq = time.perf_counter()
            sock.send_multipart([json.dumps({"op": "act", "state": states[i].tolist(), "shape": list(rgb.shape)}).encode(),
                                 np.ascontiguousarray(rgb).tobytes()])
            try:
                r = sock.recv_json()
            except zmq.error.Again:
                why = f"no reply within {a.server_timeout_ms} ms"
                break
            wait_ms = (time.perf_counter() - tq) * 1000
            if "error" in r:
                why = r["error"]
                break
            infer_ms = r.get("infer_ms", 0.0)
            q6, g = action_to_command(np.asarray(r["action"], float))
            q7 = np.append(q6, g)
            for j, (lo, hi) in enumerate(limits):
                if not lo - LIMIT_STOP <= q7[j] <= hi + LIMIT_STOP:
                    why = f"joint{j + 1} far outside its limits"
                    break
                q7[j] = min(max(q7[j], lo), hi)
            else:
                p = k.fk(q7[:6]).translation
                outside = [ax for ax, v in zip("xyz", p) if not REACH_BOX[ax][0] <= v <= REACH_BOX[ax][1]]
                jump = float(np.degrees(np.max(np.abs(q7[:6] - next_t[:6]))))
                if outside or jump > MAX_JUMP_DEG:
                    why = f"gripper outside the box ({outside})" if outside else f"jumped {jump:.0f} deg"
                    break
                q6e, _ = action_to_command(np.asarray(actions[i], float))
                dists.append(float(np.linalg.norm(p - k.fk(q6e).translation)) * 100)
                plan = r.get("plan", [])
                prev_t, next_t, interp = hold.copy(), q7, 0
                steps += 1
                now = time.perf_counter()
                if why != "end of the recorded episode":
                    break
            if why != "end of the recorded episode":
                break
        interp = min(interp + 1, int(RATE / POLICY_HZ))
        al = interp / int(RATE / POLICY_HZ)
        hold = (1 - al) * prev_t + al * next_t
        cmd = command(hold, meas, time.perf_counter(), dt)
        if time.perf_counter() - t_draw > 1.0 / POLICY_HZ:   # draw in the same loop (no separate view stage)
            t_draw = time.perf_counter()
            img = bgr.copy()
            pts = []
            for pa in plan:
                pa = np.asarray(pa, float).reshape(-1)
                if pa.size == 7:
                    q6p, gp = action_to_command(pa)
                    pp = k.fk(q6p).translation
                    pts.append((px(pp[0], pp[1]), gp))
            for (p0, _), (p1, g1) in zip(pts, pts[1:]):
                cv2.line(img, p0, p1, (0, 200, 0) if g1 > 0.02 else (0, 0, 255), 2)
            pt = k.fk(hold[:6]).translation
            cv2.circle(img, px(pt[0], pt[1]), 7, (255, 0, 255), 2)
            lines = [f"WITHOUT Holoscan: one Python loop (camera -> policy -> safety -> arm, in turn)",
                     f"policy steps {steps}   GPU inference {infer_ms:5.1f} ms   loop waited {wait_ms:5.1f} ms",
                     f"safety filter: {flt.mode}   60 Hz arm loop: largest gap {gap_max:.0f} ms, {stalls} gaps > 50 ms",
                     f"recording frame {i}/{len(files)}" + (f"   gripper vs expert {dists[-1]:4.1f} cm" if dists else "")]
            put_lines(img, lines)
            cv2.imshow(win, img)
            cv2.waitKey(1)
            if rec:
                rec.write(img, wallclock=True)    # stalls appear as frozen pictures, as long as they lasted
        time.sleep(max(0.0, 1 / RATE - (time.perf_counter() - now)))

    # ---- 3) hold where it is for a moment, report
    t_end = time.perf_counter() + 1.5
    while time.perf_counter() < t_end:
        now = time.perf_counter()
        m = meas_now()
        meas = m if m is not None else meas
        command(hold, meas, now, 1 / RATE)
        time.sleep(1 / RATE)
    if rec:
        rec.close()
    d = np.array(dists) if dists else np.zeros(1)
    print(f"[plain] stopped after {steps} steps: {why}", flush=True)
    print(f"[plain]   replay summary: gripper vs expert mean {d.mean():.1f} cm, 90% {np.percentile(d, 90):.1f} cm, "
          f"max {d.max():.1f} cm over {len(dists)} steps", flush=True)
    print(f"[plain]   60 Hz arm loop: largest gap {gap_max:.0f} ms, {stalls} gaps over 50 ms", flush=True)
    cv2.destroyAllWindows()
    link.close()


if __name__ == "__main__":
    main()
