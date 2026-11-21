"""reBot vision demo, PLAIN PYTHON version: the same work in one ordinary loop.

    while True:
        read leader -> map -> send to Isaac Sim      (control)
        grab camera frame -> GPU DNN -> draw window  (vision)

Every step waits for the one before it, so the arm can only be updated as fast as
the neural network runs. Same classes as vision_demo_holoscan.py.

    ./run_vision_demo.sh loop            # real leader
    ./run_vision_demo.sh loop --mock
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

from common import (Detector, FrameSub, LeaderDevice, Mapper, RateMeter, SimLink,
                    find_leader_port, load_cfg)


def draw(img, r, lines):
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for (x0, y0, x1, y1), lab, s in zip(r["boxes"], r["labels"], r["scores"]):
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (25, 230, 255), 1)
        cv2.putText(out, f"{lab} {s:.2f}", (int(x0), max(int(y0) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (25, 230, 255), 1)
    for key, col in (("cube", (50, 50, 255)), ("bin", (80, 255, 50))):
        if r[key]:
            x0, y0, x1, y1 = r[key]
            cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, 370 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(out, (1024, 768))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--leader-port")
    a = ap.parse_args()
    cfg = load_cfg()
    port = None if a.mock else find_leader_port(a.leader_port or cfg["leader"]["port"])

    dev = LeaderDevice(cfg, port, a.mock)
    dev.open()
    mapper, link, frames = Mapper(cfg), SimLink(cfg), FrameSub()
    det = Detector()
    print("[vision] Faster R-CNN loaded on", torch.cuda.get_device_name(0), flush=True)
    ctrl, vis = RateMeter(), RateMeter()
    det_ms, first, t_print = 0.0, True, time.time()
    next_tick = time.perf_counter()
    win = "reBot vision - PLAIN LOOP"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            # --- control: leader -> sim
            ok, deg = dev.read()
            if deg:
                link.send(mapper(deg), reset=first)
                first = False
                link.poll_state()
                ctrl.tick()
            # --- vision: frame -> DNN -> window (blocks the loop while it runs)
            f = frames.latest()
            if f is not None:
                r = det(torch.from_numpy(f[1]).cuda(non_blocking=True))
                det_ms = r["ms"]
                vis.tick()
                lines = ["PLAIN PYTHON LOOP  (one thread, steps in sequence)",
                         f"control:  {ctrl.hz:5.1f} Hz   worst gap {ctrl.max_gap_ms:4.0f} ms",
                         f"vision:   {vis.hz:5.1f} fps  Faster R-CNN {r['ms']:4.0f} ms on GB10",
                         f"cube: {'FOUND' if r['cube'] else '-'}   bin: {'FOUND' if r['bin'] else '-'}"]
                cv2.imshow(win, draw(f[1], r, lines))
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
            # same 60 Hz target as the Holoscan control lane
            next_tick = max(next_tick + 1 / 60, time.perf_counter())
            time.sleep(max(0.0, next_tick - time.perf_counter()))
            if time.time() - t_print > 1.0:
                t_print = time.time()
                print(f"[demo] control {ctrl.hz:5.1f} Hz (worst gap {ctrl.max_gap_ms:4.0f} ms) | "
                      f"vision {vis.hz:4.1f} fps, detector {det_ms:4.0f} ms", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()
        link.close()
        frames.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
