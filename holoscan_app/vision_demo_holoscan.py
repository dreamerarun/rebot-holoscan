"""reBot vision demo, HOLOSCAN version: two independent lanes.

  control lane (60 Hz):  LeaderOp -> MapOp -> SimOp              (leader -> Isaac Sim arm)
  vision lane (camera):  FrameOp -> DetectOp -> HolovizOp        (Isaac camera -> GPU DNN -> window)

The multi-thread scheduler runs the lanes side by side, so a slow neural network
never delays the arm. Compare with vision_demo_loop.py (same work, one plain loop).

    ./run_vision_demo.sh holoscan            # real leader
    ./run_vision_demo.sh holoscan --mock     # synthetic leader motion
    add --scheduler greedy to force everything onto ONE thread (Holoscan, but serial)
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta

import cupy as cp
import numpy as np
import torch
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler

from common import (Detector, FrameSub, LeaderDevice, Mapper, RateMeter, SimLink,
                    find_leader_port, load_cfg)

STATS = {"ctrl": RateMeter(), "vision": RateMeter(), "det_ms": 0.0, "sched": ""}
W, H = 640, 480


class LeaderOp(Operator):
    def __init__(self, fragment, *args, dev, **kw):
        self.dev = dev
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.output("leader")

    def start(self):
        self.dev.open()

    def stop(self):
        self.dev.close()

    def compute(self, op_input, op_output, context):
        ok, deg = self.dev.read()
        if deg:
            op_output.emit({"ok": ok, "deg": deg}, "leader")


class MapOp(Operator):
    def __init__(self, fragment, *args, cfg, **kw):
        self.mapper = Mapper(cfg)
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.input("leader")
        spec.output("q")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("leader")
        op_output.emit(self.mapper(msg["deg"]), "q")


class SimOp(Operator):
    def __init__(self, fragment, *args, cfg, **kw):
        self.cfg = cfg
        super().__init__(fragment, *args, **kw)

    def setup(self, spec: OperatorSpec):
        spec.input("q")

    def start(self):
        self.link = SimLink(self.cfg)
        self.first = True

    def stop(self):
        self.link.close()

    def compute(self, op_input, op_output, context):
        q = op_input.receive("q")
        self.link.send(q, reset=self.first)
        self.first = False
        self.link.poll_state()
        STATS["ctrl"].tick()


class FrameOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def start(self):
        self.sub = FrameSub()

    def stop(self):
        self.sub.close()

    def compute(self, op_input, op_output, context):
        f = self.sub.latest()
        if f is not None:
            op_output.emit(f, "frame")


def rect_tensor(boxes):
    """Pixel boxes -> Holoviz rectangles tensor (1, 2N, 2) in [0,1]; off-screen dummy if empty."""
    pts = [[(x0 / W, y0 / H), (x1 / W, y1 / H)] for x0, y0, x1, y1 in boxes] or [[(-1, -1), (-1, -1)]]
    return np.asarray(pts, dtype=np.float32).reshape(1, -1, 2)


class DetectOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("frame")
        spec.output("tensors")
        spec.output("specs")

    def start(self):
        self.det = Detector()
        print("[vision] Faster R-CNN loaded on", torch.cuda.get_device_name(0), flush=True)

    def compute(self, op_input, op_output, context):
        t_cam, img = op_input.receive("frame")
        img_gpu = torch.from_numpy(img).cuda(non_blocking=True)  # the one host->GPU copy
        r = self.det(img_gpu)
        STATS["vision"].tick()
        STATS["det_ms"] = r["ms"]
        c, v = STATS["ctrl"], STATS["vision"]
        lines = [f"HOLOSCAN  ({STATS['sched']})",
                 f"control lane: {c.hz:5.1f} Hz   worst gap {c.max_gap_ms:4.0f} ms",
                 f"vision lane:  {v.hz:5.1f} fps  Faster R-CNN {r['ms']:4.0f} ms on GB10",
                 f"cube: {'FOUND' if r['cube'] else '-'}   bin: {'FOUND' if r['bin'] else '-'}"]
        out = {
            "video": cp.from_dlpack(img_gpu),
            "dnn_boxes": rect_tensor(r["boxes"]),
            "cube_box": rect_tensor([r["cube"]] if r["cube"] else []),
            "bin_box": rect_tensor([r["bin"]] if r["bin"] else []),
            "hud": np.asarray([(0.015, 0.76 + 0.055 * i, 0.045) for i in range(len(lines))], dtype=np.float32),
            "dnn_text": np.asarray([(b[0] / W, max(b[1] / H - 0.04, 0), 0.035) for b in r["boxes"]]
                                   or [(-1, -1, 0.035)], dtype=np.float32),
        }
        hud = HolovizOp.InputSpec("hud", "text")
        hud.text, hud.color, hud.priority = lines, [1.0, 1.0, 1.0, 1.0], 10
        dnn = HolovizOp.InputSpec("dnn_text", "text")
        dnn.text = [f"{l} {s:.2f}" for l, s in zip(r["labels"], r["scores"])] or [" "]
        dnn.color, dnn.priority = [1.0, 0.9, 0.1, 1.0], 9
        op_output.emit(out, "tensors")
        op_output.emit([hud, dnn], "specs")


class VisionDemo(Application):
    def __init__(self, cfg, port, mock, sched):
        self.cfg, self.port, self.mock, self.sched = cfg, port, mock, sched
        super().__init__()

    def compose(self):
        dev = LeaderDevice(self.cfg, self.port, self.mock)
        leader = LeaderOp(self, PeriodicCondition(self, recess_period=timedelta(seconds=1 / 60)),
                          dev=dev, name="leader")
        mapper = MapOp(self, cfg=self.cfg, name="map")
        sim = SimOp(self, cfg=self.cfg, name="sim")
        self.add_flow(leader, mapper)
        self.add_flow(mapper, sim)

        frames = FrameOp(self, PeriodicCondition(self, recess_period=timedelta(milliseconds=5)), name="frames")
        detect = DetectOp(self, name="detect")
        viz = HolovizOp(self, name="holoviz", width=1024, height=768,
                        window_title="reBot vision - HOLOSCAN",
                        tensors=[dict(name="video", type="color"),
                                 dict(name="dnn_boxes", type="rectangles", color=[1.0, 0.9, 0.1, 1.0], line_width=2),
                                 dict(name="cube_box", type="rectangles", color=[1.0, 0.2, 0.2, 1.0], line_width=4),
                                 dict(name="bin_box", type="rectangles", color=[0.2, 1.0, 0.3, 1.0], line_width=4)])
        self.add_flow(frames, detect)
        self.add_flow(detect, viz, {("tensors", "receivers"), ("specs", "input_specs")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--leader-port")
    ap.add_argument("--scheduler", choices=["mt", "greedy"], default="mt")
    a = ap.parse_args()
    cfg = load_cfg()
    port = None if a.mock else find_leader_port(a.leader_port or cfg["leader"]["port"])
    app = VisionDemo(cfg, port, a.mock, a.scheduler)
    if a.scheduler == "mt":
        STATS["sched"] = "multi-thread scheduler: lanes run in parallel"
        app.scheduler(MultiThreadScheduler(app, name="mt", worker_thread_number=4, stop_on_deadlock=False))
    else:
        STATS["sched"] = "greedy scheduler: ONE thread"
        app.scheduler(GreedyScheduler(app, name="greedy", stop_on_deadlock=False))

    def report():
        while True:
            time.sleep(1.0)
            c, v = STATS["ctrl"], STATS["vision"]
            print(f"[demo] control {c.hz:5.1f} Hz (worst gap {c.max_gap_ms:4.0f} ms) | "
                  f"vision {v.hz:4.1f} fps, detector {STATS['det_ms']:4.0f} ms", flush=True)

    import threading
    threading.Thread(target=report, daemon=True).start()
    app.run()


if __name__ == "__main__":
    main()
