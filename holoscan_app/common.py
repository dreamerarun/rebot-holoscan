"""Shared building blocks for the reBot vision demo.

The Holoscan version (vision_demo_holoscan.py) and the plain-Python version
(vision_demo_loop.py) both use exactly these classes, so the only difference
between them is how the work is scheduled.
"""
from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

from rebot_teleop import LEADER_JOINTS, SIM_JOINTS, find_leader_port, unwrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path=None):
    return yaml.safe_load(Path(path or ROOT / "config" / "teleop.yaml").read_text())


# --------------------------------------------------------------------------- leader
class LeaderDevice:
    """102 Leader (FashionStar servos, passive) or a synthetic motion (mock)."""

    def __init__(self, cfg, port=None, mock=False):
        self.cfg, self.port, self.mock = cfg, port, mock
        self.bus = None
        self.t0 = time.perf_counter()
        self.last = None

    def open(self):
        if self.mock:
            print("[leader] MOCK leader (synthetic motion)", flush=True)
            return
        from motorbridge_smart_servo import FashionStarServo
        ids = self.cfg["leader"]["ids"]
        self.bus = FashionStarServo(self.port, baudrate=int(self.cfg["leader"]["baudrate"]))
        missing = [n for n, i in ids.items() if not self.bus.ping(i)]
        if missing:
            raise RuntimeError(f"Leader servos not answering: {missing}")
        for i in ids.values():
            self.bus.unlock(i)  # leader stays limp
            time.sleep(0.05)
        for i in ids.values():
            self.bus.reset_multi_turn(i)
        print(f"[leader] connected on {self.port}: 7/7 servos answer", flush=True)

    def close(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None

    def read(self):
        """Returns (ok, {joint: deg})."""
        if self.mock:
            t, s = time.perf_counter() - self.t0, math.sin
            deg = {"shoulder_pan": 35 * s(0.9 * t), "shoulder_lift": 60 + 25 * s(0.7 * t),
                   "elbow_flex": -70 + 25 * s(0.8 * t + 1.0), "wrist_flex": 25 * s(1.1 * t),
                   "wrist_yaw": 30 * s(1.0 * t + 0.5), "wrist_roll": 60 * s(0.5 * t),
                   "gripper": 22.5 + 22.5 * s(1.2 * t)}
            self.last = deg
            return True, deg
        try:
            ids = self.cfg["leader"]["ids"]
            mon = self.bus.sync_monitor(list(ids.values()))
            deg = {}
            for name, sid in ids.items():
                m = mon.get(sid)
                if m is None:
                    raise RuntimeError(f"no reply from {name}")
                lo, hi = self.cfg["leader"]["ranges"][name]
                deg[name] = float(np.clip(unwrap(m.angle_deg, lo, hi), lo, hi))
            self.last = deg
            return True, deg
        except Exception as exc:
            print(f"[leader] read failed: {exc}", flush=True)
            return False, self.last


# --------------------------------------------------------------------------- mapping
class Mapper:
    """Leader degrees -> sim targets (rad / m) with clamp + per-step rate limit."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.prev = None

    def __call__(self, deg):
        c, f = self.cfg, self.cfg["filter"]
        q = []
        for m in c["arm_mapping"]:
            lo, hi = c["sim_limits"][m["sim"]]
            q.append(float(np.clip(math.radians(deg[m["leader"]] * m["scale"] + m["offset_deg"]), lo, hi)))
        g = c["gripper_mapping"]
        opn = float(np.clip((deg[g["leader"]] - g["leader_closed_deg"])
                            / (g["leader_open_deg"] - g["leader_closed_deg"]), 0, 1))
        grip = g["sim_closed_m"] + opn * (g["sim_open_m"] - g["sim_closed_m"])
        q = np.array(q + [grip, -grip])
        if self.prev is not None:
            step = math.radians(f["max_step_deg"])
            d = q[:6] - self.prev[:6]
            d[np.abs(d) < math.radians(f["deadband_deg"])] = 0.0
            q[:6] = self.prev[:6] + np.clip(d, -step, step)
        self.prev = q
        return q


# --------------------------------------------------------------------------- sim I/O
class SimLink:
    def __init__(self, cfg):
        ctx = zmq.Context.instance()
        self.pub = ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.SNDHWM, 2)
        self.pub.bind(cfg["sim_link"]["cmd_endpoint"])
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.connect(cfg["sim_link"]["state_endpoint"])
        self.state = None
        self.seq = 0

    def send(self, q, hold=False, reset=False):
        self.seq += 1
        msg = {"seq": self.seq, "t": time.time(), "q": [float(v) for v in q], "names": SIM_JOINTS, "hold": hold}
        if reset:
            msg["reset"] = True
        self.pub.send_string(json.dumps(msg))

    def poll_state(self):
        try:
            while True:
                self.state = json.loads(self.sub.recv_string(zmq.NOBLOCK))
        except zmq.Again:
            pass
        return self.state

    def close(self):
        self.pub.close(0)
        self.sub.close(0)


class FrameSub:
    """Latest Isaac Sim camera frame (RGB uint8), or None if nothing new."""

    def __init__(self, endpoint="tcp://127.0.0.1:5607"):
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.connect(endpoint)

    def latest(self):
        try:
            buf = self.sock.recv(zmq.NOBLOCK, copy=True)
        except zmq.Again:
            return None
        t, w, h = struct.unpack_from("<dHH", buf, 0)
        img = np.frombuffer(buf, dtype=np.uint8, offset=12).reshape(h, w, 3)
        return t, img

    def close(self):
        self.sock.close(0)


# --------------------------------------------------------------------------- vision
COCO = None


class Detector:
    """GPU vision on the GB10:
    - Faster R-CNN (ResNet-50 FPN v2, COCO) = a real neural network, the kind of load
      a policy or perception model puts on the pipeline;
    - colour segmentation that localises the red cube and the green bin.
    """

    def __init__(self, min_score=0.5):
        import torch
        import torchvision
        global COCO
        self.torch = torch
        w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        COCO = w.meta["categories"]
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w).eval().cuda()
        self.min_score = min_score
        with torch.inference_mode():  # warm-up (cuDNN autotune, allocator)
            self.model([torch.zeros(3, 480, 640, device="cuda")])
        torch.cuda.synchronize()

    def _bbox(self, mask, min_px=30):
        """Box around the mask, ignoring the outer 2 % of pixels (stray reflections)."""
        ys, xs = self.torch.nonzero(mask, as_tuple=True)
        n = ys.numel()
        if n < min_px:
            return None
        lo, hi = int(0.02 * n), max(int(0.98 * n) - 1, 0)
        xs, ys = xs.sort().values, ys.sort().values
        return [xs[lo].item(), ys[lo].item(), xs[hi].item(), ys[hi].item()]

    def __call__(self, img_gpu):
        """img_gpu: torch uint8 (H, W, 3) on cuda. Returns dict of results in pixels."""
        torch = self.torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            x = img_gpu.permute(2, 0, 1).float().div_(255.0)
            out = self.model([x])[0]
            r, g, b = (img_gpu[..., i].int() for i in range(3))
            # ratios, not absolute values: robust to shading; excludes the arm's lime stripes
            cube = self._bbox((r > 110) & (g * 10 < r * 4) & (b * 10 < r * 4))
            bin_ = self._bbox((g > 100) & (r * 2 < g) & (b * 10 < g * 7))
            keep = out["scores"] > self.min_score
            boxes = out["boxes"][keep].tolist()
            labels = [COCO[i] for i in out["labels"][keep].tolist()]
            scores = out["scores"][keep].tolist()
        torch.cuda.synchronize()
        return {"cube": cube, "bin": bin_, "boxes": boxes, "labels": labels, "scores": scores,
                "ms": (time.perf_counter() - t0) * 1e3}


class RateMeter:
    """Rate and worst gap between events over a sliding 1 s window."""

    def __init__(self):
        self.ts = []

    def tick(self):
        now = time.perf_counter()
        self.ts.append(now)
        while self.ts and now - self.ts[0] > 1.0:
            self.ts.pop(0)

    @property
    def hz(self):
        return 0.0 if len(self.ts) < 2 else (len(self.ts) - 1) / (self.ts[-1] - self.ts[0])

    @property
    def max_gap_ms(self):
        return 0.0 if len(self.ts) < 2 else max(b - a for a, b in zip(self.ts, self.ts[1:])) * 1e3
