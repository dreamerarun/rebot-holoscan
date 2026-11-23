"""ACT policy server for the Holoscan policy app. Runs in the LeRobot venv (~/rebot_lerobot/.venv), never touches a motor.

ZMQ REP on 127.0.0.1:5610 (localhost only).
  request  multipart [json, rgb]   json {"op": "reset"}  or  {"op": "act", "state": [7], "shape": [h, w, 3]}
                                   rgb = uint8 HxWx3 RGB bytes (act only)
  reply    json {"action": [7], "plan": [[7], ...], "infer_ms": float, "refill": bool}
           action/plan are in the dataset's action convention (rebot_sim_pipeline.conventions.arm_action);
           plan = the rest of the current 50-step chunk (what ACT intends to do next), for display.

usage: policy_server.py --ckpt train/act_real_v1/checkpoints/last/pretrained_model
"""
import argparse
import json
import time

import numpy as np
import torch
import zmq
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

TASK = "Pick up the yellow cube and place it in the orange box"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint .../pretrained_model directory")
    ap.add_argument("--port", type=int, default=5610)
    a = ap.parse_args()

    policy = ACTPolicy.from_pretrained(a.ckpt)
    policy.to("cuda")
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=a.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    # warm-up: the first CUDA inference takes ~1.3 s (kernel selection) - do it now, not on the robot's first step
    t0 = time.perf_counter()
    with torch.inference_mode():
        dummy = {"observation.state": torch.zeros(1, 7), "observation.images.front": torch.zeros(1, 3, 480, 640),
                 "task": [TASK]}
        for _ in range(3):
            post(policy.select_action(pre(dummy)))
            policy.reset()
    torch.cuda.synchronize()
    print(f"[policy] warm-up done in {time.perf_counter() - t0:.1f} s", flush=True)
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{a.port}")
    print(f"[policy] ACT from {a.ckpt} on cuda, chunk {policy.config.chunk_size}, "
          f"executes {policy.config.n_action_steps} per chunk; serving 127.0.0.1:{a.port}", flush=True)

    n = 0
    while True:
        parts = sock.recv_multipart()
        req = json.loads(parts[0])
        if req.get("op") == "reset":
            policy.reset()
            sock.send_json({"ok": True})
            print("[policy] reset", flush=True)
            continue
        try:
            h, w = req["shape"][:2]
            img = np.frombuffer(parts[1], np.uint8).reshape(h, w, 3)
            obs = {"observation.state": torch.tensor(req["state"], dtype=torch.float32)[None],
                   "observation.images.front": torch.from_numpy(img.copy()).permute(2, 0, 1).float()[None] / 255.0,
                   "task": [TASK]}
            t0 = time.perf_counter()
            refill = len(policy._action_queue) == 0
            with torch.inference_mode():
                action = post(policy.select_action(pre(obs)))
                plan = [post(q).float().cpu().numpy().reshape(-1)[:7].tolist() for q in list(policy._action_queue)[::2]]  # queue items are (1, 7)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            sock.send_json({"action": action.float().cpu().numpy()[0].tolist(), "plan": plan,
                            "infer_ms": ms, "refill": refill})
            n += 1
            if refill:
                print(f"[policy] step {n}: new chunk in {ms:.0f} ms", flush=True)
        except Exception as exc:                 # never leave the client hanging: it HOLDs on an error reply
            sock.send_json({"error": f"{type(exc).__name__}: {exc}"})
            print(f"[policy] ERROR {exc}", flush=True)


if __name__ == "__main__":
    main()
