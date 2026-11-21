"""Save Isaac Sim scene-camera frames at given times (s) after start: grab_frames.py out_prefix t1 t2 ..."""
import struct, sys, time
import numpy as np, zmq
from PIL import Image
s = zmq.Context.instance().socket(zmq.SUB)
s.setsockopt(zmq.CONFLATE, 1); s.setsockopt_string(zmq.SUBSCRIBE, ""); s.connect("tcp://127.0.0.1:5607")
t0 = time.time()
for t in map(float, sys.argv[2:]):
    time.sleep(max(0, t0 + t - time.time()))
    while True:
        try:
            s.recv(zmq.NOBLOCK)  # drop stale
        except zmq.Again:
            break
    buf = s.recv()
    _, w, h = struct.unpack_from("<dHH", buf, 0)
    Image.fromarray(np.frombuffer(buf, np.uint8, offset=12).reshape(h, w, 3)).save(f"{sys.argv[1]}_{t:04.1f}.png")
    print("saved", t, flush=True)
