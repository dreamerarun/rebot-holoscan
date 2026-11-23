"""Record the Isaac twin's own scene camera (ZMQ 5607) to an mp4 until stopped (Ctrl-C / SIGINT / SIGTERM).
Independent of window layout, so the twin is always in the video.  usage: record_twin.py out.mp4"""
import signal
import sys
import time

import cv2

from common import FrameSub
from videorec import VideoRec

stop = False


def _stop(*_):
    global stop
    stop = True


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)
sub, rec, n, last = FrameSub(), VideoRec(sys.argv[1], fps=15), 0, None
t_next = time.perf_counter()
while not stop:                                   # constant 15 fps output: repeat the latest twin frame
    f = sub.latest()
    if f is not None:
        last = cv2.cvtColor(f[1], cv2.COLOR_RGB2BGR)
    if last is not None and time.perf_counter() >= t_next:
        rec.write(last)
        n += 1
        t_next += 1 / 15
    time.sleep(0.005)
rec.close()
print(f"[twin-rec] {n} frames -> {sys.argv[1]}", flush=True)
