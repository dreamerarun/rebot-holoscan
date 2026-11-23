"""Thermal guard for long GPU jobs on the GB10 (dexstro powered off at SoC 96-97 C on 2026-11-23).

Every 2 s reads the hottest thermal zone (SoC) and the GPU temperature. Above PAUSE thresholds it freezes the job's
whole process group (SIGSTOP: nothing is lost, the GPU goes idle and cools); below RESUME thresholds it continues
(SIGCONT). Exits when the job ends.   usage: thermal_guard.py <pgid>
"""
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PAUSE_SOC, PAUSE_GPU = 90.0, 83.0
RESUME_SOC, RESUME_GPU = 82.0, 76.0


def soc_temp():
    temps = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(z.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else 0.0


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:
        return 0.0


def say(m):
    print(f"{datetime.now():%H:%M:%S} {m}", flush=True)


def main():
    pgid = int(sys.argv[1])
    paused, t_paused, n_pause = False, 0.0, 0
    last_report, hot = 0.0, 0.0
    say(f"[guard] watching process group {pgid}: pause at SoC {PAUSE_SOC} C / GPU {PAUSE_GPU} C, "
        f"resume below {RESUME_SOC} / {RESUME_GPU} C")
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            say(f"[guard] job ended; paused {n_pause} times, {t_paused / 60:.1f} min in total")
            return
        s, g = soc_temp(), gpu_temp()
        if not paused and (s >= PAUSE_SOC or g >= PAUSE_GPU):
            os.killpg(pgid, signal.SIGSTOP)
            paused, t0, n_pause = True, time.time(), n_pause + 1
            if n_pause == 1:
                say(f"[guard] first pause: SoC {s:.1f} C, GPU {g:.0f} C (further pauses summarised each minute)")
        elif paused and s <= RESUME_SOC and g <= RESUME_GPU:
            os.killpg(pgid, signal.SIGCONT)
            paused = False
            t_paused += time.time() - t0
            pass
        hot = max(hot, s)
        if time.time() - last_report > 60:
            last_report = time.time()
            say(f"[guard] last minute: hottest SoC {hot:.1f} C; now SoC {s:.1f} C, GPU {g:.0f} C, "
                f"{'PAUSED' if paused else 'running'}; paused {n_pause}x / {t_paused / 60:.1f} min in total")
            hot = 0.0
        time.sleep(2)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # so the finally below runs on kill too
    try:
        main()
    finally:                                     # never leave the job frozen if the guard itself stops
        try:
            os.killpg(int(sys.argv[1]), signal.SIGCONT)
        except Exception:
            pass
