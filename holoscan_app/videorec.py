"""Frames -> H.264 mp4 through an ffmpeg pipe. ffmpeg finalises the file when the pipe closes, so the video stays
playable even when the program is stopped with Ctrl-C."""
import subprocess

import cv2


class VideoRec:
    def __init__(self, path, fps=15):
        self.path, self.fps, self.p = path, fps, None

    def write(self, bgr):
        if self.p is None:
            h, w = bgr.shape[:2]
            self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                       "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-", "-c:v", "libx264",
                                       "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", self.path],
                                      stdin=subprocess.PIPE)
        try:
            self.p.stdin.write(bgr.tobytes())
        except BrokenPipeError:
            pass

    def close(self):
        if self.p is not None:
            try:
                self.p.stdin.close()
                self.p.wait(timeout=10)
            except Exception:
                pass
            self.p = None


def put_lines(img, lines, y0=6, scale=0.45, color=(80, 255, 255)):
    """Readable text: each line once, on a dark box (an outline drawn with a thicker pen is wider and misaligned)."""
    y = y0
    for t in lines:
        (w, h), b = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (4, y), (12 + w, y + h + b + 4), (0, 0, 0), -1)
        cv2.putText(img, t, (8, y + h + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += h + b + 6
