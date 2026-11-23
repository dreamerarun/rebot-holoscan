#!/usr/bin/env bash
# Comparison videos: the SAME policy + replay episode, run (1) as the Holoscan pipeline and (2) as one plain Python
# loop, both driving the Isaac twin (never the real arm). Needs the twin (run_twin_gui.sh --vision) and the policy
# server (run_policy_server.sh <ckpt>) running.  usage: record_compare.sh <name> [episode] [server_timeout_ms]
set -u
NAME=${1:?name}; EP=${2:-13}; TMO=${3:-500}; PARTS=${4:-both}   # both | holoscan | plain
OUT=~/rebot-holoscan/videos/$NAME; mkdir -p $OUT
cd ~/rebot-holoscan
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority
# the recorder must not keep the $( ) capture open (it blocked the script until ffmpeg exited)
twin() { (cd ~/rebot-holoscan/holoscan_app && source ../env.sh && exec ../.venv-holoscan/bin/python record_twin.py "$1") < /dev/null > /dev/null 2>&1 & echo $!; }
grab() { ffmpeg -y -loglevel error -f x11grab -framerate 15 -video_size 1920x1080 -i :1 -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p "$1" < /dev/null > /dev/null 2>&1 & echo $!; }

if [ "$PARTS" != plain ]; then
echo "== 1/2 WITH Holoscan"
: > keys
G=$(grab $OUT/screen_holoscan.mp4); T=$(twin $OUT/twin_holoscan.mp4)
./run_policy.sh --replay $EP --record $OUT/overlay_holoscan.mp4 --server-timeout-ms $TMO > $OUT/holoscan.log 2>&1 &
for i in $(seq 1 120); do sleep 1; grep -q "at the observe pose" $OUT/holoscan.log && break; done
sleep 2; echo g >> keys
for i in $(seq 1 150); do sleep 1; grep -q "STOPPED" $OUT/holoscan.log && break; done
sleep 3; echo q >> keys; sleep 4
P=$(pgrep -f "policy_ap[p].py"); [ -n "$P" ] && kill -INT $P; sleep 2
kill -INT $G $T; sleep 3
grep -E "STOPPED|summary|arm loop|\[sim\]" $OUT/holoscan.log
fi

if [ "$PARTS" != holoscan ]; then
echo "== 2/2 WITHOUT Holoscan (plain loop)"
G=$(grab $OUT/screen_plain.mp4); T=$(twin $OUT/twin_plain.mp4)
source env.sh; cd holoscan_app
../.venv-holoscan/bin/python policy_plain.py --replay $EP --record $OUT/overlay_plain.mp4 --server-timeout-ms $TMO > $OUT/plain.log 2>&1
cd ..; sleep 1; kill -INT $G $T; sleep 3
grep -E "stopped|summary|arm loop" $OUT/plain.log
fi

echo "== side by side"
ffmpeg -y -loglevel error -i $OUT/overlay_holoscan.mp4 -i $OUT/overlay_plain.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2:shortest=1[v]" -map "[v]" -c:v libx264 -crf 23 -pix_fmt yuv420p $OUT/compare_overlay.mp4
ffmpeg -y -loglevel error -i $OUT/screen_holoscan.mp4 -i $OUT/screen_plain.mp4 -filter_complex "[0:v]scale=960:540[a];[1:v]scale=960:540[b];[a][b]hstack=inputs=2:shortest=1[v]" -map "[v]" -c:v libx264 -crf 25 -pix_fmt yuv420p $OUT/compare_screen.mp4
ffmpeg -y -loglevel error -i $OUT/twin_holoscan.mp4 -i $OUT/twin_plain.mp4 -filter_complex "[0:v]scale=640:-2[a];[1:v]scale=640:-2[b];[a][b]hstack=inputs=2:shortest=1[v]" -map "[v]" -c:v libx264 -crf 23 -pix_fmt yuv420p $OUT/compare_twin.mp4
ls -la $OUT/*.mp4
