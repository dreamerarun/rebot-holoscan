#!/usr/bin/env bash
# Isaac Sim reBot twin on the dexstro monitor
source ~/rebot-holoscan/env.sh
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority
cd ~/rebot-holoscan
exec env LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 .venv-isaacsim/bin/python sim/rebot_twin_gui.py "$@"
