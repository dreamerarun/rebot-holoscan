#!/usr/bin/env bash
source ~/rebot-holoscan/env.sh
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}
cd ~/rebot-holoscan/holoscan_app
exec ../.venv-holoscan/bin/python collector.py "$@"
