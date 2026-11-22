#!/usr/bin/env bash
# Record real demonstrations (leader teleop + front camera) through Holoscan. Keys: n save/start, x discard, e/s/a/p/q arm.
source ~/rebot-holoscan/env.sh
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}
cd ~/rebot-holoscan/holoscan_app
exec ../.venv-holoscan/bin/python real_record.py "$@"
