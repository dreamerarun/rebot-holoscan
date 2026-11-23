#!/usr/bin/env bash
# Camera -> table calibration (motors OFF). Keys in the window: c capture, u undo, s save, q quit.
source ~/rebot-holoscan/env.sh
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}
cd ~/rebot-holoscan/holoscan_app
exec ../.venv-holoscan/bin/python calibrate_camera.py "$@"
