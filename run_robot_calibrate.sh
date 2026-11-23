#!/usr/bin/env bash
# Robot-driven camera calibration. --sim (Isaac twin, start ./run_twin_gui.sh --vision first) or --live (real arm)
source ~/rebot-holoscan/env.sh
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}
cd ~/rebot-holoscan/holoscan_app
exec ../.venv-holoscan/bin/python robot_calibrate.py "$@"
