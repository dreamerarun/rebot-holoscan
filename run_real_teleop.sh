#!/usr/bin/env bash
# Stage 2: leader drives the REAL arm (motors start OFF; type e + Enter to enable). Start ./run_twin_gui.sh first.
source ~/rebot-holoscan/env.sh
cd ~/rebot-holoscan/holoscan_app
exec ../.venv-holoscan/bin/python real_teleop.py "$@"
