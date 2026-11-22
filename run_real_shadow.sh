#!/usr/bin/env bash
# Stage 1 SHADOW MODE (motors stay OFF). Start ./run_twin_gui.sh first. Keys: a=re-arm s=stop q=quit
source ~/rebot-holoscan/env.sh
cd ~/rebot-holoscan/holoscan_app
exec ../.venv-holoscan/bin/python real_shadow.py "$@"
