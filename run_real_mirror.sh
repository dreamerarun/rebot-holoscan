#!/usr/bin/env bash
# READ-ONLY: Isaac Sim arm mirrors the real reBot (motors never enabled). Start ./run_twin_gui.sh first.
source ~/rebot-holoscan/env.sh
cd ~/rebot-holoscan/holoscan_app
exec ../.venv-holoscan/bin/python real_mirror.py "$@"
