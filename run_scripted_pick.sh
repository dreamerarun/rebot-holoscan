#!/usr/bin/env bash
# Scripted pick-and-place on the Isaac Sim twin (start the twin with: ./run_twin_gui.sh --vision)
source ~/rebot-holoscan/env.sh
cd ~/rebot-holoscan/holoscan_app
exec ../.venv-holoscan/bin/python scripted_pick.py "$@"
