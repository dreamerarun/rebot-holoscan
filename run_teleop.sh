#!/usr/bin/env bash
# Holoscan teleop pipeline (add --mock to test without the leader)
source ~/rebot-holoscan/env.sh
cd ~/rebot-holoscan
exec .venv-holoscan/bin/python holoscan_app/rebot_teleop.py "$@"
