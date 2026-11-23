#!/usr/bin/env bash
# Holoscan policy app. usage: run_policy.sh --replay N | --live   (needs run_policy_server.sh running)
source ~/rebot-holoscan/env.sh
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}
cd ~/rebot-holoscan/holoscan_app
exec ../.venv-holoscan/bin/python policy_app.py "$@"
