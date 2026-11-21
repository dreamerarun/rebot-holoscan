#!/usr/bin/env bash
# usage: ./run_vision_demo.sh holoscan|loop [--mock] [--scheduler greedy]
source ~/rebot-holoscan/env.sh
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}
cd ~/rebot-holoscan/holoscan_app
mode=${1:-holoscan}; shift || true
case "$mode" in
  holoscan) exec ../.venv-holoscan/bin/python vision_demo_holoscan.py "$@" ;;
  loop)     exec ../.venv-holoscan/bin/python vision_demo_loop.py "$@" ;;
  *) echo "mode must be holoscan or loop"; exit 2 ;;
esac
