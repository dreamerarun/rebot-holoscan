#!/usr/bin/env bash
cd ~/rebot-holoscan
exec .venv-holoscan/bin/python tests/gripper_film.py "${1:-60}" > logs/gripper_film.log 2>&1
