#!/usr/bin/env bash
cd ~/rebot-holoscan
exec .venv-holoscan/bin/python tests/power_test.py "${1:-45}" > logs/power_test.log 2>&1
