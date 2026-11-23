#!/usr/bin/env bash
# ACT policy server (LeRobot venv, GPU, localhost:5610). usage: run_policy_server.sh [checkpoint dir]
CKPT=${1:-$HOME/rebot-holoscan/train/act_real_v1/checkpoints/last/pretrained_model}
cd ~/rebot-holoscan/holoscan_app
exec ~/rebot_lerobot/.venv/bin/python policy_server.py --ckpt "$CKPT"
