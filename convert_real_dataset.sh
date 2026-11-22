#!/usr/bin/env bash
# Staged real episodes -> LeRobot v3 dataset (runs in the LeRobot venv). usage: ./convert_real_dataset.sh [task] [--keep-staging]
set -e
TASK=${1:-cube_to_box}; shift || true
cd ~/rebot-sim-pipeline/src
~/rebot_lerobot/.venv/bin/python -m rebot_sim_pipeline.write_dataset \
  --staging ~/rebot-holoscan/data/staging/$TASK \
  --dataset-root ~/rebot-holoscan/data/lerobot/rebot_real_$TASK \
  --repo-id local/rebot_real_$TASK --keep-staging "$@"
