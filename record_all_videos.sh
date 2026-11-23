#!/usr/bin/env bash
# Re-record both comparisons (ACT, then SmolVLA) with the twin + the matching policy server. Detached; log per model.
cd ~/rebot-holoscan
server() { P=$(pgrep -f "policy_serve[r].py"); [ -n "$P" ] && kill $P; sleep 3
           setsid nohup ./run_policy_server.sh "$1" > logs/policy_server.log 2>&1 < /dev/null &
           for i in $(seq 1 90); do sleep 1; grep -q serving logs/policy_server.log && return; done; echo "server did not start"; exit 1; }
echo "== ACT"; server $HOME/rebot-holoscan/train/act_real_v1/checkpoints/025000/pretrained_model
rm -rf videos/act_final; ./record_compare.sh act_final 13 500 both
echo "== SmolVLA"; server $HOME/rebot-holoscan/train/smolvla_zeux/020000
rm -rf videos/smolvla_final; ./record_compare.sh smolvla_final 13 2000 both
echo "== ALL DONE"
