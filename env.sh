# source ~/rebot-holoscan/env.sh   -> helper commands for the reBot Holoscan + Isaac Sim stack
export REBOT_HS=~/rebot-holoscan
export OMNI_KIT_ACCEPT_EULA=YES
ulimit -s 32768 2>/dev/null
hs_py()  { "$REBOT_HS/.venv-holoscan/bin/python" "$@"; }                                   # Holoscan 4.6 + reBot SDK
sim_py() { LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" "$REBOT_HS/.venv-isaacsim/bin/python" "$@"; }  # Isaac Sim 6.2
