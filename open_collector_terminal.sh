#!/usr/bin/env bash
# Opens a terminal on the dexstro desktop running the LIVE scripted collector. usage: open_collector_terminal.sh [episodes]
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
: > ~/rebot-holoscan/keys
gnome-terminal --title "reBot collector - type keys here" --geometry 110x22+0+0 -- \
  bash -c "~/rebot-holoscan/run_collector.sh --live --episodes ${1:-1} 2>&1 | tee ~/rebot-holoscan/logs/collector_live.log; echo; echo Collector finished. Press Enter to close.; read"
