#!/usr/bin/env bash
# Terminal on the dexstro desktop running the Holoscan policy app. usage: open_policy_terminal.sh --replay N | --live
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
: > ~/rebot-holoscan/keys
gnome-terminal --title "reBot policy (Holoscan) - type keys here" --geometry 110x22+0+0 -- \
  bash -c "~/rebot-holoscan/run_policy.sh $* 2>&1 | tee ~/rebot-holoscan/logs/policy_app.log; echo; echo Policy app finished. Press Enter to close.; read"
