#!/usr/bin/env bash
# Opens a terminal on the dexstro desktop running the LIVE robot-driven calibration (keys typed there)
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
: > ~/rebot-holoscan/keys
gnome-terminal --title "reBot calibration - type keys here" --geometry 110x22+0+0 -- \
  bash -c "~/rebot-holoscan/run_robot_calibrate.sh --live 2>&1 | tee ~/rebot-holoscan/logs/robot_cal_live.log; echo; echo Calibration program finished. Press Enter to close.; read"
