#!/usr/bin/env bash
# Opens a terminal on the dexstro desktop running the recorder (keys typed there, or echo X >> ~/rebot-holoscan/keys)
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
: > ~/rebot-holoscan/keys
gnome-terminal --title "reBot recorder - type keys here" --geometry 110x20+0+0 -- \
  bash -c "~/rebot-holoscan/run_real_record.sh --keys-file /home/dexstro/rebot-holoscan/keys $* 2>&1 | tee ~/rebot-holoscan/logs/record_session.log; echo; echo Recorder finished. Press Enter to close this window.; read"
