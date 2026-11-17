#!/usr/bin/env bash
# One-terminal LIVE graph: max joint change per control tick (10 ms).
#   bash motion_step.sh          max of all 14 joints
#   bash motion_step.sh 3        joint 3 (0..13), i.e. L4
#   bash motion_step.sh all      all 14 joints
# Keep the curve under 5 deg/10ms.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 "$HERE/motion_step.py" --plot > /tmp/motion_step.log 2>&1 &
PUB=$!
trap 'kill $PUB 2>/dev/null' EXIT INT TERM
sleep 2
if [ $# -ge 1 ] && [ "$1" != "all" ]; then
  TOPIC="/motion_step/data[$1]"
elif [ $# -ge 1 ]; then
  TOPIC=""
  for i in $(seq 0 13); do TOPIC="$TOPIC /motion_step/data[$i]"; done
  TOPIC="$TOPIC /motion_step/data[14]"
else
  TOPIC="/motion_step/data[14]"
fi
xhost + >/dev/null 2>&1 || true
# shellcheck disable=SC2086
exec rqt_plot $TOPIC
