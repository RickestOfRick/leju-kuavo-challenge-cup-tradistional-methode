#!/usr/bin/env bash
# One-terminal graph: error of ALL 14 arm joints.
#   bash all_plot.sh          all 14 curves
#   bash all_plot.sh 3        only data[3]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 "$HERE/all_err.py" --plot > /tmp/all_err.log 2>&1 &
PUB=$!
trap 'kill $PUB 2>/dev/null' EXIT INT TERM
sleep 2
if [ $# -ge 1 ]; then
  TOPIC="/all_err/data[$1]"
else
  TOPIC=""
  for i in $(seq 0 13); do TOPIC="$TOPIC /all_err/data[$i]"; done
fi
xhost + >/dev/null 2>&1 || true
# shellcheck disable=SC2086
exec rqt_plot $TOPIC
