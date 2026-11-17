#!/usr/bin/env bash
# One-terminal LIVE graph: end-effector error (FK commanded vs measured).
#   bash fk_live.sh          both hands, position error in mm
#   bash fk_live.sh max      worst hand only
#   bash fk_live.sh all      both hands + rotation errors (deg)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 "$HERE/fk_live.py" --plot > /tmp/fk_live.log 2>&1 &
PUB=$!
trap 'kill $PUB 2>/dev/null' EXIT INT TERM
sleep 3
case "${1:-both}" in
  max) TOPIC="/fk_err/data[4]" ;;
  all) TOPIC="/fk_err/data[0] /fk_err/data[1] /fk_err/data[2] /fk_err/data[3]" ;;
  *)   TOPIC="/fk_err/data[0] /fk_err/data[1]" ;;
esac
xhost + >/dev/null 2>&1 || true
# shellcheck disable=SC2086
exec rqt_plot $TOPIC
