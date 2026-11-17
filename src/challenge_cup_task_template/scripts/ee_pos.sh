#!/usr/bin/env bash
# One-terminal LIVE graph: RIGHT-hand position, desired vs measured [mm]
#   /ee_target/data[0..2] = desired  x, y, z
#   /ee_pos/data[0..2]    = measured x, y, z
#
#   bash ee_pos.sh        all three axes (3 desired + 3 measured)
#   bash ee_pos.sh x      x only: desired vs measured  (y, z too)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 "$HERE/ee_pos.py" > /tmp/ee_pos.log 2>&1 &
PUB=$!
trap 'kill $PUB 2>/dev/null' EXIT INT TERM
sleep 3
case "${1:-xyz}" in
  x) TOPIC="/ee_target/data[0] /ee_pos/data[0]" ;;
  y) TOPIC="/ee_target/data[1] /ee_pos/data[1]" ;;
  z) TOPIC="/ee_target/data[2] /ee_pos/data[2]" ;;
  *) TOPIC="/ee_target/data[0] /ee_target/data[1] /ee_target/data[2] /ee_pos/data[0] /ee_pos/data[1] /ee_pos/data[2]" ;;
esac
xhost + >/dev/null 2>&1 || true
# shellcheck disable=SC2086
exec rqt_plot $TOPIC
