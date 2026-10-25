#!/usr/bin/env bash
# One command: end-effector XYZ error graph (task_sequence publishes /ee_err at 10 Hz).
#   bash ee_plot.sh          dx, dy, dz
#   bash ee_plot.sh dz       only one axis
set -u
AXIS="${1:-all}"
case "$AXIS" in
  x)   TOPIC="/ee_err/data[0]" ;;
  y)   TOPIC="/ee_err/data[1]" ;;
  z)   TOPIC="/ee_err/data[2]" ;;
  all) TOPIC="/ee_err/data[0] /ee_err/data[1] /ee_err/data[2]" ;;
  *)   TOPIC="/ee_err/data[0] /ee_err/data[1] /ee_err/data[2]" ;;
esac
xhost + >/dev/null 2>&1 || true
# shellcheck disable=SC2086
exec rqt_plot $TOPIC
