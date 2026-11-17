#!/usr/bin/env bash
# One-terminal LIVE graph: RIGHT-hand desired vs measured position (milimetres).
#   /ee_track/data[0..2] = desired  x, y, z   [mm]  (FK of commanded joints)
#   /ee_track/data[3..5] = measured x, y, z   [mm]  (FK of measured joints)
#
#   bash ee_track.sh         all three axes, desired + measured (6 curves)
#   bash ee_track.sh x       x only: desired vs measured
#   bash ee_track.sh y
#   bash ee_track.sh z
#   bash ee_track.sh err     |desired - measured| in mm (/fk_err_xyz/data[3])
set -u
case "${1:-xyz}" in
  x) TOPIC="/ee_track/data[0] /ee_track/data[3]" ;;
  y) TOPIC="/ee_track/data[1] /ee_track/data[4]" ;;
  z) TOPIC="/ee_track/data[2] /ee_track/data[5]" ;;
  err) TOPIC="/fk_err_xyz/data[3]" ;;
  *) TOPIC="/ee_track/data[0] /ee_track/data[1] /ee_track/data[2] /ee_track/data[3] /ee_track/data[4] /ee_track/data[5]" ;;
esac
xhost + >/dev/null 2>&1 || true
# shellcheck disable=SC2086
exec rqt_plot $TOPIC
