#!/usr/bin/env bash
# One-terminal LIVE graph: RIGHT-hand end-effector error in XYZ (mm, signed).
#   /fk_err_xyz/data[0..2] = dx, dy, dz  (commanded - measured)
#   /fk_err_xyz/data[3]    = norm
#
#   bash fk_xyz.sh        dx, dy, dz, norm
#   bash fk_xyz.sh x      dx only (+ norm)
#   bash fk_xyz.sh norm   magnitude only
set -u
case "${1:-xyz}" in
  x)    TOPIC="/fk_err_xyz/data[0] /fk_err_xyz/data[3]" ;;
  y)    TOPIC="/fk_err_xyz/data[1] /fk_err_xyz/data[3]" ;;
  z)    TOPIC="/fk_err_xyz/data[2] /fk_err_xyz/data[3]" ;;
  norm) TOPIC="/fk_err_xyz/data[3]" ;;
  *)    TOPIC="/fk_err_xyz/data[0] /fk_err_xyz/data[1] /fk_err_xyz/data[2] /fk_err_xyz/data[3]" ;;
esac
xhost + >/dev/null 2>&1 || true
# shellcheck disable=SC2086
exec rqt_plot $TOPIC
