#!/usr/bin/env bash
# One-terminal LIVE graph: desired vs achieved hand position (and the error).
#
#   /ee_pose = [tgt_x, tgt_y, tgt_z, act_x, act_y, act_z, err_mm, rot_deg]
#   published at 10 Hz by task_sequence.py after every absolute move (arm_test_control._verify)
#
#   bash ee_pose.sh        x/y/z desired + achieved (6 curves)
#   bash ee_pose.sh x      only x: desired vs achieved (+ error)
#   bash ee_pose.sh err    only the error in mm
set -u
case "${1:-xyz}" in
  err) TOPIC="/ee_pose/data[6]" ;;
  x)   TOPIC="/ee_pose/data[0] /ee_pose/data[3] /ee_pose/data[6]" ;;
  y)   TOPIC="/ee_pose/data[1] /ee_pose/data[4] /ee_pose/data[6]" ;;
  z)   TOPIC="/ee_pose/data[2] /ee_pose/data[5] /ee_pose/data[6]" ;;
  *)   TOPIC="/ee_pose/data[0] /ee_pose/data[1] /ee_pose/data[2] /ee_pose/data[3] /ee_pose/data[4] /ee_pose/data[5]" ;;
esac
xhost + >/dev/null 2>&1 || true
# shellcheck disable=SC2086
exec rqt_plot $TOPIC
