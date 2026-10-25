#!/usr/bin/env bash
# Launch ANY of the monitors by name (inside the container, sim running):
#   bash graphs.sh ee          -> end-effector XYZ error   (needs task_sequence running)
#   bash graphs.sh wrist       -> wrist joint errors
#   bash graphs.sh shoulder    -> shoulder joint errors
#   bash graphs.sh wristtext   -> wrist errors as text
#   bash graphs.sh eetext      -> ee error as text
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
case "${1:-ee}" in
  ee)        exec bash "$HERE/ee_plot.sh" ;;
  wrist)     exec bash "$HERE/wrist_plot.sh" ;;
  shoulder)  exec bash "$HERE/shoulder_plot.sh" ;;
  wristtext) exec python3 "$HERE/wrist_err.py" ;;
  eetext)    exec python3 "$HERE/wrist_err.py" all ;;
  *) echo "usage: bash graphs.sh [ee|wrist|shoulder|wristtext|eetext]"; exit 1 ;;
esac
