#!/usr/bin/env bash
# One-terminal shoulder-error graph.
#   bash wrist_plot.sh          -> shoulders only (L1,L2,L3,R1,R2,R3)
#   bash wrist_plot.sh all      -> all 14 arm joints
#
# Starts shoulder_err.py in the background, opens rqt_plot, and stops the publisher
# as soon as you close the plot window.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-wrist}"

if [ "$MODE" = "all" ]; then
    python3 "$HERE/shoulder_err.py" all > /tmp/shoulder_err.log 2>&1 &
    ARGS=""
    for i in $(seq 0 13); do ARGS="$ARGS /shoulder_err/data[$i]"; done
else
    python3 "$HERE/shoulder_err.py" > /tmp/shoulder_err.log 2>&1 &
    ARGS="/shoulder_err/data[0] /shoulder_err/data[1] /shoulder_err/data[2] /shoulder_err/data[3] /shoulder_err/data[4] /shoulder_err/data[5]"
fi
PUB=$!
cleanup() { kill "$PUB" 2>/dev/null; }
trap cleanup EXIT INT TERM

sleep 2
echo "[wrist_plot] publisher pid $PUB, log /tmp/shoulder_err.log"
echo "[wrist_plot] close the rqt window to stop everything"
# shellcheck disable=SC2086
rqt_plot $ARGS
