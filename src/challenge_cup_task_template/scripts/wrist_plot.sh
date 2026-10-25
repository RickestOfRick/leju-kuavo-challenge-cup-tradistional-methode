#!/usr/bin/env bash
# One-terminal wrist-error graph.
#   bash wrist_plot.sh          -> wrists only (L5,L6,L7,R5,R6,R7)
#   bash wrist_plot.sh all      -> all 14 arm joints
#
# Starts wrist_err.py in the background, opens rqt_plot, and stops the publisher
# as soon as you close the plot window.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-wrist}"

if [ "$MODE" = "all" ]; then
    python3 "$HERE/wrist_err.py" all > /tmp/wrist_err.log 2>&1 &
    ARGS=""
    for i in $(seq 0 13); do ARGS="$ARGS /wrist_err/data[$i]"; done
else
    python3 "$HERE/wrist_err.py" > /tmp/wrist_err.log 2>&1 &
    ARGS="/wrist_err/data[0] /wrist_err/data[1] /wrist_err/data[2] /wrist_err/data[3] /wrist_err/data[4] /wrist_err/data[5]"
fi
PUB=$!
cleanup() { kill "$PUB" 2>/dev/null; }
trap cleanup EXIT INT TERM

sleep 2
echo "[wrist_plot] publisher pid $PUB, log /tmp/wrist_err.log"
echo "[wrist_plot] close the rqt window to stop everything"
# shellcheck disable=SC2086
rqt_plot $ARGS
