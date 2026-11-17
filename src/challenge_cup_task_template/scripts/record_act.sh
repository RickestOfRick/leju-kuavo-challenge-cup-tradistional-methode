#!/usr/bin/env bash
# Record one ACT demonstration.
#   bash record_act.sh <name>        e.g. bash record_act.sh seed0_p0
# Stop with Ctrl-C. The bag lands in ~/kuavo_ws/bags/act/<name>.bag
set -u
NAME="${1:-demo}"
OUT_DIR="/root/kuavo_ws/bags/act"
mkdir -p "$OUT_DIR"

# camera streams + state + the action vector
MAPS=$(rostopic list 2>/dev/null | grep -E "color/image_raw/compressed" | tr '\n' ' ')
[ -z "$MAPS" ] && echo "[record_act] WARNING: no compressed color topics found (is the sim up?)"

TOPICS="/act_action /sensors_data_raw /kuavo_arm_traj /leju_claw_state $MAPS"
echo "[record_act] writing $OUT_DIR/$NAME.bag"
echo "[record_act] topics: $TOPICS"
exec rosbag record -O "$OUT_DIR/$NAME.bag" $TOPICS
