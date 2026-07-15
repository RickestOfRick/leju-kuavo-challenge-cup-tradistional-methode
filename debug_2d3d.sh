#!/bin/bash
# ============================================================================
#  Scene2 2D→3D 投影精度一键调试
#
#  用法:  bash debug_2d3d.sh [seed] [frames]
#  示例:  bash debug_2d3d.sh 3 20
#        bash debug_2d3d.sh         # 默认 seed=3, frames=15
# ============================================================================

set -e

cd "$(dirname "$0")"
WS_ROOT="$(pwd)"
SEED="${1:-3}"
FRAMES="${2:-15}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Scene2 2D→3D 投影精度调试                                  ║"
echo "║  seed=${SEED}  frames=${FRAMES}  ws=${WS_ROOT}               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# source ROS workspace（如果未 source 过）
if [ -f "${WS_ROOT}/devel/setup.bash" ]; then
    source "${WS_ROOT}/devel/setup.bash"
elif [ -f "/opt/ros/noetic/setup.bash" ]; then
    source /opt/ros/noetic/setup.bash
fi

exec python3 src/challenge_cup_simulator/test/run_debug_2d3d.py \
    --seed "${SEED}" \
    --frames "${FRAMES}" \
    --no-timer-gui \
    --repo-root "${WS_ROOT}"
