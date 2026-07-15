#!/bin/bash
# ============================================================
# 快速启动任务二（场景二：零件分拣归档）
# 用法（宿主机或容器内均可）:
#   ./start_scene2.sh                  # 默认 seed=0
#   ./start_scene2.sh 42               # 指定 seed=42
#   TIME_LIMIT=180 ./start_scene2.sh   # seed=0, 限时180秒
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 参数 ---
SEED="${1:-${SEED:-0}}"
TIME_LIMIT="${TIME_LIMIT:-}"
TIMER_GUI="${TIMER_GUI:-1}"
SCENE="scene2"
NODE_NAME="challenge_task_scene2"
WORKSPACE="/root/kuavo_ws"

# --- 颜色 ---
RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'; CYAN='\033[36m'; NC='\033[0m'

banner() {
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  挑战杯 2026 - 任务二（场景二：零件分拣归档）${NC}"
    echo -e "${CYAN}  Seed: ${SEED}${NC}"
    [[ -n "$TIME_LIMIT" ]] && echo -e "${CYAN}  限时: ${TIME_LIMIT} 秒${NC}"
    echo -e "${CYAN}============================================${NC}"
}

# --- 检测是否在容器内 ---
in_container() {
    [[ -d "$WORKSPACE" ]] && [[ -f "$WORKSPACE/devel/setup.zsh" ]]
}

# --- 容器内直接启动 ---
run_in_container() {
    banner
    echo -e "${GREEN}容器内环境，直接启动...${NC}"

    local cmd="source ${WORKSPACE}/devel/setup.zsh && rosrun challenge_cup_task_template challenge_task.py --scene ${SCENE} --seed ${SEED} --node-name ${NODE_NAME}"
    [[ -n "$TIME_LIMIT" ]] && cmd="${cmd} --time-limit ${TIME_LIMIT}"
    [[ "$TIMER_GUI" == "0" ]] && cmd="${cmd} --no-timer-gui"

    echo -e "${GREEN}执行:${NC} ${cmd}"
    echo ""
    exec zsh -c "$cmd"
}

# --- 宿主机：管理 Docker 容器 ---
run_on_host() {
    banner

    local dir_hash project_dir container_name
    project_dir="$SCRIPT_DIR"
    dir_hash=$(echo "$project_dir" | md5sum | cut -c1-8)
    container_name="kuavo_challenge_container_GPU_${dir_hash}"
    local image_name="kuavo_challenge_cup_2026:latest"
    local docker_script="${project_dir}/docker/run_with_gpu_for_challenge.sh"

    # 1. 检查镜像
    if ! docker image inspect "$image_name" >/dev/null 2>&1; then
        echo -e "${RED}错误: Docker 镜像 '${image_name}' 未找到。${NC}"
        echo "请先导入: docker load -i kuavo_challenge_cup_2026_latest.tar.gz"
        exit 1
    fi

    # 2. 检查/启动容器
    if ! docker inspect "$container_name" >/dev/null 2>&1; then
        echo -e "${YELLOW}容器未创建，正在启动 Docker 容器...${NC}"
        echo -e "${YELLOW}按 Ctrl+D 退出交互终端后，重新运行本脚本。${NC}"
        xhost + >/dev/null 2>&1
        exec bash "$docker_script"
    fi

    local status
    status=$(docker inspect -f '{{.State.Status}}' "$container_name" 2>/dev/null | tr -d '[:space:]')

    case "$status" in
        running) ;;
        exited|created)
            echo -e "${YELLOW}容器已停止，正在启动...${NC}"
            docker start "$container_name" >/dev/null
            sleep 2
            ;;
        *)
            echo -e "${RED}错误: 容器状态异常 (${status:-unknown})${NC}"
            exit 1
            ;;
    esac

    echo -e "${GREEN}容器 '${container_name}' 已就绪。${NC}"

    # 3. 构建并执行
    local cmd="source ${WORKSPACE}/devel/setup.zsh && rosrun challenge_cup_task_template challenge_task.py --scene ${SCENE} --seed ${SEED} --node-name ${NODE_NAME}"
    [[ -n "$TIME_LIMIT" ]] && cmd="${cmd} --time-limit ${TIME_LIMIT}"
    [[ "$TIMER_GUI" == "0" ]] && cmd="${cmd} --no-timer-gui"

    echo -e "${GREEN}执行:${NC} ${cmd}"
    echo ""
    docker exec -it -e DISPLAY="$DISPLAY" "$container_name" zsh -c "$cmd"
}

# --- 入口 ---
if in_container; then
    run_in_container
else
    run_on_host
fi
