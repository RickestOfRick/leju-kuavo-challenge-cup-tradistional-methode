#!/usr/bin/env bash
# Unattended ACT data collection: one sim run per seed, 4 bags each.
#   bash collect_seeds.sh                 -> seeds 0..4   (20 bags)
#   bash collect_seeds.sh 5 6             -> seeds 5..6
set -u
CONTAINER=kuavo_challenge_container_GPU_04aaf2fe
FIRST="${1:-0}"; LAST="${2:-4}"
SCRIPTS=/root/kuavo_ws/src/challenge_cup_task_template/scripts
BAGS=/root/kuavo_ws/bags/act
mkdir -p "$BAGS" 2>/dev/null || true

kill_all() {
  docker exec "$CONTAINER" bash -lc 'pkill -f "[n]odelet_manager"; pkill -f "[c]hallenge_task"; pkill -f "[r]oslaunch"; pkill -f "[r]osmaster"; pkill -f "[h]umanoid_"; sleep 3' >/dev/null 2>&1
}

for seed in $(seq "$FIRST" "$LAST"); do
  echo "================ seed $seed ================"
  kill_all
  docker exec -d "$CONTAINER" zsh -ic "source /root/kuavo_ws/devel/setup.zsh; cd /root/kuavo_ws; rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed $seed > /tmp/sim_seed$seed.log 2>&1"

  # wait until the robot publishes sensor data (max ~4 min)
  for i in $(seq 1 48); do
    if docker exec "$CONTAINER" bash -lc 'source /opt/ros/noetic/setup.bash; timeout 6 rostopic echo -n1 /sensors_data_raw/joint_data/joint_q >/dev/null 2>&1'; then
      echo "[$seed] robot ready after $((i*5)) s"; break
    fi
    sleep 5
  done
  sleep 3

  echo "[$seed] collecting ..."
  docker exec "$CONTAINER" zsh -ic "source /root/kuavo_ws/devel/setup.zsh; cd $SCRIPTS; python3 task_sequence.py --seed $seed --collect" 2>&1 | tail -25

  echo "[$seed] bags:"
  docker exec "$CONTAINER" bash -lc "ls -1 $BAGS | grep -c seed${seed}_p || true"
done

kill_all
echo "================ done ================"
docker exec "$CONTAINER" bash -lc 'ls -lh /root/kuavo_ws/bags/act/ | tail -25'
