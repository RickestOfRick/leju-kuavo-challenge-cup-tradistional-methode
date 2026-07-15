# 挑战杯 2026 Kuavo 仿真赛 — API 接口汇总

> 面向 **任务二（场景二：零件分拣归档）** 的接口速查。
> 完整字段和调用示例以各接口文档原文为准。

---

## 目录

- [1. 启动与场景管理](#1-启动与场景管理)
- [2. 行走控制（底盘/步态）](#2-行走控制底盘步态)
- [3. 手臂控制](#3-手臂控制)
- [4. 夹爪控制（二指夹爪）](#4-夹爪控制二指夹爪)
- [5. 灵巧手控制](#5-灵巧手控制)
- [6. 头部控制](#6-头部控制)
- [7. 传感器数据](#7-传感器数据)
- [8. 相机接口](#8-相机接口)
- [9. 状态估计](#9-状态估计)
- [10. RL 控制器管理](#10-rl-控制器管理)
- [11. 调试与监控](#11-调试与监控)
- [12. 禁止使用的上帝视角接口](#12-禁止使用的上帝视角接口)

---

## 1. 启动与场景管理

### 任务入口

```bash
# 场景二（零件分拣归档）
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3

# 带限时（秒）
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3 --time-limit 120

# 不弹计时窗口
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3 --time-limit 120 --no-timer-gui
```

### 快捷脚本

```bash
./scripts/start_scene2.sh         # seed=0
./scripts/start_scene2.sh 42      # seed=42
TIME_LIMIT=180 ./scripts/start_scene2.sh  # seed=0, 限时180s
```

### 手动场景生成

```bash
rosrun challenge_cup_simulator scene_builder.py --all
```

**场景配置文件**: `src/challenge_cup_simulator/config/scenes/scene2.yaml`

---

## 2. 行走控制（底盘/步态）

### 2.1 话题

| 接口 | 消息类型 | 说明 |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/Twist` | **底盘速度指令**。`linear.x/y` (m/s), `linear.z` (增量高度 m), `angular.z` (rad/s)。`angular.x/y` 不使用。 |
| `/cmd_pose` | `geometry_msgs/Twist` | **位置增量指令**。`linear.x/y/z` 基于当前位置偏移 (m)。 |
| `/humanoid_mpc_target_pose` | `ocs2_msgs/mpc_target_trajectories` | 躯干 6DoF 位姿轨迹规划。`stateTrajectory.value[]` 顺序: x,y,z,yaw,pitch,roll。 |
| `/humanoid_mpc_mode_schedule` | `ocs2_msgs/mode_schedule` | 步态序列切换。 |
| `/humanoid_mpc_gait_change` | `std_msgs/String` | 步态名称切换（仅支持 `"walk"` / `"stance"`）。 |
| `/humanoid_mpc_stop_step_num` | `std_msgs/Int32` | 从当前步数起第 N 步自动停止。每次只生效一次。 |
| `/humanoid_mpc_foot_pose_target_trajectories` | `kuavo_msgs/footPoseTargetTrajectories` | 单步脚部轨迹控制。 |

### 2.2 服务

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/humanoid_auto_gait` | — | 自动步态切换开关（默认 true）。非零 `/cmd_vel`→walk，全零→stance。 |
| `/humanoid_single_step_control` | `kuavo_msgs/singleStepControl` | 按时间序列+躯干位姿序列控制单步行走。 |
| `/humanoid_get_current_gait_name` | `kuavo_msgs/changeArmCtrlMode` | 获取当前步态名（响应 `gait_name` 字段）。 |

### 核心用法示例

```python
from geometry_msgs.msg import Twist

# 前进 0.1 m/s
twist = Twist()
twist.linear.x = 0.1
cmd_vel_pub.publish(twist)

# 停止
twist.linear.x = 0.0
cmd_vel_pub.publish(twist)

# 降低身高 0.05m
twist.linear.z = -0.05
cmd_vel_pub.publish(twist)
```

---

## 3. 手臂控制

### 3.1 话题

| 接口 | 消息类型 | 说明 |
| --- | --- | --- |
| `/kuavo_arm_traj` | `sensor_msgs/JointState` | **手臂关节轨迹**。`position[]` 前 N/2 为左臂，后 N/2 为右臂（单位: degree）。仅位置生效。 |
| `/kuavo_arm_target_poses` | `kuavo_msgs/armTargetPoses` | 手臂目标轨迹。`times[]` 全局时间递增, `values[]` 每帧关节位置。 |
| `/humanoid_mpc_target_arm` | `ocs2_msgs/mpc_target_trajectories` | 手臂 MPC 规划指令（仅在 `external_control` 模式下生效）。 |

### 3.2 服务

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/humanoid_change_arm_ctrl_mode` | `kuavo_msgs/changeArmCtrlMode` | 切换手臂控制模式: `0`=保持姿势, `1`=自动摆手, `2`=外部控制。 |
| `/humanoid_get_arm_ctrl_mode` | `kuavo_msgs/changeGaitMode` | 获取当前手臂控制模式。 |

### 手臂关节顺序（左右各 7 关节）

```
左臂: l_arm_pitch, l_arm_roll, l_arm_yaw, l_forearm_pitch, l_hand_yaw, l_hand_pitch, l_hand_roll
右臂: r_arm_pitch, r_arm_roll, r_arm_yaw, r_forearm_pitch, r_hand_yaw, r_hand_pitch, r_hand_roll
```

### 核心用法示例

```python
from sensor_msgs.msg import JointState

arm_msg = JointState()
arm_msg.name = [
    "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
    "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
    "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
    "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
]
arm_msg.position = [0.0] * 14  # 14 个关节角度 (degree)
arm_traj_pub.publish(arm_msg)
```

---

## 4. 夹爪控制（二指夹爪）

### 4.1 服务

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/control_robot_leju_claw` | `kuavo_msgs/controlLejuClaw` | **夹爪服务**。`name[]`: `"left_claw"` / `"right_claw"`; `position[]`: 0~100（0=张开, 100=闭合）; `velocity[]`: 0~100 速度; `effort[]`: 电流 A。 |

### 4.2 话题

| 接口 | 消息类型 | 说明 |
| --- | --- | --- |
| `/leju_claw_command` | `kuavo_msgs/lejuClawCommand` | 夹爪命令话题。 |
| `/leju_claw_state` | `kuavo_msgs/lejuClawState` | 夹爪状态。`state[]`: -1=Error, 0=Unknown, 1=Moving, 2=Reached, 3=Grabbed。 |
| `/gripper/state` | `sensor_msgs/JointState` | 仿真底层夹爪关节状态。 |

> ⚠️ 夹爪正在运动时，新请求会被丢弃。

### 核心用法示例

```python
import rospy
from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest

rospy.wait_for_service("/control_robot_leju_claw")
claw_srv = rospy.ServiceProxy("/control_robot_leju_claw", controlLejuClaw)
req = controlLejuClawRequest()
req.data.name = ["left_claw", "right_claw"]
req.data.position = [100.0, 100.0]  # 闭合
req.data.velocity = [50.0, 50.0]
req.data.effort = [1.0, 1.0]
resp = claw_srv(req)
```

---

## 5. 灵巧手控制

> 需要在 `kuavo.json` 中配置 `EndEffectorType` 为 `qiangnao` / `qiangnao_touch` / `revo2`。

### 5.1 话题

| 接口 | 消息类型 | 说明 |
| --- | --- | --- |
| `/control_robot_hand_position` | `kuavo_msgs/robotHandPosition` | 双手手指位置。`left/right_hand_position[6]`: 0~100（0=张开, 100=闭合）。 |
| `/dexhand/command` | `kuavo_msgs/dexhandCommand` | 双手灵巧手命令。`data[12]`: 前 6 左手，后 6 右手。 |
| `/dexhand/left/command` | `kuavo_msgs/dexhandCommand` | 左手灵巧手命令。`data[6]`。 |
| `/dexhand/right/command` | `kuavo_msgs/dexhandCommand` | 右手灵巧手命令。`data[6]`。 |
| `/dexhand/state` | `sensor_msgs/JointState` | 灵巧手状态（12 关节）。 |
| `/dexhand/touch_state` | `kuavo_msgs/dexhandTouchState` | 触觉传感器状态（仅 `qiangnao_touch`）。 |

### 5.2 服务

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/gesture/list` | — | 列出所有预设手势名称。 |
| `/gesture/execute` | — | 抢占式执行预设手势。 |
| `/gesture/execute_state` | — | 查询是否有手势正在执行。 |
| `/dexhand/change_force_level` | `kuavo_msgs/handForceLevel` | 修改灵巧手抓力程度。 |
| `/dexhand/left/enable_touch_sensor` | `kuavo_msgs/controlLejuClaw` | 开启/关闭左手触觉传感器（bitmask）。 |
| `/dexhand/right/enable_touch_sensor` | `kuavo_msgs/controlLejuClaw` | 开启/关闭右手触觉传感器（bitmask）。 |

---

## 6. 头部控制

| 接口 | 消息类型 | 说明 |
| --- | --- | --- |
| `/robot_head_motion_data` | `kuavo_msgs/robotHeadMotionData` | 头部关节控制。`joint_data[0]`: yaw [-30°, 30°], `joint_data[1]`: pitch [-25°, 25°]。 |

---

## 7. 传感器数据

| 接口 | 消息类型 | 说明 |
| --- | --- | --- |
| `/sensors_data_raw` | `kuavo_msgs/sensorsData` | **核心传感器数据**。包含 `sensor_time`、`joint_q`(rad)、`joint_v`(rad/s)、`imu_data`(gyro/acc/quat)、`end_effector_data`。 |
| `/lidar/points` | `sensor_msgs/PointCloud2` | Mid360 点云。 |
| `/livox/lidar` | `livox_ros_driver2/CustomMsg` | Livox 格式点云。 |
| `/lidar_imu` | `sensor_msgs/Imu` | 与 LiDAR 时间对齐的 IMU。 |
| `/tf` / `/tf_static` | `tf2_msgs/TFMessage` | TF 坐标变换树。 |

### `sensors_data_raw` 关节数据顺序

| 范围 | 内容 |
| --- | --- |
| 0~5 | 左腿: l_leg_roll, l_leg_yaw, l_leg_pitch, l_knee, l_foot_pitch, l_foot_roll |
| 6~11 | 右腿: r_leg_roll, r_leg_yaw, r_leg_pitch, r_knee, r_foot_pitch, r_foot_roll |
| 12~18 | 左臂: l_arm_pitch ~ l_hand_roll (7 关节) |
| 19~25 | 右臂: r_arm_pitch ~ r_hand_roll (7 关节) |
| 26~27 | 头部: head_yaw, head_pitch |

---

## 8. 相机接口

> 默认发布 **压缩图像**，降低 CPU 压力。

### 8.1 压缩图像话题

| 接口 | 消息类型 | 说明 |
| --- | --- | --- |
| `/cam_h/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 头部 RGB。 |
| `/cam_l/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 左腕 RGB。 |
| `/cam_r/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 右腕 RGB。 |
| `/cam_h/depth/image_raw/compressedDepth` | `sensor_msgs/CompressedImage` | 头部深度图。 |
| `/cam_l/depth/image_rect_raw/compressedDepth` | `sensor_msgs/CompressedImage` | 左腕深度图。 |
| `/cam_r/depth/image_rect_raw/compressedDepth` | `sensor_msgs/CompressedImage` | 右腕深度图。 |

### 8.2 相机内参

| 接口 | 消息类型 |
| --- | --- |
| `/cam_h/color/camera_info` | `sensor_msgs/CameraInfo` |
| `/cam_l/color/camera_info` | `sensor_msgs/CameraInfo` |
| `/cam_r/color/camera_info` | `sensor_msgs/CameraInfo` |

> 如需 raw 图像: `roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch raw_image:=true`

---

## 9. 状态估计

| 接口 | 消息类型 | 说明 |
| --- | --- | --- |
| `/humanoid_mpc_observation` | `ocs2_msgs/mpc_observation` | MPC 观测（状态向量、控制向量、mode 0~15）。 |
| `/state_estimate/base/pos_xyz` | — | 估计躯干位置 xyz (m)。 |
| `/state_estimate/base/linear_vel` | — | 估计躯干线速度 xyz (m/s)。 |
| `/state_estimate/base/angular_zyx` | — | 估计欧拉角 zyx/ypr (rad)。 |
| `/state_estimate/base/angular_vel_zyx` | — | 估计角速度 zyx (rad/s)。 |
| `/state_estimate/joint/pos` | — | 估计关节位置。 |
| `/state_estimate/joint/vel` | — | 估计关节速度。 |
| `/state_estimate/mode` | `std_msgs/Float64` | 估计步态 mode。 |
| `/odom` | `nav_msgs/Odometry` | 里程计。 |

### MPC 状态向量顺序 (`/humanoid_mpc_observation.state`)

| 索引 | 含义 |
| --- | --- |
| 0~2 | vcom_x, vcom_y, vcom_z |
| 3~5 | L/robotMass (角动量/质量) |
| 6~8 | p_base_x, p_base_y, p_base_z |
| 9~11 | theta_base_z, theta_base_y, theta_base_x |
| 12~ | 关节角度 |

---

## 10. RL 控制器管理

### 10.1 控制器切换服务

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/humanoid_controller/switch_controller` | `kuavo_msgs/switchController` | 切换到指定控制器（`"mpc"` 或 RL 控制器名）。 |
| `/humanoid_controller/get_controller_list` | `kuavo_msgs/getControllerList` | 获取可用控制器列表和当前激活控制器。 |
| `/humanoid_controller/switch_to_next_controller` | `kuavo_msgs/switchToNextController` | 循环切换下一个控制器。 |
| `/humanoid_controller/set_fall_down_state` | `std_srvs/SetBool` | 设置倒地/站立状态。`true`=倒地（自动切倒地起身控制器）。 |

### 10.2 RL 控制器基础服务（按控制器名）

命名空间: `/humanoid_controllers/{controller_name}/`

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `reload` | `std_srvs/Trigger` | 重新加载配置（需非 RUNNING 状态）。 |
| `isActive` | `std_srvs/Trigger` | 查询是否激活。 |
| `getState` | `std_srvs/Trigger` | 获取状态码: 0=INIT, 1=RUNNING, 2=PAUSED, 3=STOPPED。 |
| `getType` | `std_srvs/Trigger` | 获取类型码: 0=MPC, 1=倒地起身, 2=AMP。 |
| `reset` | `std_srvs/Trigger` | 重置内部状态（需非 RUNNING 状态）。 |

### 10.3 倒地起身

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/humanoid_controller/trigger_fall_stand_up` | `std_srvs/Trigger` | 触发倒地起身流程。 |

### 10.4 腰部控制

| 接口 | 消息类型 | 说明 |
| --- | --- | --- |
| `/humanoid_controller/enable_waist_control` | `std_msgs/Bool` | 启用/禁用腰部外部控制。 |
| `/robot_waist_motion_data` | `kuavo_msgs/robotWaistControl` | 腰部关节目标角度。 |

### 10.5 监控话题

| 接口 | 消息类型 | 说明 |
| --- | --- | --- |
| `/humanoid_controller/is_rl_controller_` | `std_msgs/Float64` | 1.0=RL模式, 0.0=MPC模式。 |
| `/humanoid_controller/resetting_mpc_state_` | `std_msgs/Float64` | MPC 重置状态: 0=NORMAL, 1=RESET_INIT, 2=RESET_BASE。 |

---

## 11. 调试与监控

### 11.1 性能

| 接口 | 消息类型 | 说明 |
| --- | --- | --- |
| `/monitor/frequency/mpc` | `std_msgs/Float64` | MPC 频率 (Hz)。 |
| `/monitor/frequency/wbc` | `std_msgs/Float64` | WBC 频率 (Hz)。 |
| `/monitor/time_cost/mpc` | `std_msgs/Float64` | MPC 耗时 (ms)。 |
| `/monitor/time_cost/wbc` | `std_msgs/Float64` | WBC 耗时 (ms)。 |

### 11.2 电机参数

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/hardware/change_motor_param` | `kuavo_msgs/changeMotorParam` | 修改电机 Kp/Kd（高级）。 |
| `/hardware/get_motor_param` | `kuavo_msgs/getMotorParam` | 获取电机参数。 |

---

## 12. 禁止使用的上帝视角接口

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/mujoco/qpos` | topic | MuJoCo 全量状态真值。 |
| `/ground_truth/state` | topic | 真值状态。 |
| `/set_object_position` | service | 内部物体摆放服务。 |
| `/set_object_position_lock` | service | 内部物体摆放锁服务。 |

> ⚠️ 反作弊监控会检测非白名单节点订阅/调用这些接口，触发后杀掉违规节点。

---

## 附录：任务二场景速览

**场景**: 零件分拣归档  
**零件**: 3 类共 6 个

| 类别 | 数量 | 模型 | 目标收纳区 |
| --- | --- | --- | --- |
| A 类 (`part_type_a_*`) | 2 | 钢管紧固件 | `sorting_bin_a` (蓝色) |
| B 类 (`part_type_b_*`) | 2 | T 型接头 | `sorting_bin_b` (橙色) |
| C 类 (`part_type_c_*`) | 2 | 螺丝刀 | `sorting_bin_c` (紫色) |

**核心任务流**: 识别 → 定位 → 抓取 → 搬运 → 放入对应收纳盒

**常用接口优先级**:
1. `/sensors_data_raw` — 传感器数据
2. `/cam_h/color/image_raw/compressed` — 头部相机（视觉识别）
3. `/cmd_vel` — 底盘移动
4. `/kuavo_arm_traj` — 手臂控制
5. `/control_robot_leju_claw` — 夹爪抓取/释放

---

## 参考文档

| 文档 | 内容 |
| --- | --- |
| [运动控制 API](运动控制API.md) | 行走、手臂、头部、夹爪、传感器完整接口 |
| [基础 ROS 话题说明](readme_topics.md) | OCS2 状态/控制向量、核心话题命名 |
| [RL 控制框架接口](RL控制框架ROS接口文档.md) | 控制器切换、状态查询、RL 相关服务 |
| [选手任务模板说明](../src/challenge_cup_task_template/README.md) | `challenge_task.py` 参数和启动说明 |
| [仿真器说明](../src/challenge_cup_simulator/README.md) | launch 参数、场景生成、LiDAR |
| [项目 README](../readme.md) | 整体项目说明与快速启动 |
