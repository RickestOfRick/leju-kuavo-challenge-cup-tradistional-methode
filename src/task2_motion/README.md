# task2_motion：任务二传统抓取与分类放箱

`task2_motion` 将原任务二数采脚本转换为正式任务包，输入统一为 `task2_vision` 发布的 `base_link` 工件位姿。目前使用场景生成器提供的六个工件完整真值。

输入必须包含 `part_type_a_1` 到 `part_type_c_2` 六个完整实例名。`_1/_2` 是场景中的固定对象身份，不按运行时相对位置重新编号。以后替换为 YOLO、深度和 yaw 拟合时，需要先在视觉侧确定完整实例名，并保持 `/task2_vision/objects_json` 的消息结构。

## 模块结构

- `task2_config.py`：所有可调参数，包括局部抓取偏置、交接点、固定放箱关节角、IK、插值和稳定阈值。
- `task2_perception.py`：读取并固定六个 `base_link` 工件位姿。
- `task2_geometry.py`：计算作业手、局部抓取偏置、窄边方向、抓取姿态和交接目标。
- `task2_motion.py`：WBC 控制、软 IK、二分 fallback、关节插值、稳定等待和 FK 误差。
- `task2_scheduler.py`：依次执行六个工件的抓取、必要的换手和分类放箱。
- `task2_node.py`：独立运行入口。

## 控制链路

抓取和交接末端关键点执行：

```text
base_link 目标位姿
→ 双臂 IK，活动臂求解、另一臂保持当前实际 FK
→ 只更新活动臂 7 个关节，另一臂保持上一条命令
→ 使用完整移动时长，以 100 Hz 关节插值到最终 act
→ 等待对应手臂关节速度稳定
→ FK 计算实际末端位姿
→ 绿色 INFO 输出位置和角度误差
```

抓取和交接的稳定等待超时属于软检查，只输出 WARN 并继续。

所有动作的 IK 直接失败时都会尝试当前末端到目标的 `1/2、1/4、1/8` 临时点；全部失败才终止任务。抓取和交接的临时点执行后等待稳定，放箱临时点沿用直接动作语义、不等待稳定。后续 IK 均使用新的实测 q，不使用上一轮理想 IK q。

放箱移动只使用固定关节参考 FK 定义末端目标，然后直接求最终 act IK并执行；不等待稳定，也不做动作完成后的 FK 检查。

启动预位与任务一保持一致：从当前实测关节角开始，经过配置中的 5 个路径点，以 smoothstep 做整段时间缩放，再用 Cardinal 曲线以 100 Hz 连续发布。路径点不是两次独立的关节动作，中间不分别等待稳定。

控制开始时会切换到：

```text
/arm_traj_change_mode = 2
/enable_wbc_arm_trajectory_control = 1
```

即外部轨迹控制和 WBC 手臂轨迹模式。整个任务保持该模式。

## 动作流程

```text
连续 Cardinal 关节预位
→ 抓取点上方 0.10 m
→ 抓取并闭爪
→ 抬升 0.30 m
→ 必要时在中间区域换手
→ 固定关节位姿放入 A/B/C 对应箱
→ 回工作预位
→ 全部完成后回初始位
```

场景二通过统一入口启动时会自动运行本流程：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 8
```

也可以在已经启动的场景二中单独运行：

```bash
rosrun task2_motion task2_node.py --seed 8
```
