# task2_motion：任务二传统抓取与分类放箱

`task2_motion` 将原任务二数采脚本转换为正式任务包，输入统一为 `task2_vision` 发布的 `base_link` 工件位姿。目前使用场景生成器提供的六个工件完整真值。

输入必须包含 `part_type_a_1` 到 `part_type_c_2` 六个 motion 实例名。编号由
`task2_vision` 根据 `base_link` 下的位置生成：A、B 类从右到左编号，C 类
从左到右编号。motion 使用该名字选择实例偏置；真值模式另以
`ground_truth_name` 和顶层 `motion_to_ground_truth` 保存 `.so` 原始实例名，
从而确保重新编号后仍读取对应真值对象的位置、姿态和 yaw。以后替换为
YOLO、深度和 yaw 拟合时，应直接输出相同的 motion 空间编号。

## 模块结构

- `task2_config.py`：所有可调参数，包括 base_link 抓取偏置、按 `_1/_2` 设置的放箱 Y 偏置、交接点、固定放箱关节角、IK、插值和稳定阈值。
- `task2_perception.py`：读取并固定六个 `base_link` 工件位姿。
- `task2_geometry.py`：计算作业手、base_link 抓取偏置、窄边方向、抓取姿态和交接目标。
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

`HANDOFF_MODE_BY_CLASS` 为 A/C 独立选择交接链，默认 A 使用
`source_fk_delay`，C 使用 `vision_relocalize`：

- `source_fk_delay` 读取 `HANDOFF_SOURCE_FK_RELATIVE_CONFIG`。给手到绝对
  `pre_place → place` 后读取稳定 FK，接手 pre/grasp 是相对该 FK XYZ 的偏移。
  `HANDOFF_RECEIVER_CLOSE_DELAY_BY_CLASS` 表示
  `接手 close 时刻 - 给手 open 时刻`；正数先松给手，负数先闭接手。差值由
  `/sensors_data_raw.sensor_time` 计时。该链抓取后跳过最终 lift；C 若切换到此链，
  只保留第一段小 lift。
- `vision_relocalize` 读取 `HANDOFF_CONFIG`。给手先执行正常完整 lift，再依次到
  配置中的绝对 `base_link pre_place_position → place_position`，稳定后释放，
  经 pre 撤离，随后双臂都回工作预位并稳定；只有此时才发起重定位请求。
  配置中没有固定接手 `grasp_position`；视觉节点对
  已释放的指定工件重新进行 360° RGB-D/CAD XY/yaw 搜索。拟合位置叠加该工件
  原有抓取偏置，拟合 yaw 与 `grasp_pose` 模板组合后交给接手，接手重新抓取并
  完整 lift。重定位只更新指定工件，其余五件保持此前动作锁定数据。

`HANDOFF_CONFIG` 的 pre/place 是不叠加 `HAND_EE_CALIBRATION` 的最终绝对坐标；
`HANDOFF_SOURCE_FK_RELATIVE_CONFIG` 保持原有标定和相对偏移语义。

A/B 放箱直接求最终 IK。C 类从放箱前实测 q 的当前 FK 到 release 做可配置的
笛卡尔直线等分；默认 8 段，在 `1/8..7/8` 处依次求 7 个中间 IK q 约束，
再求最终 q。所有 q 一次性预规划并连续插值，中间约束不是动作关键点，只有最终
release 等待稳定并检查 FK。

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
→ 抓取点上方 0.06 m
→ 抓取并闭爪
→ A/B 抬升 0.32 m；C 先 X-0.05/Z+0.10 m，再到 X-0.05/Z+0.34 m
→ 必要时在中间区域换手
→ A/B 直接放箱；C 经 8 等分 IK q 约束连续放箱
→ 回工作预位
→ 全部完成后回初始位
```

B 类使用 `0°/180°` 两个等价抓取 yaw，使夹爪从 T 横梁左右两侧形成 `|T|`
夹持；A/C 保持原来的 `+90°/-90°` 候选。A/B/C 的 pre-grasp FK 姿态误差
超过 `PRE_GRASP_ORIENTATION_RETRY_DEG` 时，都会实测另一个 180° 候选并保留
误差较小者。

场景二通过统一入口启动时会自动运行本流程：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 8
```

也可以在已经启动的场景二中单独运行：

```bash
rosrun task2_motion task2_node.py --seed 8
```
