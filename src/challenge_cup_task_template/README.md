# 挑战杯仿真赛选手代码提交说明

`challenge_cup_task_template` 是选手任务代码包。参赛队伍应基于本包开发算法逻辑，并在官方 Docker 仿真环境中完成自测。

正式评测时，组委会会将提交的功能包放入工作空间 `src/` 目录，编译后运行统一入口：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed <评测种子>
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed <评测种子>
rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed <评测种子>
```

## 开发入口

选手主要修改：

```text
challenge_cup_task_template/
├── CMakeLists.txt
├── package.xml
├── README.md
└── scripts/
    └── challenge_task.py
```

要求：

- 功能包名称保持 `challenge_cup_task_template` 不变；
- 入口脚本保持 `scripts/challenge_task.py` 不变；
- 三个场景统一由 `--scene` 参数选择；
- 可以在本包内新增 `src/`、`launch/`、`config/` 等辅助文件；
- 如新增第三方依赖，必须在提交包的 README 中写明安装方式和用途。

`challenge_task.py` 中已经提供场景分支位置，可按需实现：

```python
if scene == "scene1":
    pass  # 场景一：包裹称重与摆放
elif scene == "scene2":
    pass  # 场景二：分拣归档
elif scene == "scene3":
    pass  # 场景三：SMT 料盘出库
```

## 本地运行

编译并 source 工作空间后运行：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
```

`--seed` 用于本地生成不同场景实例。正式评测 seed 由组委会指定，选手不应依赖某个固定 seed 或硬编码物体位置。

## 场景一 YOLO 识别

场景一支持用内嵌 YOLO 推理识别快递包裹。YOLO 只负责输出包裹检测框，三维位置仍由相机深度图、CameraInfo 和 TF 转换得到，不读取仿真真值或禁用接口。

默认权重路径：

```text
src/challenge_cup_task_template/models/scene1_parcel.pt
```

权重文件需参赛队自行训练或提供，模板中不包含 `.pt` 模型。模型建议至少包含一个包裹类别；如果类别名中包含 `parcel`，程序只接受该类别；如果模型只有一个类别，程序会将其统一映射为 `parcel`。

在官方 Docker 容器中安装依赖：

```bash
cd /root/kuavo_ws
python3 -m pip install ultralytics==8.0.196
python3 -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```

运行示例：

```bash
# 自动模式：有模型且 ultralytics 可用时使用 YOLO，否则回退规则识别
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --scene1-perception auto

# 强制 YOLO；模型或依赖不可用时会直接报错并返回无目标
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --scene1-perception yolo

# 指定自定义模型和 GPU
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --scene1-perception yolo --yolo-model /root/kuavo_ws/src/challenge_cup_task_template/models/scene1_parcel.pt --yolo-device cuda --yolo-conf 0.45 --yolo-imgsz 640

# 无 GPU 或调试 CPU 路径
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --scene1-perception yolo --yolo-device cpu

# 完全使用原 HSV 规则识别
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --scene1-perception rule
```

可用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--scene1-perception` | `auto` | `auto` 优先 YOLO 并可回退；`yolo` 强制 YOLO；`rule` 强制规则识别。 |
| `--yolo-model` | `src/challenge_cup_task_template/models/scene1_parcel.pt` | YOLO `.pt` 权重路径。 |
| `--yolo-device` | `auto` | 可选 `auto`、`cpu`、`cuda`、`cuda:0`。 |
| `--yolo-conf` | `0.45` | YOLO 检测置信度阈值。 |
| `--yolo-imgsz` | `640` | YOLO 推理输入尺寸。 |

## 场景二分拣实现

场景二现在按“识别-定位-抓取-归档”链路执行：优先使用头部相机 YOLOv8 segmentation 识别三类零件；若没有权重或 `ultralytics` 不可用，会自动回退到 HSV 规则检测。两种路径都只使用 RGB-D、CameraInfo 和 TF 计算 `base_link` 下的三维位置，不读取仿真真值。

类别映射：

| 识别类别 | 兼容别名 | 目标箱 |
| --- | --- | --- |
| `part_type_a` | `pipe_fitting`, `steel_pipe_fasteners` | `sorting_bin_a` 蓝色箱 |
| `part_type_b` | `pipe_clamp`, `t_junction` | `sorting_bin_b` 橙色箱 |
| `part_type_c` | `screwdriver` | `sorting_bin_c` 紫色箱 |

推荐权重路径：

```text
src/challenge_cup_task_template/models/yolo/best.pt
```

程序也会自动查找：

```text
models/yolo/best.pt
models/yolo/yolov8n_seg_scene2_demo.pt
src/challenge_cup_task_template/models/yolo/yolov8n_seg_scene2_demo.pt
runs/**/weights/best.pt
```

运行示例：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
```

执行前会尝试连续检测 6 个零件，并要求连续帧位置变化小于 3cm；如果视觉完全失败，才启用公开几何兜底点，便于调试 IK 和夹爪流程。

## 比赛计时

默认不限制时长，只显示计时，便于调试。

设置 `--time-limit` 后，到达时限会自动结束当前任务节点。单位是秒：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --time-limit 120
```

如不需要弹出计时窗口，可加：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --time-limit 120 --no-timer-gui
```

计时窗口中的 `Stop Timer` 用于任务完成后停止计时显示，便于裁判查看用时。

## 常用接口

以下接口可作为开发起点。完整实物接口文档可参考官方说明，但仿真比赛最终以当前仿真环境中的话题、服务和消息定义为准：

https://kuavo.lejurobot.com/manual/basic_usage/kuavo-ros-control/docs/4%E5%BC%80%E5%8F%91%E6%8E%A5%E5%8F%A3/%E6%8E%A5%E5%8F%A3%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3/

| 接口 | 类型 | 用途 |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/Twist` | 底盘/步态速度控制 |
| `/kuavo_arm_traj` | `sensor_msgs/JointState` | 双臂关节轨迹命令 |
| `/sensors_data_raw` | `kuavo_msgs/sensorsData` | 机器人传感器数据 |
| `/lidar/points` | `sensor_msgs/PointCloud2` | 激光雷达点云 |
| `/control_robot_leju_claw` | `kuavo_msgs/controlLejuClaw` 服务 | 仿真夹爪控制 |
| `/leju_claw_command` | `kuavo_msgs/lejuClawCommand` | 夹爪命令话题 |
| `/leju_claw_state` | `kuavo_msgs/lejuClawState` | 夹爪状态话题 |

建议在容器内用下面命令核对接口：

```bash
rostopic list
rosservice list
rosmsg show kuavo_msgs/sensorsData
rossrv show kuavo_msgs/controlLejuClaw
```

## 提交内容

提交附件外层目录建议使用：

```text
参赛团队名称+挑战杯仿真赛/
```

具体命名以赛事通知为准。目录内至少包含固定功能包 `challenge_cup_task_template`：

```text
参赛团队名称+挑战杯仿真赛/
└── challenge_cup_task_template/
    ├── CMakeLists.txt
    ├── package.xml
    ├── README.md
    └── scripts/
        └── challenge_task.py
```

如果只使用官方仿真环境自带控制器，只提交 `challenge_cup_task_template` 即可。

如果修改了控制器或其他功能包，需要同时提交被修改的功能包，并保持原功能包名称不变：

```text
参赛团队名称+挑战杯仿真赛/
├── challenge_cup_task_template/
│   └── ...
└── <被修改的功能包>/
    └── ...
```

同时需要在 README 中说明：

- 修改了哪些功能包；
- 修改目的和主要内容；
- 编译方式；
- 运行方式；
- 是否需要额外依赖。

## 严禁事项

机器人必须通过自身传感器和控制接口完成任务。以下行为属于违规，可能导致对应场景成绩无效：

1. 直接读取仿真真值或物体绝对坐标；
2. 调用物体摆放、场景重置等非选手接口；
3. 修改仿真场景、模型、评分或启动相关文件；
4. 通过非物理方式移动机器人或物体；
5. 干预仿真运行状态，例如暂停、加速、跳步；
6. 依赖人工运行中干预完成任务。

请不要订阅或调用比赛禁用的上帝视角接口，例如 `/mujoco/qpos`、`/ground_truth/state` 以及物体摆放相关服务。

## 提交前检查

提交前建议逐项确认：

- `challenge_cup_task_template` 包名未修改；
- `scripts/challenge_task.py` 入口文件存在且可执行；
- 三个场景均能通过 `--scene scene1/scene2/scene3` 启动；
- 代码能在官方 Docker 环境中编译和运行；
- 没有提交 `build/`、`devel/`、`log/`、rosbag、缓存文件或大体积临时数据；
- 如修改其他功能包，已在 README 中说明修改内容和运行方式。
