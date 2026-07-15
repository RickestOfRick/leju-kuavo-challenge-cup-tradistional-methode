# Task2（Scene2：零件分拣归档）实现进度

> 整体方案链路：**YOLOv8 识别 → 深度图定位（2D→3D） → IK 抓取 → 关节空间放置**
> 
> 设计文档参考：`task2_vision_solution.md`（方案设计） | 本文档（实现进度）

---

## 目录

- [1. 整体 Pipeline](#1-整体-pipeline)
- [2. 已完成模块](#2-已完成模块)
  - [2.1 YOLO 模型训练](#21-yolo-模型训练)
  - [2.2 数据标注工具链](#22-数据标注工具链)
  - [2.3 challenge_task.py 主流程](#23-challenge_taskpy-主流程)
  - [2.4 2D→3D 验证工具](#24-2d3d-验证工具)
- [3. 已知问题](#3-已知问题)
- [4. 待完成工作](#4-待完成工作)
- [5. 文件索引](#5-文件索引)

---

## 1. 整体 Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                     任务二：零件分拣归档（固定站位）                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  【数据准备 - 已完成】                                            │
│  rosbag 录制 → 自动标注脚本 → 人工标注 → 数据增强 → YOLOv8 训练  │
│                                                                  │
│  【推理阶段 - challenge_task.py - 已完成】                         │
│                                                                  │
│  1. 头部相机 RGB  ──→ ONNX YOLOv8 ──→ [类别, cx, cy, confidence] │
│  2. 深度图 5×5 中值采样 ──→ Z（带异常值过滤）                    │
│  3. 内参反投影 ──→ 相机系 3D (cam_frame)                         │
│  4. TF lookup(cam_h → base_link) ──→ base_link 系 3D             │
│  5. TF lookup(world → base_link) ──→ world 系 3D (part_world)    │
│  6. 3轴经验修正 (dX+0.063, dZ-0.042)                             │
│  7. 施加 grasp_offset (零件中心 → 夹爪抓取点)                    │
│                                                                  │
│  8. part_world + 类别 ──→ execute_part_grasp()  ← IK 管线        │
│     ├── IK 求解末端位姿                                          │
│     ├── 预抓取点到位 → 下降 → 夹紧                               │
│     └── measure_hand_pose → 上抬 (Z + LIFT_Z_OFFSET)             │
│                                                                  │
│  9. 关节空间放置 ──→ PLACE_ACTIVE_ARM_JOINTS_DEG[arm][bin]      │
│     └── 松开夹爪 → 回 work_pose                                 │
│                                                                  │
│  循环 6 次: b_1 → b_2 → a_1 → a_2 → c_1 → c_2                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**全程无底盘移动**。抓取用 IK 解算，放置用预标定的关节角（箱子位置固定）。

---

## 2. 已完成模块

### 2.1 YOLO 模型训练

#### 模型信息

| 项目 | 详情 |
| --- | --- |
| 基础模型 | YOLOv8n（轻量级，仿真帧率优先） |
| 分类数 | 3（`part_a` 钢管紧固件 / `part_b` T型接头 / `part_c` 螺丝刀） |
| 训练集 | 54 seeds × 13 variants = 702 张 |
| 验证集 | 6 seeds × 13 variants = 78 张 |
| 输入尺寸 | 640×640 |
| Epochs | 200（patience=30 早停） |
| 部署格式 | PyTorch `.pt` + ONNX (opset 9) |
| 推理后端 | OpenCV DNN（主） / onnxruntime（备） |

#### 模型产物

| 文件 | 说明 |
| --- | --- |
| `weights/yolov8n_augmented.pt` | 训练好的 PyTorch 模型 |
| `weights/yolov8n_augmented_opset9.onnx` | ONNX 导出（无 ultralytics 依赖，直接用 OpenCV 跑） |
| `runs/train/augmented_yolov8n/` | 训练日志、checkpoint、mAP 曲线 |

#### 训练脚本

`tools/train_yolov8n.py` —— 一键训练 + 数据集切分：
- **Seed 级切分**：同一 seed 的 13 个增强变体（1 原图 + 12 增强）保持在同一个 split 中，防止数据泄露
- 支持 YOLO 内置增强（mosaic、mixup、HSV 抖动等）

### 2.2 数据标注工具链

#### 工具概览

| 工具 | 文件 | 功能 |
| --- | --- | --- |
| 🏷️ 自动标注 | `tools/scene2_auto_labeler.py` | rosbag + layout YAML → 碰撞盒投影 → YOLO 格式 |
| ✏️ 人工标注 | `tools/manual_labeler.py` | OpenCV 交互标注（拖拽画框，0/1/2 切换类别） |
| 👁️ 标注审核 | `tools/scene2_label_visualizer.py` | 交互浏览 + HTML 图库导出 + 统计 |
| 🔄 数据增强 | `tools/augment_dataset.py` | 12 种 bbox-safe 增强 → 60→780 张（13×） |
| 📸 帧提取 | `tools/extract_bag_frames.py` | rosbag → 图片序列（支持跳帧） |

#### 自动标注原理

```
仿真 GT (generated_layouts YAML)
  + 碰撞盒半边长 (视觉包围盒)
  → 8 角点世界坐标
  → 相机外参变换 → 相机坐标系
  → 内参 K 投影 → 2D bbox (min/max uv)
  → YOLO 归一化格式 (class cx cy w h)
```

**碰撞盒配置**（视觉包围盒，非物理碰撞体，已调大以保证 bbox 覆盖完整零件）：

| 类别 | class_id | 视觉包围盒半边长 (m) |
| --- | --- | --- |
| part_a 钢管紧固件 | 0 | `[0.015, 0.025, 0.010]` |
| part_b T型接头 | 1 | `[0.025, 0.030, 0.025]` |
| part_c 螺丝刀 | 2 | `[0.080, 0.015, 0.015]` |

#### 12 种数据增强 Pipeline

1. `hsv_jitter` — HSV 抖动（模拟光照变化）
2. `brightness_contrast` — 亮度/对比度
3. `rgb_shift` — RGB 通道偏移（色温）
4. `gaussian_blur` — 高斯模糊（景深）
5. `motion_blur` — 运动模糊（相机抖动）
6. `gauss_noise` — 高斯噪声（传感器噪声）
7. `hflip` — 水平翻转
8. `rotate_mild` — ±5° 旋转 + 轻微缩放
9. `rotate_moderate` — ±10° 旋转 + 缩放
10. `perspective` — 轻微透视变换
11. `combo_mild` — HSV + Brightness + Blur 组合
12. `combo_heavy` — HSV + Blur + Noise + Rotate 组合

#### 数据集目录结构

```
manual_dataset/           ← 原始标注数据
├── images/               (60 张)
└── labels/

manual_dataset_augmented/ ← 增强后数据
├── images/               (780 张: 60 × 13)
└── labels/

augmented_yolo_dataset/   ← 训练用切分数据
├── dataset.yaml
├── train/
│   ├── images/           (702 张)
│   └── labels/
└── val/
    ├── images/           (78 张)
    └── labels/
```

### 2.3 challenge_task.py 主流程

**文件**: `src/challenge_cup_task_template/scripts/challenge_task.py`

#### 核心架构

```
main() → run_scene("scene2") → _run_scene2_sorting()
```

#### ONNX 检测器 (`ONNXDetector`)

- **零额外 Python 依赖**：纯 OpenCV DNN 推理（回退方案：onnxruntime）
- 推理流程：resize + letterbox → blob → forward → 后处理（per-class NMS）
- 参数：`imgsz=640`, `conf_thresh=0.25`, `iou_thresh=0.45`（NMS）

#### 深度图解码 (`_get_head_depth`)

处理 `/cam_h/depth/image_raw/compressedDepth` 的三阶段解码：
1. 剥离 PNG 签名前的垃圾字节
2. `cv_bridge` → OpenCV → PIL 三级回退解析
3. uint16 → float32 米制转换（×0.001）

#### 2D → 3D 核心 (`_pixel_to_world`)

```python
# 1. 5×5 窗口中值深度采样（抗噪，过滤 >3m 异常值）
Z = median(depth_img[py-2:py+3, px-2:px+3])  # 排除 Z≤0 和 Z>3.0

# 2. 内参反投影 → 相机坐标系
X_cam = (cx - K[0,2]) * Z / K[0,0]
Y_cam = (cy - K[1,2]) * Z / K[1,1]
P_cam = [X_cam, Y_cam, Z]

# 3. TF 变换 → base_link
# 按优先级尝试 4 个 TF frame（URDF 光学 frame → link → 静态 Head Camera View）
# Head Camera View: 在 TF 树中是经过校准的静态 frame，直接对应 MuJoCo 相机姿态
P_base = T_cam_to_base @ [P_cam, 1]
```

**TF frame 优先级**（故意把 URDF frame 放前面，使其 fallthrough 到校准好的 Head Camera View）：
1. `cam_h_color_optical_frame`（URDF, Z 轴水平——会 fallthrough）
2. `cam_h_color_frame`（URDF——会 fallthrough）
3. `cam_h_link`（URDF——会 fallthrough）
4. **`Head Camera View`**（静态校准 frame，Z 轴朝下看桌面——实际命中）

#### 3 轴经验修正

通过 `run_debug_2d3d.py` 在 3 个 seed 上校准得出：

| 轴 | 修正值 | 说明 |
| --- | --- | --- |
| dX | **+0.063 m** | 估计值比 GT 靠后 6.3cm → 向前推 |
| dY | 0 | 可忽略 |
| dZ | **-0.042 m** | 估计值比 GT 高 4.2cm → 向下压 |

#### 分拣顺序

```python
SORT_ORDER = [
    (1, "right", "sorting_bin_b"),   # b_1 → 中箱，右手
    (1, "right", "sorting_bin_b"),   # b_2
    (0, "right", "sorting_bin_a"),   # a_1 → 右箱，右手
    (0, "right", "sorting_bin_a"),   # a_2
    (2, "left",  "sorting_bin_c"),   # c_1 → 左箱，左手
    (2, "left",  "sorting_bin_c"),   # c_2
]
```

#### 单次抓取-放置流程（每步）

```
1. 视觉检测（最多 15 次重试）
   ├── _get_head_rgb() + _get_head_depth()  ← 同一仿真帧
   └── detector.detect(img) → 筛选目标类别

2. 2D → 3D
   ├── _pixel_to_world(cx, cy, depth, K, tf_buffer) → part_in_base
   ├── TF world→base_link → part_world
   └── 施加 3轴修正 (dX+0.063, dZ-0.042)

3. 施加抓取点偏移
   └── _apply_grasp_offset_to_estimated(part_world, object_name, arm)

4. IK 抓取（复用现有管线）
   ├── execute_part_grasp(grasp_runtime, object_name, part_world, ...)
   │   ├── IK 求解 → 预抓取位姿
   │   ├── 移动到预抓取点
   │   ├── 下降到抓取深度
   │   ├── 闭合夹爪
   │   └── 返回 success

5. 上抬
   ├── measure_hand_pose() → 当前末端位姿
   └── move_arm_ik_once(lift_target = ee_pos + [0, 0, LIFT_Z_OFFSET])

6. 放置（关节空间）
   ├── PLACE_ACTIVE_ARM_JOINTS_DEG[arm][bin]
   └── _move_arm_to(joints) → 松开夹爪 → 等待 PLACE_DWELL

7. 回 work_pose
   └── _move_to_work_pose_joints(arm_hold)
```

#### IK 管线集成

`challenge_task.py` 通过动态 import 复用 `scene2_part_grasp_ik.py` 中的 `GraspRuntime` 和 `execute_part_grasp()`：

```python
# 运行时动态导入 IK/grasp 模块
_GRASP_IK_DIR = "src/challenge_cup_simulator/test/collect_scene2_dataset"
from scene2_part_grasp_ik import (
    GraspRuntime, execute_part_grasp, get_object_arm,
    get_object_lift_quat_xyzw, measure_hand_pose, move_arm_ik_once,
)
```

`GraspRuntime` 通过回调函数注入 ROS 接口（关节读取、运动执行、夹爪控制），实现与 pipeline 的解耦。

#### 可视化 Worker (`_VizWorker`)

- 独立线程，~10Hz 持续采集头部相机帧
- YOLO 检测 + 绘制 bbox → 发布到 `/scene2/yolo_detections` (ROS Image)
- 首帧和第 5N 帧保存到 `viz_output/` 目录

#### 基础设施函数（复刻 pipeline 内部）

| 函数 | 功能 |
| --- | --- |
| `_set_arm_mode()` | 切换手臂控制模式（外部控制/自动摆动） |
| `_publish_head_target()` | 设置头部相机角度（低头看桌面） |
| `GripperCommandHold` | 夹爪持续指令发布器（100Hz 线程） |
| `ArmTrajHold` | 手臂关节持续指令发布器（100Hz 线程） |
| `_publish_arm_traj_interpolation()` | 关节空间插值运动 |
| `_move_arm_to()` | 带 locked_other_arm 的关节空间移动 |
| `_enter_work_pose()` | 侧抬 → 工作姿态序列 |

#### 关键 ROS 接口

| 话题/服务 | 用途 |
| --- | --- |
| `/cam_h/color/image_raw/compressed` | RGB 图像输入 |
| `/cam_h/depth/image_raw/compressedDepth` | 深度图输入 |
| `/cam_h/color/camera_info` | 相机内参 K |
| `/tf` | TF 树（cam_h → base_link → world） |
| `/ik/two_arm_hand_pose_cmd_srv` | IK 求解服务 |
| `/kuavo_arm_traj` | 双臂关节角控制 |
| `/gripper/command` | 夹爪开合 |
| `/robot_head_motion_data` | 头部转动 |
| `/arm_traj_change_mode` | 手臂模式切换 |
| `/scene2/yolo_detections` | 可视化输出（YOLO 检测结果图） |

### 2.4 2D→3D 验证工具

**文件**: `src/challenge_cup_simulator/test/run_debug_2d3d.py`  
**启动**: `bash debug_2d3d.sh [seed] [frames]`

#### 功能

一键启动指定 seed 的 scene2 仿真，连续采集 N 帧，对比 YOLO 检测 → 深度投影（**无经验修正**）的估计值与 layout YAML 真值，输出：

- 每个零件的 `dX/dY/dZ` 误差统计（mean ± std）
- 3D Euclidean 综合误差
- 分轴独立分析（mean/std/range）
- 解读指南（mean = 系统偏差，std = 随机噪声）

#### 典型输出

```
2D→3D 投影精度统计（纯深度投影，无经验修正）
==========================================================================================
零件             帧    dX mean   ±std   dY mean   ±std   dZ mean   ±std   Euclid
------------------------------------------------------------------------------------------
part_type_a_1     15   -0.0630  0.005   -0.0030  0.003    0.0420  0.004    0.0750
...
GLOBAL            90   -0.0630  0.005   -0.0030  0.003    0.0420  0.004    0.0750
```

→ 由此得出 3 轴经验修正值 `dX=+0.063`, `dZ=-0.042`。

#### 结果分析归档

`docs/2d3d_optimization_plan.md` 记录了调试过程中的**根因分析和 6 步优化实验**，关键发现：

- TF 误差主要是**纯平移**（~7cm），非旋转
- **IK 不可达**是真正的瓶颈，而非位置估计精度
- 实验了 `solvePnP`、多点采样均值、不同 TF frame 等多种方案

---

## 3. 已知问题

### 3.1 IK 抓取部分有问题

> 用户原话："后面的抓取ik解算参考scene2数据采集,但是目前有问题"

**当前状态**：`challenge_task.py` 中的 IK 抓取代码已完整集成 `execute_part_grasp()` 调用，但实际运行时存在问题。具体情况需要进一步调试确认。

**可能原因**：

1. **深度估计精度**：虽然 3 轴校正提高了整体精度，但单帧深度噪声可能导致 `part_world` 偏差较大，IK 求解器无法收敛
2. **抓取姿态 mismatch**：`get_object_lift_quat_xyzw()` 返回的抓取四元数可能与 YOLO 估计的零件位置不匹配
3. **`scene2_part_grasp_ik.py` 依赖**：该模块原本设计用于 `scene2_data_collection_pipeline.py`（已知 GT 位置），在视觉估计位置上可能精度不足
4. **碰撞盒 vs 视觉中心偏移**：YOLO 检测的是视觉中心，而 grasp_offset 参考的是零件物理中心（碰撞盒中心），两者可能存在系统偏差

### 3.2 场景初始化偶发失败

`scene2.log` 显示某次运行时出现：

```
[ERROR] challenge_sim_launcher: 初始化 part_type_a_1 失败: set_object_position is locked
[ERROR] challenge_sim_launcher: 初始化 part_type_a_2 失败: set_object_position is locked
[FATAL] challenge_sim_launcher: 部分物体初始化失败（场景 scene2 初始化失败，禁止启动）
```

这是仿真环境的偶发问题（`set_object_position` 服务被锁），非代码 bug。重试即可。

### 3.3 深度图解码可靠性

`_get_head_depth()` 使用三级回退解析（cv_bridge → OpenCV → PIL），在边缘情况下可能解码失败。当前已有异常处理，但需关注长时间运行中的稳定性。

### 3.4 设计说明：无需 Handoff（交接）

`scene2_data_collection_pipeline.py` 原有 handoff 逻辑（左手误抓右手零件时交接给右手，反之亦然）。在视觉 pipeline 中，分拣顺序 `SORT_ORDER` 已设计为 B(右)→B(右)→A(右)→A(右)→C(左)→C(左)，**同类零件用同一只手**，因此不需要 handoff。这是设计简化，非功能缺失。

---

## 4. 待完成工作

| 优先级 | 任务 | 说明 |
| --- | --- | --- |
| 🔴 P0 | IK 抓取调试 | 定位 `execute_part_grasp()` 失败原因，修复 IK 求解/抓取执行 |
| 🔴 P0 | 端到端联调 | 确保 6 零件分拣在多种 seed 下完整通过 |
| 🟡 P1 | 多帧滤波 | 用多帧检测结果的均值/中值降低 3D 估计噪声 |
| 🟡 P1 | 抓取失败恢复 | 检测到 grasp 失败后的重试/跳过策略 |
| 🟡 P1 | YOLO 模型迭代 | 更多训练数据、更大模型（如 yolov8s）提升精度 |
| 🟢 P2 | 不确定性可视化 | RViz 中显示检测的不确定性/置信度 |
| 🟢 P2 | Scene1/Scene3 实现 | 其他两个场景的视觉+控制管线 |

---

## 5. 文件索引

### 核心入口

| 文件 | 说明 |
| --- | --- |
| `src/challenge_cup_task_template/scripts/challenge_task.py` | 三场景统一入口，scene2 主流程（1139 行） |
| `start_scene2.sh` | 一键启动脚本（自动管理 Docker 容器） |

### 数据工具

| 文件 | 说明 |
| --- | --- |
| `tools/scene2_auto_labeler.py` | 自动标注：bag + layout → YOLO 数据集 |
| `tools/manual_labeler.py` | 人工标注：OpenCV 交互工具 |
| `tools/scene2_label_visualizer.py` | 标注审核：浏览 + HTML 图库 |
| `tools/augment_dataset.py` | 数据增强：12 种 pipeline，13× 扩增 |
| `tools/extract_bag_frames.py` | 帧提取：bag → 图片 |

### 训练

| 文件 | 说明 |
| --- | --- |
| `tools/train_yolov8n.py` | YOLOv8n 训练脚本 |
| `weights/yolov8n_augmented.pt` | 训练好的 PyTorch 模型 |
| `weights/yolov8n_augmented_opset9.onnx` | ONNX 部署模型 |

### 调试

| 文件 | 说明 |
| --- | --- |
| `src/challenge_cup_simulator/test/run_debug_2d3d.py` | 2D→3D 精度一键验证 |
| `debug_2d3d.sh` | debug 工具 shell 包装 |

### 配置

| 文件 | 说明 |
| --- | --- |
| `tools/head_camera_extrinsics.yaml` | 头部相机外参校准 |
| `augmented_yolo_dataset/dataset.yaml` | YOLO 训练数据集配置 |
| `docs/2d3d_optimization_plan.md` | 2D→3D 优化实验记录（根因分析 + 6 步实验） |
| `src/challenge_cup_simulator/config/scenes/scene2.yaml` | Scene2 MuJoCo 场景定义 |

### 校准工具

| 文件 | 说明 |
| --- | --- |
| `tools/calibrate_extrinsics_v2.py` | 相机外参全局优化校准 |
| `tools/calibration_python/scripts/verify_fk_ik_offline.py` | FK/IK 离线验证 |
| `tools/calibration_python/scripts/ik_debug_single_point.py` | 单点 IK 调试 |

### 数据集

| 目录 | 说明 |
| --- | --- |
| `manual_dataset/` | 原始人工标注（~60 张） |
| `manual_dataset_augmented/` | 增强后数据（780 张） |
| `augmented_yolo_dataset/` | 训练/验证切分数据 |
| `datasets/scene2_yolo/` | 自动标注输出目录 |
| `datasets/scene2_debug/` | 调试用数据集 |

### 依赖模块（来自 scene2 数采 pipeline）

| 文件（import） | 说明 |
| --- | --- |
| `scene2_part_grasp_ik.py` | GraspRuntime + execute_part_grasp + IK 求解 |
| `scene2_data_collection_pipeline.py` | 原有的数采管线（提供常量/配置参考） |
| `challenge_sim_launcher.py` | 仿真启动器（场景初始化 + 反作弊） |

---

*最后更新：2026-07-15*
