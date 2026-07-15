# 任务二（场景二：零件分拣归档）— 传统视觉方案

> 方案链路：**YOLOv8 识别 → 深度图定位 → 复用现有 IK 抓取管线 → 关节空间放置**
>
> ⚠️ 关键发现：机器人站立位置固定，6 个零件和 3 个盒子均在双臂可达范围内，**全程不需要底盘移动**。现有 `scene2_data_collection_pipeline.py` 从不调用 `/cmd_vel`。

---

## 目录

- [1. 整体 Pipeline](#1-整体-pipeline)
- [2. 与现有 IK 管线的集成点](#2-与现有-ik-管线的集成点)
- [3. 自动化数据标注](#3-自动化数据标注)
- [4. YOLO 模型训练](#4-yolo-模型训练)
- [5. 推理与抓取（challenge_task.py 实现）](#5-推理与抓取challenge_taskpy-实现)
- [6. 已知资源与缺口](#6-已知资源与缺口)

---

## 1. 整体 Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                     任务二：零件分拣归档（固定站位）                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  【数据准备】                                                     │
│  bag → 自动标注脚本 → YOLOv8 训练                                 │
│                                                                  │
│  【推理阶段 - challenge_task.py】                                  │
│                                                                  │
│  1. 头部相机 RGB  ──→ YOLOv8 ──→ [类别, cx, cy]                 │
│  2. 深度图对应像素 ──→ Z                                         │
│  3. 内参反投影 ──→ 相机系 3D                                     │
│  4. TF ──→ base_link 系 3D (world_xyz)                          │
│                                                                  │
│  5. world_xyz + 类别 ──→ execute_part_grasp()   ← 复用 IK 管线   │
│     │                      (与数采管线完全相同的抓取逻辑)           │
│     ├── ik 求解抓取姿态                                          │
│     ├── 预抓取点到位 → 下降 → 夹紧                               │
│     └── 上抬 (Z + LIFT_Z_OFFSET)                                │
│                                                                  │
│  6. 关节空间放置 ──→ PLACE_ACTIVE_ARM_JOINTS_DEG[arm][bin]      │
│     └── 松开夹爪 → 回 work_pose                                 │
│                                                                  │
│  循环 6 次: b_1 → b_2 → a_1 → a_2 → c_1 → c_2                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**全程无底盘移动**。抓取用 IK 解算，放置用预标定的关节角（箱子位置固定）。

---

## 2. 与现有 IK 管线的集成点

### 2.1 现有管线如何工作

`scene2_data_collection_pipeline.py` 的抓取流程（简化）：

```python
# 现有：从 challenge_secret / generated_layouts 读取 hardcoded world_xyz
world_xyz = get_object_world_xyz("part_type_b_1", active_arm="right")

# 现有：调用 IK 管线执行抓取
execute_part_grasp(
    grasp_runtime,
    object_name="part_type_b_1",
    world_xyz=world_xyz,        # ← 唯一需要替换的输入
    active_arm="right",
    locked_other_arm_joints=...,
)
```

### 2.2 我们的修改

**只替换 `world_xyz` 的来源**：从硬编码布局 → YOLO + 深度图推导。其余 IK 求解、预抓取、夹取、上抬逻辑完全不动。

```python
# 修改后：YOLO + 深度图 → world_xyz
img = get_head_camera_image()
detections = yolo_model(img)              # YOLOv8 推理
target = pick_next_target(detections)     # 按分拣顺序选目标
world_xyz = detect_to_world_xyz(target, depth_img, K, tf_listener)

# ↓ 以下与现有管线完全相同 ↓
execute_part_grasp(grasp_runtime, target.name, world_xyz,
                   active_arm=target.arm, ...)
```

### 2.3 放置无需修改

放置用预标定的关节角，箱子位置固定，不受 YOLO 影响：

```python
# 直接复用 scene2_data_collection_pipeline.py 中的配置
PLACE_ACTIVE_ARM_JOINTS_DEG = {
    "right": {
        "sorting_bin_a": [-30, -10, -10, -80, 70, 0, 0],
        "sorting_bin_b": [-30,  0,  30, -80, 70, 0, 0],
    },
    "left": {
        "sorting_bin_b": [-30, 0, -30, -80, -70, 0, 0],
        "sorting_bin_c": [-30, 10, 10, -80, -70, 0, 0],
    },
}
```

### 2.4 需要新增的模块

| 模块 | 说明 |
| --- | --- |
| `detect_to_world_xyz()` | YOLO 2D + 深度 → base_link 3D |
| `pick_next_target()` | 按分拣顺序 + YOLO 检测结果选择当前目标 |
| `class_to_part_name()` | YOLO class_id → `"part_type_a"` 等，用于 IK 查询 |

其余全部复用 `scene2_part_grasp_ik.py`。

---

## 3. 自动化数据标注

### 3.1 输入

**rosbag 文件**（`.bag`），已有管线自动录制。

### 3.2 bag 录制的话题

| 话题 | 用途 |
| --- | --- |
| `/cam_h/color/image_raw/compressed` | YOLO 训练图像 |
| `/cam_h/color/camera_info` | 相机内参 K |
| `/tf` + `/tf_static` | 坐标变换 |

### 3.3 标注生成原理

仿真 ground truth → **零件 3D 碰撞盒投影到 2D 图像**：

```
零件 world 位姿 (generated_layouts YAML / challenge_secret)
  + collision_size (scene2.yaml 碰撞盒半边长)
  → 8 个角点世界坐标
  → TF 转到相机坐标系
  → 内参 K 投影到像素
  → min/max(uv) → bbox
  → YOLO 归一化格式
```

### 3.4 碰撞盒参数

| 类别 | class_id | collision_size 半边长 (m) | 零件名 |
| --- | --- | --- | --- |
| part_a（钢管紧固件） | 0 | `[0.0111, 0.02255, 0.00505]` | `part_type_a_1/2` |
| part_b（T型接头） | 1 | `[0.011825, 0.0135, 0.010155]` | `part_type_b_1/2` |
| part_c（螺丝刀） | 2 | `[0.04, 0.0125, 0.08]` ⚠️ 多段近似 | `part_type_c_1/2` |

### 3.5 标注脚本核心

```python
#!/usr/bin/env python3
"""scene2_auto_labeler.py — 从 bag 自动生成 YOLO 标注"""
import rosbag, cv2, numpy as np, yaml
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

PART_TO_CLASS = {"part_type_a": 0, "part_type_b": 1, "part_type_c": 2}
COLLISION_HALF = {
    "part_type_a": [0.0111, 0.02255, 0.00505],
    "part_type_b": [0.011825, 0.0135, 0.010155],
    "part_type_c": [0.04, 0.0125, 0.08],
}

def box_corners_world(pos, quat_wxyz, half):
    R = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    corners = np.array([[-1,-1,-1],[ 1,-1,-1],[-1, 1,-1],[ 1, 1,-1],
                        [-1,-1, 1],[ 1,-1, 1],[-1, 1, 1],[ 1, 1, 1]]) * half
    return (R @ corners.T).T + pos

def label_bag(bag_path, layout_yaml, output_dir, cam_topic, K):
    with open(layout_yaml) as f:
        layout = yaml.safe_load(f)
    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()
    for topic, msg, t in bag.read_messages(topics=[cam_topic]):
        cv_img = bridge.compressed_imgmsg_to_cv2(msg)
        H, W = cv_img.shape[:2]
        labels = []
        for part_name, data in layout["parts"].items():
            part_type = part_name.rsplit("_", 2)[0]
            T = get_tf_at(t, "cam_h_link", "world")
            if T is None: continue
            corners_w = box_corners_world(data["world_xyz"], data["quat_wxyz"],
                                          COLLISION_HALF[part_type])
            corners_c = (T[:3,:3] @ corners_w.T).T + T[:3,3]
            uv = (K @ corners_c.T).T
            uv = uv[:,:2] / uv[:,2:3]
            x1, y1 = np.clip(uv.min(axis=0), 0, [W-1, H-1])
            x2, y2 = np.clip(uv.max(axis=0), 0, [W-1, H-1])
            if x2 <= x1 or y2 <= y1: continue
            labels.append(f"{PART_TO_CLASS[part_type]} "
                          f"{(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                          f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        frame_id = str(t.to_nsec())
        cv2.imwrite(f"{output_dir}/images/{frame_id}.jpg", cv_img)
        with open(f"{output_dir}/labels/{frame_id}.txt", "w") as f:
            f.write("\n".join(labels))
    bag.close()
```

### 3.6 输出目录结构

```
datasets/scene2_yolo/
├── images/
│   ├── train/          # 80% seed
│   └── val/            # 20% seed
├── labels/
│   ├── train/          # 每行: class cx cy w h (归一化)
│   └── val/
└── data.yaml
```

### 3.7 data.yaml

```yaml
path: datasets/scene2_yolo
train: images/train
val: images/val
nc: 3
names: ["part_a", "part_b", "part_c"]
```

---

## 4. YOLO 模型训练

```bash
pip install ultralytics

yolo detect train \
    data=datasets/scene2_yolo/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    name=scene2_parts
```

| 参数 | 建议值 | 说明 |
| --- | --- | --- |
| model | `yolov8n.pt` | 轻量，仿真帧率优先 |
| imgsz | 640 | 与相机分辨率匹配 |
| epochs | 100-200 | 仿真纹理简单，易收敛 |
| 预期 mAP@0.5 | > 0.95 | 仿真光照稳定 |

---

## 5. 推理与抓取（challenge_task.py 实现）

### 5.1 相机内参获取

订阅 `/cam_h/color/camera_info`，提取 3×3 内参矩阵 K。

### 5.2 2D → 3D 反投影

```python
def detect_to_world_xyz(detection, depth_img, K, tf_listener):
    """YOLO 检测结果 + 深度图 → base_link 坐标系 3D"""
    cx, cy = int(detection.cx), int(detection.cy)
    Z = depth_img[cy, cx]  # 米

    # 1. 像素 → 相机坐标系
    X_cam = (cx - K[0,2]) * Z / K[0,0]
    Y_cam = (cy - K[1,2]) * Z / K[1,1]
    P_cam = np.array([X_cam, Y_cam, Z])

    # 2. 相机 → base_link (通过 TF)
    trans, rot = tf_listener.lookupTransform("base_link", "cam_h_link", rospy.Time(0))
    T = transform_matrix(trans, rot)
    P_base = (T @ np.append(P_cam, 1.0))[:3]
    return P_base  # 这就是 world_xyz，直接喂给 execute_part_grasp()
```

### 5.3 分拣顺序与手选择

```python
SORT_ORDER = [
    ("part_type_b", "right"),   # B 类先抓，中间
    ("part_type_b", "right"),
    ("part_type_a", "right"),   # A 类右手
    ("part_type_a", "right"),
    ("part_type_c", "left"),    # C 类左手
    ("part_type_c", "left"),
]

CLASS_TO_BIN = {
    0: "sorting_bin_a",  # part_a → 右箱
    1: "sorting_bin_b",  # part_b → 中箱
    2: "sorting_bin_c",  # part_c → 左箱
}
```

### 5.4 完整推理主循环

```python
def scene2_vision_pipeline():
    # === 初始化（与现有数采管线一致）===
    _set_arm_mode(ARM_MODE_EXTERNAL_CONTROL)
    _publish_head_target([0, 20])  # 低头看桌面
    gripper_hold = _start_gripper_hold()
    arm_hold = _start_arm_traj_hold()
    grasp_runtime = GraspRuntime(...)  # 复用 scene2_part_grasp_ik 的配置
    _enter_work_pose(arm_pub, arm_hold, grasp_runtime)

    # === 按顺序分拣 6 个零件 ===
    yolo_model = YOLO("scene2_parts.pt")
    picked = set()

    while len(picked) < 6:
        # 1. 视觉检测
        img = get_head_rgb()
        depth = get_head_depth()
        dets = yolo_model(img)[0]

        # 2. 按分拣顺序过滤未抓取目标
        candidates = []
        for det in dets.boxes:
            cls_id = int(det.cls[0])
            if cls_id not in picked:
                candidates.append(cls_id)

        if not candidates:
            _publish_head_target([15, 20])  # 转头搜索
            rospy.sleep(0.5)
            continue

        # 3. 选当前优先级最高的目标
        target_cls = min(candidates, key=lambda c: SORT_ORDER.index(c))
        target_det = [d for d in dets if int(d.cls[0]) == target_cls][0]

        # 4. 2D → 3D
        cx = int((target_det.xyxy[0] + target_det.xyxy[2]) / 2)
        cy = int((target_det.xyxy[1] + target_det.xyxy[3]) / 2)
        world_xyz = detect_to_world_xyz(cx, cy, depth, K, tf_listener)

        # 5. 确定手和箱子
        part_type, arm = SORT_ORDER[target_cls]
        bin_name = CLASS_TO_BIN[target_cls]
        object_name = f"{part_type}_{len([p for p in picked if p == target_cls]) + 1}"
        locked_other = _fixed_work_pose_other_arm_joints(arm)

        # 6. 抓取（复用 IK 管线）
        _publish_gripper_open(gripper_hold)
        execute_part_grasp(grasp_runtime, object_name, world_xyz,
                          active_arm=arm, locked_other_arm_joints=locked_other)

        # 7. 上抬
        lift_z = world_xyz[2] + LIFT_Z_OFFSET
        # ...（复用现有 lift 逻辑）

        # 8. 放置（关节空间，预标定）
        place_joints = _place_active_arm_joints(arm, bin_name)
        _move_arm_to(arm_pub, arm_hold,
                     _compose_single_arm_place_joints(arm, place_joints, locked_other))
        _publish_gripper_open(gripper_hold)
        rospy.sleep(PLACE_DWELL)
        _move_to_work_pose_joints(arm_pub, arm_hold)

        picked.add(target_cls)

    # === 收尾 ===
    _set_arm_mode(ARM_MODE_AUTO_SWING)
```

### 5.5 关键 ROS 接口（推理用）

| 接口 | 用途 |
| --- | --- |
| `/cam_h/color/image_raw/compressed` | RGB 图像 |
| `/cam_h/depth/image_raw/compressedDepth` | 深度图 |
| `/cam_h/color/camera_info` | 内参 K |
| `/tf` | TF 树（cam_h → base_link） |
| `/ik/two_arm_hand_pose_cmd_srv` | IK 求解 |
| `/ik/fk_srv` | FK 正解 |
| `/kuavo_arm_traj` | 手臂关节角控制 |
| `/gripper/command` | 夹爪开合 |
| `/robot_head_motion_data` | 头部转动 |
| `/arm_traj_change_mode` | 手臂模式切换 |

**不需要** `/cmd_vel`。

---

## 6. 已知资源与缺口

### ✅ 已有（直接复用）

| 资源 | 位置 |
| --- | --- |
| 完整抓取+放置管线（IK 求解、手选择、交接） | `scene2_part_grasp_ik.py` + `scene2_data_collection_pipeline.py` |
| 6 零件抓取 offset + 姿态配置 | `OBJECT_PART_CONFIG` in `scene2_part_grasp_ik.py` |
| 放置关节角（预标定） | `PLACE_ACTIVE_ARM_JOINTS_DEG` in `scene2_data_collection_pipeline.py` |
| 工作姿态/侧抬/归位关节序列 | `WORK_POSE_JOINTS_DEG`, `SIDE_LIFT_JOINTS_DEG`, `HOME_JOINTS_DEG` |
| 夹爪持续发布机制 | `GripperCommandHold` class |
| arm_traj 持续发布机制 | `ArmTrajHold` class |
| rosbag 录制管线 | `scene2_data_collection_pipeline.py --record-rosbag` |
| 零件 ground truth 位姿 | `challenge_secret.get_object_layout()` / `generated_layouts/` |
| 零件碰撞盒尺寸 | `scene2.yaml` |

### ✅ 已完成

| 工具 | 位置 | 说明 |
| --- | --- | --- |
| 自动标注脚本 | `tools/scene2_auto_labeler.py` | bag + layout YAML → YOLO 数据集 |
| 标注可视化 | `tools/scene2_label_visualizer.py` | 交互浏览 + HTML 图库导出 |
| 相机外参模板 | `tools/head_camera_extrinsics.yaml` | 默认外参 + 校准说明 |

### ❌ 待开发

| 缺口 | 优先级 | 说明 |
| --- | --- | --- |
| YOLO 训练 | 🔴 | 训练 scene2 三分类模型 |
| `detect_to_world_xyz()` | 🔴 | 2D+深度→base_link 3D，约 30 行 |
| 推理主循环集成到 challenge_task.py | 🔴 | 将上述流程对接现有 IK 管线 |

### 使用自动标注脚本

```bash
# 1. 先录制 rosbag（用现有 pipeline）
python3 src/challenge_cup_simulator/test/collect_scene2_dataset/scene2_data_collection_pipeline.py \
    --seed 0 --record-rosbag

# 2. 自动标注（bag → YOLO 数据集）
python3 tools/scene2_auto_labeler.py \
    --bag-dir bags/scene2/ \
    --layout-dir src/challenge_cup_simulator/test/collect_scene2_dataset/generated_layouts/ \
    --output-dir datasets/scene2_yolo/train \
    --camera cam_h

# 3. 可视化审核
python3 tools/scene2_label_visualizer.py --dataset datasets/scene2_yolo/train
python3 tools/scene2_label_visualizer.py --dataset datasets/scene2_yolo/train --html review.html

# 4. 校准外参（如果默认不准确）
python3 tools/scene2_auto_labeler.py ... \
    --extrinsics-yaml tools/head_camera_extrinsics.yaml
```
