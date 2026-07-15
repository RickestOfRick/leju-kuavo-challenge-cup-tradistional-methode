# task3_vision

`task3_vision` 用于生成场景三 SMT 料盘训练数据所需的二维真值框。它不执行目标检测，也不读取受保护布局接口，而是从运行中的 `/mujoco/qpos` 取得五个料盘的真实位姿，将真实尺寸的三维角点投影到头部 RGB 图像。

## 标注目标

场景三包含五个外观相同的料盘：

- 下层：`smt_tray_1`、`smt_tray_2`；
- 上层：`smt_tray_3`、`smt_tray_4`、`smt_tray_5`。

五个目标统一使用类别 `tray`（类别编号 `0`）。`level` 字段额外记录 `lower` 或 `upper`，供检查场景层级使用，不作为 YOLO 类别。seed 改变槽位时，qpos 会同步变化，投影框随实际料盘位置更新。

## 输出

- `/task3_vision/detections2d`：`vision_msgs/Detection2DArray` 二维框；
- `/task3_vision/objects_json`：数据采集使用的结构化结果。

`objects_json` 中每个目标包含名称、层级、`base_link` 坐标、世界坐标、完整四元数、真实尺寸和像素框。`bbox_source` 固定为 `qpos_projected_3d_corners`，`position_source` 固定为 `qpos_base_truth`。`color_stamp` 对应最近一张头部压缩 RGB 图像，可用于同步保存图像与标签。

## 启动

统一任务入口会为场景三自动启动视觉节点：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 0
```

仿真已经启动时也可以单独运行：

```bash
roslaunch task3_vision task3_vision_scene3.launch
```

打开假框检查界面：

```bash
rosrun rqt_gui rqt_gui --standalone task3_vision/Task3Vision
```

配置位于 `config/scene3_task3_vision.yaml`。料盘实际尺寸已经包含场景 XML 中的 mesh scale，不应再重复缩放。
