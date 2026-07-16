# task2_vision：任务二 base_link 位姿与 yaw 真值发布

`task2_vision` 暂时不做图像检测。节点使用与任务二数采脚本相同的接口：

```python
challenge_secret.get_object_layout("scene2", seed)
```

它读取当前 seed 下六个工件的 `pos` 和 `quat`，再读取 `/mujoco/qpos` 中机器人浮动基座的位姿，将位置和完整姿态转换到 `base_link`。统一任务入口启动场景二时会自动传入相同的 seed。

## 输出话题

主要接口：

```text
话题：/task2_vision/objects_json
类型：std_msgs/String
格式：JSON
坐标系：base_link
```

每个工件包含：

```text
objects[i].name             motion 实例名，例如 part_type_a_1
objects[i].motion_name      与 name 相同，供动作参数和偏置查表
objects[i].ground_truth_name .so 中原始真值实例名
objects[i].class_name       类别名，例如 part_type_a
objects[i].position         工件参考原点相对 base_link 的位置，单位 m
objects[i].yaw              绕 base_link Z 轴的角度，单位 rad
objects[i].yaw_deg          同一角度，单位 deg
objects[i].orientation      相对 base_link 的完整 ROS 四元数 [x,y,z,w]
objects[i].yaw_orientation  只包含 yaw 的 ROS 四元数 [x,y,z,w]
objects[i].world_position   .so 返回的原始 world_xyz
objects[i].world_quat_wxyz  .so 返回的原始 MuJoCo 四元数 [w,x,y,z]
objects[i].position_source  challenge_secret_layout_ground_truth
```

实例编号由 `task2_vision` 根据 `base_link` 下的 Y 坐标生成。Y 正方向为左：
A、B 类从右到左编号 `_1`、`_2`，C 类从左到右编号 `_1`、`_2`。
真值模式下 `.so` 的实例名可能与 motion 实例名不同，因此消息同时发布
`ground_truth_name`，顶层 `motion_to_ground_truth` 给出本次场景的完整对照。
位置、姿态和 yaw 始终来自该 `ground_truth_name` 对应的真值对象，motion
仅使用重新编号后的 `name` 选择抓取偏置。

`position` 是数采脚本 `world_xyz` 转换到 `base_link` 后的结果。它严格说是 MuJoCo 工件参考原点，不额外计算视觉包围盒的几何中心，也没有叠加抓取偏置。

另一个话题 `/task2_vision/poses` 的类型是 `geometry_msgs/PoseArray`。其中六个位姿严格按照以下顺序排列：

```text
part_type_a_1, part_type_a_2,
part_type_b_1, part_type_b_2,
part_type_c_1, part_type_c_2
```

## 读取示例

```python
#!/usr/bin/env python3

import json

import rospy
from std_msgs.msg import String


def callback(message):
    payload = json.loads(message.data)
    for item in payload.get("objects", []):
        rospy.loginfo(
            "%s center=(%.4f, %.4f, %.4f) yaw=%.2f deg",
            item["name"],
            item["position"][0],
            item["position"][1],
            item["position"][2],
            item["yaw_deg"],
        )


rospy.init_node("task2_vision_reader")
rospy.Subscriber("/task2_vision/objects_json", String, callback, queue_size=1)
rospy.spin()
```

也可以直接查看：

```bash
rostopic echo /task2_vision/objects_json
```
