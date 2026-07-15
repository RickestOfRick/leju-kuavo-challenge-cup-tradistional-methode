# task1_vision：YOLO + RANSAC 顶面拟合

## 如何读取拟合得到的 X、Y

拟合结果通过以下话题发布：

```text
话题：/task1_vision/objects_json
类型：std_msgs/String
格式：JSON
坐标系：base_link
```

每个包裹的 X、Y 位于：

```text
objects[i].position[0] = 包裹中心相对 base_link 的 X，+X 向前，单位 m
objects[i].position[1] = 包裹中心相对 base_link 的 Y，+Y 向左，单位 m
objects[i].position[2] = 包裹中心相对 base_link 的 Z，+Z 向上，单位 m
objects[i].position_source = depth_roi_top_plane_fit
```

订阅示例：

```python
#!/usr/bin/env python3

import json

import rospy
from std_msgs.msg import String


def vision_callback(msg):
    payload = json.loads(msg.data)
    if payload.get("frame_id") != "base_link":
        rospy.logwarn("忽略非 base_link 坐标：%s", payload.get("frame_id"))
        return

    for obj in payload.get("objects", []):
        if obj.get("position_source") != "depth_roi_top_plane_fit":
            continue
        position = obj.get("position", [])
        if len(position) < 2:
            continue
        x = float(position[0])
        y = float(position[1])
        rospy.loginfo(
            "%s: x=%.4f m, y=%.4f m",
            obj.get("name", "parcel"),
            x,
            y,
        )


rospy.init_node("task1_vision_xy_reader")
rospy.Subscriber(
    "/task1_vision/objects_json",
    String,
    vision_callback,
    queue_size=1,
)
rospy.spin()
```

也可以直接检查原始消息：

```bash
rostopic echo /task1_vision/objects_json
```

`position` 是 YOLO + 深度顶面拟合的包裹中心，不是二维像素框中心 `bbox_center`，也不包含抓取模块之后叠加的 `PICK_OFFSET`。YOLO 部署模式中的 qpos 不参与 `position` 计算。

## 中心怎么得到

1. YOLO 对 RGB 图像推理，经过置信度过滤和 NMS 后保留四个包裹框。
2. 在每个框的深度 ROI 中筛选有效深度和包裹前景颜色像素。
3. 使用相机内参把像素反投影为相机点云：

```text
X_camera = (u - cx) * depth / fx
Y_camera = (v - cy) * depth / fy
Z_camera = depth
```

4. 使用 TF 把点云转换到 `base_link`：

```text
P_base = R_base_camera * P_camera + t_base_camera
```

5. 体素聚类保留 ROI 中空间连续的最大点云簇，减少背景和相邻包裹干扰。
6. RANSAC 每次取三个点构造平面，只接受法线接近 `base_link +Z` 的候选。当前条件为法线倾角不超过 `12 deg`，点到平面距离不超过 `4 mm`。在有效候选中优先选择高度最高的平面，再用全部内点进行 SVD 精拟合。
7. 使用顶面内点恢复 X、Y：
   - 根据预设角度、PCA 主方向和角度搜索生成 yaw 候选；
   - 把顶面内点旋转到每个候选的局部 XY 坐标；
   - 分别取局部 X/Y 的第 5 和第 95 百分位作为可见范围；
   - 两端中点作为候选中心；
   - 结合已知包裹长宽计算越界误差，选择误差最小的 yaw 和 X/Y 中心。
8. 将拟合平面写成：

```text
normal_x * x + normal_y * y + normal_z * z + d = 0
```

在已经得到的中心 X/Y 处计算顶面高度：

```text
z_top = -(normal_x * center_x + normal_y * center_y + d) / normal_z
center_z = z_top - known_height / 2
```

这里按 `base_link` 的 Z 轴减去包裹先验高度的一半，不沿平面法线移动。最终中心为：

```text
position = [center_x, center_y, center_z]
```

四个中心全部拟合成功后，按 `base_link` 中的相对位置分配名称：X 增大为上，Y 增大为左，即右下、左下、右上、左上依次为 `parcel_1` 到 `parcel_4`。任务模块再对连续 10 帧的中心坐标分别取中值，作为实际抓取使用的固定视觉坐标。
