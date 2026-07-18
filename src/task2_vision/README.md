# task2_vision：任务二 RGB-D/CAD 位姿拟合与真值对照

`task2_vision` 支持 qpos 投影框和三类别 YOLO 部署框，并在框内运行同一套
RGB-D/CAD 位姿拟合。节点使用与任务二数采脚本相同的接口生成仅用于模式 0
以及模式 1/2 误差对照的真值：

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
objects[i].class_id         A/B/C 分别为 0/1/2
objects[i].bbox_center      头部 RGB 图像中的框中心 [u,v]，单位 pixel
objects[i].bbox_size        轴对齐框大小 [width,height]，单位 pixel
objects[i].bbox_projected   当前 qpos 是否成功投影出框
objects[i].bbox_source      mujoco_qpos_projected_visual_mesh
```

顶层 `frozen=true` 仍表示供 motion 使用的初始位姿固定不变；`bbox_frozen=false` 表示二维框根据实时 qpos 持续更新。若运行环境缺少官方 MuJoCo Python binding，节点会保留原位姿发布并将框标为未投影，不会中断 task2_motion。

实例编号由统一入口的 `TASK2_USE_YAW_PRIOR_INSTANCE_MATCHING` 选择。默认
`False`：按相对位置编号，Y 正方向为左，A、B 类从右到左，C 类从左到右；
yaw 则在该类别两个先验各自的 `+/-20 degree` 窗口内搜索，并在每一帧强制
同类两个物体一对一占用两个窗口。设为 `True` 时，每类只比较两个固定 yaw，
并通过同类两个物体的联合一一匹配生成 `_1/_2`。
真值模式下 `.so` 的实例名可能与 motion 实例名不同，因此消息同时发布
`ground_truth_name`，顶层 `motion_to_ground_truth` 给出本次场景的完整对照；
`motion_object_order` 给出与动作抓取偏置一致的显示顺序。RQT 的 `motion`
列只显示动作语义名，不把目标箱名或真值 body 名当作动作名。
原始 `/task2_vision/objects_json` 的位置、姿态和 yaw 来自该
`ground_truth_name` 对应的真值对象；模式 1 的 motion 改用
`/task2_depth_xy/objects_json` 中独立拟合的 XYZ、orientation 和 yaw。

`position` 是数采脚本 `world_xyz` 转换到 `base_link` 后的结果。它严格说是 MuJoCo 工件参考原点，不额外计算视觉包围盒的几何中心，也没有叠加抓取偏置。

另一个话题 `/task2_vision/poses` 的类型是 `geometry_msgs/PoseArray`。其中六个位姿严格按照以下顺序排列：

```text
part_type_a_1, part_type_a_2,
part_type_b_1, part_type_b_2,
part_type_c_1, part_type_c_2
```

`/task2_vision/detections2d` 使用标准 `vision_msgs/Detection2DArray` 发布相同的类别框。框由当前 `/mujoco/qpos` 对应的实际可视 mesh 顶点投影生成，因此物体被抓起后框也会更新。节点读取 XML 中的 mesh scale，支持 OBJ、二进制 STL 和 ASCII STL；只有可视 mesh 无法读取时才回退到 collision geom。

## RQT 可视化

```bash
source devel/setup.zsh
rosrun rqt_gui rqt_gui --standalone task2_vision/Task2Vision
```

界面叠加显示头部 RGB 图像中的 A/B/C 类别框，框内使用紧凑的 `A/B/C` 标签，并列出 motion 实例名、真值实例名、base_link 位姿、yaw 和像素框尺寸。界面文字全部为英文。

## 视觉模式

统一入口中的 `TASK2_VISION_MODE` 支持：

```text
0  full truth:   XYZ、orientation、yaw 全部来自真值，motion 直接读取
                 /task2_vision/objects_json，不等待深度拟合。
1  RGB-D pose:   XY、yaw 来自前景点云和 CAD 配准，Z 来自桌面平面和
                 类别静置高度，orientation 由类别静置姿态和拟合 yaw 重建，
                 ROI 来自 qpos 投影类别框，motion 等待
                 /task2_depth_xy/objects_json。
2  YOLO+RGB-D:   XYZ、yaw 和 orientation 使用与模式 1 相同的 RGB-D/CAD，
                 但类别和 ROI 来自三类别 YOLO，不使用 qpos 工件框，
                 motion 等待 /task2_depth_xy/objects_json。
```

launch 参数 `vision_mode` 使用相同语义，`instance_naming_method` 可设为
`yaw_prior` 或 `relative_position`。模式 0 不启动 `task2_depth_xy`；模式 1/2
自动启动。输出携带顶层 `vision_mode`、`vision_mode_name`、
`instance_naming_method` 和 `ready_for_motion`，motion 会核对模式和命名方式后
再接受消息。

真值工件位姿和 qpos 框每帧都使用当前 MuJoCo qpos 转换到当前 `base_link`。深度点使用该深度帧时间戳查询 `base_link <- camera` TF，直接进入当前 `base_link`；拟合结果不再经过 world、MPC、真值基座或初始化参考基座转换。模式 2 的拟合主链只使用 YOLO 类别、置信度、bbox、RGB-D/CAD 和固定 yaw 先验；当前真值工件位姿只在输出组装时补入 RQT/test 误差字段。

RQT 在模式 0 直接将真值作为当前模式的 XYZ/yaw，`dx/dy/dz/dyaw` 均显示 0；在模式 1/2 锁定前实时显示当前深度帧的拟合 XYZ/yaw 与当前 `base_link` 真值的逐轴误差、点数、双向覆盖率、IoU、GN 迭代数和深度内点数，行状态保持 `waiting`，同时显示 5 个完整有效帧的累计进度。模式 2 图像叠加的是 YOLO 框。达到 5 帧后切换为最终中位数和 `locked` 状态，之后位姿、框和误差不再变化，YOLO 模型同时释放。

## RGB-D/CAD 位姿拟合

模式 1/2 会同时启动 `task2_depth_xy_node.py`。输入和输出为：

```text
模式 1 ROI：/task2_vision/objects_json 中的 qpos 框
模式 2 ROI：头部 RGB 经 task2 YOLO 权重实时推理
输入深度图：/cam_h/depth/image_raw/compressedDepth
拟合输出：  /task2_depth_xy/objects_json
```

当前拟合链路：

1. 模式 1 的 qpos 投影框或模式 2 的 YOLO 框只确定类别和图像 ROI，不读取物体真值 XYZ、orientation 或 yaw 作为拟合初值或约束。模式 2 要求 A/B/C 每类恰好选出置信度最高的两个有效框。
2. 头部深度图反投影到 `base_link`。
3. 排除六个物体 ROI，在工作台范围内用高度直方图选择候选平面，再做带水平法向约束的 RANSAC 和最小二乘精修。
4. 将同一头部相机 RGB 缩放到深度分辨率；正常亮度用 OpenCV HSV、阴影区域用亮度无关的归一化绿色色度和绿色优势量，把地面、桌面及其 1 px 抗锯齿边缘严格标记为背景。物体前景必须同时满足离桌高度条件和非绿色条件，再通过闭运算与连通域面积过滤去除散点；完全失去色彩信息的黑色阴影仍由桌面平面高度排除。
5. Z 使用拟合桌面平面、统一标定偏置 -0.9 mm，再叠加类别静置 body 原点偏置（A=-27.2 mm、B=-0.04 mm、C=0）。类别静置姿态固定为 A/B 水平、C 绕局部 X 侧躺 90 度。
6. A/B/C 的两个固定 yaw 先验分别为 `0.05/0.82 rad`、`-0.35/0.72 rad`、`pi/2+0.22` 和 `pi/2-0.55 rad`，直接作为当前 `base_link` 下的搜索中心。yaw 先验命名和相对位置命名使用完全相同的搜索：两个先验各自 `+/-20 degree`、步长 5 degree，每一帧都比较同类两个物体的 direct/swap 总 CAD 成本并完成窗口一对一分配，再进行逐帧命名和多帧样本累计。
7. 每个 yaw 候选都用 z-buffer 渲染 CAD 可见面和轮廓。region 项计算轮廓到观测前景有符号距离，depth 项计算 CAD 可见点到 RGB-D 点的点到面残差；两项使用 Huber 权重求解 XY。窗口内候选按 `CAD 数据分 + weight*(yaw 偏离先验/sigma)^2` 排序；当前 `weight=0`，因此粗搜、3/1 degree 局部细化和同类 direct/swap 分配都完全按 CAD 数据分选择，先验只限定 `+/-20 degree` 范围。候选阶段使用 1200 个 CAD 表面点和最多 2 次 XY 迭代；最终姿态使用 5000 点和最多 6 次迭代。C 使用红色手柄参与候选评分，最终手柄修正最多 6 次迭代。
8. 最终质量门限按类别设置：A/B 允许更低的轮廓覆盖率，C 保持双向覆盖率 0.45、IoU 0.30。被拒绝的拟合仍发布诊断，但不计入 5 帧。
9. 同类窗口分配发生在每一帧计数之前；模式 2 还会在同一帧内根据拟合后的 Y 或 yaw 先验把 YOLO 临时检测映射为 motion `_1/_2`，映射完成后才进入样本。只有分配后的六个物体全部拟合成功才计为一个完整有效帧。有效帧直接进入样本，不与已有样本的中位数比较，也不在收满 5 帧后重新分配。累计 5 帧后，XYZ 分别取中位数，yaw 取跨越 +/-180 degree 安全的环形中位数并一次性锁定。

输出对象保持 task2_motion 需要的字段结构。`position[0:2]` 和 `yaw` 来自 RGB-D/CAD 拟合，`position[2]` 来自桌面平面和类别固定静置高度，`orientation` 由类别静置姿态和拟合 yaw 重建：

```text
position_source           qpos_bbox_region_depth_xyz_yaw_fit
xy_position_source        qpos_bbox_region_depth_xyz_yaw_fit
z_position_source         table_plane_class_rest_pose
yaw_source                qpos_bbox_region_depth_xyz_yaw_fit
orientation_source        qpos_bbox_region_depth_xyz_yaw_fit
```

模式 2 将上述 qpos 前缀替换为
`yolo_bbox_region_depth_xyz_yaw_fit`，Z 来源仍为
`table_plane_class_rest_pose`。

每个对象还输出 `fit_initial_xy`、`fit_yaw_initial_deg`、`fit_yaw_pca_anisotropy`、`fit_yaw_evaluations`、`fit_yaw_search_mode`、`fit_render_depth_error_m`、`fit_render_coverage`、`fit_render_observed_coverage`、`fit_render_iou`、`fit_alignment_iterations`、`fit_alignment_depth_inliers`、`fit_render_score` 和 `fit_green_rejected_pixels`。两种命名都显示 `fit_yaw_search_mode=class_yaw_prior_window_20_independent_joint_candidate`，并输出选中先验、CAD 原始 yaw 分数、先验惩罚、偏离先验角度、direct/swap 成本、分配余量和距离窗口边界的角度。

交接重定位使用独立话题
`/task2_depth_xy/handoff_relocalize_request` 和
`/task2_depth_xy/handoff_relocalize_result`。请求指定 motion 名、真值来源名、类别
及预期放置位置；节点只重新处理该物体，不清空或覆盖启动时锁定的六物体结果。
模式 1 读取对应 qpos ROI，模式 2 按需重新加载 YOLO、拟合目标类别候选并选择
最接近预期放置 XY 的结果。yaw 不使用实例先验，以
`handoff_full_yaw_step_deg` 覆盖完整 360°，再执行局部 CAD 精化。

单目标更新由 `task2_vision.target_relocalization` 统一管理会话、有效帧累计、
中位数聚合和锁定载荷合并：请求开始后主
`objects_json` 临时切为 `waiting`，只替换目标工件并逐帧发布当前拟合，其他工件
保持此前锁定值。累计 5 个有效目标帧后，目标 XYZ/yaw 的中位数写回主锁定载荷；
该值与返回给 motion 的结果完全相同。RQT 因此在重定位期间显示实时 waiting，
完成后显示 motion 实际采用的新 locked 数据，而不是最后一帧瞬时拟合。

位姿拟合本身不使用物体真值 XYZ、orientation 或 yaw。模式 1 的检测框仍由 qpos 生成；模式 2 的类别和检测框完全来自 YOLO。相机 TF 直接给出当前 `base_link` 拟合坐标，不读取 MPC 或真值基座姿态。当前真值工件位姿只在输出组装阶段计算物体拟合误差，不进入候选生成、CAD 渲染、优化、逐帧命名、多帧聚合或动作坐标转换。

`task2_motion` 已允许新的 `position_source`。统一入口按模式覆盖 motion 输入话题；单独启动 motion 时默认 `VISION_TOPIC=/task2_vision/objects_json`。未收满 5 个完整有效帧时，输出中的 `fit_position` 可供 RQT 显示，而 `position=[]`，确保动作解析器不会提前采用预热数据。锁定后节点停止解码新深度和更新拟合结果，latched 输出、RQT 坐标和 RQT 误差保持不变。

RQT 表格直接显示每个物体当前 `base_link` 下的 truth/motion `x/y/z/yaw` 以及 `dx/dy/dz/dyaw`，并显示当前帧选中的 yaw 先验窗口、direct/swap 成本、分配方式、分配余量和距窗口边界角度。motion 列是等待阶段的拟合预览或锁定后实际交给动作的位姿。图像中的白色十字表示拟合 body 原点。

## YOLO 部署

模式 2 默认读取 `models/task2_parts_best.pt`，对应类别固定为
`0=part_type_a, 1=part_type_b, 2=part_type_c`。模型来源和 SHA-256 记录在
`models/task2_parts_best.json`。ROS 节点必须从包含 `ultralytics` 的 Python
环境启动；当前 `hitirc` 环境已验证可加载该权重。推理阈值、NMS IoU、输入尺寸、
device 和 half 参数均在 `config/scene2_task2_vision.yaml` 中配置。

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
rospy.Subscriber("/task2_depth_xy/objects_json", String, callback, queue_size=1)
rospy.spin()
```

也可以直接查看：

```bash
rostopic echo /task2_depth_xy/objects_json
```
