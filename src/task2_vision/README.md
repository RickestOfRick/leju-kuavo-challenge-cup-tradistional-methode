# task2_vision：任务二生产视觉

生产节点固定使用不依赖仿真真值的链路：

```text
三类别 YOLO bbox + RGB-D/CAD 拟合 base_link XYZ/yaw
```

初始六件收集 5 个完整有效帧，XYZ 分轴取中位数，yaw 取环形中位数后锁定。
交接后的指定物体重定位继续使用
固定 ROI 和完整 360 degree RGB-D/CAD 搜索。

## 生产输入

```text
/cam_h/color/image_raw/compressed
/cam_h/depth/image_raw/compressedDepth
/cam_h/color/camera_info
TF: base_link <- camera
```

YOLO 读取 `models/task2_parts_best.pt`。CAD 所需的最小 XML 和网格全部位于
`models/cad/`，运行时不从其他
功能包读取场景或网格资产。生产节点不订阅仿真工件状态、场景布局或评分接口。

## 视觉流程

1. YOLO 每类保留置信度最高的两个框，必须凑齐 A/B/C 共六件。
2. 深度像素用 CameraInfo 反投影，并按深度帧时间查询 TF，直接转换到当前
   `base_link`。
3. 排除六个 ROI 后拟合桌面平面，再在各 ROI 内剔除绿色桌面与无效深度。
4. 每个实例分别在两个类别 yaw 先验各自 `+/-20 degree` 窗口内搜索；候选分数
   来自 CAD 轮廓、深度和 C 类手柄证据。
5. 每帧对同类两个检测与两个 yaw 先验做最小代价一对一分配，随后计入多帧聚合。
6. 收满 5 个完整有效帧后发布锁定结果并释放 YOLO 模型。

`cad_yaw_prior_regularization_weight=0` 时，先验只限定搜索窗口，窗口内完全按
CAD 数据分选择角度。

## 输出

```text
话题: /task2_depth_xy/objects_json
类型: std_msgs/String
frame_id: base_link
```

动作使用的核心字段：

```text
objects[i].name
objects[i].class_name
objects[i].position
objects[i].orientation
objects[i].yaw
objects[i].confidence
objects[i].position_source
locked
fit_coordinate_reference=current_base_link
instance_naming_method
```

锁定前 `position=[]`，`fit_position/fit_yaw` 只供 RQT 预览；锁定后 `position`、
`orientation` 和 `yaw` 是实际交给动作系统的数据。

## 交接重定位

动作系统在双臂回到工作预位并稳定后，通过：

```text
/task2_depth_xy/handoff_relocalize_request
/task2_depth_xy/handoff_relocalize_result
```

请求包含动作实例名、类别和动作预期位置。视觉节点使用配置中的
`handoff_relocalization_fixed_roi` 做完整 360 degree CAD 搜索，收集 5 个有效帧后
只替换该工件的锁定位姿，其余五件保持不变。

## RQT

```bash
source devel/setup.zsh
rosrun rqt_gui rqt_gui --standalone task2_vision/Task2Vision
```

RQT 从 `/task2_depth_xy/objects_json` 显示当前帧预览或锁定后真正交给动作系统的
数据，以及 CAD 覆盖率、IoU、深度误差、yaw 窗口分配和有效帧进度。任务包内
不读取或显示仿真真值误差。
