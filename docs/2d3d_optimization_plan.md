# 2D→3D 投影精度优化追踪

> 目标：系统性提升场景二中 YOLO 2D 检测 → 3D 抓取位置的精度
> 策略：逐项优化，每项完成后跑测试，根据 Log 分析决定下一步

---

## 问题诊断总结 (2025-07-13)

### 当前 2D→3D 链路
```
YOLOv8n detection (cx, cy) 
  → 5×5 median depth at pixel → Z
  → pinhole: X_cam=(cx-K[0,2])*Z/K[0,0], Y_cam=(cy-K[1,2])*Z/K[1,1]
  → TF cam_frame → base_link → world
  → part_world[0] += 0.10  ← 经验修正
  → IK grasp
```

### 已知系统性误差现象
- **中间零件偏前**（估计位置比实际靠前）
- **两边零件偏后**（估计位置比实际靠后）
- **左右误差不对称**
- **未考虑零件朝向**

### 根因假设优先级
1. 🔴 **TF 静态标定有旋转误差**：`camera_base → Head Camera View` 的静态 TF 不精确，且用统一 +0.10m 修正掩盖了问题
2. 🔴 **单点深度投影**：无 solvePnP / 多点约束，精度受限于单个像素的深度值
3. 🟡 **深度图噪声/压缩损失**：PNG 压缩深度 + 单帧采样
4. 🟡 **超广角相机 (fovy=85°)**：边缘像素小孔模型逼近误差
5. 🟢 **无多帧滤波**：无卡尔曼/滑动平均

---

## 优化步骤

### Step 1: 验证 TF 旋转误差假设 ✅ (已完成, 2025-07-13)

**结果：假设不成立 ❌**

数据摘要：各零件需要的 X 修正量在 0.057~0.077m 之间，均值 **0.069m**，标准差仅 0.007m。
如果存在 TF 旋转误差，不同像素位置的修正量应该显著不同（≥5cm级别），但实际差异仅 2cm。

| 零件 | 像素 | 需要修正 | 旧+0.10误差 | 新(位置变化)误差 |
|------|------|---------|------------|----------------|
| b_1 | (681,523) | 0.066 | 0.034 | 0.059 |
| b_2 | (647,527) | 0.071 | 0.029 | 0.052 |
| a_1 | (822,536) | 0.070 | 0.030 | 0.063 |
| a_2 | (875,536) | 0.077 | 0.023 | 0.059 |
| c_1 | (459,494) | 0.057 | 0.043 | 0.054 |
| c_2 | (422,523) | 0.075 | 0.026 | 0.038 |

**核心发现**：
1. TF 误差主要是纯平移（~7cm X），非旋转误差
2. 旧常数 +0.10 已经把所有零件 X 误差控制在 2-4cm，修正偏大约 3cm
3. **真正的瓶颈是 IK 可达性**：pre_grasp 执行误差 6-16cm 的零件全部抓取失败，而 1-2cm 的全部成功

**意外发现**：抓取成功率与 3D 位置估计精度的相关性不大，主要由 IK 预抓取能否到达目标点决定。

### Step 2: 修正 X 常数为最优值 + 增加 IK 可达性诊断

**改动**：将 X 修正从 +0.10 改为最优值 +0.07，同时增加 pre_grasp IK 诊断 log。

### Step 3: (待定)
### Step 4: (待定)
### Step 5: (待定)

---

## 实验记录

| 步骤 | 日期 | 改动文件 | 改动内容 | 测试结果 | 结论 |
|------|------|----------|----------|----------|------|
| Step 1 | 2025-07-13 | challenge_task.py:948-966 | 统一 +0.10 改为 position-dependent + debug log | ❌ 旋转假设不成立。各零件需修正 0.057~0.077，均值 0.069，差异仅 2cm，属纯平移误差。位置依赖修正增大了误差。真正瓶颈是 IK 可达性。 |
| Step 2 | 2025-07-13 | challenge_task.py:948 | X 修正改为最优常数 +0.07，移除位置依赖 | ✅ 全部零件 X 误差<1cm。但抓取率 3/6：右臂 b_1/b_2/a_1 均因 quat_err>60° 失败，左臂全部成功。3D 位置估计已解决，瓶颈转移到 IK 朝向。 |
| Step 3 | 2025-07-13 | scene2_part_grasp_ik.py:852 | `_move_arm_ik_once` 添加 [QUAT_DEBUG] log | ✅ b 类 target_q=[-0.707,0,0,0.707] (翻转180°) 右手无法到达，IK 牺牲朝向保位置。a 类 quat=47° 边缘通过但实际也抓不到。 | 朝向只是表象，真正问题是 Z/Y 误差大导致夹爪在零件上方闭合。需要系统诊断 2D→3D 全维度误差。 |
| Step 4 | 2025-07-13 | test/run_debug_2d3d.py (新) | 一键脚本：3 seeds (3/685/6666) 跨验证 | ✅ 三轴偏差跨 seed 高度一致：dX=-6.3cm, dY≈0, dZ=+4.2cm。dZ 有零件类型差异（螺丝刀高→dZ小，管卡矮→dZ大）。std<1mm 证明深度图噪声极低。 |
| Step 5 | 2025-07-13 | challenge_task.py:948 | 三轴修正：part_world[0]+=0.063, part_world[2]-=0.042 | ✅ 位置精准到<1cm，但 a/b 仍失败 — 原因是 IK 朝向而非位置 |
| Step 6 | 2025-07-13 | grasp_override.py(新) + challenge_task.py | B类右手narrow_offset从-90°翻到+90°；容差收紧到6cm+50° | 待测试 | — |

---

## 关键文件索引

| 文件 | 作用 |
|------|------|
| `src/challenge_cup_task_template/scripts/challenge_task.py` | 主任务脚本，含 2D→3D 全链路 |
| `src/challenge_cup_simulator/scripts/cam_info_pub.py` | 相机内参发布 (fovy=85°, fx=fy≈393) |
| `src/mujoco/src/simulate/simulate.cc` | MuJoCo 渲染 + 深度图编码 |
| `src/challenge_cup_simulator/launch/load_kuavo_mujoco_challenge.launch` | 静态 TF `camera_base → Head Camera View` |
| `src/challenge_cup_simulator/models/biped_s52/xml/biped_s52.xml:494` | MuJoCo 相机定义 (pos+quat+fovy) |
| `src/challenge_cup_simulator/models/biped_s52/urdf/biped_s52.urdf` | URDF camera_base 连杆 |
| `src/challenge_cup_simulator/test/collect_scene2_dataset/scene2_part_grasp_ik.py` | 抓取 IK 逻辑 + OBJECT_PART_CONFIG |
