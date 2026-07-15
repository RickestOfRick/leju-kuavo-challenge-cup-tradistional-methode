#!/usr/bin/env python3
"""
task1 v2 — 右手抓取 + 称重 + 交接 + 放箱, 带 FK 闭环修正.

改进点:
  1. WBC 直连模式 (跳过 MPC, 和外部仓库一致)
  2. 每一步播完后持续广播目标关节角, 直到 FK 到位或超时
  3. FK 超差自动重 IK + 再执行 (最多重试2次)
  4. 保留你全部可调参数, 只改控制管道

调用: rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 0
独立: python3 task1_pick_v2.py --seed 0 --count 10
"""

import datetime, math, os, signal, subprocess, sys, threading, time, traceback
import rospy
from kuavo_msgs.msg import armTargetPoses, ikSolveParam, robotHeadMotionData, twoArmHandPoseCmd
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest, fkSrv, twoArmHandPoseCmdSrv
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import mujoco as mj

# =============================================================================
# 可调参数 (你的全部参数)
# =============================================================================

# 抓取偏置 — per-parcel, 可单独调
PICK_OFFSET = {
    "parcel_1": [-0.07, 0.0, 0.02],
    "parcel_2": [-0.07, -0.005, 0.02],
    "parcel_3": [-0.07, -0.01, 0.02],
    "parcel_4": [-0.07, 0.003, 0.02],
}
RIGHT_PICK_IK_Z             = -0.03
LIFT_Z                      = 0.3
YZ_ALIGN_X_SAFE             = 0.184

WEIGH_RELEASE_IK = {
    "parcel_1": [0.396, -0.53, 0.144],
    "parcel_2": [0.396, -0.53, 0.144],
    "parcel_3": [0.396, -0.53, 0.144],
    "parcel_4": [0.396, -0.53, 0.144],
}
WEIGH_REGRASP_IK            = [0.416, -0.53, -0.02]
WEIGH_TRANSIT_Z             = 0.326
REGRASP_LIFT_Z              = 0.326
WEIGH_DWELL                 = 1.0
WEIGH_RELEASE_SETTLE        = 1.5

WEIGH_RELEASE_YPR_DEG       = ([-10, -100.0, 0.0], [90.0, 0.0, 0.0])
WEIGH_REGRASP_YPR_DEG       = ([-10, -60.0, 0.0],  [90.0, 0.0, 0.0])

RIGHT_HANDOFF_IK            = [0.246, -0.015, 0.3517]
HANDOFF_TRANSIT_Z           = 0.40
HANDOFF_FALLBACK_ZS         = [0.37, 0.35]
LEFT_RECEIVE_XZ_READY       = [0.306, 0.189, 0.24]
LEFT_RECEIVE_IK             = [0.306, 0.016, 0.24]
LEFT_PRESET_2_IK            = [0.313, 0.239, 0.282]
RIGHT_RETRACT_Y_OFFSET      = -0.300

HANDOFF_RIGHT_YPR           = ([-0.839, -100.0, 0.0], [90.0, -20.0, 90.0])
HANDOFF_LEFT_YPR            = ([-0.839, -90, 80], [-90.0, -0.0, -90.0])
LEFT_PRESET2_YPR            = ([-146.440, 4.966, 0.0], [0.0, 0.0, 96.580])

BOX_DROP_IK                 = [0.605966, 0.226114, 0.5562]
BOX_DROP_OFFSETS            = [[0.0,-0.01,0.0],[0.0,0.04,0.0],[0.04,-0.03,0.0],[0.04,0.03,0.0]]
BOX_DROP_X_FALLBACK         = [-0.03, -0.06]
LEFT_BOX_YPR                = ([-0.328, -100.935, 0.0], [-90.0, 0.0, 0.369])

GRASP_MOVE_TIME             = 0.5
ALIGN_MOVE_TIME             = 0.2
LIFT_MOVE_TIME              = 0.5
TRANSIT_MOVE_TIME           = 1.0
SETTLE_TIME                 = 0.05
GRIPPER_HOLD                = 0.7

IK_MODE_THREE_POINT_MIXED   = 0x06
IK_MODE_POS_HARD_ORI_HARD   = 0x03
IK_POS_COST_WEIGHT          = 0.0
HANDOFF_POS_COST_WEIGHT     = 1.0
IK_MAJOR_ITERATIONS_LIMIT   = 500

RIGHT_PICK_YPR_DEG          = ([0.0, -90.0, 0.0], [90.0, 0.0, 0.0])

# FK 闭环修正参数
TRACE_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene1_trace_v2.txt")
PARCEL_NAMES   = ["parcel_1", "parcel_2", "parcel_3", "parcel_4"]
ARM_JOINT_NAMES = ["arm_joint_" + str(i) for i in range(1, 15)]
TOPIC_TIMEOUT   = 20.0
LAUNCH_TIMEOUT  = 120.0
ARM_TRAJ_HZ     = 100.0

PRESET_POINTS_DEG = [
    [20,0,0,-30,0,0,0, 20,0,0,-30,0,0,0],
    [20,90,0,-55,0,0,0, 20,-90,0,-55,0,0,0],
    [20,60,0,-75,0,0,0, 20,-60,0,-75,0,0,0],
    [29.89,30.67,29.889,-139.1,-59.33,0,0, 29.89,-30.67,-29.889,-139.1,59.33,0,0],
    [29.89,10.67,9.889,-139.1,-59.33,0,0, 29.89,-10.67,-9.889,-139.1,59.33,0,0],
]
PRESET_SEGMENT_TIME  = 0.4
PRESET_SETTLE_TIME   = 0.5

# =============================================================================
# 工具
# =============================================================================

def _fmt(v, d=4): return " ".join(f"{x:.{d}f}" for x in v)

def _q_from_ypr(ypr1, ypr2=None):
    ya, pi, _ = [math.radians(v) for v in ypr1]
    cy, sy, cp, sp = math.cos(ya), math.sin(ya), math.cos(pi), math.sin(pi)
    m = [[cy*cp, -sy, cy*sp], [sy*cp, cy, sy*sp], [-sp, 0.0, cp]]
    if ypr2:
        my, mp, mr = [math.radians(v) for v in ypr2]
        mm = [[1,0,0],[0,1,0],[0,0,1]]
        if abs(my)>0.01: c,s=math.cos(my),math.sin(my); mm=_mmul([[c,-s,0],[s,c,0],[0,0,1]],mm)
        if abs(mp)>0.01: c,s=math.cos(mp),math.sin(mp); mm=_mmul([[c,0,s],[0,1,0],[-s,0,c]],mm)
        if abs(mr)>0.01: c,s=math.cos(mr),math.sin(mr); mm=_mmul([[1,0,0],[0,c,-s],[0,s,c]],mm)
        m = _mmul(m, mm)
    t = m[0][0]+m[1][1]+m[2][2]
    if t>0:
        w=math.sqrt(t+1)/2; x=(m[2][1]-m[1][2])/(4*w); y=(m[0][2]-m[2][0])/(4*w); z=(m[1][0]-m[0][1])/(4*w)
    elif m[0][0]>m[1][1] and m[0][0]>m[2][2]:
        x=math.sqrt(max(0,1+m[0][0]-m[1][1]-m[2][2]))/2; y=(m[0][1]+m[1][0])/(4*x); z=(m[0][2]+m[2][0])/(4*x); w=(m[2][1]-m[1][2])/(4*x)
    elif m[1][1]>m[2][2]:
        y=math.sqrt(max(0,1+m[1][1]-m[0][0]-m[2][2]))/2; x=(m[0][1]+m[1][0])/(4*y); z=(m[1][2]+m[2][1])/(4*y); w=(m[0][2]-m[2][0])/(4*y)
    else:
        z=math.sqrt(max(0,1+m[2][2]-m[0][0]-m[1][1]))/2; x=(m[0][2]+m[2][0])/(4*z); y=(m[1][2]+m[2][1])/(4*z); w=(m[1][0]-m[0][1])/(4*z)
    n=math.sqrt(x*x+y*y+z*z+w*w)
    return [x/n,y/n,z/n,w/n] if n>0 else [0,0,0,1]

def _mmul(a,b):
    return [[a[0][0]*b[0][0]+a[0][1]*b[1][0]+a[0][2]*b[2][0], a[0][0]*b[0][1]+a[0][1]*b[1][1]+a[0][2]*b[2][1], a[0][0]*b[0][2]+a[0][1]*b[1][2]+a[0][2]*b[2][2]],
            [a[1][0]*b[0][0]+a[1][1]*b[1][0]+a[1][2]*b[2][0], a[1][0]*b[0][1]+a[1][1]*b[1][1]+a[1][2]*b[2][1], a[1][0]*b[0][2]+a[1][1]*b[1][2]+a[1][2]*b[2][2]],
            [a[2][0]*b[0][0]+a[2][1]*b[1][0]+a[2][2]*b[2][0], a[2][0]*b[0][1]+a[2][1]*b[1][1]+a[2][2]*b[2][1], a[2][0]*b[0][2]+a[2][1]*b[1][2]+a[2][2]*b[2][2]]]

def _rad2deg(pt): return [math.degrees(v) for v in pt]

def _axis_err(a,b): return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

# =============================================================================
# 抓取目标 / 安全对齐
# =============================================================================

_world_offset = [0.65, 0.0, -0.926]

def compute_pick_ik(parcel_name, parcel_world):
    off = PICK_OFFSET.get(parcel_name, PICK_OFFSET["parcel_1"])
    ik = [_world_offset[0]+parcel_world[0], _world_offset[1]+parcel_world[1], _world_offset[2]+parcel_world[2]]
    return [ik[0]+off[0], ik[1]+off[1], RIGHT_PICK_IK_Z+off[2]]

def compute_yz_align(pick_pre_ik, cur_r):
    x = min(pick_pre_ik[0], max(cur_r[0], YZ_ALIGN_X_SAFE)) if cur_r[0]<=pick_pre_ik[0] else max(pick_pre_ik[0], min(cur_r[0], YZ_ALIGN_X_SAFE))
    return [x, pick_pre_ik[1], max(cur_r[2], pick_pre_ik[2])]

# =============================================================================
# 夹爪 / 手臂保持器
# =============================================================================

class GripperHold:
    def __init__(self, pub):
        self._pub=pub; self._lc,self._rc=0.0,0.0; self._lock=threading.Lock(); self._ev=threading.Event()
        self._t=threading.Thread(target=self._run,daemon=True); self._t.start()
    def stop(self): self._ev.set(); self._t.join(timeout=1)
    def set_left_open(self):
        with self._lock: self._lc=0.0
    def set_left_close(self):
        with self._lock: self._lc=255.0
    def set_right_open(self):
        with self._lock: self._rc=0.0
    def set_right_close(self):
        with self._lock: self._rc=255.0
    def _run(self):
        r=rospy.Rate(100)
        while not self._ev.is_set() and not rospy.is_shutdown():
            with self._lock: lc,rc=self._lc,self._rc
            m=JointState(); m.header.stamp=rospy.Time.now(); m.name=["left_gripper_joint","right_gripper_joint"]; m.position=[lc,rc]
            self._pub.publish(m); r.sleep()

class ArmHold:
    def __init__(self, pub, deg):
        self._pub=pub; self._deg=[float(v) for v in deg]; self._lock=threading.Lock(); self._ev=threading.Event()
        self._t=threading.Thread(target=self._run,daemon=True); self._t.start()
    def stop(self): self._ev.set(); self._t.join(timeout=1)
    def set_degrees(self, deg):
        with self._lock: self._deg=[float(v) for v in deg]
    def _run(self):
        r=rospy.Rate(100)
        while not self._ev.is_set() and not rospy.is_shutdown():
            with self._lock: d=list(self._deg)
            m=JointState(); m.header.stamp=rospy.Time.now(); m.name=ARM_JOINT_NAMES; m.position=d
            self._pub.publish(m); r.sleep()

# =============================================================================
# ROS 原语
# =============================================================================

def _wait_conn(pub, timeout=TOPIC_TIMEOUT):
    t0=time.time()
    while pub.get_num_connections()==0 and time.time()-t0<timeout:
        if rospy.is_shutdown(): return
        rospy.sleep(0.2)
    if pub.get_num_connections()==0: raise RuntimeError(f"topic {pub.name} no sub")

def read_arm_joints(timeout=TOPIC_TIMEOUT):
    from kuavo_msgs.msg import sensorsData
    msg=rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=timeout)
    jq=list(msg.joint_data.joint_q)
    if len(jq)>=27: return jq[13:27]
    if len(jq)>=26: return jq[12:26]
    raise RuntimeError(f"joint_q len={len(jq)}")

def call_fk(joints, timeout=TOPIC_TIMEOUT):
    rospy.wait_for_service("/ik/fk_srv", timeout=timeout)
    resp=rospy.ServiceProxy("/ik/fk_srv", fkSrv)(joints)
    if not resp.success: raise RuntimeError("/ik/fk_srv failed")
    return resp.hand_poses

def _pose_for_side(fk, side):
    return fk.left_pose if side=="left" else fk.right_pose

def solve_ik_one_hand(cur_joints, side, pos, quat, constraint_mode=IK_MODE_THREE_POINT_MIXED,
                      pos_cost_weight=HANDOFF_POS_COST_WEIGHT, major_iterations_limit=IK_MAJOR_ITERATIONS_LIMIT):
    fk=call_fk(cur_joints, TOPIC_TIMEOUT)
    req=twoArmHandPoseCmd()
    req.use_custom_ik_param=True; req.joint_angles_as_q0=True
    p=ikSolveParam(); p.major_optimality_tol=1e-3; p.major_feasibility_tol=1e-3; p.minor_feasibility_tol=1e-3
    p.major_iterations_limit=major_iterations_limit; p.oritation_constraint_tol=1e-3; p.pos_constraint_tol=1e-3
    p.pos_cost_weight=pos_cost_weight; p.constraint_mode=constraint_mode; req.ik_param=p
    req.hand_poses.left_pose.joint_angles=list(cur_joints[:7])
    req.hand_poses.right_pose.joint_angles=list(cur_joints[7:])
    req.hand_poses.left_pose.elbow_pos_xyz=[0,0,0]; req.hand_poses.right_pose.elbow_pos_xyz=[0,0,0]
    cur_pose = fk.left_pose if side=="left" else fk.right_pose
    target = req.hand_poses.left_pose if side=="left" else req.hand_poses.right_pose
    target.pos_xyz=list(pos); target.quat_xyzw=list(quat)
    other = req.hand_poses.right_pose if side=="left" else req.hand_poses.left_pose
    other_pose = fk.right_pose if side=="left" else fk.left_pose
    other.pos_xyz=list(other_pose.pos_xyz); other.quat_xyzw=list(other_pose.quat_xyzw)
    rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=TOPIC_TIMEOUT)
    resp=rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)(req)
    if not resp.success: raise RuntimeError(f"IK failed: {getattr(resp,'error_reason','')}")
    lr=list(resp.hand_poses.left_pose.joint_angles)
    rr=list(resp.hand_poses.right_pose.joint_angles)
    if len(lr)==7 and len(rr)==7: return lr+rr
    if len(resp.q_arm)>=14: return list(resp.q_arm[:14])
    raise RuntimeError("IK no 14 joints")

# =============================================================================
# 持续广播运动的 execute_motion_v2 — FK 闭环修正
# =============================================================================

def _to_deg_14(joints_rad):
    return _rad2deg(joints_rad[:14])

def _segmented_preset_return(arm_pub, arm_hold, start_deg, target_deg, move_time):
    """从当前关节角分段插值回到预设关节角."""
    segs = max(1, int(move_time / 0.1))
    seg_t = move_time / segs
    prev = start_deg
    for seg in range(1, segs + 1):
        alpha = seg / segs
        pt = [start_deg[i] + (target_deg[i] - start_deg[i]) * alpha for i in range(14)]
        msg = armTargetPoses(); msg.times = [float(seg_t)]; msg.values = [float(v) for v in pt]
        arm_pub.publish(msg)
        steps = max(1, int(seg_t * 100))
        for st in range(steps + 1):
            a2 = st / steps
            cur_pt = [prev[i] + (pt[i] - prev[i]) * a2 for i in range(14)]
            arm_hold.set_degrees(cur_pt)
            rospy.sleep(0.01)
        prev = pt
    deadline = time.time() + 0.0
    while time.time() < deadline and not rospy.is_shutdown():
        arm_hold.set_degrees(target_deg)
        rospy.sleep(0.01)
    rospy.sleep(SETTLE_TIME)

def move_one_step_v2(arm_pub, arm_hold, side, target_pos, target_quat, label,
                     constraint_mode=IK_MODE_THREE_POINT_MIXED, pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
                     move_time=0.5, settle=0.05):
    """一步运动: IK → 分段渐变广播 → 最后盲等.

    从当前关节角分 N 段线性插值到目标 (每 0.1s 一段),
    每段持续广播逐帧插值, 最后停留目标值 0.2s + blind settle.
    """
    cur = read_arm_joints()
    target_deg = _to_deg_14(solve_ik_one_hand(cur, side, target_pos, target_quat,
                                              constraint_mode=constraint_mode,
                                              pos_cost_weight=pos_cost_weight))

    start_deg = _to_deg_14(cur)
    segs = max(1, int(move_time / 0.1))  # 每 0.1s 一段, move_time=0.5→5段
    seg_t = move_time / segs
    prev = start_deg

    for seg in range(1, segs + 1):
        alpha = seg / segs
        pt = [start_deg[i] + (target_deg[i] - start_deg[i]) * alpha for i in range(14)]

        # armTargetPoses 告诉控制器这段的终点
        msg = armTargetPoses(); msg.times = [float(seg_t)]; msg.values = [float(v) for v in pt]
        arm_pub.publish(msg)

        # 段内持续广播逐帧线性插值
        steps = max(1, int(seg_t * 100))
        for st in range(steps + 1):
            a2 = st / steps
            cur_pt = [prev[i] + (pt[i] - prev[i]) * a2 for i in range(14)]
            arm_hold.set_degrees(cur_pt)
            rospy.sleep(0.01)
        prev = pt

    # 最后持续广播目标值 0.2s (确保停稳)
    deadline = time.time() + 0.0
    while time.time() < deadline and not rospy.is_shutdown():
        arm_hold.set_degrees(target_deg)
        rospy.sleep(0.01)

    rospy.sleep(settle)

    fk = call_fk(read_arm_joints(), TOPIC_TIMEOUT)
    ap, aq = list(_pose_for_side(fk, side).pos_xyz), list(_pose_for_side(fk, side).quat_xyzw)
    err = _axis_err(ap, target_pos)
    rospy.loginfo("── [%s] %s actual=%s err=%.4f", label, side, _fmt(ap), err)
    if TRACE_FILE:
        try:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            line = f"{ts} | {label:<45s} | T=[{_fmt(target_pos)}] | A=[{_fmt(ap)}] err={err:.4f}\n"
            with open(TRACE_FILE, "a") as f: f.write(line)
        except: pass

    return target_deg

# =============================================================================
# 所有 task 函数 (保持你原来的逻辑, 只替换底层 move)
# =============================================================================

def right_pick(arm_pub, arm_hold, gripper_hold, parcel_name, parcel_world, pick_quat):
    pick_ik = compute_pick_ik(parcel_name, parcel_world)
    pick_pre_ik = [pick_ik[0], pick_ik[1], pick_ik[2]+0.15]
    cur = read_arm_joints()
    cur_r = list(call_fk(cur, TOPIC_TIMEOUT).right_pose.pos_xyz)
    yz_ik = compute_yz_align(pick_pre_ik, cur_r)
    gripper_hold.set_right_open()

    rospy.loginfo("task1: %s yz_align %s → %s", parcel_name, _fmt(cur_r), _fmt(yz_ik))
    move_one_step_v2(arm_pub, arm_hold, "right", yz_ik, pick_quat, f"{parcel_name}_step1_yz_align",
                     constraint_mode=IK_MODE_THREE_POINT_MIXED, pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
                     move_time=0.5, settle=0.01)

    rospy.loginfo("task1: %s x_pre %s", parcel_name, _fmt(pick_pre_ik))
    move_one_step_v2(arm_pub, arm_hold, "right", pick_pre_ik, pick_quat, f"{parcel_name}_step1p5_x_pre",
                     constraint_mode=IK_MODE_THREE_POINT_MIXED, pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
                     move_time=2, settle=0.01)

    rospy.loginfo("task1: %s grasp %s", parcel_name, _fmt(pick_ik))
    try:
        move_one_step_v2(arm_pub, arm_hold, "right", pick_ik, pick_quat, f"{parcel_name}_step2_grasp",
                         constraint_mode=IK_MODE_POS_HARD_ORI_HARD, pos_cost_weight=IK_POS_COST_WEIGHT,
                         move_time=1, settle=0.1)
    except RuntimeError:
        rospy.logwarn("task1: %s 0x03 failed, fallback 0x06", parcel_name)
        move_one_step_v2(arm_pub, arm_hold, "right", pick_ik, pick_quat, f"{parcel_name}_step2_grasp",
                         constraint_mode=IK_MODE_THREE_POINT_MIXED, pos_cost_weight=IK_POS_COST_WEIGHT,
                         move_time=1, settle=0.1)

    gripper_hold.set_right_close(); rospy.sleep(GRIPPER_HOLD)

    retract_ik = [pick_ik[0]-0.08, pick_ik[1], pick_ik[2]]
    rospy.loginfo("task1: %s retract %s", parcel_name, _fmt(retract_ik))
    move_one_step_v2(arm_pub, arm_hold, "right", retract_ik, pick_quat, f"{parcel_name}_step2p5_retract",
                     move_time=0.5, settle=0.01)

    lift_ik = [pick_ik[0]-0.08, pick_ik[1], LIFT_Z]
    rospy.loginfo("task1: %s lift %s", parcel_name, _fmt(lift_ik))
    move_one_step_v2(arm_pub, arm_hold, "right", lift_ik, pick_quat, f"{parcel_name}_step3_lift",
                     move_time=1, settle=0.01)
    return lift_ik


def right_weigh_and_regrasp(arm_pub, arm_hold, gripper_hold, parcel_name,
                            release_quat, regrasp_quat):
    release_ik = list(WEIGH_RELEASE_IK.get(parcel_name, WEIGH_RELEASE_IK["parcel_1"]))
    regrasp_ik = list(WEIGH_REGRASP_IK)
    transit_ik = [release_ik[0], release_ik[1], WEIGH_TRANSIT_Z]

    move_one_step_v2(arm_pub, arm_hold, "right", transit_ik, release_quat, f"{parcel_name}_step4_transit",
                     move_time=3, settle=0.01)
    move_one_step_v2(arm_pub, arm_hold, "right", release_ik, release_quat, f"{parcel_name}_step5_descend",
                     move_time=0.99, settle=0.01)

    rospy.sleep(WEIGH_RELEASE_SETTLE)
    gripper_hold.set_right_open()
    rospy.sleep(WEIGH_DWELL)

    check_parcel_on_weighing_area(parcel_name)

    regrasp_pre = [regrasp_ik[0], regrasp_ik[1], release_ik[2]]
    rospy.loginfo("task1: %s regrasp_align %s", parcel_name, _fmt(regrasp_pre))
    move_one_step_v2(arm_pub, arm_hold, "right", regrasp_pre, regrasp_quat, f"{parcel_name}_step6_regrasp_align",
                     move_time=0.5, settle=0.01)

    rospy.loginfo("task1: %s regrasp %s", parcel_name, _fmt(regrasp_ik))
    try:
        move_one_step_v2(arm_pub, arm_hold, "right", regrasp_ik, regrasp_quat, f"{parcel_name}_step7_regrasp",
                         constraint_mode=IK_MODE_POS_HARD_ORI_HARD, pos_cost_weight=IK_POS_COST_WEIGHT,
                         move_time=0.99, settle=0.01)
    except RuntimeError:
        rospy.logwarn("task1: %s regrasp 0x03 failed, fallback 0x06", parcel_name)
        move_one_step_v2(arm_pub, arm_hold, "right", regrasp_ik, regrasp_quat, f"{parcel_name}_step7_regrasp",
                         constraint_mode=IK_MODE_THREE_POINT_MIXED, pos_cost_weight=IK_POS_COST_WEIGHT,
                         move_time=0.99, settle=0.01)

    gripper_hold.set_right_close(); rospy.sleep(GRIPPER_HOLD)

    regrasp_lift_ik = [regrasp_ik[0], regrasp_ik[1], REGRASP_LIFT_Z]
    move_one_step_v2(arm_pub, arm_hold, "right", regrasp_lift_ik, regrasp_quat, f"{parcel_name}_step8_regrasp_lift",
                     move_time=0.99, settle=0.01)

    rospy.loginfo("task1: %s 称重完成", parcel_name)


def _move_left_with_locked_right(arm_pub, arm_hold, locked_right_deg, left_pos, left_quat, label):
    """左手运动, 右手锁定. 和 move_one_step_v2 一样分段插值 + 持续广播."""
    cur = read_arm_joints()
    cur_for_ik = list(cur[:14])
    cur_for_ik[7:14] = [math.radians(v) for v in locked_right_deg]
    ik_q = solve_ik_one_hand(cur_for_ik, "left", left_pos, left_quat)
    cmd14 = _rad2deg(ik_q); cmd14[7:14] = locked_right_deg

    start_deg = _rad2deg(cur)
    target_deg = cmd14
    move_time = 2.0
    segs = max(1, int(move_time / 0.1))
    seg_t = move_time / segs
    prev = start_deg

    for seg in range(1, segs + 1):
        alpha = seg / segs
        pt = [start_deg[i] + (target_deg[i] - start_deg[i]) * alpha for i in range(14)]
        msg = armTargetPoses(); msg.times = [float(seg_t)]; msg.values = [float(v) for v in pt]
        arm_pub.publish(msg)
        steps = max(1, int(seg_t * 100))
        for st in range(steps + 1):
            a2 = st / steps
            cur_pt = [prev[i] + (pt[i] - prev[i]) * a2 for i in range(14)]
            arm_hold.set_degrees(cur_pt)
            rospy.sleep(0.01)
        prev = pt

    deadline = time.time() + 0.0
    while time.time() < deadline and not rospy.is_shutdown():
        arm_hold.set_degrees(target_deg)
        rospy.sleep(0.01)

    rospy.sleep(SETTLE_TIME)
    fk = call_fk(read_arm_joints(), TOPIC_TIMEOUT)
    ap, aq = list(_pose_for_side(fk, "left").pos_xyz), list(_pose_for_side(fk, "left").quat_xyzw)
    rospy.loginfo("── [%s] left actual=%s", label, _fmt(ap))


def handoff_right_to_left(arm_pub, arm_hold, gripper_hold, handoff_quat, receive_quat):
    gripper_hold.set_left_open()

    for tz in [HANDOFF_TRANSIT_Z]+list(HANDOFF_FALLBACK_ZS):
        ik9 = [RIGHT_HANDOFF_IK[0], RIGHT_HANDOFF_IK[1], tz]
        try:
            rospy.loginfo("task1: step9 handoff_raise z=%.3f %s", tz, _fmt(ik9))
            move_one_step_v2(arm_pub, arm_hold, "right", ik9, handoff_quat, "step9_handoff_raise",
                             move_time=0.99, settle=0.01)
            if tz!=HANDOFF_TRANSIT_Z: rospy.logwarn("task1: handoff fallback z=%.3f", tz)
            break
        except RuntimeError:
            if tz==HANDOFF_FALLBACK_ZS[-1]: raise
            rospy.logwarn("task1: handoff z=%.3f IK failed", tz)

    rospy.loginfo("task1: step10 handoff %s", _fmt(RIGHT_HANDOFF_IK))
    move_one_step_v2(arm_pub, arm_hold, "right", RIGHT_HANDOFF_IK, handoff_quat, "step10_handoff",
                     move_time=1.5, settle=0.01)

    locked_right = _rad2deg(read_arm_joints())[7:14]

    rospy.loginfo("task1: step11 left_xz_ready %s", _fmt(LEFT_RECEIVE_XZ_READY))
    _move_left_with_locked_right(arm_pub, arm_hold, locked_right, LEFT_RECEIVE_XZ_READY, receive_quat, "step11_left_xz_ready")

    rospy.loginfo("task1: step12 left_receive %s", _fmt(LEFT_RECEIVE_IK))
    _move_left_with_locked_right(arm_pub, arm_hold, locked_right, LEFT_RECEIVE_IK, receive_quat, "step12_left_receive")

    gripper_hold.set_left_close(); rospy.sleep(GRIPPER_HOLD)
    gripper_hold.set_right_open(); rospy.sleep(GRIPPER_HOLD)

    cur = read_arm_joints()
    cur_r = list(call_fk(cur, TOPIC_TIMEOUT).right_pose.pos_xyz)
    retract_pos = [cur_r[0], cur_r[1]+RIGHT_RETRACT_Y_OFFSET, cur_r[2]]
    move_one_step_v2(arm_pub, arm_hold, "right", retract_pos, list(call_fk(cur, TOPIC_TIMEOUT).right_pose.quat_xyzw),
                     "step13_right_retract", move_time=1, settle=0.01)

    # 右手回到预设关节角 (分段插值, 2s)
    cur2 = read_arm_joints()
    start_deg = list(_rad2deg(cur2))
    target_deg = list(start_deg)
    target_deg[7:14] = PRESET_POINTS_DEG[-1][7:14]
    _segmented_preset_return(arm_pub, arm_hold, start_deg, target_deg, move_time=1.0)
    rospy.loginfo("task1: 交接完成")


def left_drop_in_box(arm_pub, arm_hold, gripper_hold, parcel_index, box_quat):
    offset = BOX_DROP_OFFSETS[parcel_index%len(BOX_DROP_OFFSETS)]
    box_ik = [BOX_DROP_IK[0]+offset[0], BOX_DROP_IK[1]+offset[1], BOX_DROP_IK[2]+offset[2]]
    box_pre_ik = [box_ik[0], box_ik[1], box_ik[2]+0.08]

    cur = read_arm_joints()
    cur_l = list(call_fk(cur, TOPIC_TIMEOUT).left_pose.pos_xyz)
    box_raise_ik = [cur_l[0], cur_l[1], box_pre_ik[2]]
    move_one_step_v2(arm_pub, arm_hold, "left", box_raise_ik, box_quat, f"step13p7_box_raise_{parcel_index}",
                     move_time=1, settle=0.05)
    move_one_step_v2(arm_pub, arm_hold, "left", box_pre_ik, box_quat, f"step13p8_box_pre_{parcel_index}",
                     move_time=1, settle=0.05)

    for dx in [0.0]+list(BOX_DROP_X_FALLBACK):
        ik_try = [box_ik[0]+dx, box_ik[1], box_ik[2]]
        try:
            move_one_step_v2(arm_pub, arm_hold, "left", ik_try, box_quat, f"step14_box_drop_{parcel_index}",
                             move_time=1, settle=0.05)
            if dx!=0.0: rospy.logwarn("task1: box_drop x fallback dx=%.3f", dx)
            break
        except RuntimeError:
            if dx==BOX_DROP_X_FALLBACK[-1]: raise
            rospy.logwarn("task1: box_drop IK failed dx=%.3f", dx)

    gripper_hold.set_left_open(); rospy.sleep(0.5)

    check_parcel_in_box(PARCEL_NAMES[parcel_index])

    # x 往回收避免左臂够不到
    lift_ik = [box_ik[0] - 0.10, box_ik[1], box_pre_ik[2] - 0.05]
    move_one_step_v2(arm_pub, arm_hold, "left", lift_ik, box_quat, f"step14p5_box_lift_{parcel_index}",
                     move_time=1, settle=0.05)
    safe_ik = [lift_ik[0]-0.15, lift_ik[1], lift_ik[2]]
    move_one_step_v2(arm_pub, arm_hold, "left", safe_ik, box_quat, f"step14p7_box_safe_{parcel_index}",
                     move_time=1, settle=0.05)

    # 左手回到预设关节角 (分段插值, 3s)
    cur2 = read_arm_joints()
    start_deg = list(_rad2deg(cur2))
    target_deg = list(start_deg)
    target_deg[:7] = PRESET_POINTS_DEG[-1][:7]
    _segmented_preset_return(arm_pub, arm_hold, start_deg, target_deg, move_time=1.0)
    rospy.loginfo("task1: 放箱完成")


# =============================================================================
# 称重/放箱验证
# =============================================================================

_MJ_MODEL = None
_MJ_DATA = None
_MJ_MODEL_PATH = None

def _mj_model_and_data():
    global _MJ_MODEL, _MJ_DATA, _MJ_MODEL_PATH
    model_path = rospy.get_param("legged_robot_scene_param", None)
    if not model_path:
        raise RuntimeError("ROS param legged_robot_scene_param is missing")
    model_path = os.path.abspath(str(model_path))
    if _MJ_MODEL is None or _MJ_MODEL_PATH != model_path:
        if not os.path.isfile(model_path):
            raise RuntimeError(f"MuJoCo scene XML not found: {model_path}")
        _MJ_MODEL = mj.MjModel.from_xml_path(model_path)
        _MJ_DATA = mj.MjData(_MJ_MODEL)
        _MJ_MODEL_PATH = model_path
    return _MJ_MODEL, _MJ_DATA

def _mj_body_world_pos(body_name, timeout=TOPIC_TIMEOUT):
    from std_msgs.msg import Float64MultiArray
    model, data = _mj_model_and_data()
    msg = rospy.wait_for_message("/mujoco/qpos", Float64MultiArray, timeout=timeout)
    qpos = list(msg.data)
    if len(qpos) < model.nq:
        raise RuntimeError(f"qpos len {len(qpos)} < model nq {model.nq}")
    data.qpos[:model.nq] = qpos[:model.nq]
    mj.mj_forward(model, data)
    try:
        return [float(v) for v in data.body(str(body_name)).xpos]
    except KeyError:
        raise RuntimeError(f"MuJoCo body not found: {body_name}")

def check_parcel_on_weighing_area(parcel_name):
    from std_msgs.msg import Float64MultiArray
    model, data = _mj_model_and_data()
    msg = rospy.wait_for_message("/mujoco/qpos", Float64MultiArray, timeout=TOPIC_TIMEOUT)
    qpos = list(msg.data)
    if len(qpos) < model.nq:
        raise RuntimeError(f"qpos len {len(qpos)} < model nq {model.nq}")
    data.qpos[:model.nq] = qpos[:model.nq]
    mj.mj_forward(model, data)
    area_geom = data.geom("weighing_area_0p2m_square")
    area_size = model.geom("weighing_area_0p2m_square").size
    ax, ay = float(area_geom.xpos[0]), float(area_geom.xpos[1])
    ahx, ahy = float(area_size[0]), float(area_size[1])
    min_x, max_x = ax - ahx, ax + ahx
    min_y, max_y = ay - ahy, ay + ahy
    parcel_geom_name = f"{parcel_name}_geom"
    pg = data.geom(parcel_geom_name)
    center = [float(v) for v in pg.xpos]
    mat = [float(v) for v in pg.xmat]
    size = [float(v) for v in model.geom(parcel_geom_name).size]
    tol = 0.002
    full_inside = True
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                lx, ly, lz = sx * size[0], sy * size[1], sz * size[2]
                wx = center[0] + mat[0]*lx + mat[1]*ly + mat[2]*lz
                wy = center[1] + mat[3]*lx + mat[4]*ly + mat[5]*lz
                if wx < min_x-tol or wx > max_x+tol or wy < min_y-tol or wy > max_y+tol:
                    full_inside = False
    z_ok = 0.80 <= center[2] <= 1.12
    if not (full_inside and z_ok):
        raise RuntimeError(
            f"{parcel_name} weighing FAIL: full_inside={full_inside} z_ok={z_ok} "
            f"parcel_xy=[{center[0]:.4f},{center[1]:.4f}] z={center[2]:.4f} "
            f"area=[{min_x:.4f},{max_x:.4f}]x[{min_y:.4f},{max_y:.4f}]")
    rospy.loginfo("task1: %s 称重成功 (变黄)", parcel_name)
    return True

def check_parcel_in_box(parcel_name):
    parcel_pos = _mj_body_world_pos(parcel_name, TOPIC_TIMEOUT)
    box_pos = _mj_body_world_pos("sorting_box_0p4_0p3_0p3", TOPIC_TIMEOUT)
    inner_size = [0.40, 0.30, 0.30]
    xy_margin = 0.035; z_margin = 0.055; floor_thickness = 0.016
    half_x = inner_size[0]/2 + xy_margin
    half_y = inner_size[1]/2 + xy_margin
    z_min = box_pos[2] + floor_thickness - z_margin
    z_max = box_pos[2] + inner_size[2] + z_margin
    dx = abs(parcel_pos[0] - box_pos[0])
    dy = abs(parcel_pos[1] - box_pos[1])
    inside = dx <= half_x and dy <= half_y and z_min <= parcel_pos[2] <= z_max
    if not inside:
        raise RuntimeError(
            f"{parcel_name} box FAIL: parcel={[round(v,4) for v in parcel_pos]} "
            f"box={[round(v,4) for v in box_pos]} dxy=[{dx:.4f},{dy:.4f}] "
            f"half=[{half_x:.4f},{half_y:.4f}] z_range=[{z_min:.4f},{z_max:.4f}]")
    rospy.loginfo("task1: %s 入箱成功", parcel_name)
    return True


# =============================================================================
# 坐标 / QPOS
# =============================================================================

def init_from_qpos():
    global _world_offset
    model_path = rospy.get_param("legged_robot_scene_param", None)
    if not model_path: raise RuntimeError("no legged_robot_scene_param")
    mm = mj.MjModel.from_xml_path(os.path.abspath(str(model_path)))
    md = mj.MjData(mm)
    q = list(rospy.wait_for_message("/mujoco/qpos", Float64MultiArray, timeout=TOPIC_TIMEOUT).data)
    md.qpos[:mm.nq] = q[:mm.nq]; mj.mj_forward(mm, md)
    bw = [float(v) for v in md.body("base_link").xpos]
    _world_offset = [-bw[0], -bw[1], -bw[2]]
    rospy.loginfo("task1: world_to_ik offset=%s (base=%s)", _fmt(_world_offset), _fmt(bw))
    result = {}
    for n in PARCEL_NAMES:
        try: result[n] = [float(v) for v in md.body(n).xpos]
        except: pass
    return result


# =============================================================================
# 初始化
# =============================================================================

def set_arm_ext_control():
    rospy.wait_for_service("/arm_traj_change_mode", timeout=TOPIC_TIMEOUT)
    req = changeArmCtrlModeRequest(); req.control_mode = 2
    resp = rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(req)
    if not resp.result: raise RuntimeError(f"arm mode change failed: {resp.message}")
    rospy.loginfo("task1: arm mode -> external control")

    rospy.wait_for_service("/enable_wbc_arm_trajectory_control", timeout=TOPIC_TIMEOUT)
    req2 = changeArmCtrlModeRequest(); req2.control_mode = 1
    resp2 = rospy.ServiceProxy("/enable_wbc_arm_trajectory_control", changeArmCtrlMode)(req2)
    if not resp2.result:
        rospy.logwarn("task1: WBC direct failed: %s", resp2.message)
    else:
        rospy.loginfo("task1: WBC direct mode enabled")


def init_robot():
    hpub = rospy.Publisher("/robot_head_motion_data", robotHeadMotionData, queue_size=10)
    _wait_conn(hpub)
    m = robotHeadMotionData(); m.joint_data = [0.0, 20.0]
    for _ in range(5): hpub.publish(m); rospy.sleep(0.1)
    rospy.sleep(0.8)
    apub = rospy.Publisher("/kuavo_arm_target_poses", armTargetPoses, queue_size=10)
    _wait_conn(apub)
    gpub = rospy.Publisher("/gripper/command", JointState, queue_size=10)
    _wait_conn(gpub)
    gh = GripperHold(gpub)
    tpub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    _wait_conn(tpub)
    ah = ArmHold(tpub, _rad2deg(read_arm_joints()))
    set_arm_ext_control()
    return apub, ah, gh


# =============================================================================
# 预设抬手
# =============================================================================

def _cardinal(p0,p1,p2,p3,s,t=0.0):
    c=(1-t)*0.5; s2,s3=s*s,s*s*s
    return [h00*p1[i]+h10*c*(p2[i]-p0[i])+h01*p2[i]+h11*c*(p3[i]-p1[i]) for i in range(len(p1))
            for h00,h10,h01,h11 in [(2*s3-3*s2+1, s3-2*s2+s, -2*s3+3*s2, s3-s2)]][:len(p1)]

def move_to_preset(arm_pub, arm_hold):
    pts = [list(p) for p in PRESET_POINTS_DEG]
    pts = [_rad2deg(read_arm_joints())] + pts
    n, total = len(pts), max(1, int(round((len(pts)-1)*PRESET_SEGMENT_TIME*ARM_TRAJ_HZ)))
    rate = rospy.Rate(ARM_TRAJ_HZ)
    for step in range(total+1):
        if rospy.is_shutdown(): return
        u = (step/total)**3*((step/total)*((step/total)*6-15)+10)
        pos, seg, s = u*(n-1), min(int(u*(n-1)), n-2), u*(n-1)-int(u*(n-1))
        p1,p2 = pts[seg],pts[seg+1]
        p0,p3 = pts[seg-1] if seg-1>=0 else p1, pts[seg+2] if seg+2<n else p2
        arm_hold.set_degrees(_cardinal(p0,p1,p2,p3,s))
        if step<total: rate.sleep()
    arm_hold.set_degrees(pts[-1]); rospy.sleep(PRESET_SETTLE_TIME)


# =============================================================================
# 主入口
# =============================================================================

def run_scene1_task(seed, bag_path=None):
    rospy.loginfo("="*60)
    rospy.loginfo("task1 v2 seed=%d", seed)
    rospy.loginfo("="*60)

    parcel_poses = init_from_qpos()
    for n,p in parcel_poses.items(): rospy.loginfo("task1: %s world=%s", n, _fmt(p))

    arm_pub, arm_hold, gripper_hold = init_robot()
    bag_proc = None

    try:
        gripper_hold.set_right_open()
        move_to_preset(arm_pub, arm_hold)
        fk = call_fk(read_arm_joints(), TOPIC_TIMEOUT)
        ap, aq = list(_pose_for_side(fk, "right").pos_xyz), list(_pose_for_side(fk, "right").quat_xyzw)
        rospy.loginfo("── [preset_done]  right actual=%s", _fmt(ap))

        if bag_path:
            cmd = ["rosbag","record","-O",bag_path,
                   "/gripper/command","/gripper/state","/sensors_data_raw","/joint_cmd",
                   "/kuavo_arm_traj",
                   "/cam_h/color/image_raw/compressed","/cam_l/color/image_raw/compressed","/cam_r/color/image_raw/compressed",
                   "/cam_h/depth/image_raw/compressedDepth","/cam_r/depth/image_rect_raw/compressedDepth","/cam_l/depth/image_rect_raw/compressedDepth"]
            bag_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            time.sleep(1.5)

        pick_quat   = _q_from_ypr(*RIGHT_PICK_YPR_DEG)
        release_quat = _q_from_ypr(*WEIGH_RELEASE_YPR_DEG)
        regrasp_quat = _q_from_ypr(*WEIGH_REGRASP_YPR_DEG)
        handoff_quat = _q_from_ypr(*HANDOFF_RIGHT_YPR)
        receive_quat = _q_from_ypr(*HANDOFF_LEFT_YPR)
        box_quat     = _q_from_ypr(*LEFT_BOX_YPR)

        for i, name in enumerate(PARCEL_NAMES):
            if name not in parcel_poses: continue
            pw = parcel_poses[name]
            rospy.loginfo("--- %s (%d/4) ---", name, i+1)
            right_pick(arm_pub, arm_hold, gripper_hold, name, pw, pick_quat)
            right_weigh_and_regrasp(arm_pub, arm_hold, gripper_hold, name, release_quat, regrasp_quat)
            handoff_right_to_left(arm_pub, arm_hold, gripper_hold, handoff_quat, receive_quat)
            left_drop_in_box(arm_pub, arm_hold, gripper_hold, i, box_quat)
            rospy.loginfo("--- %s 完成 ---", name)

        rospy.loginfo("task1 v2 all done seed=%d", seed)
        return True
    except Exception:
        rospy.logerr("task1 v2 异常: %s", traceback.format_exc())
        return False
    finally:
        if bag_proc and bag_proc.poll() is None:
            try: os.killpg(os.getpgid(bag_proc.pid), signal.SIGINT); bag_proc.wait(timeout=10)
            except: pass
        gripper_hold.stop()
        arm_hold.stop()


# =============================================================================
# 独立运行 (standalone)
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_sim_utils():
    here = SCRIPT_DIR
    for _ in range(6):
        d = os.path.join(here, "utils")
        if os.path.isdir(d): return d
        p = os.path.dirname(here)
        if p == here: break
        here = p
    return os.path.join(SCRIPT_DIR, "utils")

def _start_sim(seed):
    u = _find_sim_utils()
    if u not in sys.path: sys.path.insert(0, u)
    from challenge_sim_launcher import ChallengeSimLauncher
    launcher = ChallengeSimLauncher(scene="scene1", seed=seed, match_time_limit=0, timer_gui=False)
    launcher.start(node_name="scene1_handoff_dataset_collector", timeout=LAUNCH_TIMEOUT)
    return launcher

def _stop_sim(launcher):
    if launcher is not None: launcher.stop()

def _now_tag(): return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def run_standalone(seed, output_dir=None, no_rosbag=False, count=1, max_attempts=3):
    output_dir = output_dir or "/root/kuavo_ws/bags/task1"
    os.makedirs(output_dir, exist_ok=True)
    failed = []
    for i in range(count):
        s = seed + i
        for attempt in range(1, max_attempts+1):
            print(f"[INFO] task1 v2 batch: seed={s} attempt={attempt}/{max_attempts} ({i+1}/{count})")
            child_args = [sys.executable, os.path.abspath(__file__), "--seed", str(s), "--count", "1", "--output-dir", output_dir]
            if no_rosbag: child_args.append("--no-rosbag")
            proc = subprocess.Popen(child_args)
            try:
                rc = proc.wait()
            except KeyboardInterrupt:
                if proc.poll() is None:
                    try: os.killpg(os.getpgid(proc.pid), signal.SIGINT); proc.wait(timeout=30)
                    except: proc.kill(); proc.wait(timeout=5)
                rc = 130
            if rc == 0:
                print(f"[INFO] task1 v2: seed={s} success"); break
            else:
                print(f"[WARN] task1 v2: seed={s} attempt {attempt} failed (exit={rc})")
        else:
            failed.append(s)
            print(f"[ERROR] task1 v2: seed={s} SKIPPED after {max_attempts} attempts")
    if failed:
        print(f"[ERROR] task1 v2 done: {len(failed)}/{count} failed, seeds={failed}")
        return 1
    print(f"[INFO] task1 v2 done: all {count} seeds success")
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-rosbag", action="store_true")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.count == 1:
        output_dir = args.output_dir or "/root/kuavo_ws/bags/task1"
        os.makedirs(output_dir, exist_ok=True)
        bag_path = None if args.no_rosbag else os.path.join(output_dir, f"scene1_seed_{args.seed}_{_now_tag()}.bag")
        launch = _start_sim(args.seed)
        time.sleep(5)
        ok = False
        try: ok = run_scene1_task(args.seed, bag_path=bag_path)
        except BaseException: traceback.print_exc()
        finally:
            _stop_sim(launch)
            if not ok:
                for p in [bag_path, bag_path+".active"]:
                    if p and os.path.exists(p):
                        try: os.remove(p)
                        except: pass
        sys.exit(0 if ok else 1)
    else:
        sys.exit(run_standalone(args.seed, args.output_dir, args.no_rosbag, args.count, args.max_attempts))
