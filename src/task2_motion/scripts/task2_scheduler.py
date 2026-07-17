#!/usr/bin/env python3
"""任务二六个工件的顺序抓取、交接和分类放箱调度。"""

import math
import traceback

import rospy

import task2_geometry as geometry
import task2_perception as perception
from task2_config import (
    ARM_MOVE_TIME,
    C_INITIAL_LIFT_OFFSET,
    C_PLACE_IK_SUBDIVISIONS,
    GRIPPER_CLOSE_TIME,
    HANDOFF_RECEIVER_CLOSE_DELAY_BY_CLASS,
    HOME_JOINTS_DEG,
    IK_LIFT_POS_COST_WEIGHT,
    IK_POS_COST_WEIGHT,
    LIFT_OFFSETS_BY_CLASS,
    PLACE_POSITIONS_BASE,
    PLACE_Y_OFFSETS_BY_CLASS,
    PRE_GRASP_Z_OFFSET,
    PRE_GRASP_ORIENTATION_RETRY_DEG,
    RELEASE_DWELL,
    SIDE_LIFT_JOINTS_DEG,
    WORK_POSE_JOINTS_DEG,
)
from task2_motion import RobotController, fmt


def _place_position(side, bin_name, object_name):
    try:
        position = [
            float(value) for value in PLACE_POSITIONS_BASE[side][bin_name]
        ]
    except KeyError:
        raise RuntimeError("no place XYZ for {} -> {}".format(side, bin_name))
    if len(position) != 3:
        raise ValueError(
            "place XYZ for {} -> {} must contain xyz".format(side, bin_name)
        )
    try:
        class_name, instance_text = str(object_name).rsplit("_", 1)
        instance_index = int(instance_text) - 1
        y_offsets = [
            float(value) for value in PLACE_Y_OFFSETS_BY_CLASS[class_name]
        ]
        y_offset = y_offsets[instance_index]
    except (KeyError, ValueError, IndexError):
        raise RuntimeError("no place Y offset for {}".format(object_name))
    if len(y_offsets) != 2 or instance_index not in (0, 1):
        raise ValueError(
            "place Y offsets for {} must map instances 1 and 2".format(
                class_name
            )
        )
    position[1] += y_offset
    return position


def _lift_offset(object_name):
    class_name = str(object_name).rsplit("_", 1)[0]
    try:
        offset = [float(value) for value in LIFT_OFFSETS_BY_CLASS[class_name]]
    except KeyError:
        raise RuntimeError("no lift offset for {}".format(class_name))
    if len(offset) != 3:
        raise ValueError("lift offset for {} must contain xyz".format(class_name))
    return offset


def _c_initial_lift_offset():
    offset = [float(value) for value in C_INITIAL_LIFT_OFFSET]
    if len(offset) != 3:
        raise ValueError("C initial lift offset must contain xyz")
    return offset


def _approach_pre_grasp_with_yaw_retry(
    controller,
    object_name,
    side,
    observation,
    pre_grasp,
    original_yaw_offset,
    original_grasp,
    original_lift,
    alternate_yaw_offset_for,
    grasp_for_yaw_offset,
    lift_for_yaw_offset,
    label_prefix,
):
    """Try the 180-degree yaw alternative at pre-grasp and keep the better FK."""
    controller.gripper.open(side)
    _, original_error = controller.move_pose(
        side,
        pre_grasp,
        original_grasp,
        "{}_pre_grasp".format(label_prefix),
        pos_cost_weight=IK_POS_COST_WEIGHT,
        return_orientation_error=True,
    )
    original_error_deg = math.degrees(original_error)
    if (
        observation["class_name"]
        not in ("part_type_a", "part_type_b", "part_type_c")
        or original_error_deg <= PRE_GRASP_ORIENTATION_RETRY_DEG
    ):
        return original_yaw_offset, original_grasp, original_lift

    alternative_yaw_offset = alternate_yaw_offset_for(original_yaw_offset)
    alternative_grasp = grasp_for_yaw_offset(alternative_yaw_offset)
    alternative_lift = lift_for_yaw_offset(alternative_yaw_offset)
    rospy.logwarn(
        "task2: [%s_pre_grasp] ori_err=%.2f deg > %.2f deg; "
        "returning to work pose and trying the 180-deg yaw alternative",
        label_prefix,
        original_error_deg,
        PRE_GRASP_ORIENTATION_RETRY_DEG,
    )
    controller.move_joints(
        WORK_POSE_JOINTS_DEG,
        "{}_yaw_retry_return_work".format(label_prefix),
        side=side,
        checked_keypoint=True,
    )
    alternative_error = None
    alternative_failure = None
    try:
        _, alternative_error = controller.move_pose(
            side,
            pre_grasp,
            alternative_grasp,
            "{}_pre_grasp_yaw_flipped".format(label_prefix),
            pos_cost_weight=IK_POS_COST_WEIGHT,
            return_orientation_error=True,
        )
    except RuntimeError as error:
        if rospy.is_shutdown():
            raise
        alternative_failure = str(error)

    if alternative_error is not None and alternative_error < original_error:
        alternative_error_deg = math.degrees(alternative_error)
        rospy.logwarn(
            "task2: %s selected flipped yaw at pre-grasp: "
            "ori_err %.2f -> %.2f deg",
            object_name,
            original_error_deg,
            alternative_error_deg,
        )
        return alternative_yaw_offset, alternative_grasp, alternative_lift

    if alternative_error is None:
        rospy.logwarn(
            "task2: %s flipped yaw pre-grasp failed (%s); restoring original yaw",
            object_name,
            alternative_failure,
        )
    else:
        rospy.logwarn(
            "task2: %s flipped yaw did not improve pre-grasp ori_err "
            "(original=%.2f deg flipped=%.2f deg); restoring original yaw",
            object_name,
            original_error_deg,
            math.degrees(alternative_error),
        )
    controller.move_joints(
        WORK_POSE_JOINTS_DEG,
        "{}_yaw_restore_return_work".format(label_prefix),
        side=side,
        checked_keypoint=True,
    )
    controller.move_pose(
        side,
        pre_grasp,
        original_grasp,
        "{}_pre_grasp_yaw_restored".format(label_prefix),
        pos_cost_weight=IK_POS_COST_WEIGHT,
    )
    return original_yaw_offset, original_grasp, original_lift


def _approach_initial_pre_grasp(controller, job):
    """A/B/C 在 pre-grasp 实测两个 yaw 候选，并保留 FK 误差较小者。"""
    observation = job["observation"]
    side = job["arm"]
    pre_grasp = list(job["grasp_position"])
    pre_grasp[2] += PRE_GRASP_Z_OFFSET
    yaw_offset, grasp_quaternion, lift_quaternion = (
        _approach_pre_grasp_with_yaw_retry(
            controller,
            job["object"],
            side,
            observation,
            pre_grasp,
            job["grasp_yaw_offset"],
            list(job["grasp_quaternion"]),
            list(job["lift_quaternion"]),
            lambda value: geometry.alternate_grasp_yaw_offset(
                observation, value
            ),
            lambda value: geometry.grasp_quaternion(
                observation, side, yaw_offset=value
            ),
            lambda value: geometry.lift_quaternion(
                observation, side, yaw_offset=value
            ),
            job["object"],
        )
    )
    job["grasp_yaw_offset"] = yaw_offset
    job["grasp_quaternion"] = grasp_quaternion
    job["lift_quaternion"] = lift_quaternion


def _approach_relocalized_pre_grasp(controller, job, target_arm, targets):
    """Apply the same two-yaw FK trial to the receiver's visual re-grasp."""
    observation = targets["relocalized_observation"]
    pre_grasp = targets.get("pre_grasp_position")
    if pre_grasp is None:
        pre_grasp = list(targets["grasp_position"])
        pre_grasp[2] += PRE_GRASP_Z_OFFSET
    else:
        pre_grasp = [float(value) for value in pre_grasp]
    label_prefix = "{}_vision_regrasp".format(job["object"])

    def quaternion_for_yaw_offset(yaw_offset):
        return geometry.pose_quaternion(
            targets["grasp_pose_values"],
            observation["yaw"],
            arm=target_arm,
            yaw_offset=yaw_offset,
        )

    yaw_offset, grasp_quaternion, lift_quaternion = (
        _approach_pre_grasp_with_yaw_retry(
            controller,
            job["object"],
            target_arm,
            observation,
            pre_grasp,
            targets["grasp_yaw_offset"],
            list(targets["grasp_quaternion"]),
            list(targets["lift_quaternion"]),
            geometry.flipped_yaw_offset,
            quaternion_for_yaw_offset,
            quaternion_for_yaw_offset,
            label_prefix,
        )
    )
    targets["grasp_yaw_offset"] = yaw_offset
    targets["grasp_quaternion"] = grasp_quaternion
    targets["lift_quaternion"] = lift_quaternion
    targets["pre_grasp_position"] = pre_grasp
    return targets


def _move_to_grasp(
    controller,
    side,
    grasp_position,
    grasp_quaternion,
    label_prefix,
    pre_grasp_position=None,
    pre_grasp_reached=False,
):
    if pre_grasp_position is None:
        pre_grasp = list(grasp_position)
        pre_grasp[2] += PRE_GRASP_Z_OFFSET
    else:
        pre_grasp = [float(value) for value in pre_grasp_position]
        if len(pre_grasp) != 3:
            raise ValueError("explicit pre-grasp position must contain xyz")
    if not pre_grasp_reached:
        controller.gripper.open(side)
        controller.move_pose(
            side,
            pre_grasp,
            grasp_quaternion,
            "{}_pre_grasp".format(label_prefix),
            pos_cost_weight=IK_POS_COST_WEIGHT,
        )
    controller.move_pose(
        side,
        grasp_position,
        grasp_quaternion,
        "{}_grasp".format(label_prefix),
        pos_cost_weight=IK_POS_COST_WEIGHT,
    )


def _grasp_and_close(
    controller,
    side,
    grasp_position,
    grasp_quaternion,
    label_prefix,
    pre_grasp_position=None,
    pre_grasp_reached=False,
):
    _move_to_grasp(
        controller,
        side,
        grasp_position,
        grasp_quaternion,
        label_prefix,
        pre_grasp_position=pre_grasp_position,
        pre_grasp_reached=pre_grasp_reached,
    )
    controller.gripper.close(side)
    rospy.sleep(GRIPPER_CLOSE_TIME)


def _pick_standard_at_pose(
    controller,
    object_name,
    side,
    grasp_position,
    grasp_quaternion,
    lift_quaternion,
    label_prefix,
    pre_grasp_position=None,
    pre_grasp_reached=False,
):
    _grasp_and_close(
        controller,
        side,
        grasp_position,
        grasp_quaternion,
        label_prefix,
        pre_grasp_position=pre_grasp_position,
        pre_grasp_reached=pre_grasp_reached,
    )
    lift_offset = _lift_offset(object_name)
    lift = [
        grasp_position[index] + lift_offset[index] for index in range(3)
    ]
    controller.move_pose(
        side,
        lift,
        lift_quaternion,
        "{}_lift".format(label_prefix),
        pos_cost_weight=IK_LIFT_POS_COST_WEIGHT,
    )
    rospy.loginfo(
        "task2: %s grasped with %s hand at %s",
        object_name,
        side,
        fmt(grasp_position),
    )
    return lift


def _move_c_initial_lift(
    controller,
    side,
    grasp_position,
    lift_quaternion,
    label_prefix,
):
    """只执行 C 类闭爪后的第一段低抬升。"""
    initial_offset = _c_initial_lift_offset()
    initial_lift = [
        grasp_position[index] + initial_offset[index] for index in range(3)
    ]
    controller.move_pose(
        side,
        initial_lift,
        lift_quaternion,
        "{}_initial_lift".format(label_prefix),
        pos_cost_weight=IK_LIFT_POS_COST_WEIGHT,
    )
    return initial_lift


def _pick_c_at_pose(
    controller,
    object_name,
    side,
    grasp_position,
    grasp_quaternion,
    lift_quaternion,
    label_prefix,
    pre_grasp_position=None,
    pre_grasp_reached=False,
):
    """C 类闭爪后先低抬升，再进入最终 lift。"""
    _grasp_and_close(
        controller,
        side,
        grasp_position,
        grasp_quaternion,
        label_prefix,
        pre_grasp_position=pre_grasp_position,
        pre_grasp_reached=pre_grasp_reached,
    )
    initial_lift = _move_c_initial_lift(
        controller,
        side,
        grasp_position,
        lift_quaternion,
        label_prefix,
    )
    lift_offset = _lift_offset(object_name)
    lift = [
        grasp_position[index] + lift_offset[index] for index in range(3)
    ]
    controller.move_pose(
        side,
        lift,
        lift_quaternion,
        "{}_lift".format(label_prefix),
        pos_cost_weight=IK_LIFT_POS_COST_WEIGHT,
    )
    rospy.loginfo(
        "task2: %s grasped with %s hand at %s via C initial lift=%s",
        object_name,
        side,
        fmt(grasp_position),
        fmt(initial_lift),
    )
    return lift


def _pick_at_pose(
    controller,
    object_name,
    side,
    grasp_position,
    grasp_quaternion,
    lift_quaternion,
    label_prefix,
    pre_grasp_position=None,
    pre_grasp_reached=False,
):
    picker = (
        _pick_c_at_pose
        if str(object_name).startswith("part_type_c_")
        else _pick_standard_at_pose
    )
    return picker(
        controller,
        object_name,
        side,
        grasp_position,
        grasp_quaternion,
        lift_quaternion,
        label_prefix,
        pre_grasp_position=pre_grasp_position,
        pre_grasp_reached=pre_grasp_reached,
    )


def _pick_for_source_fk_handoff(controller, job):
    """新交接模式：A 省略 lift，C 仅保留第一段低抬升。"""
    _grasp_and_close(
        controller,
        job["arm"],
        job["grasp_position"],
        job["grasp_quaternion"],
        job["object"],
        pre_grasp_reached=True,
    )
    if job["observation"]["class_name"] == "part_type_c":
        return _move_c_initial_lift(
            controller,
            job["arm"],
            job["grasp_position"],
            job["lift_quaternion"],
            job["object"],
        )
    rospy.loginfo(
        "task2: %s source hand skips final lift before relative handoff",
        job["object"],
    )
    return list(job["grasp_position"])


def _transfer_to_handoff(controller, job, target_arm, targets):
    source_arm = job["arm"]
    receiver_mode = targets["receiver_position_mode"]
    handoff_position = list(targets["place_position"])
    if receiver_mode in ("source_fk_offset", "vision_relocalize"):
        handoff_pre_position = list(targets["pre_place_position"])
        handoff_pre_label = "{}_handoff_pre_place_{}_to_{}".format(
            job["object"], source_arm, target_arm
        )
    else:
        raise RuntimeError("unknown handoff receiver mode: {}".format(receiver_mode))

    controller.move_pose(
        source_arm,
        handoff_pre_position,
        targets["place_quaternion"],
        handoff_pre_label,
        pos_cost_weight=IK_LIFT_POS_COST_WEIGHT,
    )
    source_place_joints = controller.move_pose(
        source_arm,
        handoff_position,
        targets["place_quaternion"],
        "{}_handoff_release_{}_to_{}".format(
            job["object"], source_arm, target_arm
        ),
        pos_cost_weight=IK_LIFT_POS_COST_WEIGHT,
    )
    if receiver_mode == "source_fk_offset":
        source_pose = controller.fk_pose(
            source_arm, joints=source_place_joints
        )
        targets = geometry.resolve_handoff_receiver_positions(
            targets, source_pose.pos_xyz
        )
        rospy.loginfo(
            "task2: %s relative handoff source_fk=%s "
            "receiver_pre=%s receiver_grasp=%s",
            job["object"],
            fmt(targets["source_fk_position"]),
            fmt(targets["pre_grasp_position"]),
            fmt(targets["grasp_position"]),
        )
    elif receiver_mode == "vision_relocalize":
        _release_and_retract_handoff_source(controller, job, targets)
        observation = perception.relocalize_handoff_object(
            job["observation"], targets["place_position"]
        )
        targets = geometry.resolve_relocalized_handoff_receiver(
            targets, observation, target_arm
        )
        rospy.loginfo(
            "task2: %s vision-relocalized receiver grasp=%s yaw=%.2f deg",
            job["object"],
            fmt(targets["grasp_position"]),
            math.degrees(observation["yaw"]),
        )
    else:
        raise RuntimeError("unknown handoff receiver mode: {}".format(receiver_mode))
    return targets


def _release_and_retract_handoff_source(controller, job, targets):
    """视觉重定位模式：释放撤离后双臂统一回工作预位。"""
    source_arm = job["arm"]
    _release_handoff_source_gripper(controller, source_arm)
    controller.move_pose(
        source_arm,
        targets["pre_place_position"],
        targets["place_quaternion"],
        "{}_handoff_retract_{}".format(job["object"], source_arm),
        pos_cost_weight=IK_LIFT_POS_COST_WEIGHT,
    )
    controller.move_joints(
        WORK_POSE_JOINTS_DEG,
        "{}_both_ready_for_relocalization".format(job["object"]),
        side="both",
        checked_keypoint=True,
    )
    rospy.loginfo(
        "task2: %s both arms stable at work pose; vision relocalization may start",
        job["object"],
    )


def _release_handoff_source_gripper(controller, source_arm):
    """只松开给出手，不改变其手臂位置。"""
    controller.gripper.open(source_arm)
    rospy.sleep(RELEASE_DWELL)


def _return_handoff_source_to_work(controller, job):
    """给出手从当前位置直接回工作预位。"""
    source_arm = job["arm"]
    controller.move_joints(
        WORK_POSE_JOINTS_DEG,
        "{}_source_return_work".format(job["object"]),
        side=source_arm,
        checked_keypoint=True,
    )


def _receive_source_fk_handoff(controller, job, target_arm, targets):
    """A/C 空中接手：到达 grasp 稳定后按有符号延迟切换夹爪。"""
    if targets["receiver_position_mode"] != "resolved_source_fk":
        raise ValueError("source-FK handoff receiver targets are not resolved")
    _move_to_grasp(
        controller,
        target_arm,
        targets["grasp_position"],
        targets["grasp_quaternion"],
        "{}_handoff_receive".format(job["object"]),
        pre_grasp_position=targets["pre_grasp_position"],
    )
    class_name = str(job["observation"]["class_name"])
    try:
        close_delay = float(HANDOFF_RECEIVER_CLOSE_DELAY_BY_CLASS[class_name])
    except KeyError:
        raise ValueError(
            "no handoff receiver close delay for {}".format(class_name)
        )
    if not math.isfinite(close_delay):
        raise ValueError(
            "handoff receiver close delay for {} must be finite".format(class_name)
        )
    rospy.loginfo(
        "task2: %s handoff signed receiver-close offset=%.3f s simulation time",
        job["object"],
        close_delay,
    )
    if close_delay < 0.0:
        sim_time_marker = controller.sim_time.marker()
        controller.gripper.close(target_arm)
        sim_start, sim_end = controller.sim_time.wait_duration(
            abs(close_delay), after_sequence=sim_time_marker
        )
        rospy.loginfo(
            "task2: %s receiver-close -> source-open simulated delay "
            "%.6f -> %.6f (%.3f s)",
            job["object"],
            sim_start,
            sim_end,
            sim_end - sim_start,
        )
        controller.gripper.open(job["arm"])
    else:
        sim_time_marker = (
            controller.sim_time.marker() if close_delay > 0.0 else None
        )
        controller.gripper.open(job["arm"])
        if close_delay > 0.0:
            sim_start, sim_end = controller.sim_time.wait_duration(
                close_delay, after_sequence=sim_time_marker
            )
            rospy.loginfo(
                "task2: %s source-open -> receiver-close simulated delay "
                "%.6f -> %.6f (%.3f s)",
                job["object"],
                sim_start,
                sim_end,
                sim_end - sim_start,
            )
        controller.gripper.close(target_arm)
    rospy.sleep(GRIPPER_CLOSE_TIME)

    # 给出手保持交接位，接手先带工件退回 pre，再允许给出手回工作预位。
    controller.move_pose(
        target_arm,
        targets["pre_grasp_position"],
        targets["grasp_quaternion"],
        "{}_handoff_receive_retreat_pre".format(job["object"]),
        pos_cost_weight=IK_POS_COST_WEIGHT,
    )
    _return_handoff_source_to_work(controller, job)


def _place_in_bin(
    controller,
    object_name,
    side,
    bin_name,
    held_quaternion,
):
    place_position = _place_position(side, bin_name, object_name)
    if str(object_name).startswith("part_type_c_"):
        controller.move_pose_via_ik(
            side,
            [],
            place_position,
            held_quaternion,
            "{}_place_{}".format(object_name, bin_name),
            move_time=ARM_MOVE_TIME,
            cartesian_subdivisions=C_PLACE_IK_SUBDIVISIONS,
        )
    else:
        controller.move_pose(
            side,
            place_position,
            held_quaternion,
            "{}_place_{}".format(object_name, bin_name),
            move_time=ARM_MOVE_TIME,
            checked_keypoint=True,
        )
    controller.gripper.open(side)
    rospy.sleep(RELEASE_DWELL)
    controller.move_joints(
        WORK_POSE_JOINTS_DEG,
        "{}_return_work".format(object_name),
        side=side,
        checked_keypoint=True,
    )
    rospy.loginfo("task2: %s released into %s", object_name, bin_name)


def run_scene2_task(seed, object_poses=None):
    """运行完整任务；视觉来源替换后动作层不需要修改。"""
    rospy.loginfo("=" * 60)
    rospy.loginfo("task2 seed=%d", seed)
    rospy.loginfo("=" * 60)

    controller = None
    try:
        observations = object_poses or perception.wait_for_task2_objects()
        for name in sorted(observations):
            item = observations[name]
            rospy.loginfo(
                "task2: motion=%s ground_truth=%s base_link=%s yaw=%.2f deg source=%s",
                name,
                item.get("ground_truth_name", name),
                fmt(item["position"]),
                math.degrees(item["yaw"]),
                item["position_source"],
            )

        jobs = geometry.build_jobs(observations)
        controller = RobotController()
        controller.gripper.open_all()
        controller.move_to_preset()
        controller.sync_command_to_actual()
        rospy.loginfo(
            "task2: pick order=%s",
            " -> ".join(job["object"] for job in jobs),
        )

        for index, job in enumerate(jobs, start=1):
            rospy.loginfo(
                "--- task2 %d/%d %s (%s) -> %s ---",
                index,
                len(jobs),
                job["object"],
                job["arm"],
                job["bin"],
            )
            target_arm = geometry.handoff_target_arm(job)
            handoff_targets = (
                geometry.handoff_values(job, target_arm)
                if target_arm is not None
                else None
            )
            use_source_fk_handoff = (
                handoff_targets is not None
                and handoff_targets["receiver_position_mode"]
                == "source_fk_offset"
            )
            _approach_initial_pre_grasp(controller, job)
            if use_source_fk_handoff:
                _pick_for_source_fk_handoff(controller, job)
            else:
                _pick_at_pose(
                    controller,
                    job["object"],
                    job["arm"],
                    job["grasp_position"],
                    job["grasp_quaternion"],
                    job["lift_quaternion"],
                    job["object"],
                    pre_grasp_reached=True,
                )

            if target_arm is not None:
                targets = _transfer_to_handoff(
                    controller, job, target_arm, handoff_targets
                )
                receiver_mode = targets["receiver_position_mode"]
                if receiver_mode == "resolved_source_fk":
                    _receive_source_fk_handoff(
                        controller, job, target_arm, targets
                    )
                elif receiver_mode == "resolved_vision_relocalize":
                    targets = _approach_relocalized_pre_grasp(
                        controller, job, target_arm, targets
                    )
                    _pick_at_pose(
                        controller,
                        job["object"],
                        target_arm,
                        targets["grasp_position"],
                        targets["grasp_quaternion"],
                        targets["lift_quaternion"],
                        "{}_vision_regrasp".format(job["object"]),
                        pre_grasp_position=targets["pre_grasp_position"],
                        pre_grasp_reached=True,
                    )
                else:
                    raise RuntimeError(
                        "unresolved handoff receiver mode: {}".format(receiver_mode)
                    )
                _place_in_bin(
                    controller,
                    job["object"],
                    target_arm,
                    job["bin"],
                    targets["grasp_quaternion"],
                )
            else:
                _place_in_bin(
                    controller,
                    job["object"],
                    job["arm"],
                    job["bin"],
                    job["lift_quaternion"],
                )

        controller.gripper.open_all()
        controller.move_joints(
            SIDE_LIFT_JOINTS_DEG,
            "final_side_lift",
            checked_keypoint=False,
        )
        controller.move_joints(
            HOME_JOINTS_DEG,
            "final_home",
            checked_keypoint=False,
        )
        rospy.loginfo("task2: all six parts sorted for seed=%d", seed)
        return True
    except Exception:
        if rospy.is_shutdown():
            rospy.loginfo("task2: interrupted by ROS shutdown")
            return False
        rospy.logerr("task2 exception: %s", traceback.format_exc())
        return False
    finally:
        if controller is not None:
            controller.stop()
