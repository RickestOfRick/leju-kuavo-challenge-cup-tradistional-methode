#!/usr/bin/env python3
"""任务二六个工件的顺序抓取、交接和分类放箱调度。"""

import math
import traceback

import rospy

import task2_geometry as geometry
import task2_perception as perception
from task2_config import (
    AB_PRE_GRASP_ORIENTATION_RETRY_DEG,
    ARM_MOVE_TIME,
    C_PLACE_IK_Z_FRACTIONS,
    GRIPPER_CLOSE_TIME,
    HANDOFF_APPROACH_Z_OFFSET,
    HOME_JOINTS_DEG,
    IK_LIFT_POS_COST_WEIGHT,
    IK_POS_COST_WEIGHT,
    LIFT_OFFSETS_BY_CLASS,
    PLACE_POSITIONS_BASE,
    PLACE_Y_OFFSETS_BY_CLASS,
    PRE_GRASP_Z_OFFSET,
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


def _approach_initial_pre_grasp(controller, job):
    """A/B 在 pre-grasp 实测两种窄边姿态，并保留 FK 误差较小者。"""
    object_name = job["object"]
    side = job["arm"]
    observation = job["observation"]
    pre_grasp = list(job["grasp_position"])
    pre_grasp[2] += PRE_GRASP_Z_OFFSET
    original_grasp = list(job["grasp_quaternion"])

    controller.gripper.open(side)
    _, original_error = controller.move_pose(
        side,
        pre_grasp,
        original_grasp,
        "{}_pre_grasp".format(object_name),
        pos_cost_weight=IK_POS_COST_WEIGHT,
        return_orientation_error=True,
    )
    original_error_deg = math.degrees(original_error)
    if (
        observation["class_name"] not in ("part_type_a", "part_type_b")
        or original_error_deg <= AB_PRE_GRASP_ORIENTATION_RETRY_DEG
    ):
        return

    flipped_yaw_offset = -float(job["grasp_yaw_offset"])
    flipped_grasp = geometry.grasp_quaternion(
        observation, side, yaw_offset=flipped_yaw_offset
    )
    flipped_lift = geometry.lift_quaternion(
        observation, side, yaw_offset=flipped_yaw_offset
    )
    rospy.logwarn(
        "task2: [%s_pre_grasp] ori_err=%.2f deg > %.2f deg; "
        "returning to work pose and trying the 180-deg yaw alternative",
        object_name,
        original_error_deg,
        AB_PRE_GRASP_ORIENTATION_RETRY_DEG,
    )
    controller.move_joints(
        WORK_POSE_JOINTS_DEG,
        "{}_yaw_retry_return_work".format(object_name),
        side=side,
        checked_keypoint=True,
    )
    flipped_error = None
    flipped_failure = None
    try:
        _, flipped_error = controller.move_pose(
            side,
            pre_grasp,
            flipped_grasp,
            "{}_pre_grasp_yaw_flipped".format(object_name),
            pos_cost_weight=IK_POS_COST_WEIGHT,
            return_orientation_error=True,
        )
    except RuntimeError as error:
        if rospy.is_shutdown():
            raise
        flipped_failure = str(error)

    if flipped_error is not None and flipped_error < original_error:
        flipped_error_deg = math.degrees(flipped_error)
        job["grasp_yaw_offset"] = flipped_yaw_offset
        job["grasp_quaternion"] = flipped_grasp
        job["lift_quaternion"] = flipped_lift
        rospy.logwarn(
            "task2: %s selected flipped yaw at pre-grasp: "
            "ori_err %.2f -> %.2f deg",
            object_name,
            original_error_deg,
            flipped_error_deg,
        )
        return

    if flipped_error is None:
        rospy.logwarn(
            "task2: %s flipped yaw pre-grasp failed (%s); restoring original yaw",
            object_name,
            flipped_failure,
        )
    else:
        rospy.logwarn(
            "task2: %s flipped yaw did not improve pre-grasp ori_err "
            "(original=%.2f deg flipped=%.2f deg); restoring original yaw",
            object_name,
            original_error_deg,
            math.degrees(flipped_error),
        )
    controller.move_joints(
        WORK_POSE_JOINTS_DEG,
        "{}_yaw_restore_return_work".format(object_name),
        side=side,
        checked_keypoint=True,
    )
    controller.move_pose(
        side,
        pre_grasp,
        original_grasp,
        "{}_pre_grasp_yaw_restored".format(object_name),
        pos_cost_weight=IK_POS_COST_WEIGHT,
    )


def _pick_at_pose(
    controller,
    object_name,
    side,
    grasp_position,
    grasp_quaternion,
    lift_quaternion,
    label_prefix,
    pre_grasp_reached=False,
):
    pre_grasp = list(grasp_position)
    pre_grasp[2] += PRE_GRASP_Z_OFFSET
    lift_offset = _lift_offset(object_name)
    lift = [
        grasp_position[index] + lift_offset[index] for index in range(3)
    ]

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
    controller.gripper.close(side)
    rospy.sleep(GRIPPER_CLOSE_TIME)
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


def _transfer_to_handoff(controller, job, target_arm):
    source_arm = job["arm"]
    targets = geometry.handoff_values(job, target_arm)
    handoff_position = list(targets["place_position"])
    handoff_above = list(handoff_position)
    handoff_above[2] += HANDOFF_APPROACH_Z_OFFSET

    controller.move_pose(
        source_arm,
        handoff_above,
        targets["place_quaternion"],
        "{}_handoff_above_{}_to_{}".format(
            job["object"], source_arm, target_arm
        ),
        pos_cost_weight=IK_LIFT_POS_COST_WEIGHT,
    )
    controller.move_pose(
        source_arm,
        handoff_position,
        targets["place_quaternion"],
        "{}_handoff_release_{}_to_{}".format(
            job["object"], source_arm, target_arm
        ),
        pos_cost_weight=IK_LIFT_POS_COST_WEIGHT,
    )
    controller.gripper.open(source_arm)
    rospy.sleep(RELEASE_DWELL)
    controller.move_pose(
        source_arm,
        handoff_above,
        targets["place_quaternion"],
        "{}_handoff_retract_{}".format(job["object"], source_arm),
        pos_cost_weight=IK_LIFT_POS_COST_WEIGHT,
    )
    controller.move_joints(
        WORK_POSE_JOINTS_DEG,
        "{}_source_return_work".format(job["object"]),
        side=source_arm,
        checked_keypoint=True,
    )
    return targets


def _place_in_bin(
    controller,
    object_name,
    side,
    bin_name,
    held_quaternion,
    lift_position,
):
    place_position = _place_position(side, bin_name, object_name)
    if str(object_name).startswith("part_type_c_"):
        fractions = [float(value) for value in C_PLACE_IK_Z_FRACTIONS]
        if not fractions or any(
            value <= 0.0 or value >= 1.0 for value in fractions
        ):
            raise ValueError("C place IK Z fractions must be within (0, 1)")
        if fractions != sorted(fractions):
            raise ValueError("C place IK Z fractions must be ascending")
        intermediate_positions = [
            [
                float(lift_position[0]),
                float(place_position[1]),
                float(lift_position[2])
                + fraction
                * (float(place_position[2]) - float(lift_position[2])),
            ]
            for fraction in fractions
        ]
        controller.move_pose_via_ik(
            side,
            intermediate_positions,
            place_position,
            held_quaternion,
            "{}_place_{}".format(object_name, bin_name),
            move_time=ARM_MOVE_TIME,
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
            _approach_initial_pre_grasp(controller, job)
            lift_position = _pick_at_pose(
                controller,
                job["object"],
                job["arm"],
                job["grasp_position"],
                job["grasp_quaternion"],
                job["lift_quaternion"],
                job["object"],
                pre_grasp_reached=True,
            )

            target_arm = geometry.handoff_target_arm(job)
            if target_arm is not None:
                targets = _transfer_to_handoff(controller, job, target_arm)
                lift_position = _pick_at_pose(
                    controller,
                    job["object"],
                    target_arm,
                    targets["grasp_position"],
                    targets["grasp_quaternion"],
                    targets["grasp_quaternion"],
                    "{}_regrasp".format(job["object"]),
                )
                _place_in_bin(
                    controller,
                    job["object"],
                    target_arm,
                    job["bin"],
                    targets["grasp_quaternion"],
                    lift_position,
                )
            else:
                _place_in_bin(
                    controller,
                    job["object"],
                    job["arm"],
                    job["bin"],
                    job["lift_quaternion"],
                    lift_position,
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
