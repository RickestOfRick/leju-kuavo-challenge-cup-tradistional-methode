#!/usr/bin/env python3
"""任务二六个工件的顺序抓取、交接和分类放箱调度。"""

import math
import traceback

import rospy

import task2_geometry as geometry
import task2_perception as perception
from task2_config import (
    ARM_MOVE_TIME,
    GRIPPER_CLOSE_TIME,
    HANDOFF_APPROACH_Z_OFFSET,
    HOME_JOINTS_DEG,
    IK_LIFT_POS_COST_WEIGHT,
    IK_POS_COST_WEIGHT,
    LIFT_OFFSETS_BY_CLASS,
    PLACE_POSITIONS_BASE,
    PRE_GRASP_Z_OFFSET,
    RELEASE_DWELL,
    SIDE_LIFT_JOINTS_DEG,
    WORK_POSE_JOINTS_DEG,
)
from task2_motion import RobotController, fmt


def _place_position(side, bin_name):
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


def _pick_at_pose(
    controller,
    object_name,
    side,
    grasp_position,
    grasp_quaternion,
    lift_quaternion,
    label_prefix,
):
    pre_grasp = list(grasp_position)
    pre_grasp[2] += PRE_GRASP_Z_OFFSET
    lift_offset = _lift_offset(object_name)
    lift = [
        grasp_position[index] + lift_offset[index] for index in range(3)
    ]

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
    )
    return targets


def _place_in_bin(
    controller,
    object_name,
    side,
    bin_name,
    held_quaternion,
):
    place_position = _place_position(side, bin_name)
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
                "task2: %s base_link=%s yaw=%.2f deg source=%s",
                name,
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
            _pick_at_pose(
                controller,
                job["object"],
                job["arm"],
                job["grasp_position"],
                job["grasp_quaternion"],
                job["lift_quaternion"],
                job["object"],
            )

            target_arm = geometry.handoff_target_arm(job)
            if target_arm is not None:
                targets = _transfer_to_handoff(controller, job, target_arm)
                _pick_at_pose(
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
        controller.move_joints(WORK_POSE_JOINTS_DEG, "final_work_pose")
        controller.move_joints(SIDE_LIFT_JOINTS_DEG, "final_side_lift")
        controller.move_joints(HOME_JOINTS_DEG, "final_home")
        rospy.loginfo("task2: all six parts sorted for seed=%d", seed)
        return True
    except Exception:
        rospy.logerr("task2 exception: %s", traceback.format_exc())
        return False
    finally:
        if controller is not None:
            controller.stop()
