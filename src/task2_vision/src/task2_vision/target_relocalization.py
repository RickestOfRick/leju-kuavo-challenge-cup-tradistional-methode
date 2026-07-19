"""Merge one target update into a locked multi-object pose payload."""

import copy
import statistics


def object_motion_name(item):
    return str(item.get("motion_name") or item.get("name", ""))


def create_target_session(request, base_payload, required_samples):
    """Create mutable state for a target-only relocalization request."""
    required = max(1, int(required_samples))
    return {
        "request_id": str(request["request_id"]),
        "name": str(request["name"]),
        "class_name": str(request["class_name"]),
        "expected_position": [
            float(value) for value in request["expected_position"]
        ],
        "required_samples": required,
        "samples": [],
        "attempts": 0,
        "base_payload": copy.deepcopy(base_payload),
    }


def record_target_fit(session, fit):
    """Record one attempted frame and return an immutable state snapshot."""
    session["attempts"] += 1
    if fit is not None and bool(fit.get("accepted", False)):
        session["samples"].append(copy.deepcopy(fit))
    return {
        "attempts": int(session["attempts"]),
        "samples": copy.deepcopy(session["samples"]),
        "ready": len(session["samples"]) >= int(session["required_samples"]),
    }


def aggregate_target_pose(samples, required_samples, circular_median):
    """Return the exact XYZ/yaw aggregate shared by RQT and the motion result."""
    required = max(1, int(required_samples))
    if len(samples) < required:
        raise ValueError("not enough target samples to aggregate")
    selected = list(samples[-required:])
    position = [
        float(statistics.median(float(sample["xy"][axis]) for sample in selected))
        for axis in range(2)
    ]
    position.append(
        float(statistics.median(float(sample["z"]) for sample in selected))
    )
    yaw = float(circular_median([float(sample["yaw"]) for sample in selected]))
    return selected, position, yaw


def merge_target_payload(
    base_payload,
    target_name,
    target_update,
    request_id,
    status,
    locked,
    sample_count,
    required_samples,
    stamp,
):
    """Replace exactly one object while preserving all other motion data."""
    payload = copy.deepcopy(base_payload)
    objects = []
    replaced = 0
    for source in payload.get("objects", []):
        item = copy.deepcopy(source)
        if object_motion_name(item) == str(target_name):
            item = copy.deepcopy(target_update)
            item["name"] = str(target_name)
            item["motion_name"] = str(target_name)
            item["fit_status"] = str(status)
            item["fit_diagnostic_status"] = str(status)
            item["fit_sample_count"] = int(sample_count)
            item["fit_required_samples"] = int(required_samples)
            item["fit_valid"] = bool(locked)
            replaced += 1
        objects.append(item)
    if replaced != 1:
        raise ValueError(
            "target {} matched {} payload objects".format(target_name, replaced)
        )

    payload["stamp"] = float(stamp)
    payload["objects"] = objects
    payload["frozen"] = bool(locked)
    payload["locked"] = bool(locked)
    payload["ready_for_motion"] = bool(locked)
    payload["relocalization_active"] = not bool(locked)
    payload["relocalization_request_id"] = str(request_id)
    payload["relocalization_target"] = str(target_name)
    payload["relocalization_status"] = str(status)
    payload["selection_sample_count"] = int(sample_count)
    payload["required_valid_frames"] = int(required_samples)
    return payload
