#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Count task2 object yaw and XY values from the protected scene generator.

This script deliberately does not initialize ROS or start the simulator. It
loads the same ``challenge_secret`` extension and calls the same
``get_object_layout("scene2", seed)`` entry point used by the simulator.
"""

import glob
import json
import math
import multiprocessing
import os
import re
import sys
import time
from collections import Counter

# =============================================================================
# Configuration
# =============================================================================

SCENE_NAME = "scene2"
ROBOT_VERSION = 52

# Inclusive ranges, matching the default task2 test range.
SEED_RANGES = ((-10000000, 10000000),)

# atan2 results contain floating-point tails even when the generator uses a
# discrete angle set. Values equal at this precision are counted together.
YAW_DEG_DECIMALS = 6
# 同一仿真物体的 XY 逐轴误差不超过该值时归为同一位置。
XY_MATCH_TOLERANCE = 1e-6
XY_OUTPUT_DECIMALS = 6
# 统计需要交接的工件同时落在该 base_link/world Y 中心范围内的 seed。
# 当前仿真初始基座 yaw=0、Y=0，因此 .so world Y 与 motion base_link Y 一致。
HANDOFF_ABS_Y_MAX = 0.15

# Layout statistics only reads challenge_secret data and does not start the
# simulator. Keep launcher source-integrity verification opt-in so local
# launcher edits do not block this read-only inspection.
VERIFY_SOURCE_FILES = False
PRINT_EACH_SEED = False
WRITE_JSON = True

# 0 uses every CPU available to this process. Each worker imports the same
# read-only extension and handles a continuous seed chunk locally.
PARALLEL_WORKERS = 20
SEED_CHUNK_SIZE = 20000
PROGRESS_EVERY_SEEDS = 100000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "debug_output", "task2_yaw_counts.json")

OBJECTS_BY_CLASS = {
    "part_type_a": ("part_type_a_1", "part_type_a_2"),
    "part_type_b": ("part_type_b_1", "part_type_b_2"),
    "part_type_c": ("part_type_c_1", "part_type_c_2"),
}
OBJECT_NAMES = tuple(
    object_name
    for class_name in ("part_type_a", "part_type_b", "part_type_c")
    for object_name in OBJECTS_BY_CLASS[class_name]
)

_WORKER_GET_LAYOUT = None


def _find_simulator_path():
    """Locate challenge_cup_simulator exactly as the launcher does."""
    try:
        import rospkg

        return os.path.abspath(rospkg.RosPack().get_path("challenge_cup_simulator"))
    except Exception:
        workspace_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
        candidate = os.path.join(workspace_root, "src", "challenge_cup_simulator")
        if os.path.isfile(os.path.join(candidate, "package.xml")):
            return candidate
        raise RuntimeError(
            "cannot locate challenge_cup_simulator via rospkg or repository path"
        )


def _extension_abi(path):
    """Return (major, minor) parsed from a CPython extension filename."""
    match = re.search(r"\.cpython-(\d)(\d+)[^/]*\.so$", path)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _find_secret_extension(lib_dir):
    candidates = sorted(glob.glob(os.path.join(lib_dir, "challenge_secret*.so")))
    if not candidates:
        raise RuntimeError(
            "challenge_secret extension is missing from {}".format(lib_dir)
        )

    running_abi = (sys.version_info.major, sys.version_info.minor)
    compatible = [
        path for path in candidates if _extension_abi(path) in (None, running_abi)
    ]
    if compatible:
        return compatible[0]

    required = sorted(
        set(abi for abi in map(_extension_abi, candidates) if abi is not None)
    )
    required_text = ", ".join(
        "CPython {}.{}".format(major, minor) for major, minor in required
    )
    raise RuntimeError(
        "challenge_secret ABI mismatch: running CPython {}.{}, but {} contains "
        "an extension for {}. Run this script with the ROS workspace's matching "
        "Python interpreter.".format(
            running_abi[0], running_abi[1], lib_dir, required_text or "another ABI"
        )
    )


def _load_secret_module(simulator_path):
    lib_dir = os.path.join(simulator_path, "lib")
    extension_path = _find_secret_extension(lib_dir)
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)

    try:
        import challenge_secret
    except ImportError as exc:
        raise RuntimeError("cannot import {}: {}".format(extension_path, exc)) from exc
    return challenge_secret, extension_path


def _verify_sources(challenge_secret, simulator_path):
    if not VERIFY_SOURCE_FILES:
        return
    verify = getattr(challenge_secret, "verify_source_files", None)
    if verify is None:
        raise RuntimeError("challenge_secret does not export verify_source_files")
    passed, messages = verify(simulator_path, ROBOT_VERSION)
    for message in messages:
        print("[{}] {}".format("INFO" if passed else "ERROR", message))
    if not passed:
        raise RuntimeError("challenge_cup_simulator source integrity check failed")


def _normalize_seed_ranges(seed_ranges):
    ranges = []
    for raw_start, raw_end in seed_ranges:
        start = int(raw_start)
        end = int(raw_end)
        if end < start:
            raise ValueError("invalid inclusive seed range: {}..{}".format(start, end))
        ranges.append((start, end))
    if not ranges:
        raise ValueError("SEED_RANGES selects no seeds")

    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _seed_count(seed_ranges):
    return sum(end - start + 1 for start, end in seed_ranges)


def _iter_seeds(seed_ranges):
    for start, end in seed_ranges:
        for seed in range(start, end + 1):
            yield seed


def _iter_seed_chunks(seed_ranges, chunk_size):
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("SEED_CHUNK_SIZE must be positive")
    for range_start, range_end in seed_ranges:
        start = range_start
        while start <= range_end:
            end = min(range_end, start + chunk_size - 1)
            yield start, end
            start = end + 1


def _available_cpu_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def _resolve_worker_count(seed_count):
    configured = int(PARALLEL_WORKERS)
    if configured < 0:
        raise ValueError("PARALLEL_WORKERS must be zero or positive")
    workers = _available_cpu_count() if configured == 0 else configured
    if PRINT_EACH_SEED:
        return 1
    return max(1, min(workers, int(seed_count)))


def _yaw_from_quat_wxyz(quat):
    if not isinstance(quat, (list, tuple)) or len(quat) != 4:
        raise ValueError("quat must be [w, x, y, z]")
    qw, qx, qy, qz = [float(value) for value in quat]
    if not all(math.isfinite(value) for value in (qw, qx, qy, qz)):
        raise ValueError("quat contains a non-finite value")
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        raise ValueError("quat norm is zero")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    # Keep one canonical representation: [-pi, pi), so +180 and -180 do not
    # become separate bins for the same physical orientation.
    if yaw >= math.pi:
        yaw -= 2.0 * math.pi
    return yaw


def _yaw_key_deg(yaw_rad):
    value = round(math.degrees(yaw_rad), int(YAW_DEG_DECIMALS))
    # Avoid a separate JSON/print bin named -0.0.
    return 0.0 if value == 0.0 else value


def _selected_arm_for_world_y(object_name, object_y):
    object_name = str(object_name)
    object_y = float(object_y)
    if object_name.startswith("part_type_a") and -0.2 <= object_y <= 0.1:
        return "right"
    if object_name.startswith("part_type_c") and -0.2 <= object_y <= 0.0:
        return "left"
    return "left" if object_y >= 0.0 else "right"


def _requires_handoff(object_name, object_y):
    arm = _selected_arm_for_world_y(object_name, object_y)
    if str(object_name).startswith("part_type_a"):
        return arm == "left"
    if str(object_name).startswith("part_type_c"):
        return arm == "right"
    return False


def _validate_layout(layout, seed):
    if not isinstance(layout, dict) or not layout:
        raise RuntimeError("seed {} returned an empty/non-dict layout".format(seed))
    missing = [name for name in OBJECT_NAMES if name not in layout]
    if missing:
        raise RuntimeError(
            "seed {} layout is missing: {}".format(seed, ", ".join(missing))
        )


def _counter_rows(counter):
    rows = []
    for yaw_deg, count in sorted(counter.items()):
        rows.append(
            {
                "yaw_deg": yaw_deg,
                "yaw_rad": round(math.radians(yaw_deg), 12),
                "count": int(count),
            }
        )
    return rows


def _matching_xy_keys(counter, xy):
    x, y = xy
    tolerance = float(XY_MATCH_TOLERANCE)
    return [
        key
        for key in counter
        if abs(float(key[0]) - x) <= tolerance
        and abs(float(key[1]) - y) <= tolerance
    ]


def _add_xy_count(counter, xy, count=1):
    try:
        x, y = [float(value) for value in xy[:2]]
    except (TypeError, ValueError) as exc:
        raise ValueError("XY must contain two numeric values") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("XY contains a non-finite value")
    count = int(count)
    if count <= 0:
        raise ValueError("XY count must be positive")

    matches = _matching_xy_keys(counter, (x, y))
    if not matches:
        counter[(x, y)] += count
        return

    # Keep a deterministic representative while also collapsing any matching
    # groups created independently by parallel workers.
    representative = min(matches + [(x, y)])
    total = count
    for key in matches:
        total += int(counter.pop(key))
    counter[representative] += total


def _xy_counter_rows(counter):
    return [
        {
            "x": round(float(xy[0]), int(XY_OUTPUT_DECIMALS)),
            "y": round(float(xy[1]), int(XY_OUTPUT_DECIMALS)),
            "count": int(count),
        }
        for xy, count in sorted(counter.items())
    ]


def _print_counter(title, counter):
    print(
        "\n{} ({} samples, {} yaw values)".format(
            title, sum(counter.values()), len(counter)
        )
    )
    print("  {:>14s}  {:>16s}  {:>8s}".format("yaw_deg", "yaw_rad", "count"))
    for row in _counter_rows(counter):
        yaw_deg_text = ("{:.%df}" % int(YAW_DEG_DECIMALS)).format(row["yaw_deg"])
        print(
            "  {:>14s}  {:>16.12f}  {:>8d}".format(
                yaw_deg_text,
                row["yaw_rad"],
                row["count"],
            )
        )


def _print_xy_counter(title, counter):
    print(
        "\n{} ({} samples, {} XY positions)".format(
            title, sum(counter.values()), len(counter)
        )
    )
    print("  {:>14s}  {:>14s}  {:>8s}".format("x", "y", "count"))
    value_format = "{:.%df}" % int(XY_OUTPUT_DECIMALS)
    for row in _xy_counter_rows(counter):
        print(
            "  {:>14s}  {:>14s}  {:>8d}".format(
                value_format.format(row["x"]),
                value_format.format(row["y"]),
                row["count"],
            )
        )


def _empty_counts():
    return (
        {class_name: Counter() for class_name in OBJECTS_BY_CLASS},
        {object_name: Counter() for object_name in OBJECT_NAMES},
        {object_name: Counter() for object_name in OBJECT_NAMES},
    )


def _collect_layout_counts_from_get_layout(get_layout, seeds):
    class_counts, object_counts, object_xy_counts = _empty_counts()
    yaw_cache = {}
    handoff_within_y_seed_count = 0

    for seed in seeds:
        layout = get_layout(SCENE_NAME, int(seed))
        _validate_layout(layout, seed)
        seed_values = [] if PRINT_EACH_SEED else None
        seed_has_handoff_within_y = False
        for class_name, object_names in OBJECTS_BY_CLASS.items():
            for object_name in object_names:
                item = layout[object_name]
                if (
                    not isinstance(item, dict)
                    or "quat" not in item
                    or "pos" not in item
                ):
                    raise RuntimeError(
                        "seed {} object {} has no quaternion or position".format(
                            seed, object_name
                        )
                    )
                try:
                    quat_key = tuple(float(value) for value in item["quat"])
                    yaw_deg = yaw_cache.get(quat_key)
                    if yaw_deg is None:
                        yaw_deg = _yaw_key_deg(_yaw_from_quat_wxyz(quat_key))
                        yaw_cache[quat_key] = yaw_deg
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "seed {} object {} has invalid quaternion: {}".format(
                            seed, object_name, exc
                        )
                    ) from exc
                class_counts[class_name][yaw_deg] += 1
                object_counts[object_name][yaw_deg] += 1
                try:
                    _add_xy_count(object_xy_counts[object_name], item["pos"])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "seed {} object {} has invalid position: {}".format(
                            seed, object_name, exc
                        )
                    ) from exc
                object_y = float(item["pos"][1])
                if (
                    abs(object_y) <= HANDOFF_ABS_Y_MAX
                    and _requires_handoff(object_name, object_y)
                ):
                    seed_has_handoff_within_y = True
                if seed_values is not None:
                    seed_values.append(
                        "{}=yaw:{:.6f},xy:[{:.6f},{:.6f}]".format(
                            object_name,
                            yaw_deg,
                            float(item["pos"][0]),
                            float(item["pos"][1]),
                        )
                    )
        if PRINT_EACH_SEED:
            print("seed {}: {}".format(seed, ", ".join(seed_values)))
        if seed_has_handoff_within_y:
            handoff_within_y_seed_count += 1

    return (
        class_counts,
        object_counts,
        object_xy_counts,
        handoff_within_y_seed_count,
    )


def collect_layout_counts(challenge_secret, seeds):
    get_layout = getattr(challenge_secret, "get_object_layout", None)
    if get_layout is None:
        raise RuntimeError("challenge_secret does not export get_object_layout")
    return _collect_layout_counts_from_get_layout(get_layout, seeds)


def _worker_initialize(simulator_path):
    global _WORKER_GET_LAYOUT
    challenge_secret, _ = _load_secret_module(simulator_path)
    get_layout = getattr(challenge_secret, "get_object_layout", None)
    if get_layout is None:
        raise RuntimeError("challenge_secret does not export get_object_layout")
    _WORKER_GET_LAYOUT = get_layout


def _worker_count_chunk(seed_chunk):
    if _WORKER_GET_LAYOUT is None:
        raise RuntimeError("parallel worker was not initialized")
    start, end = seed_chunk
    (
        class_counts,
        object_counts,
        object_xy_counts,
        handoff_within_y_seed_count,
    ) = _collect_layout_counts_from_get_layout(
        _WORKER_GET_LAYOUT,
        range(start, end + 1),
    )
    return (
        end - start + 1,
        class_counts,
        object_counts,
        object_xy_counts,
        handoff_within_y_seed_count,
    )


def _merge_counts(target, source):
    for name, counter in source.items():
        target[name].update(counter)


def _merge_xy_counts(target, source):
    for object_name, counter in source.items():
        for xy, count in counter.items():
            _add_xy_count(target[object_name], xy, count=count)


def collect_layout_counts_parallel(
    simulator_path, seed_ranges, seed_count, worker_count
):
    class_counts, object_counts, object_xy_counts = _empty_counts()
    handoff_within_y_seed_count = 0
    chunks = _iter_seed_chunks(seed_ranges, SEED_CHUNK_SIZE)
    processed = 0
    next_progress = max(1, int(PROGRESS_EVERY_SEEDS))
    started_at = time.monotonic()

    # The target simulator is Linux. fork avoids repeatedly starting a fresh
    # Python interpreter while still giving each process an independent GIL.
    context = multiprocessing.get_context("fork")
    sys.stdout.flush()
    with context.Pool(
        processes=worker_count,
        initializer=_worker_initialize,
        initargs=(simulator_path,),
    ) as pool:
        results = pool.imap_unordered(_worker_count_chunk, chunks, chunksize=1)
        for (
            chunk_count,
            chunk_classes,
            chunk_objects,
            chunk_object_xy,
            chunk_handoff_within_y_seed_count,
        ) in results:
            processed += int(chunk_count)
            _merge_counts(class_counts, chunk_classes)
            _merge_counts(object_counts, chunk_objects)
            _merge_xy_counts(object_xy_counts, chunk_object_xy)
            handoff_within_y_seed_count += int(
                chunk_handoff_within_y_seed_count
            )
            if processed >= next_progress or processed == seed_count:
                elapsed = max(time.monotonic() - started_at, 1e-9)
                print(
                    "progress: {}/{} ({:.1f}%), {:.0f} seeds/s".format(
                        processed,
                        seed_count,
                        100.0 * processed / float(seed_count),
                        processed / elapsed,
                    )
                )
                while next_progress <= processed:
                    next_progress += max(1, int(PROGRESS_EVERY_SEEDS))

    return (
        class_counts,
        object_counts,
        object_xy_counts,
        handoff_within_y_seed_count,
    )


def _validate_count_totals(
    seed_count,
    class_counts,
    object_counts,
    object_xy_counts,
    handoff_within_y_seed_count,
):
    if not 0 <= int(handoff_within_y_seed_count) <= int(seed_count):
        raise RuntimeError(
            "handoff-within-Y seed count outside 0..{}: {}".format(
                seed_count, handoff_within_y_seed_count
            )
        )
    for class_name, counter in class_counts.items():
        expected = 2 * int(seed_count)
        actual = sum(counter.values())
        if actual != expected:
            raise RuntimeError(
                "{} count mismatch: expected {}, got {}".format(
                    class_name, expected, actual
                )
            )
    for object_name, counter in object_counts.items():
        actual = sum(counter.values())
        if actual != int(seed_count):
            raise RuntimeError(
                "{} count mismatch: expected {}, got {}".format(
                    object_name, seed_count, actual
                )
            )
        xy_actual = sum(object_xy_counts[object_name].values())
        if xy_actual != int(seed_count):
            raise RuntimeError(
                "{} XY count mismatch: expected {}, got {}".format(
                    object_name, seed_count, xy_actual
                )
            )


def _build_report(
    simulator_path,
    extension_path,
    seed_ranges,
    seed_count,
    class_counts,
    object_counts,
    object_xy_counts,
    handoff_within_y_seed_count,
):
    return {
        "handoff_required_within_abs_y_seed_count": int(
            handoff_within_y_seed_count
        ),
        "handoff_abs_y_max_inclusive": float(HANDOFF_ABS_Y_MAX),
        "scene": SCENE_NAME,
        "source": "challenge_secret.get_object_layout",
        "simulator_path": simulator_path,
        "extension_path": extension_path,
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "seed_ranges_inclusive": [list(item) for item in seed_ranges],
        "seed_count": int(seed_count),
        "yaw_deg_decimals": int(YAW_DEG_DECIMALS),
        "xy_match_tolerance": float(XY_MATCH_TOLERANCE),
        "classes": {
            name: {
                "sample_count": sum(class_counts[name].values()),
                "yaw_counts": _counter_rows(class_counts[name]),
            }
            for name in OBJECTS_BY_CLASS
        },
        "objects": {
            name: {
                "sample_count": sum(object_counts[name].values()),
                "yaw_counts": _counter_rows(object_counts[name]),
                "xy_counts": _xy_counter_rows(object_xy_counts[name]),
            }
            for name in OBJECT_NAMES
        },
    }


def _write_report(report):
    output_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    simulator_path = _find_simulator_path()
    challenge_secret, extension_path = _load_secret_module(simulator_path)
    _verify_sources(challenge_secret, simulator_path)
    seed_ranges = _normalize_seed_ranges(SEED_RANGES)
    seed_count = _seed_count(seed_ranges)
    worker_count = _resolve_worker_count(seed_count)

    print("source: challenge_secret.get_object_layout")
    print("extension: {}".format(extension_path))
    print("scene: {}, seeds: {}".format(SCENE_NAME, seed_count))
    print("workers: {}, chunk_size: {}".format(worker_count, SEED_CHUNK_SIZE))

    if worker_count > 1:
        (
            class_counts,
            object_counts,
            object_xy_counts,
            handoff_within_y_seed_count,
        ) = collect_layout_counts_parallel(
            simulator_path,
            seed_ranges,
            seed_count,
            worker_count,
        )
    else:
        (
            class_counts,
            object_counts,
            object_xy_counts,
            handoff_within_y_seed_count,
        ) = collect_layout_counts(challenge_secret, _iter_seeds(seed_ranges))
    _validate_count_totals(
        seed_count,
        class_counts,
        object_counts,
        object_xy_counts,
        handoff_within_y_seed_count,
    )
    print(
        "\nseeds with a handoff-required object at |Y| <= {:.3f}: {}".format(
            HANDOFF_ABS_Y_MAX,
            handoff_within_y_seed_count,
        )
    )

    print("\n=== Counts by class ===")
    for class_name in OBJECTS_BY_CLASS:
        _print_counter(class_name, class_counts[class_name])

    print("\n=== Counts by source object name ===")
    for object_name in OBJECT_NAMES:
        _print_counter(object_name, object_counts[object_name])

    print("\n=== XY counts by simulator object name ===")
    for object_name in OBJECT_NAMES:
        _print_xy_counter(object_name, object_xy_counts[object_name])

    report = _build_report(
        simulator_path,
        extension_path,
        seed_ranges,
        seed_count,
        class_counts,
        object_counts,
        object_xy_counts,
        handoff_within_y_seed_count,
    )
    if WRITE_JSON:
        _write_report(report)
        print("\nJSON: {}".format(OUTPUT_PATH))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
