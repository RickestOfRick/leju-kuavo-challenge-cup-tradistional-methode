#!/usr/bin/env python3

"""校验包裹数据集并生成固定样本列表的 YOLO 配置。"""

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import config


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SAMPLE_PATTERN = re.compile(r"^seed_(\d+)_view_(\d+)$")


def _image_files(directory):
    files = []
    for suffix in IMAGE_SUFFIXES:
        files.extend(directory.glob("*" + suffix))
    return sorted(files)


def _complete_seed_ids(root):
    """读取完整种子标记，自动排除正在采集或文件不全的种子。"""
    complete = set()
    seed_dir = Path(root) / "seeds"
    for marker_path in sorted(seed_dir.glob("seed_*.json")):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            seed = int(marker.get("seed", -1))
            samples = marker.get("samples", [])
            if not (config.DATASET_SEED_START <= seed <= config.DATASET_SEED_END):
                continue
            if len(samples) != config.VIEWS_PER_SEED:
                continue
            views = {int(sample.get("view_index", -1)) for sample in samples}
            if views != set(range(config.VIEWS_PER_SEED)):
                continue
            required_files = []
            for sample in samples:
                required_files.extend((sample.get("image"), sample.get("label")))
            if not all(value and (Path(root) / value).is_file() for value in required_files):
                continue
        except (OSError, TypeError, ValueError):
            continue
        complete.add(seed)
    return complete


def _selected_sample_files(paths, complete_seed_ids, file_kind, errors):
    selected = []
    for path in paths:
        match = SAMPLE_PATTERN.match(path.stem)
        if match is None:
            errors.append("{}名称无效: {}".format(file_kind, path.name))
            continue
        seed = int(match.group(1))
        if seed in complete_seed_ids:
            selected.append(path)
    return selected


def _update_digest_from_file(digest, path):
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)


def _atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _sample_label_path(root, split, image_path):
    return Path(root) / "labels" / split / (Path(image_path).stem + ".txt")


def _update_sample_digest(digest, root, split, image_path):
    label_path = _sample_label_path(root, split, image_path)
    relative_name = "{}/{}".format(split, Path(image_path).name)
    digest.update(relative_name.encode("utf-8"))
    digest.update(b"\0")
    _update_digest_from_file(digest, image_path)
    digest.update(b"\0")
    _update_digest_from_file(digest, label_path)
    digest.update(b"\0")


def _read_labels(path):
    labels = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise RuntimeError(
                    "{}:{} 必须包含 类别 cx cy 宽度 高度".format(
                        path, line_number
                    )
                )
            try:
                class_id = int(fields[0])
                box = tuple(float(value) for value in fields[1:])
            except ValueError as exc:
                raise RuntimeError(
                    "{}:{} 标签中包含非数值字段".format(path, line_number)
                ) from exc
            if class_id < 0 or class_id >= len(config.CLASS_NAMES):
                raise RuntimeError(
                    "{}:{} 类别编号无效: {}".format(path, line_number, class_id)
                )
            if not all(math.isfinite(value) for value in box):
                raise RuntimeError(
                    "{}:{} 边界框包含非有限数值".format(path, line_number)
                )
            cx, cy, width, height = box
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                raise RuntimeError(
                    "{}:{} 中心坐标超出 [0, 1]".format(path, line_number)
                )
            if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                raise RuntimeError(
                    "{}:{} 尺寸超出 (0, 1]".format(path, line_number)
                )
            tolerance = 1e-6
            if (
                cx - width * 0.5 < -tolerance
                or cx + width * 0.5 > 1.0 + tolerance
                or cy - height * 0.5 < -tolerance
                or cy + height * 0.5 > 1.0 + tolerance
            ):
                raise RuntimeError(
                    "{}:{} 边界框超出图像范围".format(path, line_number)
                )
            labels.append((class_id,) + box)
    return labels


def _verify_image(path):
    if not config.VERIFY_IMAGE_CONTENT:
        return
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("启用图像内容校验时必须安装 Pillow") from exc
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        raise RuntimeError("无法解码图像 {}: {}".format(path, exc)) from exc
    if width <= 0 or height <= 0:
        raise RuntimeError("图像尺寸无效: {}".format(path))


def validate_dataset():
    if int(config.DATASET_SEED_START) > int(config.DATASET_SEED_END):
        raise RuntimeError("DATASET_SEED_START 不能大于 DATASET_SEED_END")
    root = config.DATASET_DIR.resolve()
    if not root.is_dir():
        raise RuntimeError("数据集目录不存在: {}".format(root))
    complete_seed_ids = _complete_seed_ids(root)

    split_counts = {}
    split_seeds = {}
    split_image_files = {}
    seed_views = defaultdict(set)
    dataset_digest = hashlib.sha256()
    errors = []

    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append("缺少 {} 图像目录或标签目录".format(split))
            continue

        images = _selected_sample_files(
            _image_files(image_dir),
            complete_seed_ids,
            "图像",
            errors,
        )
        image_stems = {path.stem for path in images}
        label_paths = _selected_sample_files(
            sorted(label_dir.glob("*.txt")),
            complete_seed_ids,
            "标签",
            errors,
        )
        label_stems = {path.stem for path in label_paths}
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        if missing_labels:
            errors.append(
                "{} 中存在无标签图像: {}".format(split, missing_labels[:10])
            )
        if missing_images:
            errors.append(
                "{} 中存在无图像标签: {}".format(split, missing_images[:10])
            )

        seeds = set()
        valid_image_paths = []
        valid_images = 0
        valid_boxes = 0
        for image_path in images:
            match = SAMPLE_PATTERN.match(image_path.stem)
            if match is None:
                errors.append("样本名称无效: {}".format(image_path.name))
                continue
            seed = int(match.group(1))
            view = int(match.group(2))
            if view < 0 or view >= config.VIEWS_PER_SEED:
                errors.append("视角编号无效: {}".format(image_path.name))
                continue
            label_path = label_dir / (image_path.stem + ".txt")
            if not label_path.is_file():
                continue
            try:
                _verify_image(image_path)
                labels = _read_labels(label_path)
                if len(labels) != config.OBJECTS_PER_IMAGE:
                    raise RuntimeError(
                        "应有 {} 个边界框，实际为 {} 个".format(
                            config.OBJECTS_PER_IMAGE, len(labels)
                        )
                    )
            except Exception as exc:
                errors.append("{}: {}".format(image_path.name, exc))
                continue
            seeds.add(seed)
            seed_views[(split, seed)].add(view)
            _update_sample_digest(dataset_digest, root, split, image_path)
            valid_image_paths.append(image_path.resolve())
            valid_images += 1
            valid_boxes += len(labels)

        split_counts[split] = {
            "images": valid_images,
            "boxes": valid_boxes,
            "seeds": len(seeds),
        }
        split_seeds[split] = seeds
        split_image_files[split] = valid_image_paths

    overlap = split_seeds.get("train", set()) & split_seeds.get("val", set())
    if overlap:
        errors.append("训练集与验证集存在种子泄漏: {}".format(sorted(overlap)))

    all_seeds = split_seeds.get("train", set()) | split_seeds.get("val", set())
    for split, seed in sorted(seed_views):
        views = seed_views[(split, seed)]
        expected = set(range(config.VIEWS_PER_SEED))
        if views != expected:
            errors.append(
                "{} 的种子 {} 包含视角 {}，应为 {}".format(
                    split, seed, sorted(views), sorted(expected)
                )
            )

    if not split_counts.get("train", {}).get("images"):
        errors.append("训练集为空")
    if not split_counts.get("val", {}).get("images"):
        errors.append("验证集为空")

    if errors:
        preview = "\n".join("- " + message for message in errors[:50])
        remaining = len(errors) - 50
        if remaining > 0:
            preview += "\n- 另有 {} 个错误".format(remaining)
        raise RuntimeError("数据集校验失败:\n" + preview)

    return {
        "root": root,
        "signature": dataset_digest.hexdigest(),
        "seed_range": [config.DATASET_SEED_START, config.DATASET_SEED_END],
        "seeds": len(all_seeds),
        "train": split_counts["train"],
        "val": split_counts["val"],
        "_image_files": split_image_files,
    }


def write_runtime_yaml(summary):
    """写入本次运行固定使用的训练集和验证集文件列表。"""
    root = config.DATASET_DIR.resolve()
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    train_list = config.RUNTIME_DIR / "train_images.txt"
    val_list = config.RUNTIME_DIR / "val_images.txt"
    _atomic_write_text(
        train_list,
        "".join("{}\n".format(path) for path in summary["_image_files"]["train"]),
    )
    _atomic_write_text(
        val_list,
        "".join("{}\n".format(path) for path in summary["_image_files"]["val"]),
    )
    names = "\n".join(
        "  {}: {}".format(index, name)
        for index, name in enumerate(config.CLASS_NAMES)
    )
    content = (
        "path: {}\n"
        "train: {}\n"
        "val: {}\n"
        "names:\n{}\n"
    ).format(root, train_list.resolve(), val_list.resolve(), names)
    _atomic_write_text(config.RUNTIME_DATA_YAML, content)
    return config.RUNTIME_DATA_YAML


def write_run_snapshot(summary, run_dir):
    """把本次训练实际使用的数据列表固化到训练目录。"""
    snapshot_dir = Path(run_dir) / "dataset_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    train_list = snapshot_dir / "train_images.txt"
    val_list = snapshot_dir / "val_images.txt"
    data_yaml = snapshot_dir / "data.yaml"
    snapshot_json = snapshot_dir / "snapshot.json"

    _atomic_write_text(
        train_list,
        "".join("{}\n".format(path) for path in summary["_image_files"]["train"]),
    )
    _atomic_write_text(
        val_list,
        "".join("{}\n".format(path) for path in summary["_image_files"]["val"]),
    )
    names = "\n".join(
        "  {}: {}".format(index, name)
        for index, name in enumerate(config.CLASS_NAMES)
    )
    yaml_content = (
        "path: {}\n"
        "train: {}\n"
        "val: {}\n"
        "names:\n{}\n"
    ).format(summary["root"], train_list.resolve(), val_list.resolve(), names)
    _atomic_write_text(data_yaml, yaml_content)

    snapshot = {
        "root": str(summary["root"]),
        "signature": summary["signature"],
        "seed_range": list(summary["seed_range"]),
        "seeds": summary["seeds"],
        "train": dict(summary["train"]),
        "val": dict(summary["val"]),
    }
    _atomic_write_text(
        snapshot_json,
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return data_yaml.resolve()


def load_run_snapshot(run_dir):
    """加载并校验某次训练固化的数据快照。"""
    snapshot_dir = Path(run_dir) / "dataset_snapshot"
    snapshot_json = snapshot_dir / "snapshot.json"
    data_yaml = snapshot_dir / "data.yaml"
    try:
        snapshot = json.loads(snapshot_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("无法读取训练数据快照: {}".format(snapshot_json)) from exc

    image_files = {}
    digest = hashlib.sha256()
    root = Path(snapshot["root"]).resolve()
    for split in ("train", "val"):
        list_path = snapshot_dir / "{}_images.txt".format(split)
        try:
            paths = [
                Path(line.strip()).resolve()
                for line in list_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError as exc:
            raise RuntimeError("无法读取数据快照列表: {}".format(list_path)) from exc
        for image_path in paths:
            label_path = _sample_label_path(root, split, image_path)
            if not image_path.is_file() or not label_path.is_file():
                raise RuntimeError("训练数据快照文件缺失: {}".format(image_path))
            _update_sample_digest(digest, root, split, image_path)
        image_files[split] = paths

    if digest.hexdigest() != snapshot.get("signature"):
        raise RuntimeError("训练数据快照内容已被修改: {}".format(snapshot_dir))
    if not data_yaml.is_file():
        raise RuntimeError("训练数据快照缺少 data.yaml: {}".format(snapshot_dir))
    snapshot["root"] = root
    snapshot["_image_files"] = image_files
    return snapshot, data_yaml.resolve()


def prepare_dataset():
    summary = validate_dataset()
    yaml_path = write_runtime_yaml(summary)
    print(
        "[正常] 数据集：范围={seed_range[0]}..{seed_range[1]}，种子={seeds}，"
        "训练图像={train[images]}/{train[boxes]}框，"
        "验证图像={val[images]}/{val[boxes]}框".format(**summary),
        flush=True,
    )
    print("[正常] 本次数据快照：{}".format(yaml_path), flush=True)
    return summary, yaml_path
