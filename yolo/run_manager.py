#!/usr/bin/env python3

"""训练运行目录、检查点恢复和结果清单管理。"""

import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import config


MANIFEST_NAME = "run_manifest.json"
TRAIN_LOCK_NAME = ".training.lock"


def now_iso():
    """返回带本地时区的当前时间。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _timestamp():
    return datetime.now().strftime(config.RUN_TIMESTAMP_FORMAT)


@contextmanager
def training_lock():
    """保证同一输出根目录同时只有一个训练脚本运行。"""
    import fcntl

    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = config.RUNS_DIR / TRAIN_LOCK_NAME
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("已有训练脚本正在使用输出目录: {}".format(config.RUNS_DIR)) from exc
        handle.seek(0)
        handle.truncate()
        handle.write("进程={}\n启动时间={}\n".format(os.getpid(), now_iso()))
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def write_json(path, payload):
    """向指定位置原子写入 JSON 文件。"""
    _atomic_write_json(path, payload)


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("无法读取运行清单 {}: {}".format(path, exc)) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("运行清单不是 JSON 对象: {}".format(path))
    return payload


def create_new_run_dir():
    """创建不会与既有结果重名的时间戳训练目录。"""
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = "{}_{}".format(config.RUN_NAME_PREFIX, _timestamp())
    for index in range(1000):
        name = base_name if index == 0 else "{}_{:02d}".format(base_name, index)
        run_dir = config.RUNS_DIR / name
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir.resolve()
    raise RuntimeError("无法分配新的训练目录: {}".format(base_name))


def checkpoint_run_dir(checkpoint):
    """从 weights/last.pt 或 weights/best.pt 推导所属训练目录。"""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.parent.name != "weights":
        raise RuntimeError("检查点必须位于训练目录的 weights 子目录: {}".format(checkpoint))
    return checkpoint.parent.parent


def _checkpoint_candidates():
    pattern = "{}*/weights/last.pt".format(config.RUN_NAME_PREFIX)
    return [
        path.resolve()
        for path in config.RUNS_DIR.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]


def resolve_resume_checkpoint():
    """按照全局训练模式解析需要恢复的 last.pt。"""
    mode = str(config.TRAIN_MODE).strip().lower()
    if mode == "resume_latest":
        candidates = _checkpoint_candidates()
        if not candidates:
            raise RuntimeError("没有找到可恢复的 last.pt")
        checkpoint = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    elif mode == "resume_path":
        if not str(config.RESUME_CHECKPOINT).strip():
            raise RuntimeError("resume_path 模式必须设置 RESUME_CHECKPOINT")
        checkpoint = Path(config.RESUME_CHECKPOINT).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = config.PROJECT_ROOT / checkpoint
        checkpoint = checkpoint.resolve()
    else:
        raise RuntimeError("当前模式不是恢复训练模式: {}".format(config.TRAIN_MODE))

    if checkpoint.name != "last.pt" or not checkpoint.is_file():
        raise RuntimeError("恢复检查点必须是有效的 last.pt: {}".format(checkpoint))
    checkpoint_run_dir(checkpoint)
    return checkpoint


def load_manifest(run_dir):
    return _read_json(Path(run_dir) / MANIFEST_NAME)


def write_manifest(run_dir, payload):
    """原子写入训练目录自己的运行清单。"""
    _atomic_write_json(Path(run_dir) / MANIFEST_NAME, payload)


def update_manifest(run_dir, **updates):
    payload = load_manifest(run_dir)
    payload.update(updates)
    write_manifest(run_dir, payload)
    return payload


def write_latest_completed(run_dir, best_weights, last_weights, dataset_signature):
    """记录最近一次完整训练，不通过目录名称猜测验证权重。"""
    payload = {
        "completed_at": now_iso(),
        "run_dir": str(Path(run_dir).resolve()),
        "best_weights": str(Path(best_weights).resolve()),
        "last_weights": str(Path(last_weights).resolve()),
        "dataset_signature": str(dataset_signature),
    }
    _atomic_write_json(config.LATEST_COMPLETED_FILE, payload)
    return payload


def resolve_validation_weights():
    """解析显式权重或最近一次完整训练的最佳权重。"""
    configured = str(config.VALIDATION_WEIGHTS).strip()
    if configured:
        weights = Path(configured).expanduser()
        if not weights.is_absolute():
            weights = config.PROJECT_ROOT / weights
        weights = weights.resolve()
    else:
        latest = _read_json(config.LATEST_COMPLETED_FILE)
        value = latest.get("best_weights")
        if not value:
            raise RuntimeError(
                "没有最近完整训练记录；请先完成一次新训练，或设置 VALIDATION_WEIGHTS"
            )
        weights = Path(value).expanduser().resolve()

    if weights.name != "best.pt" or not weights.is_file():
        raise RuntimeError("验证权重必须是有效的 best.pt: {}".format(weights))
    checkpoint_run_dir(weights)
    return weights


def allocate_timestamped_name(parent, prefix):
    """为验证等子任务分配尚不存在的时间戳名称。"""
    parent = Path(parent)
    base_name = "{}_{}".format(prefix, _timestamp())
    for index in range(1000):
        name = base_name if index == 0 else "{}_{:02d}".format(base_name, index)
        if not (parent / name).exists():
            return name
    raise RuntimeError("无法分配新的结果目录名称: {}".format(base_name))
