#!/usr/bin/env python3

"""训练单类别包裹检测模型，并管理新训练与断点恢复。"""

import os
from pathlib import Path

import config
import run_manager
from dataset import load_run_snapshot, prepare_dataset, write_run_snapshot


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("训练环境未安装 ultralytics") from exc
    return YOLO


def _dataset_record(summary, snapshot_yaml=None):
    record = {
        "root": str(summary["root"]),
        "signature": summary["signature"],
        "seed_range": list(summary["seed_range"]),
        "seeds": summary["seeds"],
        "train": dict(summary["train"]),
        "val": dict(summary["val"]),
    }
    if snapshot_yaml is not None:
        record["snapshot_yaml"] = str(Path(snapshot_yaml).resolve())
    return record


def _training_settings():
    return {
        "model": str(config.MODEL),
        "epochs": config.EPOCHS,
        "image_size": config.IMAGE_SIZE,
        "batch_size": config.BATCH_SIZE,
        "device": config.DEVICE,
        "workers": config.WORKERS,
        "patience": config.PATIENCE,
        "random_seed": config.RANDOM_SEED,
        "save_period": config.SAVE_PERIOD,
        "deterministic": config.DETERMINISTIC,
        "pretrained": config.PRETRAINED,
        "optimizer": config.OPTIMIZER,
        "cos_lr": config.COS_LR,
        "amp": config.AMP,
        "cache": config.CACHE,
    }


def _new_run_manifest(run_dir, summary, snapshot_yaml):
    return {
        "schema_version": 1,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "mode": "new",
        "status": "running",
        "process_id": os.getpid(),
        "started_at": run_manager.now_iso(),
        "dataset": _dataset_record(summary, snapshot_yaml),
        "settings": _training_settings(),
        "resume_history": [],
    }


def _prepare_resume(summary, run_dir, checkpoint):
    manifest = run_manager.load_manifest(run_dir)
    current_signature = summary["signature"]
    previous_dataset = manifest.get("dataset", {})
    snapshot_yaml = previous_dataset.get("snapshot_yaml")

    history = list(manifest.get("resume_history", []))
    history.append(
        {
            "checkpoint": str(checkpoint),
            "dataset_signature": current_signature,
            "resumed_at": run_manager.now_iso(),
        }
    )
    manifest.update(
        {
            "schema_version": 1,
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "mode": "resume",
            "status": "running",
            "process_id": os.getpid(),
            "dataset": _dataset_record(summary, snapshot_yaml),
            "resume_history": history,
        }
    )
    if "started_at" not in manifest:
        manifest["started_at"] = run_manager.now_iso()
        print("[警告] 旧训练没有运行清单，已按当前数据集补建", flush=True)
    run_manager.write_manifest(run_dir, manifest)
    return run_dir, checkpoint


def _train_new(YOLO, data_yaml, run_dir):
    model = YOLO(str(config.MODEL))
    model.train(
        data=str(data_yaml),
        project=str(config.RUNS_DIR),
        name=run_dir.name,
        # 目录由本脚本创建且已确认唯一，因此允许 Ultralytics 使用该空目录。
        exist_ok=True,
        epochs=config.EPOCHS,
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        workers=config.WORKERS,
        patience=config.PATIENCE,
        seed=config.RANDOM_SEED,
        deterministic=config.DETERMINISTIC,
        pretrained=config.PRETRAINED,
        optimizer=config.OPTIMIZER,
        cos_lr=config.COS_LR,
        amp=config.AMP,
        cache=config.CACHE,
        plots=True,
        save=True,
        save_period=config.SAVE_PERIOD,
        hsv_h=config.HSV_H,
        hsv_s=config.HSV_S,
        hsv_v=config.HSV_V,
        degrees=config.DEGREES,
        translate=config.TRANSLATE,
        scale=config.SCALE,
        fliplr=0.0,
        flipud=0.0,
        perspective=0.0,
        mosaic=config.MOSAIC,
        close_mosaic=config.CLOSE_MOSAIC,
        verbose=True,
    )
    actual_dir = Path(model.trainer.save_dir).resolve()
    if actual_dir != run_dir:
        raise RuntimeError("训练输出目录异常：预期 {}，实际 {}".format(run_dir, actual_dir))
    return model, run_dir


def _train_resume(YOLO, checkpoint, run_dir):
    model = YOLO(str(checkpoint))
    # 恢复时沿用检查点内保存的优化器、轮次和训练参数，不创建新结果目录。
    model.train(resume=True)
    actual_dir = Path(model.trainer.save_dir).resolve()
    if actual_dir != run_dir:
        raise RuntimeError("恢复输出目录异常：预期 {}，实际 {}".format(run_dir, actual_dir))
    return model, run_dir


def _finish_run(run_dir, summary):
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError("训练结束但 best.pt 或 last.pt 不完整: {}".format(run_dir))
    run_manager.update_manifest(
        run_dir,
        status="completed",
        completed_at=run_manager.now_iso(),
        best_weights=str(best.resolve()),
        last_weights=str(last.resolve()),
    )
    run_manager.write_latest_completed(
        run_dir,
        best,
        last,
        summary["signature"],
    )
    print("[完成] 训练目录：{}".format(run_dir), flush=True)
    print("[完成] 最佳权重：{}".format(best), flush=True)


def _run_training():
    YOLO = _load_yolo()
    mode = str(config.TRAIN_MODE).strip().lower()
    run_dir = None
    try:
        if mode == "new":
            summary, _ = prepare_dataset()
            run_dir = run_manager.create_new_run_dir()
            data_yaml = write_run_snapshot(summary, run_dir)
            run_manager.write_manifest(
                run_dir,
                _new_run_manifest(run_dir, summary, data_yaml),
            )
            _train_new(YOLO, data_yaml, run_dir)
        elif mode in ("resume_latest", "resume_path"):
            checkpoint = run_manager.resolve_resume_checkpoint()
            run_dir = run_manager.checkpoint_run_dir(checkpoint)
            if (run_dir / "dataset_snapshot" / "snapshot.json").is_file():
                summary, _ = load_run_snapshot(run_dir)
            else:
                summary, _ = prepare_dataset()
                print("[警告] 旧训练没有独立数据快照，将沿用其原数据配置", flush=True)
            _prepare_resume(summary, run_dir, checkpoint)
            _train_resume(YOLO, checkpoint, run_dir)
        else:
            raise RuntimeError("未知训练模式: {}".format(config.TRAIN_MODE))
        _finish_run(run_dir, summary)
    except KeyboardInterrupt:
        if run_dir is not None:
            run_manager.update_manifest(
                run_dir,
                status="interrupted",
                interrupted_at=run_manager.now_iso(),
            )
        raise
    except Exception as exc:
        if run_dir is not None:
            run_manager.update_manifest(
                run_dir,
                status="failed",
                failed_at=run_manager.now_iso(),
                error=str(exc),
            )
        raise
    return 0


def main():
    with run_manager.training_lock():
        return _run_training()


if __name__ == "__main__":
    raise SystemExit(main())
