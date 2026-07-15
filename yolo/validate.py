#!/usr/bin/env python3

"""验证指定训练权重，并将指标与预测图保存到独立目录。"""

import json
import math

import config
import run_manager
from dataset import load_run_snapshot, prepare_dataset


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("训练环境未安装 ultralytics") from exc
    return YOLO


def _numeric_metrics(metrics):
    result = {}
    for key, value in getattr(metrics, "results_dict", {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[str(key)] = number
    return result


def _check_dataset_signature(train_run_dir, summary):
    manifest = run_manager.load_manifest(train_run_dir)
    trained_signature = manifest.get("dataset", {}).get("signature")
    if (
        trained_signature
        and trained_signature != summary["signature"]
        and not config.ALLOW_VALIDATION_WITH_DATASET_CHANGE
    ):
        raise RuntimeError(
            "当前数据集与训练时不同，拒绝生成不可比较的验证结果；"
            "需要继续时请设置 ALLOW_VALIDATION_WITH_DATASET_CHANGE=True"
        )
    return manifest


def main():
    weights = run_manager.resolve_validation_weights()
    train_run_dir = run_manager.checkpoint_run_dir(weights)
    manifest = run_manager.load_manifest(train_run_dir)
    snapshot_path = train_run_dir / "dataset_snapshot" / "snapshot.json"
    if snapshot_path.is_file() and not config.ALLOW_VALIDATION_WITH_DATASET_CHANGE:
        summary, data_yaml = load_run_snapshot(train_run_dir)
    else:
        summary, data_yaml = prepare_dataset()
        manifest = _check_dataset_signature(train_run_dir, summary)

    evaluations_dir = train_run_dir / "evaluations"
    evaluation_name = run_manager.allocate_timestamped_name(
        evaluations_dir,
        config.VALIDATION_NAME_PREFIX,
    )
    evaluation_dir = evaluations_dir / evaluation_name
    evaluation_dir.mkdir(parents=True)
    evaluation_manifest_path = evaluation_dir / "evaluation_manifest.json"
    evaluation_manifest = {
        "status": "running",
        "started_at": run_manager.now_iso(),
        "training_run": str(train_run_dir),
        "weights": str(weights),
        "dataset_signature": summary["signature"],
        "dataset_seed_range": list(summary["seed_range"]),
        "dataset_seeds": summary["seeds"],
    }
    run_manager.write_json(evaluation_manifest_path, evaluation_manifest)

    try:
        YOLO = _load_yolo()
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(data_yaml),
            split="val",
            project=str(evaluations_dir),
            name=evaluation_name,
            # 目录由本脚本创建且已确认唯一，因此不会混入既有验证结果。
            exist_ok=True,
            imgsz=config.IMAGE_SIZE,
            batch=config.VAL_BATCH_SIZE,
            device=config.DEVICE,
            workers=config.WORKERS,
            conf=config.VAL_CONFIDENCE,
            iou=config.VAL_IOU,
            max_det=config.MAX_DETECTIONS,
            plots=True,
            save_json=False,
            verbose=True,
        )
        numeric = _numeric_metrics(metrics)

        if config.SAVE_VAL_PREDICTIONS:
            model.predict(
                source=[str(path) for path in summary["_image_files"]["val"]],
                project=str(evaluation_dir),
                name="predictions",
                exist_ok=True,
                imgsz=config.IMAGE_SIZE,
                device=config.DEVICE,
                conf=config.PREDICTION_CONFIDENCE,
                iou=config.VAL_IOU,
                max_det=config.MAX_DETECTIONS,
                save=True,
                verbose=False,
            )
    except KeyboardInterrupt:
        evaluation_manifest.update(
            status="interrupted",
            interrupted_at=run_manager.now_iso(),
        )
        run_manager.write_json(evaluation_manifest_path, evaluation_manifest)
        raise
    except Exception as exc:
        evaluation_manifest.update(
            status="failed",
            failed_at=run_manager.now_iso(),
            error=str(exc),
        )
        run_manager.write_json(evaluation_manifest_path, evaluation_manifest)
        raise

    evaluation_summary = {
        "completed_at": run_manager.now_iso(),
        "training_run": str(train_run_dir),
        "weights": str(weights),
        "dataset_signature": summary["signature"],
        "dataset_seed_range": list(summary["seed_range"]),
        "dataset_seeds": summary["seeds"],
        "metrics": numeric,
    }
    run_manager.write_json(evaluation_dir / "evaluation_summary.json", evaluation_summary)
    evaluation_manifest.update(
        status="completed",
        completed_at=evaluation_summary["completed_at"],
        metrics=numeric,
    )
    run_manager.write_json(evaluation_manifest_path, evaluation_manifest)

    if manifest:
        evaluations = list(manifest.get("evaluations", []))
        evaluations.append(
            {
                "completed_at": evaluation_summary["completed_at"],
                "directory": str(evaluation_dir),
                "weights": str(weights),
            }
        )
        run_manager.update_manifest(train_run_dir, evaluations=evaluations)

    print("[结果] " + json.dumps(numeric, ensure_ascii=False, sort_keys=True), flush=True)
    print("[完成] 验证目录：{}".format(evaluation_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
