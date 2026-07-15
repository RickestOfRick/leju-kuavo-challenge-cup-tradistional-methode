#!/usr/bin/env python3

"""包裹检测模型训练与验证的统一配置。"""

from pathlib import Path

YOLO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = YOLO_DIR.parent

# data.py 采集的任务一包裹数据集。
DATASET_DIR = PROJECT_ROOT / "data" / "scene1_parcel_yolo"
RUNTIME_DIR = YOLO_DIR / ".runtime"
RUNTIME_DATA_YAML = RUNTIME_DIR / "scene1_parcel_yolo.yaml"

# 数据集种子闭区间与单个种子的完整性要求。
DATASET_SEED_START = 1
DATASET_SEED_END = 400
VIEWS_PER_SEED = 3
OBJECTS_PER_IMAGE = 4
CLASS_NAMES = ("parcel",)
VERIFY_IMAGE_CONTENT = True

# 训练运行模式：new=新建时间戳运行；resume_latest=恢复最新检查点；
# resume_path=恢复 RESUME_CHECKPOINT 指定的检查点。
TRAIN_MODE = "new"
RESUME_CHECKPOINT = ""

# 每次新训练使用独立时间戳目录，避免不同训练结果相互覆盖或混合。
RUNS_DIR = YOLO_DIR / "runs"
RUN_NAME_PREFIX = "parcel_yolo26n"
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
LATEST_COMPLETED_FILE = RUNS_DIR / "latest_completed.json"

# 模型与训练参数；MODEL 也可以设置为本地 .pt 权重的绝对路径。
MODEL = "yolo26n.pt"
EPOCHS = 120
IMAGE_SIZE = 960
BATCH_SIZE = 16
DEVICE = 0
WORKERS = 4
PATIENCE = 30
RANDOM_SEED = 42
SAVE_PERIOD = 10
DETERMINISTIC = True
PRETRAINED = True
OPTIMIZER = "auto"
COS_LR = True
AMP = True
CACHE = False

# 数据增强保持接近固定头部相机的部署场景。
HSV_H = 0.015
HSV_S = 0.35
HSV_V = 0.25
DEGREES = 2.0
TRANSLATE = 0.05
SCALE = 0.15
MOSAIC = 0.5
CLOSE_MOSAIC = 10

# 验证权重留空时读取 LATEST_COMPLETED_FILE；也可以指定 best.pt 的路径。
VALIDATION_WEIGHTS = ""
ALLOW_VALIDATION_WITH_DATASET_CHANGE = False
VALIDATION_NAME_PREFIX = "validation"
VAL_BATCH_SIZE = 8
VAL_CONFIDENCE = 0.001
VAL_IOU = 0.70
MAX_DETECTIONS = 10
SAVE_VAL_PREDICTIONS = True
PREDICTION_CONFIDENCE = 0.25
