#!/usr/bin/env python3
"""YOLOv8n 训练脚本。CPU 训练，数据集小，几分钟完成。"""
import os
from ultralytics import YOLO

base = os.path.dirname(os.path.abspath(__file__))
data_yaml = os.path.join(base, "training_data", "data.yaml")

model = YOLO("yolov8n.pt")  # 预训练权重
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    name="tray_detect",
    project=os.path.join(base, "runs"),
    exist_ok=True,
    verbose=True,
)

# 导出 ONNX
#model.export(format="onnx")
print("训练完成，模型保存在: %s" % os.path.join(base, "runs", "tray_detect"))
