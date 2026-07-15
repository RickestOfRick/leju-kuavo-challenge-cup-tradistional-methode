# 包裹检测模型训练

训练和验证参数统一位于 `config.py`，脚本不接收命令行参数。默认模型为
`yolo26n.pt`，数据来自 `data/scene1_parcel_yolo/`。

每次启动前都会检查图像与标签配对、每张图四个框、每个种子三个视角，以及
训练集和验证集之间不存在种子泄漏。校验通过后会生成固定样本列表，因此采集
脚本后续新增的数据不会进入已经启动的训练。

## 选择种子范围

数据范围在 `config.py` 中按闭区间设置：

```python
DATASET_SEED_START = 1
DATASET_SEED_END = 1000
```

以上配置只使用编号 `1` 至 `1000` 范围内已经完整采集的种子，范围外的数据不会
参与完整性检查、数据签名、训练或验证。尚未采集或正在采集的种子会自动跳过，
不会要求区间内每个编号都必须存在。

## 新训练

保持以下配置：

```python
TRAIN_MODE = "new"
```

运行：

```bash
cd /home/zwx/ChallengeCup/leju-kuavo-challenge-cup-2026
conda run -n hitirc python yolo/train.py
```

每次新训练都会创建独立的时间戳目录，例如：

```text
yolo/runs/parcel_yolo26n_20260715_153012/
```

目录内包含：

- `weights/best.pt`：验证指标最佳的权重。
- `weights/last.pt`：最近保存的权重和恢复状态。
- `results.csv`：每轮训练指标。
- `args.yaml`：Ultralytics 实际使用的参数。
- `run_manifest.json`：数据签名、配置、状态、恢复历史和验证记录。

训练完整结束后，`runs/latest_completed.json` 会指向最近一次完整训练。中断或
失败的训练不会更新该文件。

## 查看训练效果

训练终端每轮都会显示损失和验证指标。主要关注：

- `box_loss`、`cls_loss`、`dfl_loss`：整体应逐步下降。
- `precision`：检出的包裹中有多少是正确的。
- `recall`：真实包裹中有多少被检出。
- `mAP50`：交并比阈值为 0.5 时的平均精度。
- `mAP50-95`：多个交并比阈值上的平均精度，更适合作为主要比较指标。

训练目录中的 `results.csv` 保存逐轮数值，`results.png` 显示损失和指标曲线，
`confusion_matrix.png`、`PR_curve.png`、`F1_curve.png` 用于检查误检、漏检以及
置信度阈值。`train_batch*.jpg` 和 `val_batch*_pred.jpg` 可直接检查标签与预测框。

完整训练后运行验证脚本：

```bash
conda run -n hitirc python yolo/validate.py
```

验证目录中的 `evaluation_summary.json` 保存可比较的数值指标，`predictions/`
保存验证集预测图。优先比较不同训练的 `mAP50-95`、召回率和实际预测图，而不是
只看训练损失。

## 恢复训练

恢复最近修改的 `last.pt`：

```python
TRAIN_MODE = "resume_latest"
```

恢复指定检查点：

```python
TRAIN_MODE = "resume_path"
RESUME_CHECKPOINT = "yolo/runs/parcel_yolo26n_20260715_153012/weights/last.pt"
```

然后仍运行 `python yolo/train.py`。恢复训练会继续使用原目录以及检查点中保存的
轮次、优化器和训练参数，不会创建新目录，也不会把它当作一次全新训练。

新版本创建的训练会从自身目录加载原始数据快照，保证恢复前后使用同一批样本。
如果希望用新增数据继续微调，应把 `MODEL` 指向原训练的 `best.pt`，并使用
`TRAIN_MODE = "new"` 创建一次新的时间戳训练，而不是使用断点恢复。

## 验证

`VALIDATION_WEIGHTS` 留空时，验证 `latest_completed.json` 记录的最佳权重：

```bash
conda run -n hitirc python yolo/validate.py
```

也可以明确指定某次训练：

```python
VALIDATION_WEIGHTS = "yolo/runs/parcel_yolo26n_20260715_153012/weights/best.pt"
```

每次验证都保存到所属训练目录下的独立时间戳目录：

```text
训练目录/evaluations/validation_20260715_170205/
```

该目录包含验证图表、`evaluation_manifest.json`、`evaluation_summary.json`，以及
可选的 `predictions/` 预测图，不会复用或覆盖其他验证结果。

默认使用训练目录内固化的数据快照，确保不同时间执行验证仍可比较。需要改用当前
配置的种子范围和最新数据验证时设置：

```python
ALLOW_VALIDATION_WITH_DATASET_CHANGE = True
```

## 注意事项

- 当前正在运行的旧训练不会因脚本文件更新而改变。
- 旧的 `yolo/runs/parcel_yolo26n/` 没有运行清单；可通过 `resume_latest` 接管，
  或在验证时把 `VALIDATION_WEIGHTS` 指向其 `best.pt`。
- 本地没有预训练权重时，Ultralytics 会尝试下载 `yolo26n.pt`。
- 显存不足时优先降低 `BATCH_SIZE`，其次降低 `IMAGE_SIZE`。
