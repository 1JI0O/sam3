# SAM3 视频时序微调可行性分析

## 结论

**SAM3 当前代码库无法做有监督视频时序微调。**

`VideoGroundingDataset` 只用于推理评估，video training 链路（collator → 模型 forward → loss 反传）完整缺失。

---

## 阻断链

### 1. 模型层：`Sam3Image.forward()` 硬断言单帧

文件：`sam3/model/sam3_image.py`

训练用的 image 模型 `Sam3Image` 的 forward 方法中：

```python
def forward(self, input: BatchedDatapoint):
    num_frames = len(input.find_inputs)
    assert num_frames == 1   # ← 多帧直接 AssertionError
```

多帧输入（来自 VideoGroundingDataset）在这里硬崩。

### 2. 视频模型：`build_sam3_video_model` 是纯推理模型

文件：`sam3/model_builder.py:664`，`sam3/model/sam3_video_inference.py`

`build_sam3_video_model()` 返回 `Sam3VideoInferenceWithInstanceInteractivity`：

```python
def forward(self, input: BatchedDatapoint, is_inference: bool = False):
    """This method is only used for benchmark eval (not used in the demo)."""
    # ... 纯推理逻辑，返回 {video_id: preds} 字典
    # ... 无 loss 计算，无 matcher
```

```python
def back_convert(self, targets):
    # Needed for retraining compatibility with trainer
    return targets   # ← 空 stub，什么都不做
```

与 image 模型的对比：

| 方面 | 训练用 Sam3Image | 视频推理模型 |
|------|-----------------|-------------|
| forward 返回值 | `SAM3Output`（多层预测，供 loss 用） | `Dict[video_id: results]`（已后处理的推理结果） |
| 有无 matcher | 有（`BinaryHungarianMatcherV2`） | 无 |
| back_convert | 实际转换目标格式 | `return targets`（空 stub） |
| 设计目的 | 训练 | 推理/评估 |

### 3. 数据层：collator 破坏时序结构

文件：`sam3/train/data/collator.py`

collator 把所有帧的图像展平成独立 tensors：

```python
# collator 内部
img_batch.extend([img.data for img in data.images])   # 所有帧展平
...
image_batch = torch.stack(img_batch)   # shape: (total_frames, C, H, W)，时序结构丢失
```

多帧变成多个独立的 single-frame，进入模型时 `num_frames = len(find_inputs)`，而 `find_inputs` 按 `query_processing_order` 分 stage。若 VideoGroundingDataset 生成了 N 个帧的 queries（`query_processing_order = 0..N-1`），collator 生成 N 个 stages，`num_frames = N`，触发 `assert num_frames == 1`。

### 4. 配置层：全仓库无视频训练 config

路径：`sam3/train/configs/`

所有含 `VideoGroundingDataset` 或 `build_sam3_video_model` 的 yaml 均为：

```yaml
mode: val      # ← 评估模式
trainer:
  data:
    train: null  # ← 无训练数据
  loss:
    default:
      _target_: sam3.train.loss.sam3_loss.DummyLoss  # ← 占位符
```

没有一个视频训练配置。

---

## VideoGroundingDataset 的实际定位

`VideoGroundingDataset` 继承自 `Sam3ImageDataset`，增加了帧采样逻辑（`num_stages_sample`、`stage_stride_min/max`、时序翻转）。在仓库中，它仅被用于：

- 视频 benchmark 评估（如 `saco_veval_sav_val.yaml`）
- 配合 `SAM3_VEVAL_API_FROM_JSON_NP` loader（无标注，仅推理）

**它不是视频训练数据集，而是视频推理数据集。**

---

## 如果要真正实现视频时序微调，需要做什么

这是研究级工作，不是配置修改：

1. **修改 `Sam3Image.forward()`**：移除 `assert num_frames == 1`，接受多帧 batch，输出每帧的预测
2. **修改 collator**：保留时序结构，输出 `(batch, num_frames, C, H, W)` 而非展平
3. **实现时序 loss**：跨帧一致性监督，或直接把每帧当独立样本（等同于图像训练）
4. **写视频训练 loader**：类似 `COCO_FROM_JSON` 但以 video 为 datapoint，逐帧返回标注
5. **写视频训练 config**：`mode: train`，数据用 VideoGroundingDataset，loss 非 dummy

---

## 现实可行的改进（不涉及时序）

在图像训练框架内，可以对齐 SAM2 的如下设计决策，有实质改善：

| 问题 | 改法 |
|------|------|
| 只监督 arm，无 gripper | `build_finetune_dataset.py` 加 gripper 双目标，两个 category |
| stem 对齐丢帧 | 改为 index-based 对齐（排序后按索引配对，与 SAM2 一致） |
| 采样过稀（stride=30） | 降为 stride=5，增大训练样本数 |
| multiplier=1 | 改为 8（对齐 SAM2 的 `multiplier: 8`） |
| 学习率过高（lr_scale=0.1） | 降为 0.01，使 vision_backbone LR ≈ 2.5e-6（接近 SAM2 的 2e-6） |

这些改动在现有图像训练框架内完全可落地，且直接对应 SAM3 比 SAM2 效果差的已知原因。
