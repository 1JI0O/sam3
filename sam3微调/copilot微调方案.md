260228

继承自[[copilot建议的路线]]

路线 A：尝试更精确的 text prompt 
这个我尝试过了，我那个prompt是尝试下来成功率最高的

其次，关于我的工作流程，我是先看看text prompt效果怎么样，不行的话再用手标点

最后，我还是希望微调sam3，这样以后我也许就不用手动去标了

---


现在所有关键信息都齐了。可以给出**完整、可执行的微调方案**。

---

## SAM3 图片模型微调：完整方案

### 核心认知先对齐

你的目标是：**微调后，`arm_text_prompt: "robot and cable"` 能在第 0 帧就准确 grounding 到机械臂**，后续视频传播不再需要手标点。

微调的对象是 **`build_sam3_image_model`**（`Sam3Image`），即 PCS 检测器部分。视频模型 `build_sam3_video_model` 的 tracker 部分不需要微调，因为它的传播能力是 memory-based 的，已经足够好。

---

## 第一步：将你现有的 mask 数据转成 COCO JSON

你已有的数据来自 `mask_1_arm` 流程：
- 原始帧：`/data/haoxiang/data/airexo2/{task}/train/{scene}/cam_xxx/color/*.png`
- arm mask：`/data/haoxiang/data/airexo2_processed/{task}/{scene}/` 下的 PNG

写一个转换脚本，**不需要采样所有帧**，每个 scene 采样 ~20-30 帧关键帧即可：

```python name=robotarm_sam/build_finetune_dataset.py
"""
将 mask_1_arm 流程产生的 arm mask PNG + 原始帧
转换成 SAM3 图片微调用的 COCO JSON 格式

用法：
    python build_finetune_dataset.py \
        --data-base /data/haoxiang/data/airexo2 \
        --mask-base /data/haoxiang/data/airexo2_processed \
        --output-dir /data/haoxiang/sam3_finetune_data \
        --text-prompt "robot and cable" \
        --sample-stride 30
"""

import argparse
import json
import glob
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


def mask_png_to_rle(mask_path: str):
    """PNG mask → COCO RLE + area"""
    mask = np.array(Image.open(mask_path).convert("L"))
    binary = (mask > 127).astype(np.uint8)
    binary_f = np.asfortranarray(binary)
    rle = mask_util.encode(binary_f)
    rle["counts"] = rle["counts"].decode("utf-8")
    area = int(binary.sum())
    return rle, area, mask.shape  # (H, W)


def mask_to_bbox_xywh(mask_path: str):
    """从 mask PNG 计算 tight bbox，COCO xywh 格式"""
    mask = np.array(Image.open(mask_path).convert("L"))
    binary = (mask > 127)
    rows, cols = np.where(binary)
    if len(rows) == 0:
        return None
    x1, y1 = int(cols.min()), int(rows.min())
    x2, y2 = int(cols.max()), int(rows.max())
    return [x1, y1, x2 - x1, y2 - y1]


def build_coco_json(
    scene_dirs,       # list of (image_dir, mask_dir)
    text_prompt: str,
    output_json: str,
    output_img_dir: str,
    sample_stride: int = 30,
    copy_images: bool = True,
):
    images, annotations = [], []
    img_id, ann_id = 1, 1

    for image_dir, mask_dir in scene_dirs:
        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            print(f"[skip] missing dir: {image_dir} or {mask_dir}")
            continue

        frame_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        # 采样关键帧：stride 间隔 + 强制包含首末帧
        sampled = frame_paths[::sample_stride]
        if frame_paths and frame_paths[-1] not in sampled:
            sampled.append(frame_paths[-1])

        for frame_path in sampled:
            fname = os.path.basename(frame_path)
            mask_path = os.path.join(mask_dir, fname)
            if not os.path.exists(mask_path):
                continue

            # 检查 mask 不为空
            bbox = mask_to_bbox_xywh(mask_path)
            if bbox is None:
                continue

            # 获取图像尺寸
            img = Image.open(frame_path)
            w, h = img.size

            # 目标路径（相对路径，供 COCO_FROM_JSON 使用）
            rel_path = os.path.relpath(frame_path, start=output_img_dir)
            if copy_images:
                dst = os.path.join(output_img_dir, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    shutil.copy2(frame_path, dst)

            images.append({
                "id": img_id,
                "file_name": rel_path,   # 相对于 img_folder 的路径
                "width": w,
                "height": h,
            })

            rle, area, _ = mask_png_to_rle(mask_path)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,      # COCO RLE 格式
                "bbox": bbox,             # xywh
                "area": area,
                "iscrowd": 0,
            })

            img_id += 1
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": text_prompt, "supercategory": "robot"}
        ],
    }
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"[done] {len(images)} images, {len(annotations)} annotations → {output_json}")
    return len(images)


def discover_scene_dirs(data_base: str, mask_base: str, cam_name: str = "cam_105422061350"):
    """自动扫描所有 task/scene 组合"""
    scene_dirs = []
    for task_dir in sorted(glob.glob(os.path.join(data_base, "task_*"))):
        task_name = os.path.basename(task_dir)
        for scene_dir in sorted(glob.glob(os.path.join(task_dir, "train", "scene_*"))):
            scene_name = os.path.basename(scene_dir)
            image_dir = os.path.join(scene_dir, cam_name, "color")
            # mask 保存位置（你的 export 路径约定）
            mask_dir = os.path.join(mask_base, task_name, scene_name)
            if os.path.isdir(image_dir) and os.path.isdir(mask_dir):
                scene_dirs.append((image_dir, mask_dir))
    return scene_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-base", default="/data/haoxiang/data/airexo2")
    parser.add_argument("--mask-base", default="/data/haoxiang/data/airexo2_processed")
    parser.add_argument("--output-dir", default="/data/haoxiang/sam3_finetune_data")
    parser.add_argument("--text-prompt", default="robot and cable")
    parser.add_argument("--sample-stride", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    all_scenes = discover_scene_dirs(args.data_base, args.mask_base)
    print(f"[scan] found {len(all_scenes)} scene dirs")

    # 按场景划分 train/val（不按帧划分，避免 leakage）
    n_val = max(1, int(len(all_scenes) * args.val_ratio))
    val_scenes = all_scenes[-n_val:]
    train_scenes = all_scenes[:-n_val]

    img_dir = os.path.join(args.output_dir, "images")

    build_coco_json(
        scene_dirs=train_scenes,
        text_prompt=args.text_prompt,
        output_json=os.path.join(args.output_dir, "train", "annotations.json"),
        output_img_dir=img_dir,
        sample_stride=args.sample_stride,
    )
    build_coco_json(
        scene_dirs=val_scenes,
        text_prompt=args.text_prompt,
        output_json=os.path.join(args.output_dir, "val", "annotations.json"),
        output_img_dir=img_dir,
        sample_stride=args.sample_stride,
    )
```

运行：

```bash
python robotarm_sam/build_finetune_dataset.py \
    --data-base /data/haoxiang/data/airexo2 \
    --mask-base /data/haoxiang/data/airexo2_processed \
    --output-dir /data/haoxiang/sam3_finetune_data \
    --text-prompt "robot and cable" \
    --sample-stride 30
```

---

## 第二步：微调训练 Config

直接仿照 `roboflow_v100_full_ft_100_images.yaml` 的结构，这是官方唯一的图片训练参考：

```yaml name=robotarm_sam/finetune_robotarm.yaml
# @package _global_
defaults:
  - _self_

# ============================================================
# 路径（只需修改这里）
# ============================================================
paths:
  dataset_root:     /data/haoxiang/sam3_finetune_data
  experiment_log_dir: /data/haoxiang/sam3_finetune_logs/robotarm
  bpe_path:         /data/haoxiang/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
  checkpoint_path:  /data/haoxiang/sam3/models/facebook/sam3/sam3.pt

# ============================================================
# scratch：超参数区
# ============================================================
scratch:
  # ⚠️ 关键：开启 segmentation loss，因为你有 mask 数据
  enable_segmentation: True

  d_model: 256
  pos_embed:
    _target_: sam3.model.position_encoding.PositionEmbeddingSine
    num_pos_feats: ${scratch.d_model}
    normalize: true
    scale: null
    temperature: 10000

  use_presence_eval: True
  original_box_postprocessor:
    _target_: sam3.eval.postprocessors.PostProcessImage
    max_dets_per_img: -1
    use_original_ids: true
    use_original_sizes_box: true
    use_presence: ${scratch.use_presence_eval}

  matcher:
    _target_: sam3.train.matcher.BinaryHungarianMatcherV2
    focal: true
    cost_class: 2.0
    cost_bbox: 5.0
    cost_giou: 2.0
    alpha: 0.25
    gamma: 2
    stable: False
  scale_by_find_batch_size: True

  resolution: 1008
  consistent_transform: False
  max_ann_per_img: 10  # 单张图里最多 arm 数（通常 1~2）

  train_norm_mean: [0.5, 0.5, 0.5]
  train_norm_std:  [0.5, 0.5, 0.5]
  val_norm_mean:   [0.5, 0.5, 0.5]
  val_norm_std:    [0.5, 0.5, 0.5]

  num_train_workers: 4
  num_val_workers: 0
  max_data_epochs: 30          # 机械臂数据少，可以多跑几轮
  target_epoch_size: 500       # 根据你的数据量调整
  hybrid_repeats: 1
  context_length: 2
  gather_pred_via_filesys: false

  # 学习率（小数据集用更小 lr，防止遗忘）
  lr_scale: 0.05               # 比官方 0.1 更保守
  lr_transformer:      ${times:8e-4,${scratch.lr_scale}}
  lr_vision_backbone:  ${times:2.5e-4,${scratch.lr_scale}}
  lr_language_backbone: ${times:5e-5,${scratch.lr_scale}}
  lrd_vision_backbone: 0.9
  wd: 0.1
  scheduler_timescale: 20
  scheduler_warmup: 10
  scheduler_cooldown: 10

  val_batch_size: 1
  train_batch_size: 1
  gradient_accumulation_steps: 4  # 有效 batch=4，显存不够时用

  collate_fn:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${scratch.hybrid_repeats}
    dict_key: all
    with_seg_masks: ${scratch.enable_segmentation}

  collate_fn_val:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${scratch.hybrid_repeats}
    dict_key: robotarm_val
    with_seg_masks: ${scratch.enable_segmentation}

# ============================================================
# Loss 配置（开启 mask loss）
# ============================================================
robotarm_loss:
  _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
  matcher: ${scratch.matcher}
  o2m_weight: 2.0
  o2m_matcher:
    _target_: sam3.train.matcher.BinaryOneToManyMatcher
    alpha: 0.3
    threshold: 0.4
    topk: 4
  use_o2m_matcher_on_o2m_aux: false
  loss_fns_find:
    - _target_: sam3.train.loss.loss_fns.Boxes
      weight_dict:
        loss_bbox: 5.0
        loss_giou: 2.0
    - _target_: sam3.train.loss.loss_fns.Labels
      weight_dict:
        loss_ce: 2.0
    # ⚠️ mask loss — 你有 mask 数据，开启它
    - _target_: sam3.train.loss.loss_fns.Masks
      weight_dict:
        loss_mask: 1.0
        loss_dice: 1.0

# ============================================================
# Train / Val Transforms
# ============================================================
train_transforms:
  - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
    transforms:
      - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
        query_filter:
          _target_: sam3.train.transforms.filter_query_transforms.FilterCrowds
      # bbox 加噪声，防止过拟合精确 bbox 位置
      - _target_: sam3.train.transforms.point_sampling.RandomizeInputBbox
        box_noise_std: 0.1
        box_noise_max: 20
      # 解码 RLE mask
      - _target_: sam3.train.transforms.segmentation.DecodeRle
      - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
        sizes:
          _target_: sam3.train.transforms.basic.get_random_resize_scales
          size: ${scratch.resolution}
          min_size: 480
          rounded: false
        max_size:
          _target_: sam3.train.transforms.basic.get_random_resize_max_size
          size: ${scratch.resolution}
        square: true
        consistent_transform: ${scratch.consistent_transform}
      - _target_: sam3.train.transforms.basic_for_api.PadToSizeAPI
        size: ${scratch.resolution}
        consistent_transform: ${scratch.consistent_transform}
      - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
      - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
        query_filter:
          _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
      - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
        mean: ${scratch.train_norm_mean}
        std:  ${scratch.train_norm_std}
      - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
        query_filter:
          _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
  - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
    query_filter:
      _target_: sam3.train.transforms.filter_query_transforms.FilterFindQueriesWithTooManyOut
      max_num_objects: ${scratch.max_ann_per_img}

val_transforms:
  - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
    transforms:
      - _target_: sam3.train.transforms.segmentation.DecodeRle
      - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
        sizes: ${scratch.resolution}
        max_size:
          _target_: sam3.train.transforms.basic.get_random_resize_max_size
          size: ${scratch.resolution}
        square: true
        consistent_transform: False
      - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
      - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
        mean: ${scratch.val_norm_mean}
        std:  ${scratch.val_norm_std}

# ============================================================
# Trainer
# ============================================================
trainer:
  _target_: sam3.train.trainer.Trainer
  skip_saving_ckpts: false   # ⚠️ 保存 checkpoint！
  empty_gpu_mem_cache_after_eval: True
  skip_first_val: True
  max_epochs: ${scratch.max_data_epochs}
  accelerator: cuda
  seed_value: 42
  val_epoch_freq: 5
  mode: train
  gradient_accumulation_steps: ${scratch.gradient_accumulation_steps}

  distributed:
    backend: nccl
    find_unused_parameters: True
    gradient_as_bucket_view: True

  loss:
    all: ${robotarm_loss}
    default:
      _target_: sam3.train.loss.sam3_loss.DummyLoss

  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        img_folder:  ${paths.dataset_root}/images
        ann_file:    ${paths.dataset_root}/train/annotations.json
        transforms:  ${train_transforms}
        load_segmentation: ${scratch.enable_segmentation}
        max_ann_per_img: 500000
        multiplier: 1
        max_train_queries: 50000
        max_val_queries: 50000
        training: true
        use_caching: False
      shuffle: True
      batch_size: ${scratch.train_batch_size}
      num_workers: ${scratch.num_train_workers}
      pin_memory: True
      drop_last: True
      collate_fn: ${scratch.collate_fn}

    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        img_folder: ${paths.dataset_root}/images
        ann_file:   ${paths.dataset_root}/val/annotations.json
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: false   # 验证时不需要 negative
          category_chunk_size: 1
          _partial_: true
        transforms: ${val_transforms}
        load_segmentation: ${scratch.enable_segmentation}
        max_ann_per_img: 100000
        multiplier: 1
        training: false
      shuffle: False
      batch_size: ${scratch.val_batch_size}
      num_workers: ${scratch.num_val_workers}
      pin_memory: True
      drop_last: False
      collate_fn: ${scratch.collate_fn_val}

  model:
    _target_: sam3.model_builder.build_sam3_image_model
    bpe_path: ${paths.bpe_path}
    device: cpus
    eval_mode: false
    enable_segmentation: ${scratch.enable_segmentation}
    checkpoint_path: ${paths.checkpoint_path}  # 从预训练权重开始 finetune

  optim:
    amp:
      enabled: True
      amp_dtype: bfloat16

    optimizer:
      _target_: torch.optim.AdamW

    gradient_clip:
      _target_: sam3.train.optim.optimizer.GradientClipper
      max_norm: 0.1
      norm_type: 2

    param_group_modifiers:
      - _target_: sam3.train.optim.optimizer.layer_decay_param_modifier
        _partial_: True
        layer_decay_value: ${scratch.lrd_vision_backbone}
        apply_to: 'backbone.vision_backbone.trunk'
        overrides:
          - pattern: '*pos_embed*'
            value: 1.0

    options:
      lr:
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${scratch.lr_transformer}
            timescale: ${scratch.scheduler_timescale}
            warmup_steps: ${scratch.scheduler_warmup}
            cooldown_steps: ${scratch.scheduler_cooldown}
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${scratch.lr_vision_backbone}
            timescale: ${scratch.scheduler_timescale}
            warmup_steps: ${scratch.scheduler_warmup}
            cooldown_steps: ${scratch.scheduler_cooldown}
          param_names:
            - 'backbone.vision_backbone.*'
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${scratch.lr_language_backbone}
            timescale: ${scratch.scheduler_timescale}
            warmup_steps: ${scratch.scheduler_warmup}
            cooldown_steps: ${scratch.scheduler_cooldown}
          param_names:
            - 'backbone.language_backbone.*'
      weight_decay:
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: ${scratch.wd}
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: 0.0
          param_names:
            - '*bias*'

# ============================================================
# Launcher
# ============================================================
launcher:
  num_nodes: 1
  gpus_per_node: 2          # 根据你的 GPU 数量调整
  experiment_log_dir: ${paths.experiment_log_dir}

submitit:
  use_cluster: False
  timeout_hour: 72
```

---

## 第三步：启动训练

```bash
cd /data/haoxiang/sam3   # SAM3 仓库根目录

python sam3/train/train.py \
    -c robotarm_sam/finetune_robotarm.yaml \
    --use-cluster 0 \
    --num-gpus 2
```

---

## 第四步：用微调后的权重替换 checkpoint

训练完成后，checkpoint 保存在 `experiment_log_dir/checkpoints/` 下，直接修改 `mask_config.yaml`：

```yaml name=robotarm_sam/mask_config.yaml url=https://github.com/1JI0O/sam3/blob/33d92a06a28337f5b2342fca268a960385334c8b/robotarm_sam/mask_config.yaml#L14
# 改为微调后的权重路径
sam3_checkpoint: "/data/haoxiang/sam3_finetune_logs/robotarm/checkpoints/checkpoint_epoch_XX.pt"
```

`mask_1_arm.ipynb` 的其余流程完全不需要改动。

---

## 关键参数说明

| 参数 | 值 | 原因 |
|---|---|---|
| `enable_segmentation: True` | 开启 | 你有 mask，开启 mask loss 训练更精确 |
| `with_seg_masks: True` | 对应开启 | collate 时传 mask |
| `lr_scale: 0.05` | 比官方 0.1 小 | 数据量小，防止 catastrophic forgetting |
| `gradient_accumulation_steps: 4` | 4步累积 | 显存不足时模拟更大 batch |
| `sample_stride: 30` | 每 30 帧取 1 帧 | 相邻帧太相似，间隔采样增加多样性 |
| `skip_saving_ckpts: false` | 保存 | 必须保存，否则无法用于推理 |
| `include_negatives: false` (val) | 关闭 | 验证集只有一个类，无 negative |