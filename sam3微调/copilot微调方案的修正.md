260228，[[copilot微调方案]]

先看具体的 anno 相关代码。现在完全清楚了你的完整数据结构。让我整合所有信息给出完整答案。

---

## 完整分析

### 你的实际目录结构（已确认）

```
airexo2/{task}/train/{scene}/cam_105422061350/color/
    00000.png  00001.png  ...  (原始帧，按帧索引命名)

airexo2_processed/{task}/{scene}_ckpt_arm/
    00000.png  00001.png  ...  (mask_1_arm 的 arm union checkpoint，未膨胀)

airexo2_processed/{task}/{scene}/
    00000.png  00001.png  ...  (mask_3_merge 最终输出：arm_only = arm - gripper，已膨胀)
```

**微调数据源应用 `_ckpt_arm/` 而不是最终的 merge 输出**，因为 `_ckpt_arm` 是纯 arm mask（未被 gripper 挖去），更接近你想让模型学习的目标。

---

### 关于"传播丢失目标"的问题

这个问题有两个独立原因，解法不同：

**原因 A：第 0 帧 grounding 的 mask 质量差** → 微调图片模型解决  
**原因 B：grounding 质量 OK 但中途目标被遮挡/相似背景混淆** → 这是 tracker 的问题，微调图片模型**无法**解决

对于原因 B，你现在的手标点 refine 方式（在丢失的帧补标点）是正确的，但可以在流程上**自动化**：检测到丢失后自动在中间关键帧补充点约束，然后重新传播。

---

## 数据准备脚本（修正版，对应你的实际路径）

```python name=robotarm_sam/build_finetune_dataset.py
"""
将 mask_1_arm 的 _ckpt_arm/ 输出 + 原始帧
转换成 SAM3 图片微调的 COCO JSON 数据集。

目录约定（来自你的 mask_config.yaml）：
  原始帧：  {data_base}/{task}/train/{scene}/cam_105422061350/color/{NNNNN}.png
  arm mask：{export_base}/{task}/{scene}_ckpt_arm/{NNNNN}.png   ← 使用这个！

输出目录结构：
  output_dir/
    images/
      task_0013/scene_0001/00000.png
      ...
    train/annotations.json
    val/annotations.json

用法示例：
  python robotarm_sam/build_finetune_dataset.py \
      --data-base   /data/haoxiang/data/airexo2 \
      --export-base /data/haoxiang/data/airexo2_processed \
      --output-dir  /data/haoxiang/sam3_finetune_data \
      --text-prompt "robot and cable" \
      --sample-stride 30 \
      --val-scenes task_0013/scene_0044 task_0013/scene_0050
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def load_binary_mask(path: str) -> np.ndarray:
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def to_rle(binary: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(binary))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def to_bbox_xywh(binary: np.ndarray):
    rows, cols = np.where(binary)
    if len(rows) == 0:
        return None
    x1, y1 = int(cols.min()), int(rows.min())
    x2, y2 = int(cols.max()), int(rows.max())
    return [x1, y1, x2 - x1, y2 - y1]


# ─────────────────────────────────────────────────────────
# 场景扫描
# ─────────────────────────────────────────────────────────

def discover_scenes(data_base: str, export_base: str,
                    cam: str = "cam_105422061350"):
    """
    返回 list of dict：
        image_dir : 原始帧目录
        mask_dir  : _ckpt_arm 目录（纯 arm mask，未膨胀）
        scene_key : "task_0013/scene_0001"
    """
    scenes = []
    for task_dir in sorted(glob.glob(os.path.join(data_base, "task_*"))):
        task = os.path.basename(task_dir)
        for scene_dir in sorted(glob.glob(os.path.join(task_dir, "train", "scene_*"))):
            scene = os.path.basename(scene_dir)
            image_dir = os.path.join(scene_dir, cam, "color")
            # ⚠️ 用 _ckpt_arm，不是 merge 后的目录
            mask_dir = os.path.join(export_base, task, f"{scene}_ckpt_arm")
            if os.path.isdir(image_dir) and os.path.isdir(mask_dir):
                scenes.append(dict(
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                    scene_key=f"{task}/{scene}",
                ))
    return scenes


# ─────────────────────────────────────────────────────────
# 构建单个 split
# ───────────────────────────────────────────────────────���─

def build_split(scenes, text_prompt, output_json, img_root,
                sample_stride=30, tag="train"):
    images, annotations = [], []
    img_id = ann_id = 1
    skip_missing = skip_empty = 0

    for s in scenes:
        image_dir = s["image_dir"]
        mask_dir  = s["mask_dir"]
        scene_key = s["scene_key"]

        frame_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not frame_paths:
            continue

        # stride 采样 + 强制包含首尾
        sampled = frame_paths[::sample_stride]
        if frame_paths[-1] not in sampled:
            sampled.append(frame_paths[-1])

        for fpath in sampled:
            fname = os.path.basename(fpath)          # e.g. "00030.png"
            mpath = os.path.join(mask_dir, fname)

            if not os.path.exists(mpath):
                skip_missing += 1
                continue

            binary = load_binary_mask(mpath)
            bbox   = to_bbox_xywh(binary)
            if bbox is None:
                skip_empty += 1
                continue

            # 复制图片到统一 img_root，保持相对路径
            rel_name = os.path.join(scene_key, fname)   # "task_0013/scene_0001/00030.png"
            dst = os.path.join(img_root, rel_name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(fpath, dst)

            img = Image.open(fpath)
            w, h = img.size

            images.append({
                "id": img_id, "file_name": rel_name,
                "width": w, "height": h,
            })
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": to_rle(binary),
                "bbox": bbox, "area": int(binary.sum()), "iscrowd": 0,
            })
            img_id += 1
            ann_id += 1

    coco = {
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": text_prompt, "supercategory": "robot"}],
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"[{tag}] {len(images)} images, {len(annotations)} annotations → {output_json}")
    print(f"       skip: {skip_missing} no-mask, {skip_empty} empty-mask")
    return len(images)


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-base",     default="/data/haoxiang/data/airexo2")
    p.add_argument("--export-base",   default="/data/haoxiang/data/airexo2_processed")
    p.add_argument("--output-dir",    default="/data/haoxiang/sam3_finetune_data")
    p.add_argument("--text-prompt",   default="robot and cable")
    p.add_argument("--sample-stride", type=int, default=30,
                   help="每隔 N 帧采一帧（30fps 时 stride=30 ≈ 1fps）")
    p.add_argument("--val-scenes",    nargs="*", default=[],
                   help="手动指定 val scene，如 task_0013/scene_0044")
    p.add_argument("--val-ratio",     type=float, default=0.15)
    args = p.parse_args()

    all_scenes = discover_scenes(args.data_base, args.export_base)
    print(f"[scan] {len(all_scenes)} scenes found")
    for s in all_scenes:
        print(f"       {s['scene_key']}")

    if not all_scenes:
        print("[error] 未找到任何场景，请检查路径")
        exit(1)

    # train / val 划分（按 scene，不按帧）
    if args.val_scenes:
        val_keys   = set(args.val_scenes)
        val_sc     = [s for s in all_scenes if s["scene_key"] in val_keys]
        train_sc   = [s for s in all_scenes if s["scene_key"] not in val_keys]
    else:
        n_val    = max(1, int(len(all_scenes) * args.val_ratio))
        val_sc   = all_scenes[-n_val:]
        train_sc = all_scenes[:-n_val]

    print(f"\n[split] train={len(train_sc)} scenes, val={len(val_sc)} scenes")

    img_root = os.path.join(args.output_dir, "images")

    print("\n[build] train ...")
    build_split(train_sc, args.text_prompt,
                os.path.join(args.output_dir, "train", "annotations.json"),
                img_root, args.sample_stride, tag="train")

    print("\n[build] val ...")
    build_split(val_sc, args.text_prompt,
                os.path.join(args.output_dir, "val", "annotations.json"),
                img_root, max(1, args.sample_stride // 2), tag="val")

    print(f"\n[done] img_root = {img_root}")
```

---

## 关于"传播中途丢失目标"的自动化处理

在 `mask_1_arm.ipynb` 现有流程的基础上，增加一个"自动检测丢失帧 → 补充点 → 重传播"的循环，写成工具函数加进 `mask_pipeline_tools.py`：

```python name=robotarm_sam/mask_pipeline_tools_addon.py
"""
添加到 mask_pipeline_tools.py 末尾的工具函数：
自动检测 arm 丢失帧，在丢失段中点补充 positive 点，重新传播。

在 mask_1_arm.ipynb 里这样调用：

    outputs_arm = auto_refine_lost_frames(
        predictor_obj      = predictor_arm,
        session_id_value   = session_id_arm,
        outputs_per_frame  = outputs_arm,
        target_obj_id      = ARM_OBJ_ID,
        video_frames       = video_frames_for_vis,
        img_width          = IMG_WIDTH,
        img_height         = IMG_HEIGHT,
        min_mask_pixels    = 500,     # 少于这个像素视为"丢失"
        max_refine_rounds  = 3,       # 最多尝试几轮自动修复
        stage_name         = "auto_refine",
    )
"""

import numpy as np
import torch


def detect_lost_frames(outputs_per_frame, target_obj_id, total_frames,
                       min_mask_pixels=500):
    """
    返回"目标丢失"的帧索引列表。
    丢失定义：该帧中 target_obj_id 的 mask 像素数 < min_mask_pixels。
    """
    lost = []
    for fi in range(total_frames):
        frame_out = outputs_per_frame.get(fi, {})
        found_pixels = 0
        for oid, mask in iter_object_masks_from_frame_output(frame_out):
            if int(oid) == int(target_obj_id):
                m = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                m = np.squeeze(m)
                found_pixels = int((m > 0).sum())
                break
        if found_pixels < min_mask_pixels:
            lost.append(fi)
    return lost


def find_lost_segments(lost_frames, total_frames):
    """
    将离散的丢失帧合并成连续段，返回 list of (start, end) 闭区间。
    """
    if not lost_frames:
        return []
    segments, seg_start = [], lost_frames[0]
    prev = lost_frames[0]
    for f in lost_frames[1:]:
        if f > prev + 1:
            segments.append((seg_start, prev))
            seg_start = f
        prev = f
    segments.append((seg_start, prev))
    return segments


def pick_refine_frame_from_good_neighbor(segment, outputs_per_frame,
                                         target_obj_id, total_frames,
                                         min_mask_pixels=500):
    """
    在丢失段两端找最近的"好帧"，返回 (frame_idx, center_xy_normalized)。
    center_xy_normalized: (x, y) 均在 [0,1]，相对于图像宽高。
    """
    seg_start, seg_end = segment

    # 向左找好帧
    left_good = None
    for fi in range(seg_start - 1, -1, -1):
        for oid, mask in iter_object_masks_from_frame_output(
                outputs_per_frame.get(fi, {})):
            if int(oid) == int(target_obj_id):
                m = np.squeeze(
                    mask.detach().cpu().numpy()
                    if isinstance(mask, torch.Tensor) else mask
                )
                if int((m > 0).sum()) >= min_mask_pixels:
                    left_good = (fi, m)
                    break
        if left_good is not None:
            break

    # 向右找好帧
    right_good = None
    for fi in range(seg_end + 1, total_frames):
        for oid, mask in iter_object_masks_from_frame_output(
                outputs_per_frame.get(fi, {})):
            if int(oid) == int(target_obj_id):
                m = np.squeeze(
                    mask.detach().cpu().numpy()
                    if isinstance(mask, torch.Tensor) else mask
                )
                if int((m > 0).sum()) >= min_mask_pixels:
                    right_good = (fi, m)
                    break
        if right_good is not None:
            break

    # 用更近的好帧的 mask 中心作为 refine 点
    # refine 注入到丢失段的中间帧
    mid_frame = (seg_start + seg_end) // 2
    best = left_good if left_good is not None else right_good
    if best is None:
        return None

    _, good_mask = best
    rows, cols = np.where(good_mask > 0)
    if len(rows) == 0:
        return None
    cy = float(rows.mean()) / good_mask.shape[0]
    cx = float(cols.mean()) / good_mask.shape[1]

    return mid_frame, (cx, cy)


def auto_refine_lost_frames(
    predictor_obj,
    session_id_value,
    outputs_per_frame,
    target_obj_id,
    video_frames,
    img_width,
    img_height,
    min_mask_pixels=500,
    max_refine_rounds=3,
    stage_name="auto_refine",
):
    """
    自动检测丢失帧段，在丢失段中间补充 positive 点，重新传播。
    返回更新后的 outputs_per_frame。

    调用位置：mask_1_arm.ipynb 里 propagate 之后、保存 checkpoint 之前。
    """
    total_frames = len(video_frames)

    for round_i in range(max_refine_rounds):
        lost = detect_lost_frames(
            outputs_per_frame, target_obj_id, total_frames, min_mask_pixels)

        if not lost:
            print(f"[{stage_name}] round {round_i}: no lost frames, done.")
            break

        segments = find_lost_segments(lost, total_frames)
        print(f"[{stage_name}] round {round_i}: "
              f"{len(lost)} lost frames in {len(segments)} segments → "
              f"{segments[:5]}{'...' if len(segments) > 5 else ''}")

        refine_added = 0
        for seg in segments:
            result = pick_refine_frame_from_good_neighbor(
                seg, outputs_per_frame, target_obj_id,
                total_frames, min_mask_pixels)
            if result is None:
                print(f"[{stage_name}]   seg {seg}: no good neighbor found, skipping")
                continue

            mid_frame, (cx, cy) = result
            print(f"[{stage_name}]   seg {seg}: inject point @ "
                  f"frame={mid_frame} rel=({cx:.3f},{cy:.3f})")

            # 注入 positive 点（归一化坐标）
            add_point_prompt(
                predictor_obj=predictor_obj,
                session_id_value=session_id_value,
                prompt={
                    "frame_index": mid_frame,
                    "obj_id":      int(target_obj_id),
                    "points_rel":  [[cx, cy]],
                    "labels":      [1],       # 1 = positive
                },
                stage_name=stage_name,
            )
            refine_added += 1

        if refine_added == 0:
            print(f"[{stage_name}] round {round_i}: no refine points added, stopping.")
            break

        # 重新双向传播
        print(f"[{stage_name}] round {round_i}: re-propagating ...")
        outputs_per_frame = propagate_bidirectional_and_merge(
            predictor_obj=predictor_obj,
            session_id_value=session_id_value,
            stage_name=f"{stage_name}/round{round_i}",
        )

    # 最终统计
    lost_final = detect_lost_frames(
        outputs_per_frame, target_obj_id, total_frames, min_mask_pixels)
    print(f"[{stage_name}] final: {len(lost_final)} lost frames remaining "
          f"(out of {total_frames})")
    if lost_final:
        segs = find_lost_segments(lost_final, total_frames)
        print(f"[{stage_name}] remaining lost segments: {segs}")

    return outputs_per_frame
```

在 `mask_1_arm.ipynb` 里，在现有的 propagate 之后、保存 checkpoint 之前插入：

```python name=mask_1_arm_auto_refine_snippet.py
# ── 原有流程：text bootstrap + point refine + 首次传播 ──
# ...（你现有的代码不变）...

# ── 新增：自动修复丢失帧 ──
from mask_pipeline_tools import auto_refine_lost_frames  # 或直接 %run

outputs_arm = auto_refine_lost_frames(
    predictor_obj     = predictor_arm,
    session_id_value  = session_id_arm,
    outputs_per_frame = outputs_arm,
    target_obj_id     = ARM_OBJ_ID,
    video_frames      = video_frames_for_vis,
    img_width         = IMG_WIDTH,
    img_height        = IMG_HEIGHT,
    min_mask_pixels   = 500,    # 根据你的 arm 大小调整（arm 很大可设 2000）
    max_refine_rounds = 3,
    stage_name        = "auto_refine/arm",
)

# ── 之后正常保存 checkpoint ──（你现有的代码不变）
```

---

## 总结：两件事分开做

| 问题 | 解法 | 状态 |
|---|---|---|
| **text bootstrap 准确率差** | 微调 `Sam3Image` 图片模型（数据来自 `_ckpt_arm/`） | 上面的 config + 数据脚本 |
| **传播中途丢帧** | `auto_refine_lost_frames()` 自动补点重传播 | 上面的工具函数 |

两者**互相独立**，可以先做自动修复（无需训练，直接用），再做微调（彻底解决根因）。微调完成后，第 0 帧质量提升，自动修复需要补的点也会减少，最终达到你想要的"不用手标"的目标。