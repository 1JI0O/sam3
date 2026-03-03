# 阶段A 数据集构建说明

本文档只覆盖“阶段A：数据集构建”，不涉及训练配置或训练执行。

## 1. 脚本入口

- 脚本：`robotarm_sam/build_finetune_dataset.py`
- 目标：从原始图像目录与 `_ckpt_arm` 掩码目录生成 COCO `train/val` 标注、split 清单与阶段A报告。

## 2. 输入约定

支持两种图像目录布局（可在配置里覆盖 `image_glob_template`）：

- 旧布局：`{data_base}/{task}/train/{scene}/cam_105422061350/color/*.png`
- 新布局：`{data_base}/{task}/{scene}/cam_105422061350/color/*.png`

脚本会优先使用配置中的 `image_glob_template`，若模板包含 `train` 且未匹配到图像，会自动回退尝试无 `train` 版本；反之亦然。

掩码默认模板：

- 掩码：`{export_base}/{task}/{scene}_ckpt_arm/*.png`

关键参数：

- `task_list`：可选，指定任务子集；为空则自动扫描。
- `sample_stride`：默认 `30`，并强制保留首尾帧。
- `val_ratio`：自动划分验证场景比例，默认 `0.2`。
- `val_scenes`：可选，手工验证场景；支持 `task/scene`（推荐）或 scene 名（全局唯一时）。
- `category_name`：默认 `robot and cable`。

## 3. 快速开始

### 3.1 使用 YAML 配置

```bash
python robotarm_sam/build_finetune_dataset.py \
  --config robotarm_sam/dataset_build_config_template.yaml
```

### 3.2 纯命令行参数

```bash
python robotarm_sam/build_finetune_dataset.py \
  --data-base /path/to/raw_data \
  --export-base /path/to/ckpt_arm_masks \
  --output-dir /path/to/output_dataset \
  --sample-stride 30 \
  --val-ratio 0.2 \
  --category-name "robot and cable"
```

### 3.3 手工指定验证场景

```bash
python robotarm_sam/build_finetune_dataset.py \
  --data-base /path/to/raw_data \
  --export-base /path/to/ckpt_arm_masks \
  --output-dir /path/to/output_dataset \
  --val-scenes task1/scene01,task2/scene09
```

## 4. 输出目录

```text
{output_dir}/
  images/
    {task}/
      {scene}/
        {frame}.png
  train/
    annotations.json
  val/
    annotations.json
  reports/
    split_manifest.json
    dataset_summary.json
    dataset_summary.md
    build_history.jsonl
```

其中：

- `split_manifest.json`：冻结的 split 清单（统计时间点：样本构建前）。
- `dataset_summary.json/.md`：包含 KPI-A1~A9、统计字段、风险触发判定。
- `build_history.jsonl`：同参数历史窗口，用于 R-A1/R-A2/R-A3 的“连续2次”判定。

## 5. 规则实现要点

1. **采样规则**：`range(0, n, sample_stride)` 后并入首尾帧并去重。
2. **空样本过滤**：
   - no-mask（无同名掩码）跳过并计数。
   - empty-mask（二值化后全0）跳过并计数。
3. **COCO导出**：
   - `bbox` 为 `xywh`。
   - `segmentation` 为压缩 RLE。
   - `area` 为前景像素数。
   - `iscrowd` 固定 `0`。
4. **scene级切分（防泄漏）**：
   - 同一 `task/scene` 不跨 split。
   - 自动划分按排序后尾部场景进入 `val`，并遵循边界公式：
     - `raw = ceil(val_ratio * N_scene)`
     - `val_scene_count = min(max(1, raw), N_scene - 1)`（当 `N_scene >= 2`）
     - 当 `N_scene < 2`：自动划分失败，KPI-A9 不通过。

## 6. 常见失败原因与排查

1. **`data_base`/`export_base` 不存在**
   - 报错：`FileNotFoundError`
   - 处理：检查路径与挂载。

2. **没有发现任何 scene 或图像帧**
   - 报错：`未发现任何可用 scene`
   - 处理：检查任务目录是否符合“旧布局/新布局”之一，并确认 `image_glob_template` 与实际路径一致。

3. **手工 `val_scenes` 不合法**
   - 报错：未知 scene 或重名 scene 需写全 `task/scene`
   - 处理：改为 `task/scene` 全名。

4. **缺少 `pycocotools`**
   - 报错：`缺少依赖 pycocotools`
   - 处理：`pip install pycocotools`。

5. **构建返回码非0**
   - 返回码 `0`：硬门槛（KPI-A1~A5）通过
   - 返回码 `3`：构建完成但硬门槛未通过
   - 返回码 `1/2`：运行或配置错误

## 7. 产物可审计字段说明

`dataset_summary.json` 至少包含：

- split 统计：scene 总数、train/val scene 数、scene 列表
- 样本统计：总帧、采样帧、有效样本、跳过计数（no-mask/empty-mask/尺寸不匹配）
- KPI-A1~A9 全量结果
- 风险触发判定（R-A1~R-A6）
- 异常说明（非法 annotation 原因计数等）

以上字段可用于 DoD 中的报告完备性检查。
