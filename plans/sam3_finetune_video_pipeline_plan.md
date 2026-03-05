# SAM3 数据转换脚本 + 训练配置修改计划

## Context

当前 SAM3 微调效果差于 SAM2 的根因（来自用户分析）：
1. 数据转换层：只用 `_ckpt_arm`，无 gripper 监督；stem 对齐容易丢帧
2. 训练范式：用 `Sam3ImageDataset`（单帧 COCO），未启用时序 `VideoGroundingDataset`
3. 超参数：学习率偏激进，multiplier=1

参考基准：`finetune_plan_sam2.md`（SAM2 用双目标 VOS 时序训练，双掩码 index 对齐，LR 保守，multiplier=8）

目标：对齐 SAM2 的核心设计决策，修改 SAM3 数据转换脚本和训练配置。

---

## 修改范围（3 个文件）

### 文件 1：`robotarm_sam/build_finetune_dataset.py`

**现存问题：**
- `DEFAULT_MASK_GLOB_TEMPLATE` 只读 `_ckpt_arm`，无 gripper
- 掩码对齐是 stem 匹配（`mask_by_frame = {image_path.stem: mask_path}`），命名不一致会丢帧
- 输出为 image-level COCO JSON（每帧独立），无法驱动视频时序训练

**改动：**

1. 新增 `DEFAULT_GRIPPER_MASK_GLOB_TEMPLATE = "{export_base}/{task}/{scene}_ckpt_gripper/*.png"`

2. `SceneInventory` 新增字段 `gripper_files: List[Path]`

3. 掩码对齐改为 **index-based**（与 SAM2 一致）：
   ```python
   # 旧：stem 匹配（fragile）
   mask_by_frame = {Path(m).stem: Path(m) for m in mask_files}
   frame = image_path.stem
   mask = mask_by_frame.get(frame)   # 可能 None → skip

   # 新：排序后按索引对齐（robust）
   arm_files     = sorted(glob(...arm...))
   gripper_files = sorted(glob(...gripper...))
   n_frames      = min(len(img_files), len(arm_files), len(gripper_files))
   for idx in range(n_frames):
       arm_mask  = arm_files[idx]
       grp_mask  = gripper_files[idx]
   ```

4. 输出格式改为 **video-level JSON**（VideoGroundingDataset 所需），替代原 image-level COCO：
   ```json
   {
     "videos":      [{"id": vid_id, "name": "task/scene", "file_names": [...]}],
     "images":      [{"id": frame_global_id, "video_id": vid_id, "frame_index": fi,
                      "file_name": "...", "width": W, "height": H}],
     "annotations": [{"id": ann_id, "image_id": frame_global_id, "video_id": vid_id,
                      "frame_index": fi, "category_id": 1_or_2, "object_id": 1_or_2,
                      "bbox": [x,y,w,h], "segmentation": {rle}, "area": float, "iscrowd": 0}],
     "categories":  [{"id": 1, "name": "robot and cable"}, {"id": 2, "name": "gripper"}]
   }
   ```
   - 每帧每scene输出 2 条 annotation（arm cat_id=1，gripper cat_id=2）
   - 输出文件：`train/video_annotations.json`，`val/video_annotations.json`

5. stride 默认值改为 **1**（全帧）：VideoGroundingDataset 自己做 `num_stages_sample` 时序子采样，数据脚本无需稀疏采样。

---

### 文件 2：`sam3/train/data/coco_json_loaders.py`

**现存问题：**
- 只有 `COCO_FROM_JSON`（image 训练）和两个 eval-only loader（无 training annotation 的视频 loader）
- 无法驱动 VideoGroundingDataset 的有标注视频训练

**改动：**

新增 `COCO_VIDEO_FROM_JSON` 类，追加至文件末尾：

```python
class COCO_VIDEO_FROM_JSON:
    """Video training API: 每个视频是一个 datapoint，
    返回逐帧 queries (query_processing_order=frame_index) 和 RLE 标注。
    供 VideoGroundingDataset 使用。"""

    def __init__(self, annotation_file, include_negatives=False):
        # 读 video JSON，建立索引：
        # self._videos            = [{id, name, file_names}]
        # self._images_by_vid     = {vid_id: [{id, frame_index, file_name, width, height}]}
        # self._anns_by_vid_frame = {(vid_id, frame_idx): [ann, ...]}
        # self._cat_id_to_name    = {1: "robot and cable", 2: "gripper"}
        ...

    def getDatapointIds(self):
        return list(range(len(self._videos)))

    def loadQueriesAndAnnotationsFromDatapoint(self, vid_idx):
        # 对每帧 × 每 category，生成一条 query：
        #   query["query_processing_order"] = frame_index
        #   query["image_id"]               = frame_index
        #   query["query_text"]             = category name
        #   query["object_ids_output"]      = [ann_id, ...]
        # 对每帧 × 每 category，生成对应 annotation：
        #   annotation["image_id"]     = frame_index
        #   annotation["bbox"]         = normalized tensor (xywh)
        #   annotation["segmentation"] = RLE
        ...

    def loadImagesFromDatapoint(self, vid_idx):
        # 返回 [{id: frame_index, file_name: ..., ...}] 供 _load_images 使用
        ...
```

关键细节：
- `annotation["bbox"]` 需归一化为 [0,1]（与 `COCO_FROM_JSON` 一致）
- `annotation["segmentation"]` 调用 `ann_to_rle()` 处理已有 RLE
- `is_exhaustive: True`（每场景2类对象均有完整标注，明确 exhaustive）

---

### 文件 3：`sam3/train/configs/robotarm/robotarm_full_ft.yaml`

**现存问题：**
- 数据集：`Sam3ImageDataset` + `COCO_FROM_JSON`（单帧）
- 模型：`build_sam3_image_model`
- multiplier=1，学习率过高，无时序采样

**改动：**

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| 训练数据集 `_target_` | `Sam3ImageDataset` | `VideoGroundingDataset` | 启用时序训练 |
| `coco_json_loader` | 无 | `COCO_VIDEO_FROM_JSON` | 对应新数据格式 |
| 验证数据集 `_target_` | `Sam3ImageDataset` | `VideoGroundingDataset` | 一致 |
| `ann_file` (train) | `train/annotations.json` | `train/video_annotations.json` | 新格式 |
| `ann_file` (val) | `val/annotations.json` | `val/video_annotations.json` | 新格式 |
| 模型 `_target_` | `build_sam3_image_model` | `build_sam3_video_model` | 时序模型 |
| `num_stages_sample` | 无 | **6** | 每次训练采 6 帧（对齐 SAM2 num_frames=6） |
| `stage_stride_min` | 无 | **1** | 最小帧间隔 |
| `stage_stride_max` | 无 | **5** | 最大帧间隔 |
| `override_query_is_exhaustive_to_true` | 无 | **true** | 确保分类 loss 有效 |
| `multiplier` (train) | 1 | **8** | 小数据集多重复（对齐 SAM2 multiplier=8） |
| `lr_scale` | 0.1 | **0.01** | LR 降低 10×（对齐 SAM2 保守策略） |
| `gpus_per_node` | 4 | 4 | 保持 |
| `train_batch_size` | 8 | **4** | 视频比图像显存高，适当降低 |
| `collate_fn dict_key` | all | all | 保持不变 |

LR 计算后变为：
- `lr_transformer = 8e-4 × 0.01 = 8e-6`
- `lr_vision_backbone = 2.5e-4 × 0.01 = 2.5e-6`（接近 SAM2 的 2.0e-6）
- `lr_language_backbone = 5e-5 × 0.01 = 5e-7`

---

## 关键文件路径

| 文件 | 作用 |
|------|------|
| `robotarm_sam/build_finetune_dataset.py` | 数据转换脚本（主改） |
| `sam3/train/data/coco_json_loaders.py` | 新增 `COCO_VIDEO_FROM_JSON`（主增） |
| `sam3/train/configs/robotarm/robotarm_full_ft.yaml` | 训练配置（主改） |
| `finetune_plan_sam2.md` | SAM2 参考配置（只读） |
| `raw_data_folder_fomat.md` | 原始数据路径结构（只读） |
| `sam3/train/data/sam3_video_dataset.py` | VideoGroundingDataset 实现（只读，接口参考） |

---

## 验证方式

1. **数据转换验证**（运行脚本后）：
   ```bash
   python robotarm_sam/build_finetune_dataset.py --config robotarm_sam/dataset_build_config_template.yaml
   # 检查：
   # - train/video_annotations.json 中 videos 数量 = 40，val = 10
   # - 每个 video 有 2 个 category 的 annotations（arm + gripper）
   # - annotations 中 category_id 1 和 2 各有相同帧数
   # - 无 skip_no_mask（index 对齐不会丢帧）
   ```

2. **Loader 验证**：
   ```python
   from sam3.train.data.coco_json_loaders import COCO_VIDEO_FROM_JSON
   loader = COCO_VIDEO_FROM_JSON("train/video_annotations.json")
   ids = loader.getDatapointIds()  # 应等于 train video 数量
   queries, anns = loader.loadQueriesAndAnnotationsFromDatapoint(0)
   # 检查：query_processing_order 覆盖多个不同帧索引
   # 检查：arm query 和 gripper query 各自有标注
   ```

3. **训练启动验证**：
   ```bash
   python sam3/train/train.py \
     -c configs/robotarm/robotarm_full_ft.yaml \
     --use-cluster 0 --num-gpus 1
   # 期望：无 KeyError，第一个 batch 正常前向，loss 不为 NaN
   ```
