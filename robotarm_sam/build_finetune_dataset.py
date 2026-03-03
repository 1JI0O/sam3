#!/usr/bin/env python3
"""
阶段A：从原始图像和 _ckpt_arm 掩码构建 COCO 训练数据。

核心目标（与 spec 对齐）：
1. scene 级 split（防泄漏），支持手工 val_scenes 与自动 val_ratio 划分。
2. 关键帧采样（默认 stride=30），强制保留首尾帧。
3. 生成 train/val COCO JSON（bbox=xywh, segmentation=压缩 RLE, area 等必填字段）。
4. 输出 split 清单、统计报告、KPI-A1~A9 与风险触发判定。
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_IMAGE_GLOB_TEMPLATE = (
    "{data_base}/{task}/train/{scene}/cam_105422061350/color/*.png"
)
DEFAULT_MASK_GLOB_TEMPLATE = "{export_base}/{task}/{scene}_ckpt_arm/*.png"
DEFAULT_CATEGORY_NAME = "robot and cable"


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path only
        raise ImportError(
            "缺少依赖 numpy。请先安装：pip install numpy"
        ) from exc
    return np


def _require_pillow_image_class():
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path only
        raise ImportError(
            "缺少依赖 Pillow。请先安装：pip install pillow"
        ) from exc
    return Image


def _require_pycocotools_mask_util():
    try:
        from pycocotools import mask as mask_util  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path only
        raise ImportError(
            "缺少依赖 pycocotools，无法进行压缩RLE编码。请先安装：pip install pycocotools"
        ) from exc
    return mask_util


@dataclass(frozen=True)
class BuildConfig:
    data_base: Path
    export_base: Path
    output_dir: Path
    task_list: Optional[List[str]]
    sample_stride: int
    val_ratio: float
    val_scenes: Optional[List[str]]
    category_name: str
    image_glob_template: str
    mask_glob_template: str
    history_file_relpath: str
    log_level: str


@dataclass(frozen=True)
class SceneInventory:
    scene_key: str
    task: str
    scene: str
    image_files: List[Path]
    mask_by_frame: Dict[str, Path]
    duplicate_mask_frames: int


@dataclass(frozen=True)
class SplitPlan:
    mode: str
    all_scenes: List[str]
    train_scenes: List[str]
    val_scenes: List[str]
    n_scene: int
    requested_val_ratio: float
    computed_val_scene_count: int
    expected_auto_val_scene_count: Optional[int]
    auto_split_failed: bool
    auto_failure_reason: Optional[str]


_NATURAL_TOKEN_SPLIT_RE = re.compile(r"(\d+)")


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("stage_a_dataset_builder")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    payload = {
        "ts": utc_now_iso(),
        "event": event,
        **kwargs,
    }
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def parse_csv_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values if values else []


def ensure_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return parse_csv_list(value)
    raise ValueError(f"期望 list 或逗号分隔字符串，实际类型为: {type(value)}")


def get_nested(dct: Dict[str, Any], path: Sequence[str]) -> Any:
    cursor: Any = dct
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def pick_config_value(
    raw: Dict[str, Any],
    candidate_paths: Sequence[Sequence[str]],
    default: Any = None,
) -> Any:
    for path in candidate_paths:
        value = get_nested(raw, path)
        if value is not None:
            return value
    return default


def load_config_file(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "读取 YAML 配置需要 PyYAML。请安装 `pip install pyyaml`，或改用 JSON 配置。"
            ) from exc
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return loaded if isinstance(loaded, dict) else {}

    raise ValueError(
        f"不支持的配置文件后缀: {config_path.suffix}，仅支持 .json/.yaml/.yml"
    )


def natural_sort_tokens(text: str) -> Tuple[Tuple[int, Any], ...]:
    """自然序：数字片段按数值比较，非数字按字典序。"""
    parts = [part for part in _NATURAL_TOKEN_SPLIT_RE.split(text) if part != ""]
    tokens: List[Tuple[int, Any]] = []
    for part in parts:
        if part.isdigit():
            tokens.append((0, int(part)))
        else:
            tokens.append((1, part))
    return tuple(tokens)


def parse_scene_key(scene_key: str) -> Tuple[str, str]:
    if "/" not in scene_key:
        raise ValueError(f"scene_key 必须是 task/scene 格式，收到: {scene_key}")
    task, scene = scene_key.split("/", 1)
    if not task or not scene:
        raise ValueError(f"scene_key 非法，收到: {scene_key}")
    return task, scene


def scene_sort_key(scene_key: str) -> Tuple[str, Tuple[Tuple[int, Any], ...], str]:
    task, scene = parse_scene_key(scene_key)
    # 稳定兜底键：完整 task/scene
    return (task, natural_sort_tokens(scene), scene_key)


def compute_auto_val_scene_count(n_scene: int, val_ratio: float) -> int:
    """
    spec 对齐：
    - raw = ceil(val_ratio * N_scene)
    - 当 N_scene >= 2: count = min(max(1, raw), N_scene - 1)
    - 当 N_scene < 2: 自动划分失败，返回 0
    """
    if n_scene < 2:
        return 0
    raw_count = math.ceil(val_ratio * n_scene)
    return min(max(1, raw_count), n_scene - 1)


def resolve_manual_val_scenes(
    requested: List[str],
    all_scenes: List[str],
) -> List[str]:
    """
    支持两种手工输入：
    1) task/scene（推荐，唯一）
    2) scene（仅当在全局唯一时可用）
    """
    all_scene_set = set(all_scenes)
    scene_name_to_keys: Dict[str, List[str]] = {}
    for key in all_scenes:
        _, scene = parse_scene_key(key)
        scene_name_to_keys.setdefault(scene, []).append(key)

    resolved: List[str] = []
    for item in requested:
        if "/" in item:
            if item not in all_scene_set:
                raise ValueError(
                    f"手工 val_scenes 含未知 scene: {item}，可选值示例: {all_scenes[:5]}"
                )
            resolved.append(item)
            continue

        candidates = scene_name_to_keys.get(item, [])
        if not candidates:
            raise ValueError(f"手工 val_scenes 含未知 scene 名称: {item}")
        if len(candidates) > 1:
            raise ValueError(
                f"scene 名称 `{item}` 在多个 task 下重复，需使用 task/scene 全名。"
            )
        resolved.append(candidates[0])

    # 去重后按 spec 排序规则返回，保证稳定性
    dedup = sorted(set(resolved), key=scene_sort_key)
    return dedup


def split_scenes(
    all_scenes: Iterable[str],
    val_ratio: float,
    manual_val_scenes: Optional[List[str]] = None,
) -> SplitPlan:
    all_scene_sorted = sorted(set(all_scenes), key=scene_sort_key)
    n_scene = len(all_scene_sorted)
    expected_auto = compute_auto_val_scene_count(n_scene, val_ratio) if n_scene >= 2 else None

    if manual_val_scenes is not None and len(manual_val_scenes) > 0:
        mode = "manual"
        val_scenes = resolve_manual_val_scenes(manual_val_scenes, all_scene_sorted)
        val_scene_set = set(val_scenes)
        train_scenes = [s for s in all_scene_sorted if s not in val_scene_set]
        return SplitPlan(
            mode=mode,
            all_scenes=all_scene_sorted,
            train_scenes=train_scenes,
            val_scenes=val_scenes,
            n_scene=n_scene,
            requested_val_ratio=val_ratio,
            computed_val_scene_count=len(val_scenes),
            expected_auto_val_scene_count=expected_auto,
            auto_split_failed=False,
            auto_failure_reason=None,
        )

    # 自动划分
    mode = "auto"
    if n_scene < 2:
        return SplitPlan(
            mode=mode,
            all_scenes=all_scene_sorted,
            train_scenes=all_scene_sorted,
            val_scenes=[],
            n_scene=n_scene,
            requested_val_ratio=val_ratio,
            computed_val_scene_count=0,
            expected_auto_val_scene_count=0,
            auto_split_failed=True,
            auto_failure_reason="N_scene < 2，触发 R-A3 条件（自动划分失败）",
        )

    val_scene_count = compute_auto_val_scene_count(n_scene, val_ratio)
    val_scenes = all_scene_sorted[-val_scene_count:]
    train_scenes = all_scene_sorted[:-val_scene_count]

    return SplitPlan(
        mode=mode,
        all_scenes=all_scene_sorted,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        n_scene=n_scene,
        requested_val_ratio=val_ratio,
        computed_val_scene_count=val_scene_count,
        expected_auto_val_scene_count=val_scene_count,
        auto_split_failed=False,
        auto_failure_reason=None,
    )


def sample_indices_with_first_last(frame_count: int, sample_stride: int) -> List[int]:
    if frame_count <= 0:
        return []
    if sample_stride <= 0:
        raise ValueError(f"sample_stride 必须为正整数，收到: {sample_stride}")

    sampled = list(range(0, frame_count, sample_stride))
    sampled.append(0)
    sampled.append(frame_count - 1)

    # 保序去重
    dedup: List[int] = []
    seen = set()
    for idx in sampled:
        if idx not in seen:
            dedup.append(idx)
            seen.add(idx)
    return dedup


def build_mask_index(mask_files: Sequence[Path]) -> Tuple[Dict[str, Path], int]:
    mask_by_frame: Dict[str, Path] = {}
    duplicate = 0
    for path in sorted(mask_files, key=lambda p: natural_sort_tokens(p.stem)):
        frame = path.stem
        if frame in mask_by_frame:
            duplicate += 1
            continue
        mask_by_frame[frame] = path
    return mask_by_frame, duplicate


def compute_tight_bbox_xywh(binary_mask: Any) -> List[float]:
    np = _require_numpy()

    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("空 mask 无法计算 bbox")

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return [float(x_min), float(y_min), float(width), float(height)]


def encode_compressed_rle(binary_mask: Any) -> Dict[str, Any]:
    np = _require_numpy()
    mask_util = _require_pycocotools_mask_util()

    # pycocotools 需要 Fortran order
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("utf-8")
    return {"counts": rle["counts"], "size": list(rle["size"])}


def validate_annotation(ann: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    for field in [
        "id",
        "image_id",
        "category_id",
        "bbox",
        "segmentation",
        "area",
        "iscrowd",
    ]:
        if field not in ann:
            reasons.append(f"missing:{field}")

    bbox = ann.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        reasons.append("invalid:bbox_shape")
    else:
        x, y, w, h = bbox
        if not all(math.isfinite(float(v)) for v in [x, y, w, h]):
            reasons.append("invalid:bbox_non_finite")
        if w <= 0 or h <= 0:
            reasons.append("invalid:bbox_non_positive_wh")

    seg = ann.get("segmentation")
    if not isinstance(seg, dict):
        reasons.append("invalid:segmentation_not_dict")
    else:
        counts = seg.get("counts")
        size = seg.get("size")
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        if not isinstance(counts, str) or len(counts) == 0:
            reasons.append("invalid:segmentation_counts")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or not all(isinstance(v, int) and v > 0 for v in size)
        ):
            reasons.append("invalid:segmentation_size")

    area = ann.get("area")
    if not isinstance(area, (int, float)) or area <= 0:
        reasons.append("invalid:area")

    iscrowd = ann.get("iscrowd")
    if iscrowd != 0:
        reasons.append("invalid:iscrowd")

    return (len(reasons) == 0, reasons)


def make_empty_coco(category_name: str) -> Dict[str, Any]:
    return {
        "info": {
            "description": "SAM3 finetune stage-A dataset",
            "version": "1.0.0",
            "date_created": utc_now_iso(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": category_name}],
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_markdown_summary(path: Path, summary: Dict[str, Any]) -> None:
    kpis = summary["kpis"]
    risks = summary["risk_evaluation"]

    lines = [
        "# 阶段A数据集构建报告",
        "",
        f"- 生成时间(UTC): `{summary['generated_at_utc']}`",
        f"- 构建状态: `{summary['build_status']}`",
        f"- 硬门槛(A1~A5): `{summary['hard_gate']['pass']}`",
        "",
        "## 输入参数",
        "",
        "```json",
        json.dumps(summary["input_config"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Split清单",
        "",
        f"- mode: `{summary['split']['mode']}`",
        f"- N_scene: `{summary['split']['n_scene']}`",
        f"- train_scene_count: `{summary['split']['train_scene_count']}`",
        f"- val_scene_count: `{summary['split']['val_scene_count']}`",
        f"- train_scenes: `{summary['split']['train_scenes']}`",
        f"- val_scenes: `{summary['split']['val_scenes']}`",
        "",
        "## KPI",
        "",
        "| KPI | pass | value | threshold |",
        "|---|---:|---:|---|",
    ]

    for kpi_id in [
        "KPI-A1",
        "KPI-A2",
        "KPI-A3",
        "KPI-A4",
        "KPI-A5",
        "KPI-A6",
        "KPI-A7",
        "KPI-A8",
        "KPI-A9",
    ]:
        item = kpis[kpi_id]
        lines.append(
            f"| {kpi_id} | {item['pass']} | {item['value']} | {item['threshold']} |"
        )

    lines += [
        "",
        "## 风险触发判定",
        "",
        "| 风险ID | triggered | condition |",
        "|---|---:|---|",
    ]

    for risk_id in ["R-A1", "R-A2", "R-A3", "R-A4", "R-A5", "R-A6"]:
        item = risks[risk_id]
        lines.append(f"| {risk_id} | {item['triggered']} | {item['condition']} |")

    lines += [
        "",
        "## 统计明细",
        "",
        "```json",
        json.dumps(summary["counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 备注",
        "",
        "- R-A1/R-A2/R-A3 触发口径依赖“同参数连续2次构建”窗口，本次报告会引用最近一次同参数历史。",
        "- 若无历史记录，则这三项仅能给出“本次是否超阈值”，不会触发连续2次告警。",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def discover_tasks(config: BuildConfig) -> List[str]:
    if config.task_list is not None and len(config.task_list) > 0:
        return sorted(set(config.task_list))

    tasks: List[str] = []
    if not config.data_base.exists():
        return tasks

    for child in config.data_base.iterdir():
        if not child.is_dir():
            continue

        train_root = child / "train"
        if train_root.is_dir():
            tasks.append(child.name)
            continue

        # 兼容新布局：{data_base}/{task}/{scene}/...
        has_scene_dirs = any(
            p.is_dir() and p.name.startswith("scene")
            for p in child.iterdir()
        )
        if has_scene_dirs:
            tasks.append(child.name)

    return sorted(set(tasks))


def discover_scene_inventory(
    config: BuildConfig,
    tasks: List[str],
    logger: logging.Logger,
) -> Dict[str, SceneInventory]:
    inventory: Dict[str, SceneInventory] = {}

    for task in tasks:
        task_root = config.data_base / task
        if not task_root.is_dir():
            log_event(
                logger,
                "task_skipped_not_found",
                task=task,
                task_root=str(task_root),
            )
            continue

        train_root = task_root / "train"
        train_scene_dirs = [p for p in train_root.iterdir() if p.is_dir()] if train_root.is_dir() else []

        if train_scene_dirs:
            scene_root = train_root
            scene_root_source = "train_subdir"
            scene_dirs = train_scene_dirs
        else:
            # 兼容新布局：{data_base}/{task}/{scene}/...
            scene_root = task_root
            scene_root_source = "task_root"
            scene_dirs = [p for p in task_root.iterdir() if p.is_dir() and p.name != "train"]

        log_event(
            logger,
            "task_scene_root_resolved",
            task=task,
            scene_root=str(scene_root),
            scene_root_source=scene_root_source,
            scene_dir_count=len(scene_dirs),
        )

        scene_dirs = sorted(scene_dirs, key=lambda p: natural_sort_tokens(p.name))

        for scene_dir in scene_dirs:
            scene = scene_dir.name

            format_kwargs = {
                "data_base": config.data_base.as_posix(),
                "export_base": config.export_base.as_posix(),
                "task": task,
                "scene": scene,
            }

            primary_image_pattern = config.image_glob_template.format(**format_kwargs)
            image_pattern_templates = [config.image_glob_template]
            if "{task}/train/{scene}" in config.image_glob_template:
                image_pattern_templates.append(
                    config.image_glob_template.replace("{task}/train/{scene}", "{task}/{scene}")
                )
            elif "{task}/{scene}" in config.image_glob_template:
                image_pattern_templates.append(
                    config.image_glob_template.replace("{task}/{scene}", "{task}/train/{scene}")
                )

            image_files: List[Path] = []
            matched_image_pattern: Optional[str] = None
            seen_patterns = set()
            for image_template in image_pattern_templates:
                image_pattern = image_template.format(**format_kwargs)
                if image_pattern in seen_patterns:
                    continue
                seen_patterns.add(image_pattern)

                candidate_files = [Path(p) for p in glob.glob(image_pattern)]
                candidate_files = sorted(
                    candidate_files,
                    key=lambda p: natural_sort_tokens(p.stem),
                )
                if candidate_files:
                    image_files = candidate_files
                    matched_image_pattern = image_pattern
                    break

            if not image_files:
                continue

            if matched_image_pattern is not None and matched_image_pattern != primary_image_pattern:
                log_event(
                    logger,
                    "image_glob_template_fallback_used",
                    task=task,
                    scene=scene,
                    primary_pattern=primary_image_pattern,
                    matched_pattern=matched_image_pattern,
                )

            mask_pattern = config.mask_glob_template.format(**format_kwargs)
            mask_files = [Path(p) for p in glob.glob(mask_pattern)]
            mask_by_frame, dup_count = build_mask_index(mask_files)

            scene_key = f"{task}/{scene}"
            inventory[scene_key] = SceneInventory(
                scene_key=scene_key,
                task=task,
                scene=scene,
                image_files=image_files,
                mask_by_frame=mask_by_frame,
                duplicate_mask_frames=dup_count,
            )

    return inventory


def flatten_kpi_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def load_history(history_path: Path) -> List[Dict[str, Any]]:
    if not history_path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                # 容忍历史脏行，不中断当前构建
                continue
    return rows


def append_history(history_path: Path, record: Dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_previous_same_baseline(
    history_rows: List[Dict[str, Any]],
    baseline_signature: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    for row in reversed(history_rows):
        if row.get("baseline_signature") == baseline_signature:
            return row
    return None


def evaluate_risks(
    current_obs: Dict[str, Any],
    prev_obs: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    def prev_value(key: str, default: Any = None) -> Any:
        if prev_obs is None:
            return default
        return prev_obs.get(key, default)

    ra1_current = current_obs["kpi_a7_no_mask_ratio"] > 0.20
    ra1_prev = bool(prev_value("kpi_a7_no_mask_ratio", 0.0) > 0.20)
    ra1_trigger = bool(ra1_current and prev_obs is not None and ra1_prev)

    ra2_current = current_obs["kpi_a8_empty_mask_ratio"] > 0.30
    ra2_prev = bool(prev_value("kpi_a8_empty_mask_ratio", 0.0) > 0.30)
    ra2_trigger = bool(ra2_current and prev_obs is not None and ra2_prev)

    ra3_current = not current_obs["kpi_a9_pass"]
    ra3_prev = bool(prev_obs is not None and not bool(prev_value("kpi_a9_pass", True)))
    ra3_trigger = bool(ra3_current and prev_obs is not None and ra3_prev)

    ra4_trigger = bool(
        (current_obs["kpi_a4_completeness_ratio"] < 1.0)
        or (current_obs["invalid_segmentation_count"] > 0)
    )

    ra5_trigger = False
    if prev_obs is not None:
        ra5_trigger = bool(
            current_obs["train_scene_count"] != prev_obs.get("train_scene_count")
            or current_obs["val_scene_count"] != prev_obs.get("val_scene_count")
        )

    ra6_trigger = bool(current_obs["write_failures"] > 0 and not current_obs["kpi_a1_pass"])

    return {
        "R-A1": {
            "triggered": ra1_trigger,
            "condition": "KPI-A7 > 20% 且连续2次同参数构建",
            "current_over_threshold": ra1_current,
            "previous_over_threshold": ra1_prev if prev_obs is not None else None,
        },
        "R-A2": {
            "triggered": ra2_trigger,
            "condition": "KPI-A8 > 30% 且连续2次同参数构建",
            "current_over_threshold": ra2_current,
            "previous_over_threshold": ra2_prev if prev_obs is not None else None,
        },
        "R-A3": {
            "triggered": ra3_trigger,
            "condition": "KPI-A9 不通过 且连续2次同参数构建",
            "current_failed": ra3_current,
            "previous_failed": ra3_prev if prev_obs is not None else None,
        },
        "R-A4": {
            "triggered": ra4_trigger,
            "condition": "当次构建 KPI-A4 < 100% 或 segmentation 非法",
            "kpi_a4_completeness_ratio": current_obs["kpi_a4_completeness_ratio"],
            "invalid_segmentation_count": current_obs["invalid_segmentation_count"],
        },
        "R-A5": {
            "triggered": ra5_trigger,
            "condition": "同参数重复构建，train/val scene_count 不一致",
            "current_train_scene_count": current_obs["train_scene_count"],
            "previous_train_scene_count": prev_value("train_scene_count"),
            "current_val_scene_count": current_obs["val_scene_count"],
            "previous_val_scene_count": prev_value("val_scene_count"),
        },
        "R-A6": {
            "triggered": ra6_trigger,
            "condition": "写出失败且导致 KPI-A1 不满足",
            "write_failures": current_obs["write_failures"],
            "kpi_a1_pass": current_obs["kpi_a1_pass"],
        },
    }


def build_config(args: argparse.Namespace) -> BuildConfig:
    raw: Dict[str, Any] = {}
    if args.config is not None:
        raw = load_config_file(Path(args.config))

    cli_task_list = parse_csv_list(args.task_list)
    cli_val_scenes = parse_csv_list(args.val_scenes)

    data_base = args.data_base or pick_config_value(
        raw,
        [
            ("inputs", "data_base"),
            ("data_base",),
        ],
    )
    export_base = args.export_base or pick_config_value(
        raw,
        [
            ("inputs", "export_base"),
            ("export_base",),
        ],
    )
    output_dir = args.output_dir or pick_config_value(
        raw,
        [
            ("inputs", "output_dir"),
            ("output_dir",),
        ],
    )

    if not data_base or not export_base or not output_dir:
        raise ValueError(
            "必须提供 data_base/export_base/output_dir（可来自 CLI 或 --config 文件）"
        )

    task_list = (
        cli_task_list
        if cli_task_list is not None
        else ensure_list(
            pick_config_value(
                raw,
                [
                    ("inputs", "task_list"),
                    ("task_list",),
                ],
            )
        )
    )

    sample_stride = args.sample_stride
    if sample_stride is None:
        sample_stride = pick_config_value(
            raw,
            [
                ("sampling", "sample_stride"),
                ("sample_stride",),
            ],
            default=30,
        )

    val_ratio = args.val_ratio
    if val_ratio is None:
        val_ratio = pick_config_value(
            raw,
            [
                ("split", "val_ratio"),
                ("val_ratio",),
            ],
            default=0.2,
        )

    val_scenes = (
        cli_val_scenes
        if cli_val_scenes is not None
        else ensure_list(
            pick_config_value(
                raw,
                [
                    ("split", "val_scenes"),
                    ("val_scenes",),
                ],
            )
        )
    )

    category_name = args.category_name or pick_config_value(
        raw,
        [
            ("category", "name"),
            ("category_name",),
        ],
        default=DEFAULT_CATEGORY_NAME,
    )

    image_glob_template = args.image_glob_template or pick_config_value(
        raw,
        [
            ("paths", "image_glob_template"),
            ("image_glob_template",),
        ],
        default=DEFAULT_IMAGE_GLOB_TEMPLATE,
    )

    mask_glob_template = args.mask_glob_template or pick_config_value(
        raw,
        [
            ("paths", "mask_glob_template"),
            ("mask_glob_template",),
        ],
        default=DEFAULT_MASK_GLOB_TEMPLATE,
    )

    history_file_relpath = pick_config_value(
        raw,
        [
            ("reports", "history_file"),
            ("history_file",),
        ],
        default="reports/build_history.jsonl",
    )

    log_level = args.log_level or pick_config_value(
        raw,
        [("log_level",)],
        default="INFO",
    )

    sample_stride = int(sample_stride)
    val_ratio = float(val_ratio)

    if sample_stride <= 0:
        raise ValueError(f"sample_stride 必须 > 0，收到: {sample_stride}")
    if not (0.0 < val_ratio <= 1.0):
        raise ValueError(f"val_ratio 必须在 (0, 1]，收到: {val_ratio}")
    if not str(category_name).strip():
        raise ValueError("category_name 不能为空")

    val_scenes = val_scenes if val_scenes and len(val_scenes) > 0 else None
    task_list = task_list if task_list and len(task_list) > 0 else None

    return BuildConfig(
        data_base=Path(data_base),
        export_base=Path(export_base),
        output_dir=Path(output_dir),
        task_list=task_list,
        sample_stride=sample_stride,
        val_ratio=val_ratio,
        val_scenes=val_scenes,
        category_name=str(category_name),
        image_glob_template=str(image_glob_template),
        mask_glob_template=str(mask_glob_template),
        history_file_relpath=str(history_file_relpath),
        log_level=str(log_level),
    )


def build_dataset(config: BuildConfig, logger: logging.Logger) -> Dict[str, Any]:
    np = _require_numpy()
    Image = _require_pillow_image_class()

    if not config.data_base.exists():
        raise FileNotFoundError(f"data_base 不存在: {config.data_base}")
    if not config.export_base.exists():
        raise FileNotFoundError(f"export_base 不存在: {config.export_base}")

    output_dir = config.output_dir
    images_output_root = output_dir / "images"
    train_json_path = output_dir / "train" / "annotations.json"
    val_json_path = output_dir / "val" / "annotations.json"
    reports_dir = output_dir / "reports"
    split_manifest_path = reports_dir / "split_manifest.json"
    summary_json_path = reports_dir / "dataset_summary.json"
    summary_md_path = reports_dir / "dataset_summary.md"
    history_path = output_dir / config.history_file_relpath

    tasks = discover_tasks(config)
    if not tasks:
        raise RuntimeError("未发现可用任务(task)。请检查 data_base 或 task_list 配置。")

    log_event(logger, "task_discovered", tasks=tasks, task_count=len(tasks))

    scene_inventory = discover_scene_inventory(config, tasks, logger)
    if not scene_inventory:
        raise RuntimeError("未发现任何可用 scene（包含图像帧）。请检查路径模板与输入目录。")

    all_scene_keys = sorted(scene_inventory.keys(), key=scene_sort_key)
    split_plan = split_scenes(all_scene_keys, config.val_ratio, config.val_scenes)

    log_event(
        logger,
        "split_planned",
        mode=split_plan.mode,
        n_scene=split_plan.n_scene,
        train_scene_count=len(split_plan.train_scenes),
        val_scene_count=len(split_plan.val_scenes),
        auto_split_failed=split_plan.auto_split_failed,
        auto_failure_reason=split_plan.auto_failure_reason,
    )

    split_manifest = {
        "generated_at_utc": utc_now_iso(),
        "mode": split_plan.mode,
        "n_scene": split_plan.n_scene,
        "val_ratio": split_plan.requested_val_ratio,
        "all_scenes": split_plan.all_scenes,
        "train_scenes": split_plan.train_scenes,
        "val_scenes": split_plan.val_scenes,
        "computed_val_scene_count": split_plan.computed_val_scene_count,
        "expected_auto_val_scene_count": split_plan.expected_auto_val_scene_count,
        "auto_split_failed": split_plan.auto_split_failed,
        "auto_failure_reason": split_plan.auto_failure_reason,
        "sort_rule": {
            "task": "lexicographic asc",
            "scene": "natural asc within task",
            "stable_tie_breaker": "full task/scene",
            "auto_pick": "tail scenes",
        },
    }
    write_json(split_manifest_path, split_manifest)

    coco_train = make_empty_coco(config.category_name)
    coco_val = make_empty_coco(config.category_name)

    scene_to_split = {scene: "train" for scene in split_plan.train_scenes}
    scene_to_split.update({scene: "val" for scene in split_plan.val_scenes})

    counters: Dict[str, Any] = {
        "scene_count_total": len(all_scene_keys),
        "frame_count_total": 0,
        "sampled_frame_total": 0,
        "valid_sample_total": 0,
        "skip_no_mask_total": 0,
        "skip_empty_mask_total": 0,
        "skip_mask_size_mismatch_total": 0,
        "duplicate_mask_frame_total": 0,
        "write_failures": 0,
        "annotation_invalid_total": 0,
        "annotation_invalid_reasons": {},
        "split": {
            "train": {
                "images": 0,
                "annotations": 0,
                "sampled_frames": 0,
                "valid_samples": 0,
                "skip_no_mask": 0,
                "skip_empty_mask": 0,
                "skip_mask_size_mismatch": 0,
            },
            "val": {
                "images": 0,
                "annotations": 0,
                "sampled_frames": 0,
                "valid_samples": 0,
                "skip_no_mask": 0,
                "skip_empty_mask": 0,
                "skip_mask_size_mismatch": 0,
            },
        },
        "scene_stats": {},
    }

    next_image_id = 1
    next_annotation_id = 1

    for scene_key in all_scene_keys:
        inv = scene_inventory[scene_key]
        split_name = scene_to_split.get(scene_key)
        if split_name is None:
            # 正常不应发生，防御性处理
            continue

        counters["duplicate_mask_frame_total"] += inv.duplicate_mask_frames

        frame_count = len(inv.image_files)
        sampled_indices = sample_indices_with_first_last(frame_count, config.sample_stride)
        sampled_images = [inv.image_files[i] for i in sampled_indices]

        scene_stat = {
            "task": inv.task,
            "scene": inv.scene,
            "split": split_name,
            "frame_count": frame_count,
            "sampled_frames": len(sampled_images),
            "valid_samples": 0,
            "skip_no_mask": 0,
            "skip_empty_mask": 0,
            "skip_mask_size_mismatch": 0,
            "duplicate_mask_frames": inv.duplicate_mask_frames,
        }

        counters["frame_count_total"] += frame_count
        counters["sampled_frame_total"] += len(sampled_images)
        counters["split"][split_name]["sampled_frames"] += len(sampled_images)

        for image_path in sampled_images:
            frame = image_path.stem
            mask_path = inv.mask_by_frame.get(frame)
            if mask_path is None:
                counters["skip_no_mask_total"] += 1
                counters["split"][split_name]["skip_no_mask"] += 1
                scene_stat["skip_no_mask"] += 1
                continue

            try:
                with Image.open(mask_path) as mask_img:
                    mask_array = np.array(mask_img)
            except Exception:
                counters["skip_no_mask_total"] += 1
                counters["split"][split_name]["skip_no_mask"] += 1
                scene_stat["skip_no_mask"] += 1
                continue

            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]
            binary = (mask_array > 0).astype(np.uint8)

            if int(binary.sum()) == 0:
                counters["skip_empty_mask_total"] += 1
                counters["split"][split_name]["skip_empty_mask"] += 1
                scene_stat["skip_empty_mask"] += 1
                continue

            with Image.open(image_path) as image_obj:
                width, height = image_obj.size

            if binary.shape[0] != height or binary.shape[1] != width:
                counters["skip_mask_size_mismatch_total"] += 1
                counters["split"][split_name]["skip_mask_size_mismatch"] += 1
                scene_stat["skip_mask_size_mismatch"] += 1
                continue

            bbox = compute_tight_bbox_xywh(binary)
            segmentation = encode_compressed_rle(binary)
            area = int(binary.sum())

            image_relpath = Path(inv.task) / inv.scene / image_path.name
            output_image_path = images_output_root / image_relpath
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(image_path, output_image_path)
            except Exception as exc:
                counters["write_failures"] += 1
                log_event(
                    logger,
                    "image_copy_failed",
                    scene_key=scene_key,
                    image=str(image_path),
                    target=str(output_image_path),
                    error=str(exc),
                )
                continue

            image_uid = f"{inv.task}__{inv.scene}__{frame}"
            image_entry = {
                "id": next_image_id,
                "file_name": image_relpath.as_posix(),
                "width": width,
                "height": height,
                "source_uid": image_uid,
                "task": inv.task,
                "scene": inv.scene,
                "frame": frame,
            }
            annotation_entry = {
                "id": next_annotation_id,
                "image_id": next_image_id,
                "category_id": 1,
                "bbox": bbox,
                "segmentation": segmentation,
                "area": area,
                "iscrowd": 0,
                "source_uid": image_uid,
            }

            target_coco = coco_train if split_name == "train" else coco_val
            target_coco["images"].append(image_entry)
            target_coco["annotations"].append(annotation_entry)

            next_image_id += 1
            next_annotation_id += 1

            counters["valid_sample_total"] += 1
            counters["split"][split_name]["valid_samples"] += 1
            scene_stat["valid_samples"] += 1

        counters["scene_stats"][scene_key] = scene_stat

    counters["split"]["train"]["images"] = len(coco_train["images"])
    counters["split"]["train"]["annotations"] = len(coco_train["annotations"])
    counters["split"]["val"]["images"] = len(coco_val["images"])
    counters["split"]["val"]["annotations"] = len(coco_val["annotations"])

    # KPI A4 完整性检查
    all_annotations = coco_train["annotations"] + coco_val["annotations"]
    ann_valid_count = 0
    invalid_reason_counter: Dict[str, int] = {}
    for ann in all_annotations:
        valid, reasons = validate_annotation(ann)
        if valid:
            ann_valid_count += 1
            continue
        for reason in reasons:
            invalid_reason_counter[reason] = invalid_reason_counter.get(reason, 0) + 1

    counters["annotation_invalid_total"] = len(all_annotations) - ann_valid_count
    counters["annotation_invalid_reasons"] = invalid_reason_counter

    # 写出 COCO JSON
    write_json(train_json_path, coco_train)
    write_json(val_json_path, coco_val)

    train_images = len(coco_train["images"])
    val_images = len(coco_val["images"])
    train_anns = len(coco_train["annotations"])
    val_anns = len(coco_val["annotations"])

    sampled_total = int(counters["sampled_frame_total"])
    valid_total = int(counters["valid_sample_total"])
    no_mask_total = int(counters["skip_no_mask_total"])
    empty_mask_total = int(counters["skip_empty_mask_total"])

    effective_sampling_rate = (valid_total / sampled_total) if sampled_total > 0 else 0.0
    no_mask_ratio = (no_mask_total / sampled_total) if sampled_total > 0 else 0.0
    empty_mask_ratio = (empty_mask_total / sampled_total) if sampled_total > 0 else 0.0

    # KPI A1~A5
    kpi_a1_pass = bool(train_images > 0 and val_images > 0)
    kpi_a2_pass = bool(train_images == train_anns and val_images == val_anns)
    scene_overlap = sorted(set(split_plan.train_scenes).intersection(set(split_plan.val_scenes)))
    kpi_a3_pass = len(scene_overlap) == 0

    ann_total = len(all_annotations)
    kpi_a4_ratio = (ann_valid_count / ann_total) if ann_total > 0 else 0.0
    kpi_a4_pass = bool(kpi_a4_ratio == 1.0)

    categories_ok = (
        len(coco_train["categories"]) == 1
        and len(coco_val["categories"]) == 1
        and coco_train["categories"][0].get("id") == 1
        and coco_val["categories"][0].get("id") == 1
        and coco_train["categories"][0].get("name") == config.category_name
        and coco_val["categories"][0].get("name") == config.category_name
    )
    kpi_a5_pass = bool(categories_ok)

    # KPI A9
    n_scene = split_plan.n_scene
    train_scene_count = len(split_plan.train_scenes)
    val_scene_count = len(split_plan.val_scenes)

    all_tasks = {parse_scene_key(s)[0] for s in split_plan.all_scenes}
    val_tasks = {parse_scene_key(s)[0] for s in split_plan.val_scenes}
    val_task_coverage_rate = (len(val_tasks) / len(all_tasks)) if len(all_tasks) > 0 else 0.0

    a9_auto_count_ok = True
    expected_auto_count = None
    if split_plan.mode == "auto":
        if n_scene >= 2:
            expected_auto_count = compute_auto_val_scene_count(n_scene, config.val_ratio)
            a9_auto_count_ok = val_scene_count == expected_auto_count
        else:
            expected_auto_count = 0
            a9_auto_count_ok = False

    if n_scene < 2:
        kpi_a9_pass = False
    else:
        kpi_a9_pass = bool(
            train_scene_count >= 1
            and val_scene_count >= 1
            and val_task_coverage_rate >= max(0.2, config.val_ratio)
            and a9_auto_count_ok
        )

    kpis: Dict[str, Dict[str, Any]] = {
        "KPI-A1": {
            "name": "split 非空",
            "pass": kpi_a1_pass,
            "value": {"train_images": train_images, "val_images": val_images},
            "threshold": "train_images>0 且 val_images>0",
        },
        "KPI-A2": {
            "name": "样本一致性",
            "pass": kpi_a2_pass,
            "value": {
                "train_images": train_images,
                "train_annotations": train_anns,
                "val_images": val_images,
                "val_annotations": val_anns,
            },
            "threshold": "每个 split 中 len(images)==len(annotations)",
        },
        "KPI-A3": {
            "name": "场景隔离",
            "pass": kpi_a3_pass,
            "value": {"scene_overlap": scene_overlap},
            "threshold": "train_scenes ∩ val_scenes == 空集",
        },
        "KPI-A4": {
            "name": "标注完整性",
            "pass": kpi_a4_pass,
            "value": {
                "annotation_total": ann_total,
                "annotation_valid": ann_valid_count,
                "completeness_ratio": round(kpi_a4_ratio, 6),
                "invalid_reasons": invalid_reason_counter,
            },
            "threshold": "每条 annotation 具备 bbox/segmentation/category_id/image_id 等必填字段",
        },
        "KPI-A5": {
            "name": "类别一致性",
            "pass": kpi_a5_pass,
            "value": {
                "train_categories": coco_train["categories"],
                "val_categories": coco_val["categories"],
            },
            "threshold": "categories 仅1类且名称为目标类名",
        },
        "KPI-A6": {
            "name": "有效采样率",
            "pass": bool(effective_sampling_rate >= 0.60),
            "value": round(effective_sampling_rate, 6),
            "threshold": "建议 >= 0.60",
        },
        "KPI-A7": {
            "name": "no-mask 比例",
            "pass": bool(no_mask_ratio <= 0.20),
            "value": round(no_mask_ratio, 6),
            "threshold": "建议 <= 0.20",
        },
        "KPI-A8": {
            "name": "empty-mask 比例",
            "pass": bool(empty_mask_ratio <= 0.30),
            "value": round(empty_mask_ratio, 6),
            "threshold": "建议 <= 0.30",
        },
        "KPI-A9": {
            "name": "scene 覆盖率",
            "pass": kpi_a9_pass,
            "value": {
                "N_scene": n_scene,
                "train_scene_count": train_scene_count,
                "val_scene_count": val_scene_count,
                "val_task_coverage_rate": round(val_task_coverage_rate, 6),
                "required_min_coverage": max(0.2, config.val_ratio),
                "split_mode": split_plan.mode,
                "auto_expected_val_scene_count": expected_auto_count,
                "auto_count_ok": a9_auto_count_ok,
                "n_scene_lt_2": n_scene < 2,
            },
            "threshold": (
                "train_scene_count>=1 且 val_scene_count>=1 且 "
                "val_task_coverage_rate>=max(0.2,val_ratio)，"
                "自动划分时 val_scene_count 满足边界公式"
            ),
        },
    }

    hard_gate_pass = bool(
        kpis["KPI-A1"]["pass"]
        and kpis["KPI-A2"]["pass"]
        and kpis["KPI-A3"]["pass"]
        and kpis["KPI-A4"]["pass"]
        and kpis["KPI-A5"]["pass"]
    )

    baseline_signature = {
        "task_list": tasks,
        "sample_stride": config.sample_stride,
        "val_ratio": config.val_ratio,
        "category_name": config.category_name,
    }

    current_obs = {
        "kpi_a1_pass": kpi_a1_pass,
        "kpi_a4_completeness_ratio": round(kpi_a4_ratio, 6),
        "kpi_a7_no_mask_ratio": round(no_mask_ratio, 6),
        "kpi_a8_empty_mask_ratio": round(empty_mask_ratio, 6),
        "kpi_a9_pass": kpi_a9_pass,
        "invalid_segmentation_count": counters["annotation_invalid_total"],
        "train_scene_count": train_scene_count,
        "val_scene_count": val_scene_count,
        "write_failures": counters["write_failures"],
    }

    history_rows = load_history(history_path)
    previous_record = find_previous_same_baseline(history_rows, baseline_signature)
    previous_obs = previous_record.get("observations") if previous_record else None
    risk_evaluation = evaluate_risks(current_obs=current_obs, prev_obs=previous_obs)

    history_record = {
        "generated_at_utc": utc_now_iso(),
        "baseline_signature": baseline_signature,
        "observations": current_obs,
        "hard_gate_pass": hard_gate_pass,
        "split_mode": split_plan.mode,
    }
    append_history(history_path, history_record)

    summary = {
        "generated_at_utc": utc_now_iso(),
        "build_status": "PASS" if hard_gate_pass else "FAIL",
        "hard_gate": {
            "pass": hard_gate_pass,
            "hard_kpi_ids": ["KPI-A1", "KPI-A2", "KPI-A3", "KPI-A4", "KPI-A5"],
            "failed": [k for k in ["KPI-A1", "KPI-A2", "KPI-A3", "KPI-A4", "KPI-A5"] if not kpis[k]["pass"]],
        },
        "input_config": {
            "data_base": str(config.data_base),
            "export_base": str(config.export_base),
            "output_dir": str(config.output_dir),
            "task_list": config.task_list,
            "resolved_tasks": tasks,
            "sample_stride": config.sample_stride,
            "val_ratio": config.val_ratio,
            "val_scenes": config.val_scenes,
            "category_name": config.category_name,
            "image_glob_template": config.image_glob_template,
            "mask_glob_template": config.mask_glob_template,
        },
        "split": {
            "mode": split_plan.mode,
            "n_scene": n_scene,
            "train_scene_count": train_scene_count,
            "val_scene_count": val_scene_count,
            "all_scenes": split_plan.all_scenes,
            "train_scenes": split_plan.train_scenes,
            "val_scenes": split_plan.val_scenes,
            "auto_split_failed": split_plan.auto_split_failed,
            "auto_failure_reason": split_plan.auto_failure_reason,
        },
        "counts": counters,
        "kpis": {
            k: {
                **v,
                "value": flatten_kpi_value(v["value"]),
            }
            for k, v in kpis.items()
        },
        "risk_evaluation": {
            **risk_evaluation,
            "history_window": {
                "has_previous_same_baseline": previous_record is not None,
                "previous_generated_at_utc": previous_record.get("generated_at_utc") if previous_record else None,
                "baseline_signature": baseline_signature,
            },
        },
        "artifacts": {
            "train_json": str(train_json_path),
            "val_json": str(val_json_path),
            "split_manifest": str(split_manifest_path),
            "summary_json": str(summary_json_path),
            "summary_md": str(summary_md_path),
            "history_file": str(history_path),
        },
    }

    write_json(summary_json_path, summary)
    write_markdown_summary(summary_md_path, summary)

    log_event(
        logger,
        "build_finished",
        build_status=summary["build_status"],
        hard_gate_pass=hard_gate_pass,
        train_images=train_images,
        val_images=val_images,
        sampled_total=sampled_total,
        valid_total=valid_total,
        summary_json=str(summary_json_path),
    )

    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段A数据集构建：原图+_ckpt_arm 掩码 -> COCO(train/val)+split+报告"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径，支持 .json/.yaml/.yml。CLI 参数会覆盖配置文件同名项。",
    )

    parser.add_argument("--data-base", type=str, default=None, help="原图根目录")
    parser.add_argument("--export-base", type=str, default=None, help="掩码根目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出根目录")

    parser.add_argument(
        "--task-list",
        type=str,
        default=None,
        help="任务列表（逗号分隔），例如 task_a,task_b；不传则自动扫描 data_base。",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help="采样步长，默认 30，且始终保留首尾帧。",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="自动划分 val 比例，默认 0.2。",
    )
    parser.add_argument(
        "--val-scenes",
        type=str,
        default=None,
        help=(
            "手工验证场景（逗号分隔）。支持 task/scene（推荐）或 scene（全局唯一时）。"
        ),
    )
    parser.add_argument(
        "--category-name",
        type=str,
        default=None,
        help=f"类别名，默认 `{DEFAULT_CATEGORY_NAME}`。",
    )

    parser.add_argument(
        "--image-glob-template",
        type=str,
        default=None,
        help=(
            "图像glob模板，支持占位符 {data_base}/{task}/{scene}/{export_base}。"
            f"默认: {DEFAULT_IMAGE_GLOB_TEMPLATE}"
        ),
    )
    parser.add_argument(
        "--mask-glob-template",
        type=str,
        default=None,
        help=(
            "掩码glob模板，支持占位符 {data_base}/{task}/{scene}/{export_base}。"
            f"默认: {DEFAULT_MASK_GLOB_TEMPLATE}"
        ),
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="日志级别（DEBUG/INFO/WARNING/ERROR），默认 INFO。",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        config = build_config(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ts": utc_now_iso(),
                    "event": "config_error",
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2

    logger = setup_logger(config.log_level)
    log_event(
        logger,
        "build_start",
        data_base=str(config.data_base),
        export_base=str(config.export_base),
        output_dir=str(config.output_dir),
        sample_stride=config.sample_stride,
        val_ratio=config.val_ratio,
        val_scenes=config.val_scenes,
        category_name=config.category_name,
    )

    try:
        summary = build_dataset(config=config, logger=logger)
    except Exception as exc:
        log_event(logger, "build_failed", error=str(exc))
        return 1

    log_event(
        logger,
        "final_summary",
        build_status=summary["build_status"],
        hard_gate_pass=summary["hard_gate"]["pass"],
        failed_hard_kpis=summary["hard_gate"]["failed"],
        summary_json=summary["artifacts"]["summary_json"],
    )

    return 0 if summary["hard_gate"]["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
