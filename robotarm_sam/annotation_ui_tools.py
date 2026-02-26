from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional
import json


AnnotationObjectSpecs = Dict[str, Dict[str, Any]]
AnnotationStore = Dict[str, Dict[int, List[Dict[str, int]]]]
PromptList = List[Dict[str, Any]]


ANNOTATION_PROMPT_KEYS = (
    "ARM_CABLE_INITIAL_PROMPTS",
    "GRIPPER_LEFT_KEYFRAME_PROMPTS",
    "GRIPPER_RIGHT_KEYFRAME_PROMPTS",
)

OPTIONAL_ANNOTATION_PROMPT_KEYS = (
    "ARM_CABLE_2_INITIAL_PROMPTS",
)

LEGACY_ARM_PROMPT_KEYS = (
    "ARM_LEFT_INITIAL_PROMPTS",
    "ARM_RIGHT_INITIAL_PROMPTS",
)


@dataclass
class AnnotationUIState:
    annotation_store: AnnotationStore
    object_specs: AnnotationObjectSpecs
    frame_min: int
    frame_max: int
    img_width: int
    img_height: int
    export_prompts: Dict[str, PromptList] = field(default_factory=dict)
    use_visual_annotation_export: bool = False


def build_annotation_object_specs(
    arm_cable_obj_id: int,
    gripper_left_obj_id: int,
    gripper_right_obj_id: int,
    arm_cable_obj_id_2: Optional[int] = None,
) -> AnnotationObjectSpecs:
    specs: AnnotationObjectSpecs = {
        "arm_cable": {
            "display": "机械臂+线缆",
            "obj_id": int(arm_cable_obj_id),
            "target": "ARM_CABLE_INITIAL_PROMPTS",
        },
        "gripper_left": {
            "display": "左夹爪",
            "obj_id": int(gripper_left_obj_id),
            "target": "GRIPPER_LEFT_KEYFRAME_PROMPTS",
        },
        "gripper_right": {
            "display": "右夹爪",
            "obj_id": int(gripper_right_obj_id),
            "target": "GRIPPER_RIGHT_KEYFRAME_PROMPTS",
        },
    }
    if arm_cable_obj_id_2 is not None:
        specs["arm_cable_2"] = {
            "display": "第二机械臂",
            "obj_id": int(arm_cable_obj_id_2),
            "target": "ARM_CABLE_2_INITIAL_PROMPTS",
        }
    return specs


def create_annotation_store(object_specs: Mapping[str, Mapping[str, Any]]) -> AnnotationStore:
    return {str(k): {} for k in object_specs.keys()}


def append_click(
    store: MutableMapping[str, Dict[int, List[Dict[str, int]]]],
    obj_key: str,
    frame_idx: int,
    x: int,
    y: int,
    label: int,
) -> None:
    frame_idx = int(frame_idx)
    store[obj_key].setdefault(frame_idx, []).append(
        {"x": int(x), "y": int(y), "label": int(label)}
    )



def seed_store_from_prompt_list(
    store: MutableMapping[str, Dict[int, List[Dict[str, int]]]],
    obj_key: str,
    prompt_list: Optional[List[Dict[str, Any]]],
    img_w: int,
    img_h: int,
) -> None:
    for entry in (prompt_list or []):
        frame_idx = int(entry["frame_index"])
        coord_type = entry.get("coord_type", "abs")
        points = entry.get("points", [])
        labels = entry.get("labels", [])
        for p, lb in zip(points, labels):
            if coord_type == "rel":
                x = int(round(float(p[0]) * max(img_w - 1, 1)))
                y = int(round(float(p[1]) * max(img_h - 1, 1)))
            else:
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
            x = max(0, min(int(img_w) - 1, x))
            y = max(0, min(int(img_h) - 1, y))
            append_click(store, obj_key, frame_idx, x, y, int(lb))


def seed_store_from_prompt_map(
    store: MutableMapping[str, Dict[int, List[Dict[str, int]]]],
    object_specs: Mapping[str, Mapping[str, Any]],
    prompt_map: Mapping[str, List[Dict[str, Any]]],
    img_w: int,
    img_h: int,
) -> None:
    for obj_key, spec in object_specs.items():
        target_name = str(spec["target"])
        seed_store_from_prompt_list(
            store=store,
            obj_key=str(obj_key),
            prompt_list=prompt_map.get(target_name, []),
            img_w=img_w,
            img_h=img_h,
        )


def prompt_list_from_store(
    store: Mapping[str, Mapping[int, List[Dict[str, int]]]],
    obj_key: str,
    obj_id: int,
) -> PromptList:
    prompt_list: PromptList = []
    for frame_idx in sorted(store[obj_key].keys()):
        clicks = store[obj_key][frame_idx]
        if len(clicks) == 0:
            continue
        prompt_list.append(
            {
                "frame_index": int(frame_idx),
                "obj_id": int(obj_id),
                "coord_type": "abs",
                "points": [[int(c["x"]), int(c["y"])] for c in clicks],
                "labels": [int(c["label"]) for c in clicks],
            }
        )
    return prompt_list


def store_to_export_prompts(
    annotation_store: Mapping[str, Mapping[int, List[Dict[str, int]]]],
    object_specs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, PromptList]:
    export_prompts: Dict[str, PromptList] = {}
    for obj_key, spec in object_specs.items():
        export_prompts[str(spec["target"])] = prompt_list_from_store(
            store=annotation_store,
            obj_key=str(obj_key),
            obj_id=int(spec["obj_id"]),
        )
    return export_prompts


def validate_export_prompt_map(prompt_map: Mapping[str, Any]) -> Dict[str, PromptList]:
    if not isinstance(prompt_map, Mapping):
        raise ValueError("[annotation] prompts 必须为包含对象提示的 dict")

    has_all_new_keys = all(k in prompt_map for k in ANNOTATION_PROMPT_KEYS)
    has_any_legacy_arm_keys = any(k in prompt_map for k in LEGACY_ARM_PROMPT_KEYS)

    normalized: Dict[str, PromptList] = {}

    if has_all_new_keys:
        for key in ANNOTATION_PROMPT_KEYS:
            prompt_list_raw = prompt_map[key]
            if not isinstance(prompt_list_raw, (list, tuple)):
                raise ValueError(f"[annotation] prompts[{key}] 必须为 list")
            normalized[key] = list(prompt_list_raw)

        if has_any_legacy_arm_keys:
            print(
                "[annotation][warn] 检测到旧字段 ARM_LEFT/ARM_RIGHT_*；"
                "当前已优先使用 ARM_CABLE_INITIAL_PROMPTS，旧字段将被忽略。"
            )

        for opt_key in OPTIONAL_ANNOTATION_PROMPT_KEYS:
            if opt_key in prompt_map:
                val = prompt_map[opt_key]
                if not isinstance(val, (list, tuple)):
                    raise ValueError(f"[annotation] prompts[{opt_key}] 必须为 list")
                normalized[opt_key] = list(val)
            else:
                normalized[opt_key] = []

        return normalized

    if has_any_legacy_arm_keys:
        missing_legacy = [k for k in LEGACY_ARM_PROMPT_KEYS if k not in prompt_map]
        if missing_legacy:
            raise ValueError(
                "[annotation] 检测到旧版 arm 字段但不完整，缺少: "
                + ", ".join(missing_legacy)
                + "；请补齐旧字段后自动迁移，或改用新字段 ARM_CABLE_INITIAL_PROMPTS。"
            )

        gripper_missing = [
            k
            for k in ("GRIPPER_LEFT_KEYFRAME_PROMPTS", "GRIPPER_RIGHT_KEYFRAME_PROMPTS")
            if k not in prompt_map
        ]
        if gripper_missing:
            raise ValueError(
                "[annotation] prompts 缺少必要 gripper 键: "
                + ", ".join(gripper_missing)
                + "；旧版兼容仅自动合并 arm 字段。"
            )

        arm_left_raw = prompt_map["ARM_LEFT_INITIAL_PROMPTS"]
        arm_right_raw = prompt_map["ARM_RIGHT_INITIAL_PROMPTS"]
        if not isinstance(arm_left_raw, (list, tuple)):
            raise ValueError("[annotation] prompts[ARM_LEFT_INITIAL_PROMPTS] 必须为 list")
        if not isinstance(arm_right_raw, (list, tuple)):
            raise ValueError("[annotation] prompts[ARM_RIGHT_INITIAL_PROMPTS] 必须为 list")

        print(
            "[annotation][warn] 检测到旧字段 ARM_LEFT_INITIAL_PROMPTS/ARM_RIGHT_INITIAL_PROMPTS；"
            "已自动合并为 ARM_CABLE_INITIAL_PROMPTS。"
        )

        normalized["ARM_CABLE_INITIAL_PROMPTS"] = list(arm_left_raw) + list(arm_right_raw)

        for key in ("GRIPPER_LEFT_KEYFRAME_PROMPTS", "GRIPPER_RIGHT_KEYFRAME_PROMPTS"):
            prompt_list_raw = prompt_map[key]
            if not isinstance(prompt_list_raw, (list, tuple)):
                raise ValueError(f"[annotation] prompts[{key}] 必须为 list")
            normalized[key] = list(prompt_list_raw)

        for opt_key in OPTIONAL_ANNOTATION_PROMPT_KEYS:
            if opt_key in prompt_map:
                val = prompt_map[opt_key]
                if not isinstance(val, (list, tuple)):
                    raise ValueError(f"[annotation] prompts[{opt_key}] 必须为 list")
                normalized[opt_key] = list(val)
            else:
                normalized[opt_key] = []

        return normalized

    missing = [k for k in ANNOTATION_PROMPT_KEYS if k not in prompt_map]
    raise ValueError(
        "[annotation] prompts 缺少必要键: "
        + ", ".join(missing)
        + f"；必须包含: {', '.join(ANNOTATION_PROMPT_KEYS)}"
    )


def _median(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0)


def summarize_gripper_store_spatial_side_consistency(
    annotation_store: Mapping[str, Mapping[int, List[Dict[str, int]]]],
    img_width: int,
    *,
    left_key: str = "gripper_left",
    right_key: str = "gripper_right",
    dominance_ratio: float = 0.6,
) -> Dict[str, Any]:
    img_w = max(float(img_width), 1.0)

    def _collect_side_stats(side_key: str, side_name: str) -> Dict[str, Any]:
        x_all: List[float] = []
        x_pos: List[float] = []
        entries = 0

        frame_map = annotation_store.get(side_key, {})
        for clicks in frame_map.values():
            if not isinstance(clicks, list):
                continue
            if len(clicks) > 0:
                entries += 1
            for c in clicks:
                if not isinstance(c, Mapping):
                    continue
                try:
                    x = float(c["x"])
                    lb = int(c.get("label", 1))
                except Exception:
                    continue
                x_all.append(x)
                if lb == 1:
                    x_pos.append(x)

        basis_x = list(x_pos) if len(x_pos) > 0 else list(x_all)
        basis = "positive" if len(x_pos) > 0 else "all"
        if len(basis_x) == 0:
            return {
                "side_name": side_name,
                "entries": int(entries),
                "all_points": 0,
                "pos_points": 0,
                "basis": basis,
                "basis_points": 0,
                "x_mean": None,
                "x_median": None,
                "left_ratio": None,
                "right_ratio": None,
                "dominant_side": "unknown",
            }

        half = img_w / 2.0
        left_count = sum(1 for x in basis_x if x < half)
        right_count = sum(1 for x in basis_x if x >= half)
        total = len(basis_x)

        dominant = "unknown"
        left_ratio = float(left_count / total)
        right_ratio = float(right_count / total)
        if left_ratio >= float(dominance_ratio):
            dominant = "left"
        elif right_ratio >= float(dominance_ratio):
            dominant = "right"

        return {
            "side_name": side_name,
            "entries": int(entries),
            "all_points": int(len(x_all)),
            "pos_points": int(len(x_pos)),
            "basis": basis,
            "basis_points": int(total),
            "x_mean": float(sum(basis_x) / total),
            "x_median": _median(basis_x),
            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            "dominant_side": dominant,
        }

    left_stats = _collect_side_stats(left_key, "left")
    right_stats = _collect_side_stats(right_key, "right")

    conflict_type = "none"
    warning = ""
    if left_stats["basis_points"] > 0 and right_stats["basis_points"] > 0:
        if left_stats["dominant_side"] == "right" and right_stats["dominant_side"] == "left":
            conflict_type = "likely_swapped"
            warning = (
                "left 点主要在右半区且 right 点主要在左半区，请复核左右对象选择。"
            )
    elif left_stats["basis_points"] > 0 and right_stats["basis_points"] == 0:
        if left_stats["dominant_side"] == "right":
            conflict_type = "left_points_mainly_right"
            warning = "仅 left 有点且主要在右半区，请复核 left/right 选择。"
    elif right_stats["basis_points"] > 0 and left_stats["basis_points"] == 0:
        if right_stats["dominant_side"] == "left":
            conflict_type = "right_points_mainly_left"
            warning = "仅 right 有点且主要在左半区，请复核 left/right 选择。"

    return {
        "left": left_stats,
        "right": right_stats,
        "dominance_ratio": float(dominance_ratio),
        "conflict_type": conflict_type,
        "warning": warning,
    }


def summarize_prompt_map(prompt_map: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    entries_per_object: Dict[str, int] = {}
    points_per_object: Dict[str, int] = {}
    frame_point_counts_per_object: Dict[str, Dict[int, int]] = {}
    unique_frames = set()

    all_summary_keys = list(ANNOTATION_PROMPT_KEYS) + [
        k for k in OPTIONAL_ANNOTATION_PROMPT_KEYS if k in prompt_map
    ]

    for key in all_summary_keys:
        prompt_list = prompt_map.get(key, [])
        if not isinstance(prompt_list, list):
            raise ValueError(f"[annotation] prompts[{key}] 必须为 list")

        entries_per_object[key] = len(prompt_list)
        point_count = 0
        frame_point_counts: Dict[int, int] = {}
        for idx, entry in enumerate(prompt_list):
            if not isinstance(entry, Mapping):
                raise ValueError(f"[annotation] prompts[{key}][{idx}] 必须为 dict")
            if "frame_index" not in entry:
                raise ValueError(f"[annotation] prompts[{key}][{idx}] 缺少 frame_index")

            try:
                frame_idx = int(entry["frame_index"])
                unique_frames.add(frame_idx)
            except Exception as e:
                raise ValueError(
                    f"[annotation] prompts[{key}][{idx}].frame_index 无法转换为整数"
                ) from e

            points = entry.get("points", [])
            if not isinstance(points, list):
                raise ValueError(f"[annotation] prompts[{key}][{idx}].points 必须为 list")
            point_count += len(points)
            frame_point_counts[frame_idx] = int(frame_point_counts.get(frame_idx, 0) + len(points))

        points_per_object[key] = point_count
        frame_point_counts_per_object[key] = {
            int(f): int(frame_point_counts[f]) for f in sorted(frame_point_counts.keys())
        }

    left_key = "GRIPPER_LEFT_KEYFRAME_PROMPTS"
    right_key = "GRIPPER_RIGHT_KEYFRAME_PROMPTS"
    left_counts = frame_point_counts_per_object.get(left_key, {})
    right_counts = frame_point_counts_per_object.get(right_key, {})
    gripper_frame_count_warnings: List[str] = []
    for frame_idx in sorted(set(left_counts.keys()) | set(right_counts.keys())):
        left_count = int(left_counts.get(frame_idx, 0))
        right_count = int(right_counts.get(frame_idx, 0))
        if left_count != right_count:
            gripper_frame_count_warnings.append(
                f"frame={frame_idx}: left_points={left_count}, right_points={right_count}"
            )

    return {
        "entries_per_object": entries_per_object,
        "points_per_object": points_per_object,
        "frame_point_counts_per_object": frame_point_counts_per_object,
        "gripper_frame_count_warnings": gripper_frame_count_warnings,
        "num_frames_with_prompts": len(unique_frames),
    }


def save_annotation_prompts_json(
    export_prompts: Mapping[str, Any],
    json_path: str,
    status_prefix: str = "[annotation]",
) -> Dict[str, Any]:
    prompts_to_save = validate_export_prompt_map(export_prompts)
    summary = summarize_prompt_map(prompts_to_save)

    payload: Dict[str, Any] = {
        "format": "sam3.annotation_prompts.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompts": prompts_to_save,
        "stats": summary,
    }

    out_path = Path(json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"{status_prefix} JSON 导出成功: {out_path}")
    print(f"{status_prefix} 对象条目统计: {summary['entries_per_object']}")
    print(f"{status_prefix} 对象点数统计: {summary['points_per_object']}")
    print(f"{status_prefix} 按对象逐帧点数统计: {summary['frame_point_counts_per_object']}")
    print(f"{status_prefix} 含标注帧数: {summary['num_frames_with_prompts']}")
    for warn in summary.get("gripper_frame_count_warnings", []):
        print(f"{status_prefix} [warn][gripper_frame_count] {warn}")

    return payload


def load_annotation_prompts_json(
    json_path: str,
    status_prefix: str = "[annotation]",
) -> Dict[str, PromptList]:
    in_path = Path(json_path)
    if not in_path.exists():
        raise FileNotFoundError(
            f"{status_prefix} JSON 文件不存在: {in_path}；已阻断流程，请先完成导出或修正路径"
        )

    try:
        with in_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{status_prefix} JSON 解析失败: {in_path}（{e.msg} at line {e.lineno}, col {e.colno}）"
        ) from e

    if not isinstance(payload, dict):
        raise ValueError(f"{status_prefix} JSON 顶层必须为 dict: {in_path}")

    if "prompts" in payload:
        prompts_raw = payload["prompts"]
    else:
        prompts_raw = payload

    created_at_utc = payload.get("created_at_utc") if isinstance(payload, dict) else None
    prompts = validate_export_prompt_map(prompts_raw)
    summary = summarize_prompt_map(prompts)

    print(f"{status_prefix} JSON 读取成功: {in_path}")
    print(f"{status_prefix} JSON created_at_utc: {created_at_utc}")
    print(f"{status_prefix} 对象条目统计: {summary['entries_per_object']}")
    print(f"{status_prefix} 对象点数统计: {summary['points_per_object']}")
    print(f"{status_prefix} 按对象逐帧点数统计: {summary['frame_point_counts_per_object']}")
    print(f"{status_prefix} 含标注帧数: {summary['num_frames_with_prompts']}")
    for warn in summary.get("gripper_frame_count_warnings", []):
        print(f"{status_prefix} [warn][gripper_frame_count] {warn}")

    return prompts


def create_annotation_ui(
    *,
    video_frames_for_vis: List[Any],
    total_frames: int,
    img_width: int,
    img_height: int,
    object_specs: AnnotationObjectSpecs,
    annotation_store: AnnotationStore,
    on_export: Optional[Callable[[Dict[str, PromptList]], None]] = None,
    auto_display: bool = True,
    status_prefix: str = "[annotation]",
    export_json_path: Optional[str] = None,
    save_json_on_export: bool = True,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "widget_ready": False,
        "widget_error": None,
        "controls": None,
        "state": None,
        "figure": None,
        "axis": None,
    }

    frame_min = 0
    frame_max = max(0, int(total_frames) - 1)
    for obj_key in object_specs.keys():
        annotation_store[str(obj_key)] = {}
    print(f"{status_prefix} 已清空历史标注缓存：本次会话从空白标注开始")

    state = AnnotationUIState(
        annotation_store=annotation_store,
        object_specs=dict(object_specs),
        frame_min=frame_min,
        frame_max=frame_max,
        img_width=int(img_width),
        img_height=int(img_height),
    )
    result["state"] = state

    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            raise RuntimeError("当前不在 IPython/Jupyter 环境")

        ip.run_line_magic("matplotlib", "widget")

        import ipywidgets as widgets
        from IPython.display import display
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as e:  # pragma: no cover - notebook runtime fallback path
        result["widget_error"] = e
        return result

    frame_input = widgets.IntText(
        value=0,
        description="Frame",
        layout=widgets.Layout(width="220px"),
    )

    object_options = [
        (
            f"{spec['display']} (obj_id={spec['obj_id']})",
            key,
        )
        for key, spec in object_specs.items()
    ]
    if len(object_options) == 0:
        raise ValueError("[annotation] object_specs 不能为空")

    object_dropdown = widgets.Dropdown(
        options=object_options,
        value=object_options[0][1],
        description="Object",
        layout=widgets.Layout(width="380px"),
    )

    label_toggle = widgets.ToggleButtons(
        options=[("positive(1)", 1), ("negative(0)", 0)],
        value=1,
        description="Point Label",
        layout=widgets.Layout(width="380px"),
    )
    clear_btn = widgets.Button(
        description="Clear Current Obj@Frame",
        button_style="warning",
        layout=widgets.Layout(width="220px"),
    )
    clear_all_btn = widgets.Button(
        description="Clear All Annotations",
        button_style="danger",
        layout=widgets.Layout(width="220px"),
    )
    refresh_btn = widgets.Button(
        description="Refresh",
        button_style="",
        layout=widgets.Layout(width="100px"),
    )
    export_btn = widgets.Button(
        description="Export Prompts",
        button_style="success",
        layout=widgets.Layout(width="140px"),
    )

    status_out = widgets.Output(layout=widgets.Layout(border="1px solid #aaa"))
    export_out = widgets.Output(layout=widgets.Layout(border="1px solid #aaa"))

    def _configure_matplotlib_fonts() -> None:
        candidate_fonts = [
            "SimHei",
            "Microsoft YaHei",
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "Noto Sans CJK TC",
            "Source Han Sans SC",
            "WenQuanYi Zen Hei",
            "PingFang SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]

        installed_font_names = {f.name for f in font_manager.fontManager.ttflist}
        preferred_fonts = [f for f in candidate_fonts if f in installed_font_names]
        fallback_chain = preferred_fonts if preferred_fonts else ["DejaVu Sans"]

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = fallback_chain
        plt.rcParams["axes.unicode_minus"] = False

        resolved_path = font_manager.findfont(
            font_manager.FontProperties(family=fallback_chain),
            fallback_to_default=True,
        )
        resolved_name = font_manager.FontProperties(fname=resolved_path).get_name()
        print(
            f"{status_prefix} matplotlib 字体={resolved_name} "
            f"(fallback={fallback_chain}) unicode_minus={plt.rcParams['axes.unicode_minus']}"
        )

    _configure_matplotlib_fonts()

    plt.close("all")
    ann_fig, ann_ax = plt.subplots(1, 1, figsize=(9, 6))
    ann_fig.canvas.toolbar_visible = True

    def _clamp_frame_value(v: Any) -> int:
        return max(frame_min, min(frame_max, int(v)))

    frame_sync_lock = {"active": False}
    render_state: Dict[str, Any] = {
        "point_artists": [],
        "text_artists": [],
        "last_frame_idx": None,
        "last_click_diag": None,
    }

    def _sync_frame_value(v: Any) -> int:
        clamped = _clamp_frame_value(v)
        if int(frame_input.value) != clamped:
            frame_sync_lock["active"] = True
            try:
                frame_input.value = clamped
            finally:
                frame_sync_lock["active"] = False
        return clamped

    def _current_ctx() -> tuple[int, str, int]:
        frame_idx = _sync_frame_value(frame_input.value)
        return int(frame_idx), str(object_dropdown.value), int(label_toggle.value)

    def _clear_previous_point_artists() -> None:
        for artist in render_state["point_artists"] + render_state["text_artists"]:
            try:
                artist.remove()
            except Exception:
                pass
        render_state["point_artists"].clear()
        render_state["text_artists"].clear()

    def _draw_annotation_canvas(
        force_draw: bool = False,
        *,
        trigger: str = "manual",
        full_reset: bool = False,
    ) -> None:
        frame_idx, obj_key, _ = _current_ctx()
        frame = video_frames_for_vis[frame_idx]

        _clear_previous_point_artists()

        if full_reset or render_state["last_frame_idx"] != frame_idx:
            ann_ax.cla()
            ann_ax.imshow(frame)
            ann_ax.set_axis_off()
            render_state["last_frame_idx"] = frame_idx

        ann_ax.set_title(
            f"Frame={frame_idx} | Object={object_specs[obj_key]['display']} "
            f"(obj_id={object_specs[obj_key]['obj_id']}) | "
            f"CurrentLabel={label_toggle.value}"
        )

        clicks = annotation_store[obj_key].get(frame_idx, [])
        for idx, c in enumerate(clicks):
            x, y, lb = int(c["x"]), int(c["y"]), int(c["label"])
            color = "lime" if lb == 1 else "red"
            marker = "o" if lb == 1 else "x"
            point_artist = ann_ax.scatter(
                [x],
                [y],
                marker=marker,
                c=color,
                s=64,
                linewidths=2,
            )
            text_artist = ann_ax.text(
                x + 6,
                y,
                f"{idx}:{lb}",
                color="white",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
            )
            render_state["point_artists"].append(point_artist)
            render_state["text_artists"].append(text_artist)

        with status_out:
            status_out.clear_output()
            print(
                f"{status_prefix} 当前对象={object_specs[obj_key]['display']} "
                f"frame={frame_idx} 点数={len(clicks)} trigger={trigger}"
            )
            if render_state.get("last_click_diag"):
                print(render_state["last_click_diag"])
            print(f"{status_prefix} 点击图像可添加点；绿色=o=positive(1)，红色=x=negative(0)")
            print(f"{status_prefix} 同一对象同一帧可连续点多个点；切换对象/帧后数据严格隔离。")

        if force_draw:
            ann_fig.canvas.draw()
        else:
            ann_fig.canvas.draw_idle()

    def _on_canvas_click(event: Any) -> None:
        if event.inaxes != ann_ax or event.xdata is None or event.ydata is None:
            return

        frame_idx, obj_key, point_label = _current_ctx()
        x = int(round(float(event.xdata)))
        y = int(round(float(event.ydata)))
        x = max(0, min(int(img_width) - 1, x))
        y = max(0, min(int(img_height) - 1, y))

        append_click(annotation_store, obj_key, frame_idx, x, y, point_label)

        current_points = len(annotation_store[obj_key].get(frame_idx, []))
        click_diag = (
            f"{status_prefix} [diag][write_target] frame_idx={frame_idx} object_key={obj_key} "
            f"point={[x, y]} label={point_label} current_obj_frame_points={current_points}"
        )
        render_state["last_click_diag"] = click_diag
        print(click_diag)

        _draw_annotation_canvas(trigger="click_add", force_draw=True)

    def _on_clear_clicked(_: Any) -> None:
        frame_idx, obj_key, _ = _current_ctx()
        if frame_idx in annotation_store[obj_key]:
            annotation_store[obj_key].pop(frame_idx, None)
        _draw_annotation_canvas(trigger="clear", force_draw=True, full_reset=True)

    def _on_clear_all_clicked(_: Any) -> None:
        for k in list(annotation_store.keys()):
            annotation_store[k] = {}
        render_state["last_click_diag"] = f"{status_prefix} [diag] 已清空所有对象、所有帧标注"
        _draw_annotation_canvas(trigger="clear_all", force_draw=True, full_reset=True)

    def _on_refresh_clicked(_: Any) -> None:
        _draw_annotation_canvas(trigger="refresh", force_draw=True, full_reset=True)

    def _on_frame_value_changed(change: Dict[str, Any]) -> None:
        if frame_sync_lock["active"]:
            return
        if change.get("name") != "value":
            return
        _sync_frame_value(change["new"])
        _draw_annotation_canvas(trigger="frame_change", force_draw=True, full_reset=True)

    def _on_object_changed(change: Dict[str, Any]) -> None:
        if change.get("name") != "value":
            return
        _draw_annotation_canvas(trigger="object_change", force_draw=True, full_reset=True)

    def _on_label_changed(change: Dict[str, Any]) -> None:
        if change.get("name") != "value":
            return
        _draw_annotation_canvas(trigger="label_change")

    def _on_export_clicked(_: Any) -> None:
        state.export_prompts = store_to_export_prompts(annotation_store, object_specs)
        state.use_visual_annotation_export = True

        if save_json_on_export and export_json_path:
            save_annotation_prompts_json(
                export_prompts=state.export_prompts,
                json_path=export_json_path,
                status_prefix=status_prefix,
            )

        if on_export is not None:
            on_export(state.export_prompts)

        with export_out:
            export_out.clear_output()
            summary = summarize_prompt_map(state.export_prompts)
            print(f"{status_prefix} 导出完成，已自动设置 USE_VISUAL_ANNOTATION_EXPORT=True")
            print(f"{status_prefix} 对象条目统计: {summary['entries_per_object']}")
            print(f"{status_prefix} 对象点数统计: {summary['points_per_object']}")
            print(f"{status_prefix} 按对象逐帧点数统计: {summary['frame_point_counts_per_object']}")
            print(f"{status_prefix} 含标注帧数: {summary['num_frames_with_prompts']}")
            for warn in summary.get("gripper_frame_count_warnings", []):
                print(f"{status_prefix} [warn][gripper_frame_count] {warn}")

            spatial_diag = summarize_gripper_store_spatial_side_consistency(
                annotation_store=annotation_store,
                img_width=img_width,
                left_key="gripper_left",
                right_key="gripper_right",
                dominance_ratio=0.6,
            )
            left_stats = spatial_diag["left"]
            right_stats = spatial_diag["right"]
            print(
                f"{status_prefix} [diag][spatial][left] entries={left_stats['entries']} "
                f"all_points={left_stats['all_points']} pos_points={left_stats['pos_points']} "
                f"basis={left_stats['basis']} basis_points={left_stats['basis_points']} "
                f"x_mean={left_stats['x_mean']} x_median={left_stats['x_median']} "
                f"left_ratio={left_stats['left_ratio']} right_ratio={left_stats['right_ratio']} "
                f"dominant={left_stats['dominant_side']}"
            )
            print(
                f"{status_prefix} [diag][spatial][right] entries={right_stats['entries']} "
                f"all_points={right_stats['all_points']} pos_points={right_stats['pos_points']} "
                f"basis={right_stats['basis']} basis_points={right_stats['basis_points']} "
                f"x_mean={right_stats['x_mean']} x_median={right_stats['x_median']} "
                f"left_ratio={right_stats['left_ratio']} right_ratio={right_stats['right_ratio']} "
                f"dominant={right_stats['dominant_side']}"
            )
            if spatial_diag.get("conflict_type") != "none":
                print(
                    f"{status_prefix} [diag][spatial][warn] conflict_type={spatial_diag['conflict_type']} "
                    f"message={spatial_diag['warning']}"
                )

            if save_json_on_export and export_json_path:
                print(f"{status_prefix} JSON 文件(覆盖写入): {export_json_path}")
            print(f"{status_prefix} 导出结构（可直接被 Stage A/B 消费）:")
            print(json.dumps(state.export_prompts, ensure_ascii=False, indent=2))

    frame_input.observe(_on_frame_value_changed, names="value")
    object_dropdown.observe(_on_object_changed, names="value")
    label_toggle.observe(_on_label_changed, names="value")
    clear_btn.on_click(_on_clear_clicked)
    clear_all_btn.on_click(_on_clear_all_clicked)
    refresh_btn.on_click(_on_refresh_clicked)
    export_btn.on_click(_on_export_clicked)
    ann_fig.canvas.mpl_connect("button_press_event", _on_canvas_click)

    controls = widgets.VBox(
        [
            widgets.HBox([frame_input, object_dropdown]),
            widgets.HBox([label_toggle]),
            widgets.HBox([clear_btn, clear_all_btn, refresh_btn, export_btn]),
            status_out,
            export_out,
        ]
    )

    result.update(
        {
            "widget_ready": True,
            "controls": controls,
            "figure": ann_fig,
            "axis": ann_ax,
            "draw": _draw_annotation_canvas,
        }
    )

    if auto_display:
        display(controls)
        _draw_annotation_canvas(trigger="init")
        plt.show()

    return result
