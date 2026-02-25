from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional
import json


AnnotationObjectSpecs = Dict[str, Dict[str, Any]]
AnnotationStore = Dict[str, Dict[int, List[Dict[str, int]]]]
PromptList = List[Dict[str, Any]]


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
    arm_left_obj_id: int,
    arm_right_obj_id: int,
    gripper_left_obj_id: int,
    gripper_right_obj_id: int,
) -> AnnotationObjectSpecs:
    return {
        "arm_left": {
            "display": "左臂",
            "obj_id": int(arm_left_obj_id),
            "target": "ARM_LEFT_INITIAL_PROMPTS",
        },
        "arm_right": {
            "display": "右臂",
            "obj_id": int(arm_right_obj_id),
            "target": "ARM_RIGHT_INITIAL_PROMPTS",
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
    except Exception as e:  # pragma: no cover - notebook runtime fallback path
        result["widget_error"] = e
        return result

    frame_input = widgets.IntText(
        value=0,
        description="Frame",
        layout=widgets.Layout(width="220px"),
    )

    object_dropdown = widgets.Dropdown(
        options=[
            (
                f"{spec['display']} (obj_id={spec['obj_id']})",
                key,
            )
            for key, spec in object_specs.items()
        ],
        value="arm_left",
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

    plt.close("all")
    ann_fig, ann_ax = plt.subplots(1, 1, figsize=(9, 6))
    ann_fig.canvas.toolbar_visible = True

    def _clamp_frame_value(v: Any) -> int:
        return max(frame_min, min(frame_max, int(v)))

    frame_sync_lock = {"active": False}

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

    def _draw_annotation_canvas(force_draw: bool = False) -> None:
        frame_idx, obj_key, _ = _current_ctx()
        frame = video_frames_for_vis[frame_idx]

        ann_ax.clear()
        ann_ax.imshow(frame)
        ann_ax.set_title(
            f"Frame={frame_idx} | Object={object_specs[obj_key]['display']} "
            f"(obj_id={object_specs[obj_key]['obj_id']}) | "
            f"CurrentLabel={label_toggle.value}"
        )
        ann_ax.set_axis_off()

        clicks = annotation_store[obj_key].get(frame_idx, [])
        for idx, c in enumerate(clicks):
            x, y, lb = int(c["x"]), int(c["y"]), int(c["label"])
            color = "lime" if lb == 1 else "red"
            marker = "o" if lb == 1 else "x"
            ann_ax.plot(x, y, marker=marker, color=color, markersize=8, markeredgewidth=2)
            ann_ax.text(
                x + 6,
                y,
                f"{idx}:{lb}",
                color="white",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
            )

        with status_out:
            status_out.clear_output()
            print(
                f"{status_prefix} 当前对象={object_specs[obj_key]['display']} "
                f"frame={frame_idx} 点数={len(clicks)}"
            )
            print(f"{status_prefix} 点击图像可添加点；绿色=o=positive(1)，红色=x=negative(0)")

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
        _draw_annotation_canvas()

    def _on_clear_clicked(_: Any) -> None:
        frame_idx, obj_key, _ = _current_ctx()
        if frame_idx in annotation_store[obj_key]:
            annotation_store[obj_key].pop(frame_idx, None)
        _draw_annotation_canvas()

    def _on_refresh_clicked(_: Any) -> None:
        _draw_annotation_canvas(force_draw=True)

    def _on_frame_value_changed(change: Dict[str, Any]) -> None:
        if frame_sync_lock["active"]:
            return
        if change.get("name") != "value":
            return
        _sync_frame_value(change["new"])
        _draw_annotation_canvas()

    def _on_export_clicked(_: Any) -> None:
        state.export_prompts = store_to_export_prompts(annotation_store, object_specs)
        state.use_visual_annotation_export = True

        if on_export is not None:
            on_export(state.export_prompts)

        with export_out:
            export_out.clear_output()
            summary = {k: len(v) for k, v in state.export_prompts.items()}
            print(f"{status_prefix} 导出完成，已自动设置 USE_VISUAL_ANNOTATION_EXPORT=True")
            print(f"{status_prefix} 各对象关键帧条目数: {summary}")
            print(f"{status_prefix} 导出结构（可直接被 Stage A/B 消费）:")
            print(json.dumps(state.export_prompts, ensure_ascii=False, indent=2))

    frame_input.observe(_on_frame_value_changed, names="value")
    object_dropdown.observe(lambda _: _draw_annotation_canvas(), names="value")
    label_toggle.observe(lambda _: _draw_annotation_canvas(), names="value")
    clear_btn.on_click(_on_clear_clicked)
    refresh_btn.on_click(_on_refresh_clicked)
    export_btn.on_click(_on_export_clicked)
    ann_fig.canvas.mpl_connect("button_press_event", _on_canvas_click)

    controls = widgets.VBox(
        [
            widgets.HBox([frame_input, object_dropdown]),
            widgets.HBox([label_toggle]),
            widgets.HBox([clear_btn, refresh_btn, export_btn]),
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
        _draw_annotation_canvas()
        plt.show()

    return result
