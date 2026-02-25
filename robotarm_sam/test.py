import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
import sam3
import torch
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
video_path = f"{sam3_root}/assets/videos/camera0_rgb.mp4"

# use all available GPUs on the machine
gpus_to_use = range(torch.cuda.device_count())

from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(
    checkpoint_path="/data/haoxiang/sam3/models/facebook/sam3/sam3.pt",
    gpus_to_use=gpus_to_use,
    # 关闭 temporal disambiguation，避免 hotstart 阶段把候选 obj_id 提前过滤掉
    apply_temporal_disambiguation=False,
)

from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

# font size for axes titles
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12

def propagate_in_video(predictor, session_id, propagation_direction="forward"):
    # 使用单向传播，避免 "both" 模式下同一 frame 被后续方向覆盖
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction=propagation_direction,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def print_frame_obj_ids(outputs_per_frame, max_frames=10):
    frame_indices = sorted(outputs_per_frame.keys())[:max_frames]
    for idx in frame_indices:
        obj_ids = outputs_per_frame[idx].get("out_obj_ids", [])
        print(f"frame {idx}: obj_ids={obj_ids}")


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

# load "video_frames_for_vis" for visualization purposes (they are not used by the model)
if isinstance(video_path, str) and video_path.endswith(".mp4"):
    cap = cv2.VideoCapture(video_path)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
        video_frames_for_vis.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
        print(
            f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
            f"falling back to lexicographic sort."
        )
        video_frames_for_vis.sort()

response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]

# note: in case you already ran one text prompt and now want to switch to another text prompt
# it's required to reset the session first (otherwise the results would be wrong)
_ = predictor.handle_request(
    request=dict(
        type="reset_session",
        session_id=session_id,
    )
)

prompt_text_str = "robot arm"
# prompt_text_str = "black end-effector"
frame_idx = 0  # add a text prompt on frame 0
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=prompt_text_str,
    )
)
out = response["outputs"]

plt.close("all")
visualize_formatted_frame_output(
    frame_idx,
    video_frames_for_vis,
    outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
    titles=["SAM 3 Dense Tracking outputs"],
    figsize=(6, 4),
)

# propagate from frame 0 to the end and collect outputs
outputs_per_frame = propagate_in_video(
    predictor,
    session_id,
    propagation_direction="forward",
)
print(f"num propagated frames: {len(outputs_per_frame)}")
print_frame_obj_ids(outputs_per_frame, max_frames=20)

# reformat for visualization
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

vis_frame_stride = 60
plt.close("all")
for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
    visualize_formatted_frame_output(
        frame_idx,
        video_frames_for_vis,
        outputs_list=[outputs_per_frame],
        titles=["SAM 3 Dense Tracking outputs"],
        figsize=(6, 4),
    )