下面按“**作用点** + **流程位置**”总结 [`arm_cable_obj_id`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:87)。

### 1) 配置层（流程起点）
- 在对象 ID 配置区定义主 arm 目标 ID：[`arm_cable_obj_id`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:87)。
- 与 [`gripper_left_obj_id`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:88)、[`gripper_right_obj_id`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:89) 共同形成三对象 ID 约定。

### 2) 校验层（初始化 gate）
- 进入统一校验函数 [`validate_obj_id_constraints()`](robotarm_sam/mask_pipeline_tools.py:187)。
- 作用：
  1. 检查 3 个 ID 都是 int（见 [`validate_obj_id_constraints()`](robotarm_sam/mask_pipeline_tools.py:201)）。
  2. 检查 3 个 ID 互不重复（见 [`validate_obj_id_constraints()`](robotarm_sam/mask_pipeline_tools.py:205)）。
  3. 检查 arm 提示列表里的每条 [`obj_id`](robotarm_sam/mask_pipeline_tools.py:215) 都等于 [`arm_cable_obj_id`](robotarm_sam/mask_pipeline_tools.py:220)。

### 3) 标注 UI 层（对象定义）
- 通过 [`build_annotation_object_specs()`](robotarm_sam/annotation_ui_tools.py:39) 把 [`arm_cable_obj_id`](robotarm_sam/annotation_ui_tools.py:40) 绑定到“机械臂+线缆”对象规格（见 [`build_annotation_object_specs()`](robotarm_sam/annotation_ui_tools.py:45)）。
- 作用是“标注对象路由”，即 arm 标注写入目标列表（`ARM_CABLE_INITIAL_PROMPTS`）。

### 4) Stage A 推理层（bootstrap + refinement）
- Stage A 先打印 [`stage_a_active_obj_ids`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:428)（包含 [`arm_cable_obj_id`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:428)）。
- 先走文本 bootstrap：[`add_text_prompt()`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:442)，文本来自 [`ARM_CABLE_BOOTSTRAP_TEXT_PROMPT`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:93)。
- 关键点：文本路径底层调用 [`add_text_prompt()`](robotarm_sam/mask_pipeline_tools.py:447) -> [`handle_request(... text=...)`](robotarm_sam/mask_pipeline_tools.py:451)，这里**不传** arm 的 [`obj_id`](robotarm_sam/mask_pipeline_tools.py:456)。
- 之后 arm 点提示 refinement 才会显式绑定 ID：[`apply_prompt_list()`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:465) -> [`add_point_prompt()`](robotarm_sam/mask_pipeline_tools.py:423) -> [`obj_id=prompt["obj_id"]`](robotarm_sam/mask_pipeline_tools.py:434)。

### 5) Stage B 诊断层（期望 ID 集）
- 仅用于构造“期望并集”诊断：[`stage_b_union_obj_ids`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:775) = arm + 左右 gripper。
- 作用偏“检查/打印语义”，不是硬过滤器。

### 6) 导出层（真正的下游控制）
- 在 arm-only 导出中，[`arm_cable_obj_id`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:897) 被作为 arm 掩码来源 ID：[`arm_obj_ids=[arm_cable_obj_id]`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:897)。
- 执行函数 [`save_arm_only_masks_for_propainter()`](robotarm_sam/mask_pipeline_tools.py:705) 会把 [`arm_obj_ids`](robotarm_sam/mask_pipeline_tools.py:706) 与 [`gripper_obj_ids`](robotarm_sam/mask_pipeline_tools.py:707) 分别并集，再做 arm 减 gripper。

---

## 整体逻辑流程（按时序）
1. 定义 ID：[`arm_cable_obj_id`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:87) + 左右 gripper。
2. 初始化校验：[`validate_obj_id_constraints()`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:255)。
3. 构建标注对象规格：[`build_annotation_object_specs()`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:357)。
4. Stage A 文本 bootstrap（不绑定 arm 的 [`obj_id`](robotarm_sam/mask_pipeline_tools.py:456)）。
5. Stage A 点提示 refinement（绑定到 [`arm_cable_obj_id`](robotarm_sam/mask_pipeline_tools.py:220) 对应的提示）。
6. Stage B 注入左右 gripper 独立 ID，并传播。
7. 导出时以 [`arm_obj_ids=[arm_cable_obj_id]`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:897) 作为 arm 主通道。

一句话归纳：[`arm_cable_obj_id`](robotarm_sam/generate_mask_airexo_data_gripper_points.ipynb:87) 在这套流程里是“**arm 主通道/校验与导出锚点 ID**”，不是“文本阶段只允许单 arm 实例”的强约束器。