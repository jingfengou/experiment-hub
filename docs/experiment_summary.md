# Experiment Summary & Prompts

## MathCanvas / BAGEL-Canvas
- Rotation (1k samples, dataset: `.../dataset/data_modified_with_subject.json`)
  - base: 0.3070 (`outputs/rotation_mathcanvas_ddp`)
  - steps: 0.5240 (`outputs/rotation_mathcanvas_steps_ddp`)
  - steps_nofinal: 0.2710 (`outputs/rotation_mathcanvas_steps_nofinal_ddp`)
  - interleave base: 0.2420 (`outputs/rotation_mathcanvas_interleave_base/rotation_base_default`)
  - interleave nofinal: 0.2580 (`outputs/rotation_mathcanvas_interleave_nofinal/rotation_nofinal_default`)
  - interleave base (MoT): 0.2580 (`outputs/rotation_mathcanvas_interleave_base_mot/rotation_base_mot_default`)
  - Prompt:
    ```
    You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer.
    The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags,
    respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.
    <image>

    Question: {Question}
    A) {choice1}
    B) {choice2}
    C) {choice3}
    D) {choice4}

    Answer:
    ```

- 3Dviews (1k samples, dataset: `.../3Dviews/thinking_chain_dataset/dataset.json`)
  - base non-interleave: 0.2370 (`outputs/3dviews_mathcanvas_ddp`)
  - steps non-interleave: 0.3230 (`outputs/3dviews_mathcanvas_steps_ddp`)
  - base interleave: 0.2370 (`outputs/3dviews_interleave_base/3dviews_base_default`)
  - steps interleave: 0.2980 (`outputs/3dviews_interleave_steps/3dviews_steps_default`)
  - Prompt (base): same as rotation above.
  - Prompt addition for steps: 
    ```
    Step hints (sanitized, options hidden):
    - Step 1: {cleaned_thought}
    - Step 2: ...

    Answer:
    ```

- SpatialViz (MoT, 566 samples from `SpatialViz/test-00000-of-00001.parquet`)
  - interleave base: 0.3110 (`outputs/spatialviz_mathcanvas_interleave_mot/run_1208_0301/spatialviz_mot_default`)
  - Case samples: `MentalAnimation-ArrowMoving-Level0-0-3-3-2` GT=D/Pred=D（推理链较长但收敛）；`...-1-3-3-2` GT=C/Pred=D（方向更新理解错误）；`...-2-3-3-2` GT=A/Pred=A。

## Qwen3-VL (official offline)
- Rotation (1k samples, dataset: `.../dataset/data_modified_with_subject.json`)
  - 8B base: 0.3160 (`outputs/qwen3vl8b_rotation_base_official`)
  - 8B nofinal: 0.2200 (`outputs/qwen3vl8b_rotation_nofinal_official`)
  - Prompt:
    ```
    You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer.
    The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags,
    respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.
    Keep the reasoning concise (<=3 sentences) and always output both tags even if unsure; pick the most likely option.
    <image>
    Question: {Question}
    A) ...
    B) ...
    C) ...
    D) ...
    [nofinal mode adds rotation steps text]
    Answer:
    ```

## Qwen3-VL (API)
- Rotation (dataset: `.../dataset/data_modified_with_subject.json`)
  - qwen3-vl-235b base: 0.3300 (`outputs/rotation_api_qwen3vl235b_base`)
  - qwen3-vl-plus nofinal-steps: 0.0200 (`outputs/rotation_api_qwen3vl_plus_nofinal_steps`, 907 missing responses)
  - Prompt (base, `rotation_api_infer.py`):
    ```
    {Question}
    Choices:
    A. {optA}
    B. {optB}
    C. {optC}
    D. {optD}
    请根据图像与选项，直接给出正确选项字母，并用一两句话说明理由。
    ```
  - Prompt (nofinal-steps, `rotation_api_infer_nofinal_steps.py`):
    ```
    {Question}
    Choices:
    A. ...
    B. ...
    C. ...
    D. ...
    提示：先绕 Z 轴旋转 180 度后的中间状态如附图所示，再绕 X 轴旋转 180 度，判断最终对应的选项。
    请根据图像与选项，直接给出正确选项字母，并用一两句话说明理由。
    ```

## Pending / Missing
- ROVER 推理已跑（outputs/rover/rover_default）但未跑官方评测。
- 若需补全：检查 qwen3-vl-plus API 的失败请求（907 个缺失 response），重新调用或重试限速；可再跑 qwen3-vl-8b/32b API 以对齐。
