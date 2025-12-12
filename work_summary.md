# Work Summary

## 2025-11-23
- Rotation baseline inference (no step hints), 2×GPU, log to `logs/rotation_mathcanvas_ddp.log`:
  ```bash
  cd /workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas
  bash scripts/inference/run_rotation_mathcanvas_ddp.sh
  ```
- Rotation with step text+images, 2×GPU, log to `logs/rotation_mathcanvas_steps_ddp.log`:
  ```bash
  cd /workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas
  bash scripts/inference/run_rotation_mathcanvas_steps_ddp.sh
  ```
- Accuracy eval on outputs:
  ```bash
  cd /workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas
  python scripts/evaluation/eval_rotation_outputs.py \
    --output-dir outputs/rotation_mathcanvas_ddp \
    --dataset-json /workspace/oujingfeng/project/datasets/mydatasets/dataset/data_modified_with_subject.json
  ```

## 2025-11-25
- ROVER 数据格式确认：在 `bagel-canvas` 环境运行 `inspect_rover_dataset.py --split train[:5] --save rover_schema_sample.json`，核对字段（id/prompt/keywords/target_description、image/image2/target_image）。
- 导出 ROVER 数据为 MathCanvas 统一格式：新增 `ROVER/prepare_rover_for_mathcanvas.py`，将 HF ROVER-IG 导出到 `MathCanvas/BAGEL-Canvas/data_handlers/rover/{rover_dataset.jsonl, images/}`，指令包含 prompt+target_description+keywords。
- 推理脚本：新增 `MathCanvas/BAGEL-Canvas/scripts/inference/infer_rover.sh`，指向权重 `/workspace/oujingfeng/modelckpt/BAGEL-Canvas`，数据 `data_handlers/rover/rover_dataset.jsonl`，输出 `outputs/rover/rover_default`（不追加时间戳）。
- 结果转存：新增 `ROVER/export_rover_generations.py`，将推理目录中的最后一步图像/文本导出为 `gen_{task_id}.png/.txt` 到 `ROVER_GEN_DIR`，便于运行 `evaluate_rover.py` 评测。
- 完整流程示例（需 bagel-canvas 环境，可选设置 `HF_ENDPOINT` 镜像）：先导出数据 `python prepare_rover_for_mathcanvas.py --split train --max_samples 0`，再 `bash scripts/inference/infer_rover.sh`，最后 `python export_rover_generations.py --inference_dir ... --dest_dir $ROVER_GEN_DIR` 后运行 ROVER 评测。
- 实际命令串（bagel-canvas 环境）：
  ```bash
  # 1) 可选：镜像
  export HF_ENDPOINT=https://hf-mirror.com
  # 2) 数据导出到 MathCanvas/BAGEL-Canvas/data_handlers/rover
  cd /workspace/oujingfeng/project/think_with_generated_images/ROVER
  python prepare_rover_for_mathcanvas.py --split train --max_samples 0
  # 3) 推理生成（输出到 outputs/rover/rover_default）
  cd /workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/inference/infer_rover.sh
  # 4) 转存为 ROVER 评测格式（需设置 ROVER_GEN_DIR）
  export ROVER_GEN_DIR=/workspace/oujingfeng/project/think_with_generated_images/ROVER/gen_outputs
  python /workspace/oujingfeng/project/think_with_generated_images/ROVER/export_rover_generations.py \
    --inference_dir /workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas/outputs/rover/rover_default \
    --dest_dir $ROVER_GEN_DIR
  # 5) 运行官方评测
  cd /workspace/oujingfeng/project/think_with_generated_images/ROVER
  python evaluate_rover.py --output_dir results
  ```

## 2025-11-26
- Qwen3-VL 云 API 旋转基准推理：新增 `Qwen3-VL/evaluation/rotation_api_infer.py` 与运行脚本 `run_rotation_api.sh`，支持 `qwen-vl-8b/32b`，输入为 rotation JSON+图像，输出 request/response。
- MathCanvas 旋转基准交错推理改造：新增数据准备 `scripts/inference/prepare_rotation_interleave_jsonl.py`（base/steps/nofinal 三模式），运行脚本改为调用交错推理 `mathcanvas_interleave_reasoner.py`：
  - `scripts/inference/run_rotation_mathcanvas_ddp.sh`（base，无步骤图，输出 `outputs/rotation_mathcanvas_interleave_base`）
  - `scripts/inference/run_rotation_mathcanvas_steps_ddp.sh`（含全部步骤图，输出 `outputs/rotation_mathcanvas_interleave_steps`）
  - `scripts/inference/run_rotation_mathcanvas_steps_nofinal_ddp.sh`（去掉最终步骤图，输出 `outputs/rotation_mathcanvas_interleave_nofinal`）
  运行前需用预处理脚本生成对应 JSONL+images。
- 交错推理命令示例（bagel-canvas 环境）：
  ```bash
  # 生成不同模式的 JSONL（按需选择模式）
  python scripts/inference/prepare_rotation_interleave_jsonl.py \
    --dataset-json /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json \
    --dataset-root /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset \
    --output-root data_handlers/rotation_interleave \
    --mode nofinal \
    --max-samples 1000
  # 运行交错推理（示例：去掉最终步骤图）
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/inference/run_rotation_mathcanvas_steps_nofinal_ddp.sh
  # base/steps 模式则运行对应脚本 run_rotation_mathcanvas_ddp.sh / run_rotation_mathcanvas_steps_ddp.sh
  ```
- ROVER 交错推理与评估：`prepare_rover_for_mathcanvas.py` 转数据，`scripts/inference/infer_rover.sh` 调用交错推理，`export_rover_generations.py` 转为 `gen_{task_id}.png/.txt`，可直接用 `python evaluate_rover.py --output_dir results` 评测（已有结果可复用）。
- Qwen3-VL 官方模型离线推理脚本：新增 `evaluation/run_qwen3vl_rotation_official.py` 与 `run_qwen3vl_rotation_official.sh`，默认跑 8B/32B Thinking，base & nofinal 前 1000 样本，输出至 `outputs/qwen3vl8b_rotation_base_official[_32b]`、`outputs/qwen3vl8b_rotation_nofinal_official[_32b]`。可选 `HF_ENDPOINT/HF_TOKEN`，温度默认 0.7（温度为0时自动用 greedy）。

## 2025-11-27
- 旋转基准精度评估：新增 `Qwen3-VL/evaluation/eval_qwen3vl_rotation_outputs.py`（解析 `<answer>`、生成 eval jsonl/summary），对 8B base/nofinal 输出完成首轮评估。
- Qwen3-VL 官方推理改进：默认 hf-mirror，prompt 追加“简短思维链+必须给出 `<think>/<answer>` 即使不确定”；优先加载 `Combined_image`；进度用 tqdm；采样默认温度降至 0.3、max_new_tokens=512（可覆写）。
- 运行脚本修复：`evaluation/run_qwen3vl_rotation_official.sh` 增加采样参数导出、greedy 开关处理，避免无效参数报错。
- MathCanvas 数据准备对齐：`prepare_rotation_interleave_jsonl.py`/`prepare_rotation_nofinal_jsonl.py` 统一为推理 prompt 格式；nofinal 模式文本里自然加入中间步骤描述，保留步骤图（去掉最后一步图）。
- MathCanvas/BAGEL-Canvas 旋转实验精度汇总：生成 `outputs/rotation_eval_summary.md`，评估 base/steps/steps_nofinal（非交替）及 interleave base/nofinal，精度分别为 0.3110 / 0.5230 / 0.2700 / 0.2430 / 0.2570。
- Qwen3-VL 8B Instruct 旋转基准：评估 `outputs/qwen3vl8b_rotation_base_official` 与 `..._nofinal_official`，准确率调整为 0.3160 / 0.2200（改进解析避免将量词 a 误判为答案）；生成总结 `outputs/qwen3vl8b_rotation_eval_summary.md`。

## 2025-11-28
- MathCanvas 3Dviews 四模式推理脚本落地（默认 2 卡）：非图文交替 `infer_3dviews_mathcanvas_ddp.py` / `run_3dviews_mathcanvas_ddp.sh`（无步骤）、`infer_3dviews_mathcanvas_steps_ddp.py` / `run_3dviews_mathcanvas_steps_ddp.sh`（带步骤图/文、过滤 option/answer 关键词）。
- 图文交替数据与运行：`prepare_3dviews_interleave_jsonl.py` 生成 base/steps JSONL+拷贝图，`run_3dviews_mathcanvas_interleave_base.sh` / `run_3dviews_mathcanvas_interleave_steps.sh` 调用 `mathcanvas_interleave_reasoner.py`。
- Prompt 细节：统一 rotation 模板，带 `<think>/<answer>` 标签；steps 模式在 prompt 中追加“Step hints (sanitized, options hidden)”并按步骤依次喂入图/文；过滤规则用正则 `(option|选项|answer|正确|正确选项|final answer)` 按句拆分后删除含关键字句子。
- 路径修正：上述脚本与预处理文件已放回 `MathCanvas/BAGEL-Canvas/scripts/inference/`，清理了误放的 `/workspace/scripts/` 目录。

## 2025-12-03
- 新增 MoT 权重跑 rotation base 交替推理脚本 `scripts/inference/run_rotation_mathcanvas_mot_ddp.sh`，默认指向 `/workspace/oujingfeng/modelckpt/BAGEL-7B-MoT`，输出 `outputs/rotation_mathcanvas_interleave_base_mot`，日志 `logs/rotation_mathcanvas_mot_ddp.log`。
- 修正 rotation 交替数据准备的 prompt 生成模板（`prepare_rotation_interleave_jsonl.py`）：题干 `<image>` + `Question:` + 选项 + `Answer:`，与 infer 脚本一致。
- 处理 `ae.safetensors` 缺少 metadata 导致 Accelerate 报错的问题：用 safetensors 重写 metadata `format=pt`，路径不变。
- MoT 推理已启动（2 卡 H100），可能有 NCCL 设备映射警告和未使用权重提示，均可忽略；实时日志 `tail -f logs/rotation_mathcanvas_mot_ddp.log`，结果目录 `outputs/rotation_mathcanvas_interleave_base_mot/rotation_base_mot_default`。

## 2025-12-XX
- Rotation Hunyuan3D 中间步骤替换与运行指令（集中在 experiment-hub）：
  - 新脚本：`experiment-hub/mathcanvas/hunyuan3d/scripts/prepare_rotation_hunyuan3d_override.py`，将 Hunyuan3D 投影替换原数据集的步骤图，生成新数据根和 JSON。
  - 生成命令：
    ```bash
    cd /workspace/oujingfeng/experiment-hub
    python mathcanvas/hunyuan3d/scripts/prepare_rotation_hunyuan3d_override.py \
      --dataset-json /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json \
      --dataset-root /workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset \
      --hunyuan-root /workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas/outputs/hunyuan3d_rotation_proj \
      --output-root /workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas/data_handlers/rotation_hunyuan3d \
      --max-samples 1000
    ```
  - 非交替推理：
    ```bash
    cd /workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/inference/run_rotation_mathcanvas_steps_ddp.sh \
      DATA_JSON=/workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas/data_handlers/rotation_hunyuan3d/data_modified_with_subject.json \
      DATA_ROOT=/workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas/data_handlers/rotation_hunyuan3d
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/inference/run_rotation_mathcanvas_steps_nofinal_ddp.sh \
      DATA_JSON=/workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas/data_handlers/rotation_hunyuan3d/data_modified_with_subject.json \
      DATA_ROOT=/workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas/data_handlers/rotation_hunyuan3d
    ```
  - 交替数据准备：
    ```bash
    python scripts/inference/prepare_rotation_interleave_jsonl.py \
      --dataset-json data_handlers/rotation_hunyuan3d/data_modified_with_subject.json \
      --dataset-root data_handlers/rotation_hunyuan3d \
      --output-root data_handlers/rotation_interleave_hunyuan3d \
      --mode steps --max-samples 1000
    python scripts/inference/prepare_rotation_interleave_jsonl.py \
      --dataset-json data_handlers/rotation_hunyuan3d/data_modified_with_subject.json \
      --dataset-root data_handlers/rotation_hunyuan3d \
      --output-root data_handlers/rotation_interleave_hunyuan3d \
      --mode nofinal --max-samples 1000
    ```
  - 交替推理：
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/inference/run_rotation_mathcanvas_steps_ddp.sh \
      DATA_JSON=data_handlers/rotation_interleave_hunyuan3d/steps/rotation_steps.jsonl \
      DATA_ROOT=data_handlers/rotation_interleave_hunyuan3d/steps
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/inference/run_rotation_mathcanvas_steps_nofinal_ddp.sh \
      DATA_JSON=data_handlers/rotation_interleave_hunyuan3d/nofinal/rotation_nofinal.jsonl \
      DATA_ROOT=data_handlers/rotation_interleave_hunyuan3d/nofinal
    ```
  - 评估可沿用 `scripts/evaluation/eval_rotation_outputs.py`，指定对应输出目录和新 JSON。

## 2025-12-05
- 评估 MoT 权重交错推理 `outputs/rotation_mathcanvas_interleave_base_mot/rotation_base_mot_default`：total=1000、correct=261、acc=0.2610，未生成中间图（action 未触发 image 分支，`max_iterations=10` 全部为文本）。
- 更新 rotation 报告：`mathcanvas/rotation/reports/rotation_eval_summary.md` 与 `docs/rotation_report.md` 记录 MoT 交错无中间图、默认迭代说明，并补充 base/steps/steps_nofinal 含步骤文本+图的定义。
- 统一旋转模式释义至 Qwen 报告：`qwen3-vl/reports/qwen3vl8b_rotation_eval_summary.md`、`rotation_api_eval_summary.md` 明确 base=题干图；nofinal=题干+步骤图+步骤文本但去掉最后一步由模型补全。

## 2025-12-08
- 修复 MoT 交替推理的终止判定：在 `mathcanvas/rotation/scripts/mathcanvas_interleave_reasoner.py` 与 `MathCanvas/BAGEL-Canvas/mathcanvas_interleave_reasoner.py` 将 `<|im_end|>` (`eos_token_id`) 也映射为 end，避免 MoT 输出被截断却不早停/不画图。
- SpatialViz + MoT 支持：新增 `scripts/inference/prepare_spatialviz_from_parquet.py`（从 `datasets/SpatialViz/test-00000-of-00001.parquet` 生成带系统 prompt 的 JSONL），更新 `scripts/inference/run_spatialviz_mathcanvas_mot_ddp.sh`（默认系统 prompt、自动生成 JSONL、指向 MoT 权重）。
- 运行指令示例：
  - Rotation MoT 重跑（自定义输出防覆盖）：
    ```bash
    cd /workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas
    export OUTPUT_DIR=outputs/rotation_mathcanvas_interleave_base_mot/rotation_base_mot_rerun_$(date +%m%d_%H%M)
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/inference/run_rotation_mathcanvas_mot_ddp.sh
    ```
  - SpatialViz + MoT（需 pyarrow/pandas 读取 parquet，自定义输出防覆盖）：
    ```bash
    cd /workspace/oujingfeng/project/think_with_generated_images/MathCanvas/BAGEL-Canvas
    export OUTPUT_DIR=outputs/spatialviz_mathcanvas_interleave_mot/run_$(date +%m%d_%H%M)
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/inference/run_spatialviz_mathcanvas_mot_ddp.sh
    ```

## 2025-12-09
- MoT 交错 rotation 评估：`outputs/rotation_mathcanvas_interleave_base_mot/rotation_base_mot_default` 重新评测，total=1000、correct=258、acc=0.2580，仍无中间图；`mathcanvas/rotation/reports/rotation_eval_summary.md` 与 `docs/rotation_report.md` 已更新并补充错误案例（sample_0001 GT=D/Pred=B 等）。
- SpatialViz + MoT 评估：`outputs/spatialviz_mathcanvas_interleave_mot/run_1208_0301/spatialviz_mot_default`，566 样本，acc=0.311；代表性案例写入 `docs/experiment_summary.md`（如 MentalAnimation-ArrowMoving-Level0-0-3-3-2 GT=D/Pred=D，Level0-1-3-3-2 GT=C/Pred=D）。
