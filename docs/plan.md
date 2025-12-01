# Plan for MathCanvas rotation inference

- [x] Inspect Orthus rotation prompt format and dataset layout (`project/Orthus/inference/interleave_generation_rotation_test_8gpu.py`) to mirror prompt and image resolution logic.
- [x] Add a MathCanvas/BAGEL-Canvas inference script for the rotation dataset using the same prompt style, loading the Bagel checkpoint and tokenizer configs.
- [x] Provide a runnable shell wrapper pointing to the downloaded checkpoint in `/workspace/oujingfeng/modelckpt/BAGEL-Canvas` and the rotation dataset JSON.
- [ ] Run end-to-end verification once networking/compute are confirmed available for the target GPU setup.

# MathCanvas 评测 ROVER 计划（2025-11-25）

- 标题：使用 MathCanvas/BAGEL-Canvas 评测 ROVER 基准

- [ ] 明确 ROVER 基准的输入输出格式与所需字段（任务 id、prompt/keywords、参考图像路径），对齐 MathCanvas 推理接口的数据需求与目录布局。
- [ ] 准备推理数据与脚本：将 ROVER 任务转成 `UniHandler` 可读格式或新增 handler，指定模型权重 `/workspace/oujingfeng/modelckpt/BAGEL-Canvas`，规划生成目录 `ROVER_GEN_DIR`，输出 `gen_{task_id}.png/.txt`。
- [ ] 运行 MathCanvas 推理批量生成结果，抽检若干样本确保命名与内容符合评测约定。
- [ ] 配置并执行 ROVER 官方评测脚本（如 `python evaluate_rover.py --output_dir rover_results`），生成 `rover_metrics.jsonl`/`rover_summary.json`。
- [ ] 汇总得分与异常日志，记录可复现命令、资源占用和后续优化点。

# Qwen3-VL 旋转基准 API 推理计划（2025-11-25）

- 标题：使用 Qwen3-VL 8B/32B 云 API 跑 rotation bench 并沉淀脚本。
- [x] 了解 rotation 数据结构与图像路径规则（Task/Level/Image_id/question.png 等）。
- [x] 编写 API 推理脚本 `Qwen3-VL/evaluation/rotation_api_infer.py`（支持模型切换、限速、存 request/response）。
- [x] 编写运行脚本 `Qwen3-VL/evaluation/run_rotation_api.sh`（可配置 MODEL/OUTPUT_DIR/MAX_SAMPLES 等）。
- [ ] 试跑少量样本验证接口连通与输出格式，并记录耗时/费用建议。

# MathCanvas 旋转基准（缺失最终 step 图）计划（2025-11-25）

- 标题：提供带 step 文本/中间图但不含最终 step 图的推理脚本，让模型自行生成最终结果再作答。
- [x] 编写推理脚本 `scripts/inference/infer_rotation_mathcanvas_steps_nofinal_ddp.py`（过滤最后一步图像，仍保留步骤文本）。
- [x] 编写运行脚本 `scripts/inference/run_rotation_mathcanvas_steps_nofinal_ddp.sh`（2×GPU，指向本地权重与数据）。
- [ ] 小规模试跑并对比含最终图/不含最终图的输出差异，评估对答案准确率的影响。

# MathCanvas 3Dviews 推理计划（2025-11-28）

- [x] 新增 3Dviews 非图文交替推理脚本（无步骤/含步骤，步骤文本过滤 option/answer 关键词），默认 2 卡可跑。
- [x] 新增 3Dviews 图文交替数据预处理与运行脚本（base/steps，步骤文本同样过滤），指向本地权重。
- [ ] 小样本验证四种模式（非交替 base/steps、交替 base/steps）路径与输出格式，确认显存/耗时。
- [ ] 评估 3Dviews 四种模式的准确率，汇总日志与命令。

# Rotation 3D 重建+投影方案（2025-11-30）

- [x] 方案：用 Hunyuan3D-Shape v2.1 从 rotation 题干图生成 3D 网格，按 rotation steps 旋转后渲染多视角 2D 图，供后续 VLM 推理使用。
- [x] 实现：新增脚本 `scripts/inference/generate_rotation_hunyuan3d_projections.py`（默认 front/back/left/right/top/iso 渲染，支持应用 Rotation_steps）。
- [ ] 验证：小样本跑通（需安装 trimesh/pyrender/PyOpenGL，设置 `PYOPENGL_PLATFORM=egl` 如无显示），确认生成的 glb 和投影图质量/角度。
- [ ] 融合：将渲染图接入 rotation 推理/评测流程（选择投影视角或组合）并对比精度提升效果。
