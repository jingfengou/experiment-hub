# Rotation Benchmark Accuracy (MathCanvas/BAGEL-Canvas)

| Mode                     | Pipeline       | Total | Correct | Missing_pred | Accuracy | Output Path |
|--------------------------|----------------|-------|---------|--------------|----------|-------------|
| base                     | non-interleave | 1000  | 307     | 0            | 0.3070   | outputs/rotation_mathcanvas_ddp |
| steps (all step images)  | non-interleave | 1000  | 524     | 0            | 0.5240   | outputs/rotation_mathcanvas_steps_ddp |
| steps_nofinal            | non-interleave | 1000  | 271     | 0            | 0.2710   | outputs/rotation_mathcanvas_steps_nofinal_ddp |
| base                     | interleave     | 1000  | 242     | 2            | 0.2420   | outputs/rotation_mathcanvas_interleave_base/rotation_base_default |
| nofinal                  | interleave     | 1000  | 258     | 2            | 0.2580   | outputs/rotation_mathcanvas_interleave_nofinal/rotation_nofinal_default |
| base (MoT, 10 iters)     | interleave     | 1000  | 258     | 0            | 0.2580   | outputs/rotation_mathcanvas_interleave_base_mot/rotation_base_mot_default |

- 评估基于数据集 `/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json`。
- 解析策略：优先 `<answer>…</answer>`，否则匹配含 answer/option/choice 等提示的 A-D，或末行单独大写 A-D，最后取文本末尾独立的大写 A-D（避免将量词 a 误判为答案）。
- MoT 交错跑的是默认 `max_iterations=10`（未提前终止），因此每样本存 10 段文本推理后再取答案。
- 本次 MoT 交错运行未生成中间图（action token 未触发 image 分支），样本目录仅有题干原图与 reasoning_result.json，如需中间图需修正 action 映射/结束条件后重跑。
- 模式含义：base=只有题干图；steps=题干+全部步骤图+步骤文本；steps_nofinal=题干+步骤图+步骤文本但移除最后一步图文，让模型补完；interleave 表示图文交错喂入推理（非交错即一次性提供）。

## Prompt（base / steps / steps_nofinal 非交替）
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

## Prompt（interleave base/nofinal，经 `mathcanvas_interleave_reasoner.py`，nofinal 去掉最后一步图像）
- 文本部分同上；nofinal 模式在数据准备时移除最后一步图像，保留前序步骤图/文本。
