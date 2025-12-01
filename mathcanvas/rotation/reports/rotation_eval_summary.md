# Rotation Benchmark Accuracy (MathCanvas/BAGEL-Canvas)

| Mode                     | Pipeline       | Total | Correct | Missing_pred | Accuracy | Output Path |
|--------------------------|----------------|-------|---------|--------------|----------|-------------|
| base                     | non-interleave | 1000  | 307     | 0            | 0.3070   | outputs/rotation_mathcanvas_ddp |
| steps (all step images)  | non-interleave | 1000  | 524     | 0            | 0.5240   | outputs/rotation_mathcanvas_steps_ddp |
| steps_nofinal            | non-interleave | 1000  | 271     | 0            | 0.2710   | outputs/rotation_mathcanvas_steps_nofinal_ddp |
| base                     | interleave     | 1000  | 242     | 2            | 0.2420   | outputs/rotation_mathcanvas_interleave_base/rotation_base_default |
| nofinal                  | interleave     | 1000  | 258     | 2            | 0.2580   | outputs/rotation_mathcanvas_interleave_nofinal/rotation_nofinal_default |

- 评估基于数据集 `/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json`。
- 解析策略：优先 `<answer>…</answer>`，否则匹配含 answer/option/choice 等提示的 A-D，或末行单独大写 A-D，最后取文本末尾独立的大写 A-D（避免将量词 a 误判为答案）。

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
