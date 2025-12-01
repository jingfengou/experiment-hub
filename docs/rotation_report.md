# Rotation Benchmark Summary

Dataset: `/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json`

## MathCanvas / BAGEL-Canvas
| Mode / Pipeline          | Total | Correct | Missing_pred | Accuracy | Output Path |
|--------------------------|-------|---------|--------------|----------|-------------|
| base / non-interleave    | 1000  | 307     | 0            | 0.3070   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_ddp |
| steps / non-interleave   | 1000  | 524     | 0            | 0.5240   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_ddp |
| steps_nofinal / non-int  | 1000  | 271     | 0            | 0.2710   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_nofinal_ddp |
| base / interleave        | 1000  | 242     | 2            | 0.2420   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_interleave_base/rotation_base_default |
| nofinal / interleave     | 1000  | 258     | 2            | 0.2580   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_interleave_nofinal/rotation_nofinal_default |

Prompt (all modes share文本；nofinal 移除最后一步图像):
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

## Qwen3-VL (Official Offline, 8B)
| Mode     | Total | Correct | Missing_pred | Accuracy | Output Path |
|----------|-------|---------|--------------|----------|-------------|
| base     | 1000  | 316     | 0            | 0.3160   | Qwen3-VL/outputs/qwen3vl8b_rotation_base_official |
| nofinal  | 1000  | 220     | 0            | 0.2200   | Qwen3-VL/outputs/qwen3vl8b_rotation_nofinal_official |

Prompt (nofinal 追加步骤文本):
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
Answer:
```

## Qwen3-VL (API)
| Model / Mode                          | Total | Correct | Missing_pred | Accuracy | Output Path |
|---------------------------------------|-------|---------|--------------|----------|-------------|
| qwen3-vl-235b base                    | 1000  | 330     | 1            | 0.3300   | Qwen3-VL/outputs/rotation_api_qwen3vl235b_base |
| qwen3-vl-plus nofinal-steps           | 1000  | 20      | 907          | 0.0200   | Qwen3-VL/outputs/rotation_api_qwen3vl_plus_nofinal_steps |

Prompt (统一与 MathCanvas，nofinal 模式额外附步骤图/文):
```
You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer.
The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags,
respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.
<image>

Question: {Question}
A) {optA}
B) {optB}
C) {optC}
D) {optD}
Answer:
```

Notes:
- qwen3-vl-plus run缺 907 个 response.json，导致低准确率；需重跑 API 请求以填补缺失。否则可参考 235B 结果作为 API 基线。
