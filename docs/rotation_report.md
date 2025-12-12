# Rotation Benchmark Summary

Dataset: `/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json`

模式速记：base=只有题干图；steps=题干+全部步骤图+步骤文本；steps_nofinal=题干+步骤图+步骤文本但移除最后一步图文让模型补完；interleave 表示图文交错喂入（非交错一次性提供）。

MoT 交错运行说明：默认 `max_iterations=10` 未提前终止，action token 未触发 image 分支，输出目录仅含题干原图与 reasoning_result.json（无中间生成图）；如需中间图需修正 action 映射/结束条件后重跑。

## MathCanvas / BAGEL-Canvas
| Mode / Pipeline          | Total | Correct | Missing_pred | Accuracy | Output Path |
|--------------------------|-------|---------|--------------|----------|-------------|
| base / non-interleave    | 1000  | 307     | 0            | 0.3070   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_ddp |
| steps / non-interleave   | 1000  | 524     | 0            | 0.5240   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_ddp |
| steps_nofinal / non-int  | 1000  | 271     | 0            | 0.2710   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_nofinal_ddp |
| base / interleave        | 1000  | 242     | 2            | 0.2420   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_interleave_base/rotation_base_default |
| nofinal / interleave     | 1000  | 258     | 2            | 0.2580   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_interleave_nofinal/rotation_nofinal_default |
| base / interleave (MoT)  | 1000  | 258     | 0            | 0.2580   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_interleave_base_mot/rotation_base_mot_default |
| steps (Hunyuan, final off) | 1000 | 390   | 0 | 0.3900 | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_ddp_hy_nogen |
| steps_nofinal (Hunyuan, final on) | 1000 | 291 | 0 | 0.2910 | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_nofinal_ddp_hy |

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

Case study（MoT，base 交错，`rotation_base_mot_default`，均未生成中间图）：
- sample_0001：GT=D，Pred=B，长链推理后仍落到错误选项（多次重复思考未收敛）。
- sample_0003：GT=B，Pred=D，堆叠旋转描述混乱，最终随机落到 D。
- sample_0042：GT=B，Pred=D，错误地将 X/Y 轴两次 180° 视作 90° 旋转，导致镜像方向判定错误。
- sample_0100：GT=C，Pred=C，能按两次 180° 的组合保持三层结构的相对位置，给出正确选项。

Case study (rotation, Hunyuan steps_nofinal, final on)：
- sample00000: GT=D, Pred=C，尾句 `The final answer is \boxed{C}.`
  ![sample00000](MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_nofinal_ddp_hy/sample00000/images/gen_final.png)
- sample00001: GT=D, Pred=A，尾句 `Final answer: \boxed{A}`
  ![sample00001](MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_nofinal_ddp_hy/sample00001/images/gen_final.png)
- sample00002: GT=B, Pred=D，尾句 `...answer is \boxed{D}.`
  ![sample00002](MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_nofinal_ddp_hy/sample00002/images/gen_final.png)
- sample00003: GT=C, Pred=C，尾句 `...correct choice is \boxed{C}.`
  ![sample00003](MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_nofinal_ddp_hy/sample00003/images/gen_final.png)
- sample00004: GT=D, Pred=D，尾句 `...answer is \boxed{D}.`
  ![sample00004](MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_nofinal_ddp_hy/sample00004/images/gen_final.png)

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
