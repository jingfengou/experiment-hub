# Rotation Benchmark with Hunyuan3D Step Images

Dataset: `MathCanvas/BAGEL-Canvas/data_handlers/rotation_hunyuan3d/data_modified_with_subject.json`

| Mode (Hunyuan steps)                 | Final image gen | Total | Correct | Accuracy | Output Path |
|--------------------------------------|-----------------|-------|---------|----------|-------------|
| steps (all steps, final gen **off**) | off             | 1000  | 390     | 0.3900   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_ddp_hy_nogen |
| steps_nofinal (remove last step img) | on              | 1000  | 291     | 0.2910   | MathCanvas/BAGEL-Canvas/outputs/rotation_mathcanvas_steps_nofinal_ddp_hy |

Prompt（steps 模式，含全部步骤；nofinal 则去掉最后一步描述/图）：
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
Step 1: {description_1}
Step 2: {description_2}
...
Answer:
```

Case studies（steps_nofinal_ddp_hy，前 5 条）：
- sample00000: GT=D，Pred=C，`The final answer is \boxed{C}.`
- sample00001: GT=D，Pred=A，`Final answer: \boxed{A}`
- sample00002: GT=B，Pred=D，`The final configuration corresponds to option D...`
- sample00003: GT=C，Pred=C，`The final orientation matches option C...`
- sample00004: GT=D，Pred=D，`The final configuration corresponds to option D...`
