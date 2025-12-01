# 3Dviews Benchmark Accuracy (MathCanvas/BAGEL-Canvas)

Dataset: `/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/3Dviews/thinking_chain_dataset/dataset.json`

| Mode        | Pipeline       | Total | Correct | Missing_pred | Accuracy | Output Path |
|-------------|----------------|-------|---------|--------------|----------|-------------|
| base        | non-interleave | 1000  | 237     | 0            | 0.2370   | outputs/3dviews_mathcanvas_ddp |
| steps       | non-interleave | 1000  | 323     | 0            | 0.3230   | outputs/3dviews_mathcanvas_steps_ddp |
| base        | interleave     | 1000  | 237     | 0            | 0.2370   | outputs/3dviews_interleave_base/3dviews_base_default |
| steps       | interleave     | 1000  | 298     | 0            | 0.2980   | outputs/3dviews_interleave_steps/3dviews_steps_default |

Prompt (base, non-interleave & interleave):
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

Prompt (steps mode adds sanitized hints):
```
...same as above...
Step hints (sanitized, options hidden):
- Step 1: {cleaned_thought}
- Step 2: ...

Answer:
```
