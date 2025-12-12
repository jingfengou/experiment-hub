# Qwen3-VL-8B-Instruct Rotation Benchmark Accuracy

| Mode     | Total | Correct | Missing_pred | Accuracy | Output Path |
|----------|-------|---------|--------------|----------|-------------|
| base     | 1000  | 316     | 0            | 0.3160   | outputs/qwen3vl8b_rotation_base_official |
| nofinal  | 1000  | 220     | 0            | 0.2200   | outputs/qwen3vl8b_rotation_nofinal_official |

- 数据集：`/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json`
- 解析：优先 `<answer>…</answer>`，否则匹配含 “answer/option/choice” 等提示的 A-D，或末行单独大写 A-D，最后再取文本末尾的大写独立 A-D（避免将量词 a 判为答案）。
- 模式含义：base=只有题干图；nofinal=题干+步骤图+步骤文本但移除最后一步图文，让模型补完。

## Prompt（base）
```
You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer.
The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags,
respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.
Keep the reasoning concise (<=3 sentences) and always output both tags even if unsure; pick the most likely option.
<image>
Question: {Question}
A) {choice1}
B) {choice2}
C) {choice3}
D) {choice4}
Answer:
```

## Prompt（nofinal）
- 文本同上，追加步骤描述：
```
Step 1: {description_1}
Step 2: {description_2}
...
Answer:
```
