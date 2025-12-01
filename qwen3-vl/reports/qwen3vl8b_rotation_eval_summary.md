# Qwen3-VL-8B-Instruct Rotation Benchmark Accuracy

| Mode     | Total | Correct | Missing_pred | Accuracy | Output Path |
|----------|-------|---------|--------------|----------|-------------|
| base     | 1000  | 316     | 0            | 0.3160   | outputs/qwen3vl8b_rotation_base_official |
| nofinal  | 1000  | 220     | 0            | 0.2200   | outputs/qwen3vl8b_rotation_nofinal_official |

- 数据集：`/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json`
- 解析：优先 `<answer>…</answer>`，否则匹配含 “answer/option/choice” 等提示的 A-D，或末行单独大写 A-D，最后再取文本末尾的大写独立 A-D（避免将量词 a 判为答案）。
