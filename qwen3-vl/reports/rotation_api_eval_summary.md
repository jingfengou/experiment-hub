# Rotation Benchmark Accuracy (Qwen3-VL API)

Dataset: `/workspace/oujingfeng/project/think_with_generated_images/datasets/mydatasets/dataset/data_modified_with_subject.json`

| Model/Mode                          | Total | Correct | Missing_pred | Accuracy | Output Path |
|-------------------------------------|-------|---------|--------------|----------|-------------|
| qwen3-vl-plus (nofinal steps input) | 1000  | 20      | 907          | 0.0200   | outputs/rotation_api_qwen3vl_plus_nofinal_steps |
| qwen3-vl-235b (base)                | 1000  | 330     | 1            | 0.3300   | outputs/rotation_api_qwen3vl235b_base |

Notes:
- qwen3-vl-plus run缺少 907 个 response.json，导致大量 missing_pred。
- 模式含义：base=只有题干图；nofinal steps=题干+步骤图+步骤文本但移除最后一步图文，让模型补完。

Prompt (base, `rotation_api_infer.py`):
```
{Question}
Choices:
A. {optA}
B. {optB}
C. {optC}
D. {optD}
请根据图像与选项，直接给出正确选项字母，并用一两句话说明理由。
```

Prompt (nofinal steps, `rotation_api_infer_nofinal_steps.py` adds step image/text):
```
{Question}
Choices:
A. ...
B. ...
C. ...
D. ...
提示：先绕 Z 轴旋转 180 度后的中间状态如附图所示，再绕 X 轴旋转 180 度，判断最终对应的选项。
请根据图像与选项，直接给出正确选项字母，并用一两句话说明理由。
```
