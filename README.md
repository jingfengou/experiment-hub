# Experiment Hub

集中存放各模型/实验类型的脚本与评测结果，方便统一 git 管理。所有源码仍运行于原项目目录，本仓库保存可重复的脚本副本、报告和同步工具。

## 目录
- `mathcanvas/rotation`：MathCanvas/BAGEL-Canvas 旋转基准脚本与精度报告。
- `mathcanvas/3dviews`：MathCanvas 3Dviews 交错/非交错脚本。
- `mathcanvas/hunyuan3d`：旋转题目 3D 重建与多视角投影脚本。
- `qwen3-vl/rotation_api`：Qwen3-VL 云 API 旋转推理脚本。
- `qwen3-vl/rotation_official`：Qwen3-VL 官方模型离线推理与评测脚本。
- `qwen3-vl/reports`：Qwen3-VL 旋转基准评测摘要。
- `rover`：ROVER 数据转存、生成导出与官方评测脚本。
- `docs`：个人计划与工作总结副本。

## 使用方式
1. 确认原始路径（默认 `/workspace/oujingfeng/project/think_with_generated_images`）。若有改动，请编辑 `sync_from_sources.sh` 顶部的路径。
2. 运行同步脚本，把最新脚本/报告拷贝进本仓库：
   ```bash
   cd /workspace/oujingfeng/experiment-hub
   ./sync_from_sources.sh
   ```
3. 在本仓库正常使用 git 提交。需要更新时再次运行同步脚本。

## 同步范围
- MathCanvas：旋转/3Dviews 推理与数据准备脚本、Hunyuan3D 投影脚本、旋转评测摘要。
- Qwen3-VL：旋转 API、离线推理脚本与评测摘要。
- ROVER：数据预处理、生成导出、官方评测脚本与 schema 示例。
- 文档：`plan.md` 与 `work_summary.md` 副本。

如需扩展同步的文件，将其源路径添加到 `sync_from_sources.sh` 即可。
