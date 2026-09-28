# v10：固定 Kronecker 项数实验

这是[当前阶段稿](../稿件/当前阶段稿/给老师_阶段性初稿.md)所用的固定结构实验。`anc_core.py` 是算法实现；`run_experiments.py` 生成实验数据；`analyze.py` 从已有数据生成统计与图；`test_math.py` 检查数值关系。

| 路径 | 内容 |
| --- | --- |
| `data/` | 各工况逐次 ANR 曲线及元数据（NPZ） |
| `results/` | 逐次指标和汇总统计（CSV/TXT） |
| `figures/` | 根据结果绘制的图（PNG/PDF） |
| `logs/` | 历史运行和测试记录 |

从仓库根目录可运行 `python -B experiments_v10/test_math.py`。它会更新 `logs/test_math.txt`。历史数据对应 `main`（E1–E4）、`ksweep`（E5）、`rsweep`（E6）、`ablation`（E7）和 `init`（E8）；脚本也有 `smoke` 模式。参数与输出文件名以 `run_experiments.py` 为准。

**重跑前先备份。**`run_experiments.py` 会写入 `data/`、`results/` 和 `logs/` 的固定路径；`analyze.py` 会重写汇总和图。不要在原目录直接重跑并把新文件当成历史结果。新机制应放在独立实验目录，用新的输出目录和协议记录，见[新增实验指南](../docs/新增实验指南.md)。
