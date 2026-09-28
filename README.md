# Kronecker 结构化 RFF 非线性主动降噪研究

论文、实验实现、原始数据及动态项数选择原型的统一版本库。

## 主要入口

| 内容 | 路径 |
|---|---|
| 当前阶段论文 | [给老师_阶段性初稿.md](稿件/当前阶段稿/给老师_阶段性初稿.md) |
| 当前论文 PDF | [Overleaf_阶段性初稿_编译版.pdf](稿件/当前阶段稿/Overleaf_阶段性初稿_编译版.pdf) |
| LaTeX 源 | [main_stage_draft.tex](稿件/当前阶段稿/overleaf_stage_update/main_stage_draft.tex) |
| 固定项数实验 | [experiments_v10](experiments_v10/) |
| 动态项数方法说明 | [experiments_v11/README.md](experiments_v11/README.md) |
| 动态方法开发记录 | [开发记录_20260928.md](experiments_v11/开发记录_20260928.md) |
| 动态方法试运行报告 | [pilot REPORT.md](experiments_v11/outputs/20260928_110901_119741_pilot/REPORT.md) |
| 下一步工作 | [NEXT_STEPS.md](NEXT_STEPS.md) |
| 旧版目录索引 | [README_目录索引.md](README_目录索引.md) |

## 当前进展

当前论文与 v10 的固定 Kronecker 项数实现对应。v11 已实现独立的增长/剪枝候选评估、顺序因子更新、物理通道验证和平滑切换。

v11 已通过 12 项正确性测试，并完成 3 工况 × 3 次独立开发试验。部分工况接受了增长，但本轮没有接受剪枝；包含搜索开销的总成本尚未显示一致优势。此处保留实际结果，不把原型写成已经验证的论文创新成果。

## 运行与复现

使用 Python 3.12 和 NumPy；绘图另需 Matplotlib。本次实验版本记录在每次输出的 `protocol.json` 中。仓库不包含本机安装的依赖。

```shell
python -m pip install -r requirements.txt
python -B experiments_v11/test_adaptive.py
python -B experiments_v11/run_adaptive.py --profile smoke
python -B experiments_v11/run_adaptive.py --profile pilot
```

每次新运行会创建时间戳目录，不覆盖历史数据。输出包含输入/噪声/特征映射、曲线、项数轨迹、成本、结构事件、协议与源代码快照。

原 v10 脚本及结果保留用于追溯；直接运行 v10 的原入口可能覆盖其同名输出，重做历史实验时请先另存结果目录。

## Git 同步

后续完成本项目工作后，提交本次范围内的修改并推送到本仓库，核对远程提交号。具体约定见 [AGENTS.md](AGENTS.md)。

原始实验数据、图和 PDF 一并追踪；依赖目录、缓存、系统文件和凭据文件不上传。`.gitattributes` 保留源文件原始换行字节，以维护实验协议中的 SHA-256 校验。
