# Kronecker 结构化 RFF 非线性主动降噪研究

本仓库保存论文、可复现实验和后续方法原型。**当前阶段稿依据 v10 固定项数实验；v11 是尚在开发的在线项数选择原型。**两套结果不能混用。

## 从哪里开始

| 想做什么 | 入口 | 状态 |
| --- | --- | --- |
| 阅读当前论文 | [阶段性初稿](稿件/当前阶段稿/给老师_阶段性初稿.md) · [PDF](稿件/当前阶段稿/Overleaf_阶段性初稿_编译版.pdf) · [LaTeX](稿件/当前阶段稿/overleaf_stage_update/main_stage_draft.tex) | 对应 v10 |
| 查固定项数实验 | [v10 说明](experiments_v10/README.md) | 当前阶段稿的实验依据 |
| 开发动态项数方法 | [v11 方法与运行说明](experiments_v11/README.md) · [开发记录](experiments_v11/开发记录_20260928.md) · [运行结果索引](experiments_v11/outputs/README.md) | 开发原型，尚无正式优越性结论 |
| 计划新实验或创新点 | [新增实验指南](docs/新增实验指南.md) · [下一步工作](NEXT_STEPS.md) | 先保存独立方案与结果 |
| 查原始 MATLAB、旧稿和审稿记录 | [历史材料说明](archive/README.md) · [稿件](稿件/) · [审稿与工作记录](审稿与工作记录/) | 仅用于追溯 |

## 仓库结构

```text
experiments_v10/    固定项数基线：代码、逐次数据、统计、图和日志
experiments_v11/    动态项数原型：代码、测试、开发记录和独立运行目录
稿件/               当前阶段稿及 v09 历史稿
审稿与工作记录/     审稿清单和工作记录
archive/            原 MATLAB 脚本与 v09 PDF 图形检查工具
docs/               后续实验的目录、记录和复现约定
```

## 快速检查

从仓库根目录运行，使用 Python 3.12：

```shell
python -m pip install -r requirements.txt
python -B experiments_v11/test_adaptive.py
python -B experiments_v11/run_adaptive.py --profile smoke
```

v11 每次运行会建立带时间戳的输出目录，并保存协议、随机信息、实际输入、逐次数据及源码快照。`smoke` 是流程检查，`pilot` 是开发试验；[结果索引](experiments_v11/outputs/README.md)标明了各次运行的用途。v10 的实验脚本和数值测试会写入固定文件名，重做前先按 [v10 说明](experiments_v10/README.md)备份历史结果。

## 当前结论与后续更新

v11 已通过 12 项正确性测试，完成 3 工况 × 3 次开发试验。部分工况接受增长，本轮未接受剪枝；计入搜索开销后，总成本尚未显示一致优势。具体数值和边界见 [pilot 报告](experiments_v11/outputs/20260928_110901_119741_pilot/REPORT.md)。

新增方法时，保留 v10、v11 的历史代码和结果，另建清晰的实验入口与独立输出；记录对照组、随机种子、参数、源码版本和成本口径。具体格式见 [新增实验指南](docs/新增实验指南.md)。原始数据、图和 PDF 随仓库版本管理；本机依赖、缓存与凭据由 `.gitignore` 排除。
