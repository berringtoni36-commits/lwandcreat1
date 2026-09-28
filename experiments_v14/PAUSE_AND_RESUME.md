# N14 暂停点与续做计划（2026-09-28，Asia/Shanghai）

本项目按用户要求暂停计算并上传**当前代码和计划**。这里保存的是研究中间状态，绝不把开发结果或未跑完的选择集当作论文结论。Git 提交只是可恢复检查点，不表示验证完成。

## 当前结论与明确边界

- 旧 v12 双向增减项数方案未同时满足质量和完整成本门槛；v13/P13 的固定项数容量闸门亦未通过。保留失败记录，不改写历史。
- N14 改为一个较窄、已在开发数据上实现的机制：原始 500 维 RFF 的 R4 因子控制器先经当前系数稳定带符号排序、计数 QR/Jacobi 秩 2 截断及未来虚拟次级残差验收，完成一次 R4→R2；之后每次成功切换经过固定 50,000 点，再提出 **R2→R2 坐标重排**。后者不是继续降项，也没有可靠增长。
- N14 **新开发流** E2/E3 各 5 次、共 40 个分段，周期重排的末窗 ANR 全部优于同输入固定 R4（最小优势 0.420 dB），完整标量乘法成本约为固定 R4 的 0.852。等项数因子重平衡消融未复现收益。十条独立 Python 重放、相同前缀不同未来的因果性测试和 E2/E3 各一条完整隐藏 MATLAB 重演已通过。[开发报告](resort_diagnostic/DEVELOPMENT_REPORT.md)、[独立审计](dual_rate/REPORT.md)、[MATLAB 复核](matlab_replay/REPORT_RESORT.md)。
- E1 的已知真实秩 `1→4→1` 开发实验是严重反例：周期 R2 在高秩与返回低秩两段相对固定 R4 退化约 24.4/23.5 dB。当前方法**不能**写成通用双向自适应项数或结构变化安全算法。完整 500 维 RFF-MCC 在 E2/E3 开发数据上仍比 N14 的 ANR 更好且标量乘法更少，不能写“优于所有 RFF 方法”。[定向查新](prior_art/REPORT.md)还发现 RFF+鲁棒 ANC、Kronecker+MCC+ANC、虚拟次级通道候选选择及排序促张量化均有先例；不能声称这些基本组合首次提出。
- 控制因子用 MCC 影响函数更新；候选接受准则是削顶的**模型推算虚拟次级残差绝对值**。候选并未在验证期实际驱动扬声器。论文必须准确区分。

## 正式选择集冻结与暂停进度

正式选择集是独立于上述开发流的 E2/E3 各 8 次配对运行，固定 R1–R8/full500 与周期方法均扫描五档 `μ={0.05,0.1,0.2,0.4,0.8}`。唯一方案、成本/ANR 口径、所有停止规则、源码和协议已在[冻结网格](selection/freeze_candidate_grid.json)及[独立封签](selection/freeze_selection_seal.json)锁定：

| 冻结对象 | SHA-256 |
|---|---|
| `freeze_candidate_grid.json` | `e3a71d920896126ffd6c4b02d08f503893ed5e5677def3f70029291084d84d83` |
| `freeze_selection_seal.json` | `7ca013d29c16b6425e64780f6f747a817606ff20ff7e643e64434b1c3e3bdd55` |
| 周期方案 `selector_resort.py` | `3be39ad3665aeab96e9f068135f9cab9b04fbafda960421682be398fd4e3f356` |

在用户要求暂停时，已停止全部选择进程，确认没有后台 `selection_runner.py`。本机结果目录为 `experiments_v14/outputs/n14_select_e3a71d920896126f/`：

| 状态 | 已完成/待续的配对工况运行 |
|---|---|
| 完整且保存 `result.json` | E2 run00、01、02、05、06；E3 run00、01、04、05；共 **9/16** |
| 中途停止、部分五档动态结果已保存 | E2 run03、07；E3 run02、06；共 **4/16** |
| 尚未开始 | E2 run04；E3 run03、07；共 **3/16** |

四个中断运行的 `attempts` 目录有 `.start.json` 而无对应 `.finish.json`；这是用户主动暂停，不是假装完整运行。已写入的原子结果可由同一冻结模型核查、恢复；没有 `.tmp` 文件。**尚未运行 800 次评价的完整选择分析，没有确定动态步长，也没有生成任何确认集随机流或论文正式主结果。** 不查看部分选择结果调参，保留原定门槛。

为了这次能及时上传，Git 保存源码、协议、封签、测试、小型 JSON/文本报告和图表；本机约 9.3 GiB 的新 `.npz` 逐点轨迹和约 0.1 GiB 的 MATLAB `.mat` 中间件**不随此次代码检查点上传**，也不会从本机删除。仓库的[数据清单](DATA_AVAILABILITY.md)列明可从冻结种子/源码重建的原始工况、尚只在本机保存的中断现场，以及不能把仓库本身误读成已包含全部数值数组的限制。新机器需要从头重跑选择集；同一机器可续跑七个未完成的工况运行。

## 恢复命令与停止规则

在**同一工作机**，先核对 `git rev-parse HEAD` 为本次上传提交、工作树源码与冻结网格 SHA 对应，安装/确认 NumPy 与 Numba，再执行：

```powershell
$py = 'C:\Users\nxtrol\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -B experiments_v14/selection/check_freeze.py --freeze experiments_v14/selection/freeze_candidate_grid.json
& $py -B experiments_v14/protocol/test_generator.py
& $py -B experiments_v14/selection/test_selection.py
& $py -B experiments_v14/resort_diagnostic/test_resort.py
foreach ($spec in @(@('E2',3),@('E2',4),@('E2',7),@('E3',2),@('E3',3),@('E3',6),@('E3',7))) {
  & $py -B experiments_v14/selection/selection_runner.py --freeze experiments_v14/selection/freeze_candidate_grid.json --unlock-select --case $spec[0] --run $spec[1]
  if ($LASTEXITCODE -ne 0) { throw "Selection run failed: $($spec[0]) $($spec[1])" }
}
& $py -B experiments_v14/selection/selection_analysis.py --freeze experiments_v14/selection/freeze_candidate_grid.json --unlock-select
```

在**新机器或无本机原始轨迹**时，同一冻结网格下将 E2/E3 的 run00–07 全部重新运行，再执行分析；不能仅复制小 JSON 摘要冒充完整恢复。选择分析逐文件验证 SHA 和 45 固定+5 动态评价。冻结的唯一门槛是：25 个配对步长组合中至少一个在 ≥7/8 配对运行同时满足四段相对固定 R4 的 ANR 退化≤+0.5 dB、两者各段 ANR≤−6 dB、每工况总在线成本/R4≤0.9、80k 前真实完成 R4→R2 并保持 R2、物理和账本完整性；固定 R4 的绝对底线亦须≥7/8 配对通过。完整失败组合也保存。

若选择**通过**：将唯一 `(μ_E2,μ_E3)`、全部固定方法的选择步长、选择结果 SHA、确认算法/分析器/环境哈希写入新的不可变确认封签，之后首次生成 E2/E3 各 20 对 `confirm` 流；主检验为 20 对中 `K≥17` 全条件共同通过，当前 N14 alpha 分配为 `0.05/12`。再运行 E1 与 C1–C4 各 20 次描述/固定模式桥接、MATLAB 正式轨迹复核、消融、耗时/内存与图表，最后按[论文整合计划](../experiments_v12/PAPER_INTEGRATION.md)更新原稿并编译 PDF。`confirmation/` 中的骨架**尚未对最后一次改动复测**，只能作为续写起点，不可直接解锁确认。

若选择**不通过**：原样保存完整结果与失败原因，**不生成 N14 确认数据**。按性能/成本/结构事件分解原因；任何修改算法、阈值、输入或统计定义都要另立新假设与新的选择/确认随机流，不能在已看的 N14 选择数据上继续筛到成功。若未来仍要求原目标的真实双向增长和剪枝，必须设计并独立验证恢复到高容量的机制；本轮 E1 已明确暴露当前不足。

## 下一轮论文与 Git 工作

先根据完整独立结果决定能写的主张，再把固定结构原稿的系统模型、顺序 MCC 更新保留，新增周期重排的置换等价性、R4→R2 截断、候选虚拟残差及完整标量乘法账本。原固定结构后验残差性质不能延伸成切换时的稳定性定理。比较图至少包含四段 ANR/结构事件、R1–R8/full500 前沿、候选与重排成本、同 R 重参数化消融、E1 迁移限制。论文不能写成已经验证通用自适应项数。

用户已授权将本次暂停检查点上传 GitHub；未来完成确认与论文整合后，应另作最终提交并推送。保留本次暂停后未完成的任务，不因上传完成而宣布实验完成。
