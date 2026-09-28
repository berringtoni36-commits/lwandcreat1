# N14 正式选择阶段：已冻结，按用户要求暂停

唯一候选源为 [`selector_resort.py`](../resort_diagnostic/selector_resort.py)：初次因果 R4→R2 后，每 50000 点再提出一次带符号坐标重排并通过同样的谱和虚拟物理残差验证。新 dev 的 E2/E3 五运行共 40 段优于固定 R4、成本比约 0.8518；等 R 因子重参数化消融、独立十条重放及 E2/E3 各一条全长 MATLAB 复核已完成。开发结果不能作为正式结论。源码、协议和审计已由 `freeze_candidate_grid.json` 与独立封签冻结；首次选择运行按用户要求中断于 **9/16 完整、4/16 可恢复中断、3/16 未开始**。完整状态和续做命令见[暂停说明](../PAUSE_AND_RESUME.md)，原始数组只在本机保存。

## 冻结闸门

`selection_runner.py` 要求同时有显式 `--unlock-select`、完整的 `freeze_candidate_grid.json`，以及**分立的** `freeze_selection_seal.json`。在第一次派生 select 随机数前，它验证：N14/`select` 身份、E2/E3 各八次、四段各 100000、末窗 5000、五档 μ、唯一 `once + signed_sort + 谱 veto + 100 点渐变 + 每 50000 点周期重排` 配置、固定/动态选择规则、成本口径、协议和全部源码/依赖的实际 SHA-256。网格指向已审阅的[周期重排开发报告](../resort_diagnostic/DEVELOPMENT_REPORT.md)；封签精确绑定网格与该报告的 SHA，并记录 `proceed_to_select` 决定。任何源码/协议/审计后来改动会使检查失败；`confirm` 没有本目录入口。

只读检查使用 `python experiments_v14/selection/check_freeze.py --freeze experiments_v14/selection/freeze_candidate_grid.json`；它核对哈希/网格/封签但不生成数据。`--print-draft-template` 与 `--print-draft-seal-template` 仅供将来的新假设参考，不改变此次冻结。

## 一份配对选择运行如何保存

正式授权后，选择输入才由 `SeedSequence([20260928,14,2,case_id,run_id,component_id])` 直接生成；E2/E3 数据公式调用 N14 原生成器中的 `e2e3_primary` 和 α-stable 噪声函数，独立流组件编号仍是参考1、映射2、初值3、噪声4、教师5、结构41。每个工况/运行保存实际 `input.npz`、数组和文件 SHA。八列相同初值前缀同时供 R1–R8 和动态活跃 R4；full500 零初值。

审计过的 45 分支固定 Numba 核一次产生 R1–R8/full500 × 五档 μ 的四段 ANR 与 45 条执行器/物理误差轨迹；复用其逐点 FIR 舍入界校验与完整固定成本账本。纯 `selector_resort.simulate` 对唯一冻结的 periodic resort 策略按五档 μ 各跑一次，每档保存物理输出与误差、R/候选/退役历史、**初次降项和后续 R2→R2 重排**事件、逐点成本类别、四段 ANR、物理重演和 SHA。固定与动态逐文件原子写入；每一阶段完成后可跳过，孤立的轨迹文件或元数据可通过相同冻结模型重新计算并核对再恢复。任何旧文件与新结果不符都停止而不覆盖。16 份工况运行落于 `experiments_v14/outputs/n14_select_<grid_sha前16位>/caseE2/E3_run00..07/`；输出目录会复制实际冻结文件、已审阅开发审计和源码快照。

两个工况各收齐八次后，`selection_analysis.py` 才会核对 16 份文件 SHA 和全部 800 次评价：九种固定方法各以该工况的 `8×4` 平均 ANR 锁定**单一** μ；动态的五档 E2×五档 E3 组成 25 个配对联合 `J_i` 扫描，逐对要求四段相对/绝对 ANR、各工况 0.9 在线成本、实际且保留到终点的 80k 前 R4→R2、物理完整性。后续 R2→R2 重排必须保留在实际事件与在线成本账本，不能冒充新的降项。固定 R4 的两工况四段底线也须至少 7/8 配对运行通过。合格组合按两工况八运行平均成本比、最大逐段退化、较小两 μ 依次择一；不合格 25 组和失败原因完整保留。研究搜索额外报告 720 固定+80 动态评价的总计算及耗时；它与每次部署的**在线**成本账本分开。

本目录测试命令为 `python experiments_v14/selection/test_selection.py`。它用 N14 **dev 的 80 点烟雾数组**测试缺失冻结文件先拒绝、45 固定+5 动态的纯核返回、输入/固定/单档动态阶段的故障注入与恢复、以及 25 组合规则；另在 E2 dev run0 的 78110 点前缀逐事件检查首次降项与后续同 R 重排。烟雾文件位于临时目录并随测试删除。测试成功不等于独立选择通过。当前只在用户恢复实验时继续执行：

```text
python experiments_v14/selection/selection_runner.py --freeze experiments_v14/selection/freeze_candidate_grid.json --unlock-select --case E2 --run 3
python experiments_v14/selection/selection_analysis.py --freeze experiments_v14/selection/freeze_candidate_grid.json --unlock-select
```
