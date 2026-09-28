# 因果在线排序压缩：E2/E3 开发流诊断

**状态与范围。** 主方法是一次因果的 R4→R2 排序压缩及候选门控，并非已证实的双向动态项数算法。只使用 `main_dev_round1` 的 E2/E3 各 5 条开发输入和同目录固定项数开发扫；未读取 select/confirm。控制器数值代码冻结为 [`online_selector.py`](online_selector.py)，SHA256 `60F6FA3F0AFE5B9FFC03EF4B610BC446EFD7A9E5ED391D621FCE6F9B1CC78846`。无文件读取的 `simulate(x,d,v,Om,ph,A0,B0,mu,config)` 可直接接入新随机流；本报告所有性能结论仍仅来自旧开发流。

## 方法及因果时序

原 500 维稠密 RFF、`S=[0,0,1,.5]`、25×20 Kronecker 因子及 v10 顺序 B→A MCC 更新均不变。E2 的 μ=.1、E3 的 μ=.05，是先前五档步长开发扫中固定 R4 对本组 5×4 段平均 ANR 的最优值；在线方法与固定 R4 用同一 μ。没有使用任何已知段界。每个采样点先用更新前因子产生主/候选/退出分支执行器输出，再通过真实执行器历史与次级 FIR 形成物理误差 `e=d-S*y+v`，然后由可观测的 `dv=e+S*y` 更新各分支。这与精确次级模型下的 `d+v` 数值等价；`d` 单独只用于离线 ANR 分母。

在预定 n=20,000，用当前 R4 权重 `w=vec(B Aᵀ)` 的**有符号稳定排序** `p=argsort(w)` 重排全部特征；即使不截断，`z[p]·w[p]=z·w`。将重排后的 20×25 矩阵用有乘法计数的 MGS2 QR 和一侧 Jacobi SVD 分解，取最佳 R2 截断。若相对 Frobenius 尾误差超过 .15，或 Jacobi 在 32 轮内不收敛，则拒绝并计入已发生的分解成本；绝不回退到未计数的库 SVD。通过谱筛后，在未来 2,000 点训练候选、随后 2,000 点比较削顶绝对物理残差；候选/执行分支残差比不高于 `10^(.5/20)` 才接受。接受后的 100 点将原始执行器输出从旧 R4 线性渐变到新 R2，两个分支均持续更新；真实 FIR 只过滤混合后的执行器序列。渐变终点用真实执行器最近四点重新对齐新分支私有历史，并核对之后三点。

`once` 臂在这次提案后不再尝试增长或剪枝。`gated_growth` 臂另用已观测物理残差的快/慢稳健 EWMA 报警；仅在 R2 后提出 R4，要求未来验证窗至少改善 .5 dB，拒绝后 100,000 点不再提出。它保留作双向构想的开发消融；本数据上没有一次增长被接受。

## 同口径结果

判据逐段同时要求相对同 μ 固定 R4 的末 5,000 点 ANR 差 ≤+0.5 dB，及整段平均实乘法 ≤0.9×固定 R4。所有 10 条运行、40 段均通过；每条一次压缩候选在 n=24,000 接受，随后 n=24,001–24,100 平滑切换。

| 开发臂 | 合格段 | 最坏 ANR 差 | 最坏逐段成本比 | 全程平均成本比 | 接受剪枝 / 接受增长 / 拒绝提案 |
|---|---:|---:|---:|---:|---:|
| 一次排序压缩 | 40/40 | +0.4549 dB | 0.8767 | 约 0.8399 | 10 / 0 / 0 |
| 报警增长消融 | 40/40 | +0.4549 dB | 0.8940 | 约 0.8572 | 10 / 0 / 40 |

E2/E3 一次压缩臂的最坏段差分别为 +0.395/+0.455 dB。报警增长没有质量收益，平均成本却增加，故**不把它称为已验证的动态增减项方法**。每条开发运行的全部提案、验证比、是否接收、截断误差、QR/Jacobi 乘法及每段 ANR/成本记录在相应 `*_summary.json`；每采样点分项账本和真实/各分支物理轨迹记录在 `*_trace.npz`。汇总见 [`audit_results.json`](audit_results.json) 与 [`comparison.json`](comparison.json)。

固定 R4 成本为 19,132 次实乘/点。主方法接受后（含 12 次/点管理）为 15,834 次/点；整条平均约 16,068 次/点，其中候选并行训练/验证平均 33.22、100 点退出分支 1.658、管理 12.06、排序分解 0.80 次/点，均已摊入。候选和退出期间额外的私有次级 FIR、真实混合 FIR、渐变乘法、拒绝提案亦计入。R2 常驻 90 个因子系数，最大并行峰值 270；固定 R4 是 180。一次臂平均影响函数指数运算 2.0205 次/点，固定 R4 为 2 次/点。排序比较与特征索引搬运在乘法口径为零，仍会造成真实时间/存储开销；报告同时保留约 15–20 秒/40 万点的 Python 墙钟时间，不能把乘法比例当作硬件加速比。

固定 R1–R8 在五档 μ 中分别按本组 5×4 开发平均 ANR 选一个 μ。相对调优 R4，固定 R1、R2、R3 的最坏段差在 E2 分别为 +5.385、+2.879、+1.509 dB，在 E3 为 +5.131、+2.604、+1.104 dB；固定 R3 成本已达 0.913×R4。故固定低项数不支配在线排序压缩。**但原始 full500 MCC 是重要反例：**即使仅使用原默认 μ，成本为 14,508 次实乘/点（0.758×R4）、系数 500，E2/E3 每一段 ANR 都优于本方法；本方法对它的平均差分别为 +2.83/+2.76 dB。可辩护的收益是相对固定 R4 的项数/因子存储压缩及其质量保持，不能声称相对 full500 的乘法或 ANR 优势。

## 门控负对照与边界

同样在 20k 当前 R4 状态，对 E2/E3 各 5 条开发流做恒等与固定随机排列提案。[`negative_controls.json`](negative_controls.json) 记录：10 个 signed-sort 候选均进入物理候选验证，R2 尾误差 0.0269–0.0724；10 个恒等排列在有界 Jacobi 内不收敛，显式拒绝（仅作事后核查的库 SVD 尾误差 0.352–0.617）；10 个固定随机排列被 .15 谱阈值拒绝，尾误差 0.748–0.857。只在事后核查中把阈值变为 .10、.20、.30，三组的分离结果不变；未从确认流选择阈值。

**仅靠短窗物理门控不够。** 在 E2 run00 禁用谱筛的可复现实验中，随机排列 R2 的训练后验证比仍为 1.0027，错误地被接受，长期最坏段差 +2.970 dB；证据保留为 `E2_run00_tuned_once_fixed_random_no_spectral_veto_summary.json` 和对应轨迹。谱筛是本门控可信度的必要组成，不代表 .15 保证未来 ANC 质量。当前开发数据没有一个通过谱筛而长期显著失败的 signed-sort 候选；负对照只能检验明显坏的排列/分解，不能证明物理短窗能识别所有细微失败。

## 独立核查与重现

`audit.py --arm both` 对 20 条臂轨迹重新求成本和末窗 ANR，并核对各提案时间、候选 4,000 点、100 点渐变双分支、渐变后三点私有/真实 FIR 对齐、初始 20k 与 v10 固定 R4 的逐点重放。真实 FIR 最大绝对误差 `7.55e-15`，重算 ANR 最大误差 `6.57e-6` dB。`fixed_bridge_smoke.py` 在 C1–C4 各 4,000 点上验证关闭结构管理的同一 `simulate` 与 v10 固定 R4 的执行器和物理误差逐点最大差 `<5e-15`，且账本恰为 19,132 次/点；`causality_test.py` 用相同前 23,000 点、不同后 3,000 点，证实前缀输出、成本和事件完全一致。结果见 [`fixed_bridge_smoke.json`](fixed_bridge_smoke.json)、[`causality_test.json`](causality_test.json)。

从项目根目录执行，Python 路径为 `C:\Users\nxtrol\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`：

```powershell
$py = 'C:\Users\nxtrol\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
foreach ($case in 'E2','E3') { foreach ($arm in 'once','gated_growth') { foreach ($run in 0..4) { & $py -B experiments_v13/online_selector/online_selector.py --case $case --run $run --arm $arm } } }
& $py -B experiments_v13/online_selector/negative_controls.py
& $py -B experiments_v13/online_selector/audit.py --arm both
& $py -B experiments_v13/online_selector/comparison.py
& $py -B experiments_v13/online_selector/fixed_bridge_smoke.py
& $py -B experiments_v13/online_selector/causality_test.py
```

下一步只能在新的独立开发/选择随机流上按冻结配置测试泛化。E1 的真实增长及重排回退需要另外设计和验证；当前 E2/E3 的 40 次被拒增长提案不构成成功的双向机制证据。
