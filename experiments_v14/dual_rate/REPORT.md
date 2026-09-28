# N14 开发诊断：双步长、R3 容量与周期 R2 重排独立审计

本目录只读取 N14 **dev** 的 E2/E3 各 5 条实际输入和配对固定前沿；未读取 selection/confirm，未改冻结 60F6 核心、`resort_diagnostic/` 或 v10–v12，未提交 Git。结论是：R4→R2 后单独提高 R2 步长无可靠改善；R3 一直运行可补质量但超出 0.9 成本线。周期性 signed-sort R2→R2 机制在这 10 条开发流的 40 个末窗全部满足相对固定 R4 μ=.05 的 +0.5 dB 门槛，且计数成本均低于 0.9；该结果仍是开发证据，需冻结后由独立流验证。

## 双步长与 R3

冻结 one-shot R4 μ=.05 → R2 μ=.05 的逐运行最大分段差（dB）为 E2 `[+.774,+.379,+.254,+.504,+.589]`、E3 `[+.238,+.582,+.256,+.188,+.164]`，相对同输入固定 R4 μ=.05 分别 2/5、4/5 条全段通过。E2 run0 的第 3 段差随时间累积，不是 100 点切换瞬态；详细前后窗统计见 `diagnosis.json`。

只把剪枝后步长调至 μ=.1，保持 R4 预热 μ=.05、映射、候选流程与物理 FIR 不变，在全部 10 条新 dev 流的逐运行最大差为：

| 条件 | run00–04 最大分段差 dB | 全段通过 | 最大分段成本/R4 |
|---|---|---:|---:|
| E2，R2 μ=.1 | +.957, +.526, +.413, +.709, +.736 | 1/5 | .8766 |
| E3，R2 μ=.1 | +.375, +.873, +.398, +.322, +.399 | 4/5 | .8766 |

`controller.py` 的 `mu_after=.05` E2 run0 桥接与冻结 60F6 的 `drive/error/anr/R/cost_total` 全轨迹逐点差均为 0；同一次物理 FIR 重放误差 1.23e−15。双步长方向据此停止，没有继续调参。

另做有限的**容量上界诊断**：20k 一次 signed-sort R4→R3，后续始终 R3，μ=.05，针对 E2 run0/3/4 与 E3 run1。四条最大分段质量差依次 +.136/−.118/+.081/−.128 dB，但平均成本/R4 均约 .9220、第一段约 .9458。R3 稳态（含管理）的 17489/19132=.9141 已超过 .9；即使只在一段的高风险区用 R3，若为 4k 训练/验证候选额外计约 199 乘法/段样本，则 R3 占该段比例须低于约 71.6%。此处没有实现或验证能因果检测并及时回落的 R2→R3→R2 控制器，不能把上界视为合格在线方法。

## 周期 signed-sort 主方法：独立重放与因果审计

父实验源码 `../resort_diagnostic/selector_resort.py` 的 SHA256 为 `3be39ad3665aeab96e9f068135f9cab9b04fbafda960421682be398fd4e3f356`。它与冻结 60F6 仅有三处实质差别：读取 `resort_interval`、一次成功提案后安排下一次时点、容许 R2 时再提 R2 候选。物理执行、在线因子更新、谱阈值、候选验证与 ramp 代码未变。`run_resort.py` 仅把 `segment_bounds` 用于离线指标，没有传给 `simulate`。控制器当前样本先执行执行器、由真实次级 FIR 得到观测误差，再用 `e+S*y=d+v` 更新；代码对该等式逐样本断言。结构决策只依赖既往及当样本观测。计划时点均为 20/74/128/182/236/290/344/398k，系接受后固定 50k 时钟，不与 100k 段界绑定。前 7 个提案各用 2k 候选训练、2k 后续物理验证并接受，且有 100 点混合输出；398k 的末次候选到 400k 尚未决，仍计入训练与成本。

`audit_saved_resort.py` 核验 10 份输入 SHA 与 N14 dev manifest、固定前沿输入 SHA、模拟器 SHA、全部结构时点和保存的物理 FIR 上界。`audit_resort.py` 对 10 条各自重新运行 400k 点，断言四段末窗 ANR、事件字典与成本比和保存结果逐值一致。独立重放物理 FIR 最大绝对差 5.77e−14；ramp 后前 3 点内部/真实次级输出最大差为 0。10 次重放各约 15–21 秒（此为实测运行时间，不等于乘法计数）。

与**同输入固定 R4 μ=.05** 配对，40 个末窗的最差质量差为 **−.420064 dB**，即全部改善；全程乘法成本/R4 为 **.851743–.851773**。E2 run0 逐样本账本均值：主体 16016.608、候选 249.142、retiring 6.641、管理 12.455、QR/Jacobi 重排 6.703、真实次级 FIR 4，合计 16295.548 乘法/样本，对照 R4 的 19132。该条四段成本比分别 .883926/.842239/.842236/.838571。其余九条提案/接受时间、秩与候选历程相同，只有计数 QR/Jacobi 成本有差异；按保存事件计数精确推导，10 条的最大分段成本比 **.883955**。E3 run4 的逐样本重放交叉核查已加在 `resort_audit_E3_run04_signed_sort.json`。

这套成本是预先约定的**标量乘法账本**：它包含主体 filtered-x、候选、验证管理、双分支、ramp 与实际 FIR、计数 QR/Jacobi；排序的比较与内存移动、除法、平方根和 wall-clock 不在乘法指标中。`resort_saved_audit.json` 给出全部 10 条逐段质量/成本与 SHA；`resort_audit_*_signed_sort.json` 给出独立重放的完整事件和账本。

## 机制消融

为检验是否只是“每 50k 新开一个 R2 分支”，`resort_control.py` 是父源码的隔离副本，只在**首次 signed-sort R4→R2 后**改变维护候选：`clone` 精确复制当前因子，`thin_qr` 以计数的两遍 Gram–Schmidt 对 `A=QR` 做 `A'=Q,B'=BRᵀ`，故 `B'A'ᵀ=BAᵀ`（逐次断言 1e−10）。两者保留原特征顺序、同一训练/物理验证/ramp 时序，都有 7 次接受，成本比约 .85144（比 signed-sort .85175 略低）。下表为第 3 段末窗 ANR，越负越好：

| 开发运行 | 周期 signed-sort | 薄 QR 同矩阵 | 直接复制 |
|---|---:|---:|---:|
| E2 run0 | −9.243 | −7.808 | −7.719 |
| E2 run4 | −8.853 | −7.844 | — |
| E3 run0 | −8.205 | −7.048 | −7.053 |
| E3 run1 | −7.973 | −6.689 | — |

固定随机重排的 E2/E3 run0 在首个 R2 维护提案的 Frobenius 谱尾误差分别 .80/.81，超过 .15 因果谱阈值，被拒；它没有形成等训练预算的有效对照。恒等排列在原大矩阵 Jacobi 核遇到秩亏不收敛，故使用薄 QR 对照。这些结果排除了单纯重复候选训练、切换与同矩阵因子重参数化对主要增益的解释；支持**signed-sort 引入的新坐标重构**是关键，但不证明它是唯一有效的重排。全 10 条同项数分解对照可参照父实验记录。

与固定 R2 μ=.05 相比，周期法 40 段至少好 2.204 dB，平均好 3.037 dB；它同时比固定 full500 μ=.05 平均差约 2.003 dB、最差差 3.050 dB。应描述为 E2/E3 开发流上的低成本 R4 配对改进，而非全维质量上限已达到。50k 间隔与该开发流有关，独立选择评估前不能当泛化证据。

E1 dev run0 的独立边界测试见 `../resort_diagnostic/E1_run00_boundary.json`：三个末窗固定 R4 为 −33.96/−31.29/−34.30 dB，周期 R2 为 −31.07/−6.87/−10.84 dB，后两段退化 +24.42/+23.46 dB。该机制**不是通用双向项数调节**，也没有证明高非线性 E1 的增长与恢复。

## 复现

在仓库根目录用 `C:\Users\nxtrol\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`：

```powershell
& 'C:\Users\nxtrol\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' experiments_v14/dual_rate/audit_saved_resort.py
& 'C:\Users\nxtrol\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' experiments_v14/dual_rate/audit_resort.py --case E2 --run 0 --maintenance signed_sort
& 'C:\Users\nxtrol\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' experiments_v14/dual_rate/audit_resort.py --case E2 --run 0 --maintenance thin_qr
& 'C:\Users\nxtrol\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' experiments_v14/dual_rate/run.py --case E2 --runs 0 1 2 3 4 --post-mu 0.1
& 'C:\Users\nxtrol\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' experiments_v14/dual_rate/run.py --case E2 --runs 0 3 4 --post-mu 0.05 --target-rank 3
```

其他 E3 运行只需替换 `--case` 和 `--runs`。输出 JSON 保留输入 SHA、控制器 SHA、事件、逐段指标、物理核查与运行耗时；`.npz` 保存双步长及 R3 轨迹。所有诊断仅作新 dev 探索。
