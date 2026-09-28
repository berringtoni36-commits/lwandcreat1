# N14 MATLAB 跨语言复核

当前状态：**E2/E3 新 dev run00 的 60F6 自然接受全长复核及新增 50k 周期因果重排全长复核均已通过**，分别见 [`REPORT.md`](REPORT.md) 和 [`REPORT_RESORT.md`](REPORT_RESORT.md)。本页其余内容记录此前 E2 短段强制渐变数值探针；它不是全长自然事件的证据。本目录仅读取 `phase=dev` 保存输入，未读取 select/confirm，也不更改主算法。fixture 对输入文件、各数组与算法源码逐一记录 SHA；60F6 全长复核锁定 selector SHA `60f6fa3f0afe5b9ffc03ef4b610bc446efd7a9e5ed391d621fce6f9b1cc78846`。

## 已完成的独立数值链

[`export_fixture.py`](export_fixture.py) 从 E2 新 dev run00 的相同 `x,d,v,omega,rff_phase,initial_A/B` 导出前 24,200 点。Python 使用归档 v10 顺序 MCC 更新 R4；在零基样本 20,000 **更新后**，以当时 R4 权重带符号稳定排序，调用当前 selector 的计数 QR/单边 Jacobi SVD 得到 R2 候选。候选从样本 20,001 才运行。MATLAB 的 [`replay_short.m`](replay_short.m) 自行逐样本计算 RFF、filtered-x、R4 与候选因子和各自的次级 FIR，并自行重做排序、QR/Jacobi、重构及乘法计数。

为单独核对执行器渐变，短段 fixture **强制**让两个分支在 2k 训练、2k 验证窗口后于样本 24,001–24,100 以 γ=0.01,0.02,…,1 混合执行器输出，再通过唯一的实际次级 FIR。它不是 selector 的接受事件，也没有实现谱 veto、物理验证门控或在线探测逻辑。谱尾在当前参数 `μ=0.1` 下为 `0.0770027782473128`，低于当前 `0.15` veto 阈值；这一事实只说明短段候选可继续用于数值试验。

隐藏 MATLAB 由 [`run_hidden.ps1`](run_hidden.ps1) 使用 `Start-Process -WindowStyle Hidden -Wait`、MATLAB `-batch -noFigureWindows` 启动，输出独立的 stdout、stderr、diary log 与 `.mat` 结果。当前 E2 短段比对见 [`E2_run00_provisional_short_comparison.json`](E2_run00_provisional_short_comparison.json)：R4/候选输出、实际执行器、次级输出、物理误差、ANR、特征及因子最大绝对差 `1.60e-14`；排列完全一致，Jacobi `10` 扫和计数 `347,295` 次乘法完全一致。强制渐变系数及实际 FIR 也一致。这是相同输入与初值上的逐样本跨语言重演，不是读取 Python 输出后在 MATLAB 播放。

## 本机复现

从仓库根目录执行，以下使用本机 Python 运行时；脚本会从 `experiments_v12/.deps` 使用保存的 SciPy：

```powershell
$python = 'D:\software\Xiaomi MiMo\resources\runtimes\win32-x64\python\python.exe'
& $python -B experiments_v14/matlab_replay/export_fixture.py --case E2 --run 0 --mu 0.1 --tag provisional
& experiments_v14/matlab_replay/run_hidden.ps1 -Fixture experiments_v14/matlab_replay/E2_run00_provisional_short.mat -Stem E2_run00_provisional_short
& $python -B experiments_v14/matlab_replay/compare_short.py --stem E2_run00_provisional_short
```

MATLAB 可执行路径是 `D:\software\bin\matlab.exe`，始终隐藏启动；`run_hidden.ps1` 将 `N14_REPLAY_FIXTURE/RESULT/LOG` 作为环境变量传给 MATLAB。若重跑，stdout/stderr/log/result 会在本目录按相同 stem 更新。

## 60F6 自然事件全长验证

[`export_full_fixture.py`](export_full_fixture.py)、[`replay_full_60f6.m`](replay_full_60f6.m)、[`n14_counted_prune.m`](n14_counted_prune.m) 与 [`compare_full.py`](compare_full.py) 已对 E2/E3 新 dev run00 的各 400,000 点进行独立重演。MATLAB 自然提案、验证并接受 R4→R2，逐点物理输出、结构状态、六项成本账本与 Python 保存的 dev 轨迹吻合。详细证据及运行命令见 [`REPORT.md`](REPORT.md)。
