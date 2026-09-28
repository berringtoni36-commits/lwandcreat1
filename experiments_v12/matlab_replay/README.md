# 短序列动态控制器 MATLAB 独立重放

`probe.py` 从当前 Python `AdaptiveKronV12` 与 `KronRFF` 生成两条 120 样本的开发样例。MAT 文件保存实际参考输入 `x`、干扰 `d`、噪声 `v`、RFF 频率与相位、初始因子，以及 Python 在第 9 个零基样本提出增长时实际抽到的新 A 列。MATLAB 从这些共同输入和初始化出发，独立逐样本计算 RFF、次级路径 filtered-x、两次顺序 MCC 因子更新、候选训练、在线先评分后学习的验证损失、增长决策、渐变切换和实际输出 FIR；`z_expected`、`q_expected`、`trace_expected` 仅用于事后比对。

两个样例均用 `D1=3, D2=2, D=6, M=4, R:1→2`、`mu_a=mu_b=0.4`、`sigma=2`、`lambda_cost=0.001`、`seed=42`、`growth_seed=314159`、`S=[0,0,1,0.5]`。增长接受样例使用 `teacher_gain=1`，拒绝样例使用 `teacher_gain=0`。样例特意覆盖一次接受和一次拒绝，未经确认集种子选择。

在仓库根目录执行：

```powershell
& 'C:\Users\nxtrol\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B experiments_v12/matlab_replay/probe.py
& experiments_v12/run_hidden_matlab.ps1 -Script experiments_v12/matlab_replay/replay_dynamic.m -OutputPrefix experiments_v12/matlab_replay/growth_accept -Fixture experiments_v12/matlab_replay/growth_accept.mat -Result experiments_v12/matlab_replay/growth_accept_result.mat
& experiments_v12/run_hidden_matlab.ps1 -Script experiments_v12/matlab_replay/replay_dynamic.m -OutputPrefix experiments_v12/matlab_replay/growth_reject -Fixture experiments_v12/matlab_replay/growth_reject.mat -Result experiments_v12/matlab_replay/growth_reject_result.mat
```

启动脚本使用 `Start-Process -WindowStyle Hidden`、`-batch`、`-noFigureWindows` 和日志重定向。`growth_*.log`、`growth_*.stdout.txt`、`growth_*.stderr.txt` 是原始 MATLAB 运行记录；`growth_*_result.mat` 保存完整 MATLAB 轨迹、最终因子和误差。

| 样例 | RFF 最大误差 | 逐样本轨迹最大误差 | 实际次级输出最大误差 | 最终因子最大误差 | 结构事件，零基样本 |
|---|---:|---:|---:|---:|---|
| 接受 | 5.14e-16 | 6.67e-16 | 6.67e-16 | 2.78e-16 | 提出 9，验证开始 39，接受 79，切换完成 84 |
| 拒绝 | 5.14e-16 | 5.56e-16 | 3.34e-16 | 3.06e-16 | 提出 9，验证开始 39，拒绝 79，无切换 |

比对轨迹包含旧/候选项数、实际控制输出、实际次级输出、渐变权重、阶段、两分支的 A/B 因子和；另逐项比对决策损失、惩罚后分数和最终全部因子。运行时断言误差阈值与全部结构事件匹配。

该核对只覆盖随机增长的短序列、单次候选和实际输出 FIR；未覆盖剪枝、残差方向增长、反复提案、多运行统计或长时间真实 ANC 闭环稳定性。候选新 A 列作为共同初始化由 Python 保存，MATLAB 不复现 NumPy 随机数生成器。

本次样例的 Python 控制器源码 SHA-256：`adaptive_core.py` 为 `B229E64B7BE393CDAE1F145222E9BFB0E71EE18CD8C0B4EF663E6A582B08E517`；`anc_core.py` 为 `868C9776FBCE3B4DE4AEFBD5056F51B2F6FB5D52605AE40D0E5CE80996BA9F41`。工作树 Git HEAD 为 `a77f7f1ce6ec9a3e1ab14e805e4d999f6cb5ec98`；源码哈希用于区别未提交的开发修改。
