# MATLAB R2025b 静默复核记录

启动方式：`run_hidden_matlab.ps1` 使用 `Start-Process -WindowStyle Hidden`、`-wait -batch`、`-noFigureWindows`，并把 stdout、stderr 与 MATLAB 日志写入文件；三次检查的退出码均为 0。

| 数据 | 检查范围 | 最大物理输出差 | 最大 ANR 差 | 候选判据不一致 |
|---|---|---:|---:|---:|
| `outputs/matlab_fixed_fixture.mat` | 固定 R4：1000 步 RFF、滤波特征、顺序因子更新 | 3.36e-15 | 不适用 | 不适用 |
| `outputs/matlab_dynamic_audit_fixture.mat` | E2 开发运行：实际 FIR、ANR、成本和决策 | 2.37e-7 | 4.90e-6 dB | 0 / 47 |
| `E1_select_run00.mat` | E1 选择集运行：实际 FIR、ANR、成本和决策 | 5.96e-8 | 5.26e-6 dB | 0 / 33 |

固定 R4 的最大 A、B 因子差分别为 1.51e-14、4.86e-16。两条动态审计的逐样本成本求和差均为 0。

动态审计读取 Python 保存的控制输出、账本和验证损失；MATLAB 从控制输出独立计算次级通道、物理 ANR 与接受不等式。它**没有**重新计算动态因子训练、候选降阶和完整结构事件生成，因此只能作为部分跨语言复核。`E1_select_run00_result.mat` 和对应日志保存了直接输出。
