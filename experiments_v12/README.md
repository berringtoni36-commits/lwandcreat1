# v12 在线 Kronecker 项数研究

v12 是独立开发目录。`experiments_v10/` 对应当前论文的固定项数结果；`experiments_v11/` 是最初的动态原型。两者的代码和运行记录均不由 v12 覆盖。

## 已实现

- 保留 v10 的顺序因子 MCC 更新和 v11 的独立物理候选验证、实际输出 FIR 历史及渐变切换。
- 增长、剪枝分别记录连续拒绝；候选冷却从 4000 点开始，最高退避至 16000 点。候选等待时没有额外分支更新。
- 剪枝候选可选原删列、库 QR/SVD 或可计数 QR/Jacobi SVD 的 Frobenius 降阶；后两者只改变后台分支，仍需要新数据上的物理残差验证。
- 增长候选可使用原随机方向，或只根据过去残差梯度选择方向；加入时仍保持零输出。该方向的计算与分解均计入成本。
- E1 的目标矩阵结构为已知的 1→4→1；E2/E3 为原初级通道非线性系数分段变化；C1–C4 对应原论文四种静态工况；Fclean/Fburst 使用配对输入。
- 固定基线支持 R=1,…,8、完整 500 维 RFF-MCC、规范缩放的 180 维 RFF-MCC、v11 原型。各方法使用相同输入、RFF 映射和对应初始因子前缀。
- 每次运行保存真实输入、映射、初始化、ANR、物理输出、结构事件、成本账本、运行摘要、源码快照和版本信息；`--output` 可恢复相同协议下尚未完成的运行。

## 快速检查

使用 Python 3.12、NumPy 2.3.5、SciPy 1.18.1、Matplotlib 3.11.2。输入和结果保存在各次输出目录；本机 `.deps/` 仅放数值库，不纳入版本管理。

```powershell
python -B experiments_v12/test_adaptive.py
python -B experiments_v12/diagnose_v11.py
python -B experiments_v12/run.py --phase dev --cases E2 --runs 1 --T 6000 --workers 1 --baselines essential
python -B experiments_v12/analyze.py experiments_v12/outputs/某次运行目录
```

正式运行须提供选择集通过后冻结的 JSON 配置。以下是格式示意，不是已经确定的参数：

```json
{
  "selection": {"cooldown": 4000, "max_backoff": 16000, "prune_mode": "counted"},
  "mu_by_method": {"fixed_R4": 0.2, "adaptive": 0.2}
}
```

```powershell
python -B experiments_v12/run.py --phase confirm --cases E1 E2 E3 C1 C2 C3 C4 Fclean Fburst --config <选择后冻结的配置文件> --workers 4
```

确认阶段强制每工况 20 次、全长度及全部固定基线。目前尚无配置通过独立选择集，故尚未创建冻结配置或使用正式确认随机流。开发短运行仍标为 `dev`，不能改名为正式结果。

## 指标与限制

- ANR 与当前稿一致：指数平滑后的绝对物理误差与扰动幅度之比取 20log10；每段统计末 5000 点，越负越好。
- 主成本为全程总乘法/采样，固定 R4 为 19132；动态成本包含候选控制器、筛选、验证、实际输出、过渡及分解费用。平均 R、候选占用、驻留因子和运行时间另报。
- `prune_mode=svd` 使用 NumPy 数值库，账本每次剪枝提案暂收取 **1,000,000 次乘法的公开预算费用**，并非库函数的精确计数。`prune_mode=counted` 使用显式两轮重新正交 QR 和最多 32 轮 Jacobi SVD，逐操作累计理论标量乘法、除法和开方。正式总乘法比较应采用后者；硬件耗时另报。
- `growth_mode=residual` 目前用库 SVD 求梯度方向，每次收取同样的公开预算费用；若选作正式算法，仍需改为可计数内核。
- 开发阶段的统计上界仅用于设计选择；正式结论需独立确认，并满足每段性能、总成本、完整固定 R 前沿、机制与 MATLAB 复核的共同要求。
- E2/E3 的多项式系数变化并不保证所需矩阵秩变化；E1 是已知结构机制检查，不替代真实 ANC 证据。
- 完整 500 维 RFF-MCC 的计算量更低且当前降噪更好；资源主张限定为相对于固定 Kronecker 项数的比较，并保留完整 RFF-MCC 作背景基线。

## MATLAB 静默复核

`run_hidden_matlab.ps1` 使用隐藏窗口、`-batch`、`-noFigureWindows` 和独立日志。先用 `make_matlab_fixture.py` 从一条已保存的运行生成同输入样例，再执行 `matlab/verify_fixed_r4.m`。脚本检查 RFF、滤波特征、物理输出及完整顺序因子更新。

当前 MATLAB 复核覆盖固定 R4 的确定性样例；`make_matlab_audit.py` 与 `matlab/audit_dynamic.m` 还可独立核查已保存动态运行的实际物理 FIR、ANR、乘法账本求和及候选接受判据。它尚未独立复算动态因子学习，所以不能声称已经完成整条动态算法的跨语言复现。

## 结果入口

- `V11_DIAGNOSIS.json`：原 v11 的九次试验诊断，仅用于开发。
- `outputs/main_dev_round1/`：E2/E3 的首轮长程开发试验。以该目录的 `REPORT.md` 为准。
- `outputs/mechanism_dev_round1/`：E1 的结构机制开发试验。
- `outputs/selection_e1_v500/` 等：三套配置在新随机流上的 E1 选择集；均未同时通过预设性能与成本上界。
- `FEASIBILITY_DIAGNOSTIC.md`：E2/E3 固定项数逐段容量差的开发集诊断。
- `PAPER_INTEGRATION.md`：算法写入现有论文的条件及“主张—证据”对照。

每轮提交、推送和状态报告按根目录 `AGENTS.md` 执行，保留其他人已暂存或修改的文件。
