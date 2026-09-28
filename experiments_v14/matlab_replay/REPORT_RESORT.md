# N14 周期因果重排的 MATLAB 全长数值复核（开发流）

E2/E3 新 dev run00、`μ=0.05`、50,000 点固定间隔的 R2→R2 因果 signed-sort 重排，均已在隐藏 MATLAB 进程中从相同原始输入和初值独立重演 400,000 点。两条轨迹各自自然产生 8 次提案、7 次验证后接受及 7 次 100 点真实执行器渐变；第八次在零基 `n=398000` 提案，至流结束尚在训练，未虚构验证或接受。MATLAB 自行计算 RFF、顺序 MCC、权重排序、计数 QR/Jacobi SVD、谱 veto、候选 2k 训练与 2k 验证、物理 FIR、状态转换和成本。MATLAB fixture 只含实际输入与初值，不含 Python 控制输出。

仅使用 `experiments_v14/outputs/n14_dev_inputs` 中 `phase=dev` 的 E2/E3 run00，输入逐数组与 manifest 核验；Python 参考代码 [`selector_resort.py`](../resort_diagnostic/selector_resort.py) SHA-256 为 `3be39ad3665aeab96e9f068135f9cab9b04fbafda960421682be398fd4e3f356`，冻结 60F6 对照源码 SHA-256 为 `60f6fa3f0afe5b9ffc03ef4b610bc446efd7a9e5ed391d621fce6f9b1cc78846`。未读 select/confirm，未改核心算法，也未提交 Git。

| 校验量 | E2 run00 | E3 run00 |
|---|---:|---:|
| `drive` 全长最大绝对差 | `5.33e-15` | `5.77e-15` |
| 实际物理误差最大绝对差 | `5.55e-15` | `7.11e-15` |
| 实际次级输出最大绝对差 | `5.55e-15` | `6.22e-15` |
| 逐点 ANR 最大差（Python float32 轨迹） | `7.35e-6` dB | `4.40e-6` dB |
| 四段末 5,000 点 ANR 最大差 | `3.06e-7` dB | `4.52e-7` dB |
| 8 次提案谱尾最大差 | `4.16e-16` | `2.91e-16` |
| 7 次验证比最大差 | `4.44e-16` | `8.88e-16` |
| 六条逐点乘法账本及 `exp_evals` | 完全相同 | 完全相同 |
| `R`、候选/退役 R、驻留因子数、γ | 完全相同 | 完全相同 |
| 全程平均乘法/点 | `16,295.54842` | `16,295.95226` |
| 与固定 R4 主体账本 `19,132` 的比 | `0.85174307` | `0.85176418` |

两条轨迹的提案下标均为 `20000, 74000, 128000, 182000, 236000, 290000, 344000, 398000`，接受下标均为 `24000, 78000, 132000, 186000, 240000, 294000, 348000`；每次验证点数恰为 2,000。各提案的 `from_R`、QR/Jacobi 扫次、重排乘法数和谱尾，以及各决定的接受状态、验证比和渐变起止点，均与 Python 事件逐项一致。末次候选在 `398001–399999` 共 1,999 个执行点继续更新：其每点候选成本 `3,322` 次乘法、最后一点总成本 `19,160` 次乘法，在 MATLAB 与 Python 逐点完全相同。尽管末次候选尚未进入验证，`n=398000` 的排序/SVD 一次成本也已入账。

| 四段末窗物理 ANR，dB | 第 1 段 | 第 2 段 | 第 3 段 | 第 4 段 |
|---|---:|---:|---:|---:|
| E2 Python 参考 | -9.902494 | -9.862249 | -9.242975 | -11.171450 |
| E3 Python 参考 | -9.666608 | -9.385987 | -8.205435 | -9.941375 |

四段平均乘法/点为 E2 `[16911.27096,16113.72570,16113.65874,16043.53828]`、E3 `[16911.78864,16113.76834,16114.06162,16044.19044]`；MATLAB 与 Python 各段数值完全相同。E2 的全程六项账本均值依次为 `core=16016.608275`、`candidate=249.141695`、`retiring=6.641`、`management=12.45475`、`reorder=6.7027`、`total=16295.54842`，其中总成本还含每点实际次级 FIR 的 4 次乘法；E3 仅重排分解均值变为 `7.10654`，总成本变为 `16295.95226`。物理 FIR 独立重构最大误差 E2 `8.88e-16`、E3 `1.53e-14`；每次渐变后私人和实际 FIR 的误差均为零。

在重排第一次提案之前的零基样本 `0–73999`，同一 Python 输入以原冻结 60F6 `once/signed_sort` 再运行，所有保存的输出、结构、执行器状态和逐点成本与新机制**逐元素零差**；`n=74000` 才出现新机制特有的下一次排序。MATLAB 到新机制 Python 在此前缀同样满足全长数值容差。该前缀核查保证新增机制没有改变首次剪枝或先前控制流程。

比对容差为：`drive/physical_error/ys_actual` 和渐变后 FIR `<1e-9`，逐点 float32 ANR 与四段末窗 ANR `<1e-5` dB，谱尾/验证比 `<1e-8`，平均成本 `<1e-8`；整数事件、逐点结构与六项乘法账本要求严格相等。E2/E3 两份 [逐项比较 JSON](E2_run00_mu0.05_resort50k_comparison.json)、[E3 比较 JSON](E3_run00_mu0.05_resort50k_comparison.json) 的 `pass_full` 均为 `true`，没有数值失败。脚本准备阶段曾修正导出器末尾打印参数和比较器成本字段命名，均未涉及 MATLAB/控制器逻辑；修正后的两条完整运行与比较均退出码为零，MATLAB stderr 为空。

从仓库根目录复现；如下 Python 路径为本机使用的运行时，所需 SciPy 由脚本从 `experiments_v12/.deps` 加载：

```powershell
$python = 'D:\software\Xiaomi MiMo\resources\runtimes\win32-x64\python\python.exe'
& $python -B experiments_v14/matlab_replay/export_resort_fixture.py --case E2 --run 0
& $python -B experiments_v14/matlab_replay/export_resort_fixture.py --case E3 --run 0
& experiments_v14/matlab_replay/run_hidden.ps1 -Fixture experiments_v14/matlab_replay/E2_run00_mu0.05_resort50k.mat -Stem E2_run00_mu0.05_resort50k -EntryPoint replay_resort50k
& experiments_v14/matlab_replay/run_hidden.ps1 -Fixture experiments_v14/matlab_replay/E3_run00_mu0.05_resort50k.mat -Stem E3_run00_mu0.05_resort50k -EntryPoint replay_resort50k
& $python -B experiments_v14/matlab_replay/compare_resort.py --stem E2_run00_mu0.05_resort50k
& $python -B experiments_v14/matlab_replay/compare_resort.py --stem E3_run00_mu0.05_resort50k
```

[`run_hidden.ps1`](run_hidden.ps1) 使用 `Start-Process -WindowStyle Hidden -Wait` 调用 `D:\software\bin\matlab.exe -noFigureWindows -batch ...`，图窗默认隐藏，并分别保存 `.stdout.txt`、`.stderr.txt`、`.log` 和 `_result.mat`。[`export_resort_fixture.py`](export_resort_fixture.py) 校验 dev 来源并保存独立 Python 参考轨迹；[`replay_resort50k.m`](replay_resort50k.m) 是 MATLAB 全长重演；[`compare_resort.py`](compare_resort.py) 事后执行逐点和事件核对。

这是两个开发流 run00 的实现一致性复核，并非独立统计确认或其他运行保证。乘法账本遵循算法级标量计数；排序比较、内存搬移、余弦/指数/除法、状态存储与真实执行墙钟没有换算为乘法加速。固定 50k 提案时钟和参考次级路径也是本诊断的模型条件。
