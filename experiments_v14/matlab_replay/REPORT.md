# N14 60F6 方法的独立 MATLAB 全长复核（开发流）

**E2 与 E3 新开发集 run00 的自然接受事件和全长数值轨迹均通过跨语言复核。** MATLAB 用同一实际输入与初值，从头逐样本计算稠密 500 维 RFF、filtered-x、顺序 MCC、带符号排序与计数 QR/Jacobi SVD、谱 veto、严格后续的 2k 训练与 2k 验证、真实执行器上的 100 点渐变、物理次级 FIR 和成本账本。它不读取 Python 的输出轨迹来产生控制输出；Python 保存轨迹只在事后由比较脚本读取。

仅使用 `experiments_v14/outputs/n14_dev_inputs` 的 `phase=dev` E2/E3 run00。两个对照结果均为 `experiments_v14/dev_eval_final` 中同一输入上的 `once/signed_sort`、R4→R2、谱阈值 0.15、100 点渐变，步长均 `μ=0.1`。输入 NPZ、数组、结果 NPZ 和 Python selector 各有 SHA 校验；selector SHA 固定为 `60f6fa3f0afe5b9ffc03ef4b610bc446efd7a9e5ed391d621fce6f9b1cc78846`。未读 select/confirm，未改主算法，未提交。

| 项目 | E2 run00 | E3 run00 |
|---|---:|---:|
| 0 基提案样本 | 20,000 | 20,000 |
| 排序截断谱尾 | 0.0770027782473 | 0.0334409611730 |
| QR/Jacobi 扫次 | 10 | 9 |
| 重排分解计数乘法 | 347,295 | 322,723 |
| 2k 验证末样本 / 后续验证点数 | 24,000 / 2,000 | 24,000 / 2,000 |
| 验证候选/活跃比 | 0.9971321171715 | 1.0003570405271 |
| 决定 | 接受 R4→R2 | 接受 R4→R2 |
| 执行器渐变样本 | 24,001–24,100 | 24,001–24,100 |
| 全程乘法/点 | 16,068.4169425 | 16,068.3555125 |
| 对固定 R4 主体成本比 | 0.8398713 | 0.8398680 |

MATLAB 自主产生的排列、事件下标、接受决定、验证点数、Jacobi 扫次和分解乘法数与 60F6 Python 完全一致；谱尾最大差 `1.11e-16`，验证比最大差 `6.66e-16`。**R、候选 R、退役 R、执行器混合系数与六条逐点成本账本全部逐元素一致**：`core,candidate,retiring,management,reorder,total` 的最大差均为零；各段平均成本和全程平均成本也完全一致。

| 最大绝对差 | E2 | E3 |
|---|---:|---:|
| 全长执行器输出 `drive` | `1.18e-14` | `9.33e-15` |
| 实际次级输出 / 物理误差 | `1.31e-14` | `1.07e-14` |
| 逐点 ANR（Python 保存为 float32） | `7.35e-6` dB | `4.40e-6` dB |
| 四段末 5,000 点 ANR 均值 | `4.29e-7` dB | `2.80e-7` dB |
| 事件验证比 | `6.66e-16` | `0` |
| 物理 FIR 重构误差（MATLAB 内部） | `1.11e-15` | `1.85e-13` |
| 渐变后私人/实际 FIR 核对 | `0` | `0` |

Python 参考的四段末窗物理 ANR 分别为 E2 `[-9.793079,-9.110039,-7.987780,-10.029786]` dB、E3 `[-9.201671,-8.251750,-6.571315,-8.593356]` dB；MATLAB 各段对应差值均小于上表容差。E2 第一段平均成本 `16,771.66777` 乘/点，E3 为 `16,771.42205`；后三段两者均为 `15,834`，与逐点账本相符。逐点 ANR 的微小差异主要受 Python 参考轨迹以 float32 保存影响，因子、物理输出与误差以 double 计算并达到约 `1e-14` 的一致性。

复核要求为：`drive/error/ys_actual` 最大差 `<1e-9`，Python float32 `anr` 最大差 `<1e-5` dB，四段末窗 `<1e-5` dB；整数结构/成本迹完全一致；事件连续指标 `<1e-8`；平均成本 `<1e-8`；实际 FIR 与渐变后核对 `<1e-9`。E2/E3 均通过，没有数值检查失败。此前 24,200 点的 [`provisional` 短 fixture](E2_run00_provisional_short_comparison.json) 使用**强制**切换，只作为 RFF/SVD/MCC/执行器数值单元检查；本报告的全长结果则由 MATLAB 自行按冻结逻辑自然提案、验证并接受。没有把强制切换当作算法事件。

## 可复现文件与命令

[`export_full_fixture.py`](export_full_fixture.py) 只把新 dev 的输入/初值写入 MATLAB fixture，要求 dev 结果的 `once/signed_sort` 配置、一次真实接受与 60F6 SHA 一致。[`replay_full_60f6.m`](replay_full_60f6.m) 独立重演控制器；[`n14_counted_prune.m`](n14_counted_prune.m) 逐操作计数 QR/Jacobi；[`compare_full.py`](compare_full.py) 最后才读取 Python dev trace 对比。完整逐点差值和 SHA 见 [E2 比较 JSON](E2_run00_mu0.1_60f6_full_full_comparison.json)、[E3 比较 JSON](E3_run00_mu0.1_60f6_full_full_comparison.json)。同 stem 的 `.stdout.txt`、`.stderr.txt`、`.log` 均保存在此目录；两次 MATLAB 退出码为零，stderr 为空。

从仓库根目录运行（使用本机 Python 运行时）：

```powershell
$sha = '60f6fa3f0afe5b9ffc03ef4b610bc446efd7a9e5ed391d621fce6f9b1cc78846'
$python = 'D:\software\Xiaomi MiMo\resources\runtimes\win32-x64\python\python.exe'
& $python -B experiments_v14/matlab_replay/export_full_fixture.py --case E2 --run 0 --mu 0.1 --selector-sha $sha
& $python -B experiments_v14/matlab_replay/export_full_fixture.py --case E3 --run 0 --mu 0.1 --selector-sha $sha
& experiments_v14/matlab_replay/run_hidden.ps1 -Fixture experiments_v14/matlab_replay/E2_run00_mu0.1_60f6_full.mat -Stem E2_run00_mu0.1_60f6_full -EntryPoint replay_full_60f6
& experiments_v14/matlab_replay/run_hidden.ps1 -Fixture experiments_v14/matlab_replay/E3_run00_mu0.1_60f6_full.mat -Stem E3_run00_mu0.1_60f6_full -EntryPoint replay_full_60f6
& $python -B experiments_v14/matlab_replay/compare_full.py --stem E2_run00_mu0.1_60f6_full
& $python -B experiments_v14/matlab_replay/compare_full.py --stem E3_run00_mu0.1_60f6_full
```

MATLAB 固定通过 [`run_hidden.ps1`](run_hidden.ps1) 的 `Start-Process -WindowStyle Hidden -Wait` 调用 `D:\software\bin\matlab.exe -noFigureWindows -batch ...`，所有图窗默认隐藏，并保存 stdout、stderr、diary log。比较脚本在任何整数账本、事件、数值容差或来源 SHA 不一致时返回失败并保存差异 JSON。

范围限于 E2/E3 新开发集各 **run00 一条**、一次被接受的 R4→R2；这验证 60F6 实现的跨语言数值和账本一致性，不能把两个 run 当成统计确认、其他运行的证明或真实硬件加速。乘法账本是算法级标量计数，排序比较、内存搬移、余弦/指数/除法及墙钟另计。
