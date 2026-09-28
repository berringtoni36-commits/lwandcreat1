# 在线 RFF 坐标重排与 Kronecker ANC：正式写作前定向查新

检索日期：2026-09-28。范围是“由当前控制器系数因果排序固定稠密 RFF 的 500 个坐标，改变 25×20 张量布局，在低项数 Kronecker 控制器中继续学习；用次级通道模型评估候选并核算搜索成本”。对照既有 [v13 特征矩阵](../../experiments_v13/prior_art/feature_matrix.md)。下表只用原论文、作者全文、会议/期刊官网及 DOI；这是限定主题的文献核对，不是穷尽检索，也不支持“首次”判断。

## 先把本项目对象说准

当前 [周期重排开发实现](../resort_diagnostic/selector_resort.py) 固定原 500 维稠密 RFF 的频率和相位；由**当前**因子合成的带符号系数稳定排序，同时重排特征坐标，形成新的 25×20 矩阵并以计数 QR/Jacobi SVD 初始化候选。全系数若同步逆置换，`wᵀz=(Pw)ᵀ(Pz)` 严格不变；R2 截断之后函数可改变。第一次接受是 R4→R2，其后固定 50k 提案时钟尝试 **R2→R2 坐标重排**，并非继续减少 Kronecker 项数，也没有经证实的在线增长。因此可研究的是“随时间更新的低秩坐标布局”，不能把 R2→R2 写成“自适应项数进一步降低”。[MATLAB 开发流数值复核](../matlab_replay/REPORT_RESORT.md) 只证明 E2/E3 run00 实现与账本一致，不能充当独立效果确认。

**损失名称必须区分。** 两因子的顺序参数更新使用 Gaussian MCC 影响函数；候选门控在后续 2k 验证样本上累积的是 `min(|dv−ŷ_s,branch|, cap)` 的**削顶绝对虚拟残差**，再比较候选/活跃比，并非 MCC 目标的候选比较。这里 `dv` 由真实执行器下的实测残差与**假定已知且准确**的次级路径输出重构；候选的 `ŷ_s,branch` 由其私人执行器历史通过次级路径模型推算。候选在验证期没有实际驱动扬声器，故不能写“两个控制器分别取得物理麦克风实测误差”或“物理 MCC 门控”。接受后才经 100 点混合输出作用于模拟物理次级 FIR。这一区分可直接从 [更新与门控代码](../resort_diagnostic/selector_resort.py) 核查。

## 最接近的八篇一手工作

| 论文与精确来源 | 实际重合 | 对本项目的可辩护区别与写作限制 |
|---|---|---|
| **Ye 等，EMRFF-GMCC ANC，Applied Acoustics 239:110807 (2025)**；[出版方原页](https://www.sciencedirect.com/science/article/pii/S0003682X25002798)，[DOI:10.1016/j.apacoust.2025.110807](https://doi.org/10.1016/j.apacoust.2025.110807)。 | 多重 RFF、GMCC、前馈 ANC、物理次级路径应用与管道实验同处一个方法；出版方还把此前 RFFxLMS 列为已有工作。 | **不能声称 RFF+correntropy+ANC 首次结合，更不能声称本项目已有实物管道验证。** 本项目保留预固定稠密映射、按当前控制系数重排其坐标并测试低项数布局；出版方可见摘要/节选未报告这个具体程序。全文访问有限，不能把“未见”升级为排他性新颖性结论。GMCC 与本项目 Gaussian MCC 更新不同。 |
| **Ye、Zhao，NKP 子带自适应滤波与 ANC，IEEE/ACM TASLP，期刊 DOI 2025、作者全文 2026**；[作者全文 §VI-D](https://arxiv.org/html/2601.10078)，[DOI:10.1109/TASLPRO.2025.3649394](https://doi.org/10.1109/TASLPRO.2025.3649394)。 | §VI-D 明确实施并比较 `NKP-FxNSAF-MCC`：Kronecker 因子、filtered-x、MCC、主/次 FIR 和 ANR 同现，并给出复杂度分析。 | **不能称 Kronecker+filtered-x+MCC+ANC 的首次组合。** 该全文的输入是子带信号，未给出预固定 500 维稠密 RFF 的系数驱动任意坐标重排、对应候选 SVD 与当前方案的门控账本；其固定/可调项数也不能直接等同于此处 R2。 |
| **Spiriti、Morici、Piroddi，*A gradient-free adaptation method for nonlinear active noise control*, Journal of Sound and Vibration 333:13–30 (2014)**；[作者全文](https://re.public.polimi.it/retrieve/e0c31c0d-6caf-4599-e053-1705fe0aef77/A%20gradient-free%20adaptation%20method%20for%20nonlinear%20active%20noise%20control_11311-762725_Piroddi.pdf)，[出版方与 DOI:10.1016/j.jsv.2013.09.006](https://www.sciencedirect.com/science/article/pii/S0022460X13007402)。同一研究线另见 [ECC 2013 原文](https://skoge.folk.ntnu.no/prost/proceedings/ecc-2013/data/papers/0323.pdf)、[DOI:10.23919/ECC.2013.6669272](https://doi.org/10.23919/ECC.2013.6669272)。 | 利用次级路径模型虚拟重构误差，并行评估结构增项/删项候选，在控制器本体继续工作时择优；原文明确讨论并行搜索开销及降采样搜索。 | **不能称 ANC 中“后台虚拟候选—验证—增删结构”是新想法，也不能称首个把搜索成本带入设计。** 其候选是 NARX 回归项与梯度自由误差评估，不是固定 RFF 坐标的带符号重排、Kronecker 因子顺序 MCC 或当前逐点乘法分账。这里所谓“物理候选”也同样依赖次级路径模型，不能暗示反事实候选被物理执行。 |
| **Koike-Akino、Liu、Wang，*EinSort: Sorting is All We Need for Tensorizing LLM*，作者预印本 (2026)**；[原文 §2.3–2.5 与附录](https://arxiv.org/html/2606.08565)，[arXiv DOI:10.48550/arXiv.2606.08565](https://doi.org/10.48550/arXiv.2606.08565)。 | 将权重索引排序、逆置换与低秩张量化结合；展示排序后奇异谱和低秩误差改善，明确计入置换存储。附录还讨论测试时秩/张量核更新的可能性。 | **不能称“排序暴露可张量化低秩”首次。** 本项目不同之处是用在线 ANC 当前系数选排列，对同一固定 RFF 特征同步排列，在新样本上继续控制并以虚拟次级残差验收。EinSort 主实验为 LLM 权重/KV 压缩；其附录使“完全没有在线更新”这种断言也不安全。 |
| **Kwon 等，*NeuKron: Constant-Size Lossy Compression of Sparse Reorderable Matrices and Tensors*, WWW (2023)**；[作者原文](https://arxiv.org/html/2302.04570)，[ACM DOI:10.1145/3543507.3583226](https://doi.org/10.1145/3543507.3583226)。 | 重排行列与 Kronecker 类结构化近似交替优化，提高可压缩性；原文给出更新复杂度。 | **不能称重排促进 Kronecker 类压缩首次。** 它面向可重排行列的稀疏矩阵/张量，不是对 500 个独立稠密 RFF 特征做跨行列任意置换；本项目的排序决定来自随时间变化的 ANC 系数。仅整行/整列置换不会改变矩阵奇异值，本项目改变有效秩须跨行列重新布置坐标。 |
| **Zhu 等，*Cascaded Random Fourier Filter for Robust Nonlinear Active Noise Control*, IEEE/ACM TASLP 30:2188–2200 (2022)**；[DOI:10.1109/TASLP.2021.3126943](https://doi.org/10.1109/TASLP.2021.3126943)、[作者单位论文目录](https://faculty.swjtu.edu.cn/zhaohaiquan/zh_CN/zdylm/161950/list/index.htm)。 | 既有随机 Fourier filtered-x ANC，进一步以级联/双线性结构降低复杂度，并用广义双曲正割鲁棒准则处理脉冲噪声及作成本比较。 | **不能称 RFF 非线性 ANC 或低复杂度级联结构首次。** 它的鲁棒准则不是本项目的 Gaussian MCC，出版摘要未给出当前系数排序同一 RFF 坐标与 R2→R2 时变布局；级联与 Kronecker 项数不可直接视为同一结构。 |
| **Bhattacharjee、George，*Nearest Kronecker Product Decomposition Based Linear-in-The-Parameters Nonlinear Filters*, IEEE/ACM TASLP 29:2111–2122 (2021)**；[DOI:10.1109/TASLP.2021.3084755](https://doi.org/10.1109/TASLP.2021.3084755)。 | 高维非线性功能展开的系数以 NKP 短因子表示，并考察非线性 ANC 中的复杂度和降噪。 | **不能称非线性 ANC 的 Kronecker 因子化首次。** 可区分的对象是固定 Gaussian RFF 特征的在线坐标排序与虚拟候选门控；它的功能展开、损失和控制细节不应被写成与本项目相同。 |
| **Wesel、Batselier，*Large-Scale Learning with Fourier Features and Tensor Decompositions*, NeurIPS (2021)**；[会议正式论文](https://proceedings.neurips.cc/paper/2021/hash/92a08bf918f44ccd961477be30023da1-Abstract.html)。 | Fourier 特征与低秩张量权重联合学习，说明“Fourier+张量因子”并非独有。 | 其特征是**确定性张量积 Fourier 特征**、正则化平方损失与块坐标下降；本项目是固定稠密随机 RFF、重排后重新矩阵化、逐样本 ANC。不能把不同映射的低秩收益当作本项目固定映射的证明。 |

## 对旧矩阵的更新与可写边界

[旧矩阵](../../experiments_v13/prior_art/feature_matrix.md) 中把稠密 RFF 任意坐标置换列为“尚未实现”的历史状态已经过时；当前 v14 已有周期因果 R2→R2 的开发诊断。旧矩阵也未收录上表的 **Ye 2025 EMRFF-GMCC ANC**、**Spiriti/Morici/Piroddi 2014 虚拟候选结构选择**和 **Zhu 2022 级联 RFF 鲁棒 ANC**，写作时必须补引。旧矩阵关于 EinSort、NeuKron、NKP-FxNSAF-MCC 及确定性 Fourier 张量学习的排他性警示仍成立。此处未改写旧文件。

可安全描述的**方法对象**是：在同一预固定稠密 RFF 映射下，用当前因子控制器的带符号系数因果选择 500 坐标排列，保留特征值并重新形成低秩系数网格；以次级路径模型构造候选的虚拟残差，在严格后续样本上训练/验收，经渐变切换真实执行器，同时把 RFF 主体、候选与退役分支、分解、探测、验证和切换的**标量乘法**合计。该具体组合是待比较的方法定义，不是“首次”主张；上表原作已覆盖其多数要素。正式经验结果只有在预先冻结规则后以未参与设计的数据、同映射同初值优化过的固定秩对照及完整成本口径检验成立，才可写为“在所测 ANC 条件和指定 ANR 容差下，包含结构搜索的标量乘法少于该对照”。当前两个 run00 开发重演只可用于实现核查。

尤其不得写“在线自适应增减项数已经成立”：所核查 v14 分支在初次 R4→R2 后仅进行 R2→R2 重排，缺少可靠的增项与再剪枝成功证据。不得写“候选使用 MCC 物理实测损失”：MCC 属于因子更新，门控用削顶绝对**模型推算**残差。不得把 500 维 RFF 继续全部计算的流程称作 RFF 特征数削减。一个 500 索引的持久排列需约 1,000 B（16 位）、2,000 B（32 位）或 4,000 B（64 位）；活动、候选和退役排列若同时保存还会叠加。已有标量乘法账本计入排序后 QR/Jacobi 和影子训练，但排序比较、内存读写/搬移、置换存储、余弦/指数/除法及墙钟时间尚未折合为统一硬件成本，不能据乘法比直接宣称端到端加速。EinSort 也强调置换存储不能免费处理。

本次英文主题检索覆盖 `online feature permutation tensorization`、`RFF GMCC ANC`、`Kronecker MCC filtered-x ANC`、`nonlinear ANC virtual model selection`、`sorting low-rank tensorization`；使用原论文/DOI/出版页核验具体机制。Elsevier 和部分 IEEE 全文访问有限，因此对其“未报告”仅限可核查摘要与节选；没有做专利、引文网络的穷尽追踪。下一轮写作前宜对 Ye 2025 与 Zhu 2022 取得全文并逐式核查，再以已冻结的选择协议产生独立证据；本报告不触碰该协议或正式数据。
