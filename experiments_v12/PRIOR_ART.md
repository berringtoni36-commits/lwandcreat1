# 在线项数选择：定向文献对照

这是方法定位清单，不是穷尽性查新或“首次提出”的证明。论文引言写作前仍需核对各论文全文、定义和后续引用。

| 已有工作 | 与本研究的重合处 | 当前稿需要限定的区别 |
|---|---|---|
| [Fractional tap-length algorithm for adaptive filters under MCC](https://bhxb.buaa.edu.cn/bhzk/en/article/doi/10.13700/j.bh.1001-5965.2015.0137)，2016 | 已经利用 MCC 处理自适应滤波器长度与复杂度的权衡 | 不能笼统声称“首次把 MCC 与在线结构规模学习结合”；需要聚焦 RFF 系数的 Kronecker 项数、物理 ANC 验证和完整候选成本 |
| [Nearest Kronecker Product Decomposition Based Linear-in-The-Parameters Nonlinear Filters](https://doi.org/10.1109/TASLP.2021.3084755)，2021 | Kronecker 分解已经用于高维非线性展开，并用于非线性 ANC | 不能笼统声称“首次将 Kronecker 用于非线性 ANC”；应区分特征映射、更新方式和在线项数选择 |
| [NKP based multichannel filtered-x affine projection algorithm for ANC](https://www.sciencedirect.com/science/article/pii/S0888327024009531)，2025 | NKP、filtered-x 和 ANC 均有已有算法，且包含部分更新以减少开销 | 需要直接比较其成本定义与本文 RFF 特征映射及结构候选；不能将全部成本收益归因于 Kronecker 本身 |
| [AdaLoRA](https://openreview.net/pdf?id=lq62uWRJjiY)，2023 | 自适应低秩与预算分配已在机器学习中存在 | “自适应秩”这个概念本身不是创新；本文仅讨论在线 ANC 控制器的项数选择与鲁棒物理误差判据 |

目前可检验的候选贡献是：**在固定特征映射和顺序 MCC 因子学习外，加入在线 Kronecker 项数候选，用独立的物理通道历史评估，以可审计的完整成本比较固定项数前沿。**

若正式 ANC 工况无法在预设性能和成本门槛下取得优势，文稿保留原固定项数论文主线；动态方法作为尚在研究的方法记录，不填入摘要中的已验证贡献。
