# N14 E2/E3 新开发输入

本目录当前主层保存 E2、E3 各 5 条、每条 400000 点的 **dev 输入数组**，没有控制器训练、选择或确认输出。每份 `case*_run*.npz` 包含 `x,d,v,omega,rff_phase,initial_A,initial_B,segment_bounds`；同名 JSON 保存 SHA-256、SeedSequence 键和逐项生成校验。`manifest.json` 汇总 10 份文件，`source_context.json` 和 `sources/` 保留生成时的协议、生成器、测试、物化脚本与 v10 数值核快照。

物化脚本对 10/10 份数组都用 v12 原路径独立复算并取得**逐点精确相等**：`P*x` 与四段二次/三次初级路径、E2 Gaussian/E3 α-stable 测量噪声、500 维 RFF 抽样及 8 列因子初值。随机流改用 N14 独立命名空间，因此这些数组不等于旧 v12 开发数据。重新执行 `materialize_dev.py` 会核对已有文件及其哈希，而不会静默覆盖。

`pre_protocol_clarification/` 保存首次物化时的同值数组与旧协议文本 SHA 快照；当选择规则从草案中的多配置写法澄清为**唯一策略、25 个动态步长对**后，主层按新协议 SHA 完整重建。该归档仅供追溯，不计入当前 `manifest.json` 的 10 份主数据。

没有物化 select/confirm；E1/C1–C4 目前只通过生成器短序列测试，未在此目录生成正式长数组。
