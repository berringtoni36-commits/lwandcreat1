"""Recheck and summarize the five P13 development-only fixed frontiers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from run_frontier import HERE, MU, assess_capacity, atomic_json, costs, source_hashes


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != source_hashes():
        raise RuntimeError("Source hash differs from frozen development run")
    if manifest["cost_ledger"] != costs():
        raise RuntimeError("Cost ledger differs from run manifest")
    rows = []
    all_rank = []
    all_full = []
    fir = []
    for run in range(5):
        out = HERE / f"run{run:02d}"
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        if result["phase"] != "dev" or result["run"] != run or result["T"] != 320_000:
            raise RuntimeError(f"Unexpected run identity {run}")
        if result["source_sha256"] != manifest["source_sha256"]:
            raise RuntimeError(f"Run {run} source mismatch")
        for stem in ("inputs", "traces"):
            if sha(out / f"{stem}.npz") != result[f"{stem}_sha256"]:
                raise RuntimeError(f"Run {run} {stem} checksum mismatch")
        anr = np.asarray(result["segment_anr_db"])
        if anr.shape != (45, 3) or not np.isfinite(anr).all():
            raise RuntimeError(f"Run {run} ANR malformed")
        fresh = assess_capacity(anr)
        if fresh != result["capacity"]:
            raise RuntimeError(f"Run {run} capacity decision mismatch")
        if not 0 <= result["physical_fir_max_abs_gap"] <= 1e-5:
            raise RuntimeError(f"Run {run} physical FIR mismatch")
        rows.append(dict(run=run, required=fresh["required_rank_by_stage"],
                         pass_run=fresh["capacity_gate_pass"],
                         r4_floor=fresh["fixed_R4_absolute_floor"],
                         best_anr_db=fresh["best_anr_db"],
                         physical_fir_max_abs_gap=result["physical_fir_max_abs_gap"],
                         inputs_sha256=result["inputs_sha256"],
                         traces_sha256=result["traces_sha256"]))
        all_rank.append(np.asarray(fresh["rank_oracle_anr_db"]))
        all_full.append(np.min(anr[40:], axis=0))
        fir.append(result["physical_fir_max_abs_gap"])
    rank_anr = np.mean(all_rank, axis=0)
    full_anr = np.mean(all_full, axis=0)
    n_pass = sum(row["pass_run"] for row in rows)
    summary = dict(phase="dev", case="P13", status="stop_before_dynamic_tuning",
                   n_runs=5, n_capacity_pass=n_pass, required_pass=4,
                   gate_pass=n_pass >= 4, run_rows=rows,
                   mean_rank_oracle_anr_db=rank_anr.tolist(),
                   mean_full500_oracle_anr_db=full_anr.tolist(),
                   max_physical_fir_gap=max(fir),
                   cost_ledger=manifest["cost_ledger"],
                   frozen_source_sha256=manifest["source_sha256"],
                   manifest_sha256=sha(HERE / "manifest.json"),
                   analysis_source_sha256=sha(Path(__file__)))
    atomic_json(HERE / "summary.json", summary)
    lines = [
        "# P13 固定秩开发前沿：容量闸门未通过",
        "",
        "**结论：5 次 dev 运行的所需秩均为 7→7→7；通过数 0/5，低于预注册的 4/5。",
        "按 N13 协议停止动态调参，不解锁选择或确认。** 这些数值只是开发集的容量筛查，",
        "不构成新方法的统计确认。未修改 P13 生成公式、ANR 底线或所需秩定义。",
        "",
        "逐段所需秩 `R*` 是同一运行中，固定 R1–R8 各自在五档步长",
        "`{0.05,0.10,0.20,0.40,0.80}` 上的逐段最优 ANR；最小的 R 须离该段",
        "固定前沿最优不超过 0.5 dB，且达到 −6 dB 绝对底线。这个乐观分段步长",
        "是筛查上界，不是单一在线步长的性能。三段为 `[0,120000)`、",
        "`[120000,200000)`、`[200000,320000)`，各取末 10000 点 ANR。",
        "",
        "| dev 运行 | 低段 R* | 高段 R* | 末低段 R* | R4 三段≤−6 dB | 通过容量闸门 |",
        "|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        r = row["required"]
        lines.append(f"| {row['run']} | {r[0]} | {r[1]} | {r[2]} | "
                     f"{'是' if row['r4_floor'] else '否'} | "
                     f"{'是' if row['pass_run'] else '否'} |")
    lines += [
        "",
        "以下为五次运行的**逐段、逐秩乐观 ANR 平均值**（dB，越负越好）；",
        "每格可由 `summary.json` 和各运行 `result.json` 复算。",
        "",
        "| 方法 | 低段 | 高段 | 末低段 | 每点标量乘法 |",
        "|---|---:|---:|---:|---:|",
    ]
    for rank in range(1, 9):
        a = rank_anr[rank - 1]
        c = manifest["cost_ledger"][f"R{rank}"]["total_per_sample"]
        lines.append(f"| R{rank} | {a[0]:.3f} | {a[1]:.3f} | {a[2]:.3f} | {c:,} |")
    lines.append(f"| full500 | {full_anr[0]:.3f} | {full_anr[1]:.3f} | "
                 f"{full_anr[2]:.3f} | 14,508 |")
    lines += [
        "",
        "仅作开发集描述：full500 的逐段步长乐观 ANR 平均值在三段均优于",
        "R1–R8，且其参考乘法 14508/点低于 R2–R8。这不是单一步长",
        "的配对确认结论，但进一步削弱了在原稠密映射上宣称 Kronecker",
        "成本—性能优势的依据。",
        "",
        "每条固定 R 的参考乘法账本为 `12512+1655R/点`：RFF 投影 10000、",
        "特征缩放 500、filtered-x 2000、顺序因子核心 `8+1655R`、物理输出",
        "FIR 4。full500 为 `14508/点`。总乘法是每点值乘以 320000；",
        "各组件和整数总数逐运行保存在 `result.json`。共用的 RFF/filtered-x",
        "在同一次联合仿真中只实际计算一次，但单方法成本比较各自承担一次。",
        "",
        "所有 45 条固定轨迹共享同一运行的 `x,d,v,Ω,φ`；R1–R8 和五档步长",
        "使用相同的初始因子前缀，full500 从零权重开始。非字典教师的 P13",
        "初级路径由 `generator.py` 在抽取 RFF 映射前计算。顺序 MCC 更新",
        "与 v10 原核的 60 点短序列逐方法、逐点对照通过。保存全部输入、",
        "45 条执行器/物理误差轨迹和源码快照；每运行还用保存的执行器输出",
        "独立重演次级 FIR。最大物理误差差值为",
        f"`{max(fir):.3g}`（float32 轨迹存储量化内）。",
        "`manifest.json` 固定源码、协议、环境和成本账本 SHA-256；",
        "`summary.json` 核对五次输入及轨迹文件 SHA-256，若有不一致即失败。",
        "",
        "首次 run0 后发现 manifest 的 Python tuple/JSON list 序列化比较错误。",
        "修正元数据比较后，原始 run0 的源码快照与结果原样归档在",
        "`pre_metadata_fix/`，并在新哈希下重跑 run0；两次 R* 均为",
        "`[7,7,7]`。这个修正不涉及数值内核、输入、生成器或门槛。",
        "run1–4 在新哈希下一次完成。没有运行动态策略、选择集或确认集。",
        "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(n_capacity_pass=n_pass, required_pass=4,
                          gate_pass=n_pass >= 4,
                          required=[row["required"] for row in rows]), ensure_ascii=False))


if __name__ == "__main__":
    main()
