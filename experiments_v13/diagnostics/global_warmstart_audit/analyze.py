"""Aggregate independently replayed E2/E3 development compression runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from replay import (BRANCH_NAMES, DEV, HERE, METHODS, SOURCE, SWITCH, T,
                    MU_BY_CASE)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def muls_rank(rank):
    return 12_512 + 1_655 * rank


def main():
    rows = {}
    truncations = {name: [] for name in BRANCH_NAMES}
    max_legacy_gap = 0.0
    max_original_gap = 0.0
    max_trunc_gap = 0.0
    max_fir_gap = 0.0
    max_cost_gap = 0.0
    for case in MU_BY_CASE:
        rows[case] = []
        for run in range(5):
            audit_path = HERE / f"{case}_run{run:02d}_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            source_input = DEV / f"case{case}_run{run:02d}.npz"
            original_path = SOURCE / f"{case}_run{run:02d}.json"
            original = json.loads(original_path.read_text(encoding="utf-8"))
            legacy = json.loads((DEV / f"case{case}_run{run:02d}_summary.json").read_text(
                encoding="utf-8"))
            if audit["phase"] != "dev" or original["phase"] != "dev":
                raise RuntimeError("Unexpected phase")
            if audit["source_inputs_sha256"] != sha(source_input):
                raise RuntimeError("Input hash mismatch")
            if audit["original_json_sha256"] != sha(original_path):
                raise RuntimeError("Original JSON changed since independent replay")
            if audit["replay_script_sha256"] != sha(HERE / "replay.py"):
                raise RuntimeError("Replay script changed since result")
            if audit["method_names"] != METHODS:
                raise RuntimeError("Method list mismatch")
            a = np.asarray(audit["segment_anr_db"])
            if a.shape != (15, 4) or not np.isfinite(a).all():
                raise RuntimeError("ANR shape or finite check failed")
            for name in BRANCH_NAMES:
                truncations[name].append(audit["checks"][name]["relative_initial_truncation"])
            for index, label in ((0, "fixed_R1"), (1, "fixed_R2"),
                                 (3, "fixed_R4"), (7, "fixed_R8"),
                                 (8, "full_RFF_MCC")):
                legacy_anr = np.array([stage[label] for stage in legacy["segment_anr"]])
                max_legacy_gap = max(max_legacy_gap,
                                     float(np.max(np.abs(a[index] - legacy_anr))))
            max_original_gap = max(max_original_gap,
                                   audit["comparisons_to_original"]["fixed_R4_max_anr_gap_db"])
            max_fir_gap = max(max_fir_gap, audit["physical_fir_max_abs_gap"])
            for name, data in original["branches"].items():
                check = audit["comparisons_to_original"][name]
                max_original_gap = max(max_original_gap, check["max_delta_gap_db"])
                max_trunc_gap = max(max_trunc_gap, check["truncation_gap"])
                rank = int(name[-1])
                body = (SWITCH * muls_rank(4) + (T-SWITCH) * muls_rank(rank)) / T
                conventional = body + (500*4 + rank*45 + 1_000_000) / T
                max_cost_gap = max(max_cost_gap,
                    abs(data["body_cost_ratio"] - body/muls_rank(4)),
                    abs(data["conventional_cost_ratio"] - conventional/muls_rank(4)))
                if data["quality_within_half_db"] != bool(np.all(
                        a[METHODS.index(name)] - a[3] <= .5)):
                    raise RuntimeError("Quality flag differs")
            rows[case].append(a)
    if max_legacy_gap > 2e-5 or max_original_gap > 2e-5 or max_trunc_gap > 2e-8:
        raise RuntimeError("Replay differs from original or legacy baseline")
    if max_fir_gap > 1e-10 or max_cost_gap > 1e-12:
        raise RuntimeError("Physical FIR or cost mismatch")
    arrays = {case: np.stack(rows[case]) for case in rows}
    mean_truncation = {name: float(np.mean(series)) for name, series in truncations.items()}
    values = {}
    for case, a in arrays.items():
        signed2 = a[:, 9] - a[:, 3]
        signed3 = a[:, 10] - a[:, 3]
        values[case] = dict(
            mean_anr_db=a.mean(axis=0).tolist(),
            sorted_R2_delta_db=signed2.tolist(),
            sorted_R3_delta_db=signed3.tolist(),
            sorted_R2_pass_count=int(np.all(signed2 <= .5, axis=1).sum()),
            sorted_R3_pass_count=int(np.all(signed3 <= .5, axis=1).sum()),
            full500_dominates_sorted_R2_count=int(np.all(a[:, 8] <= a[:, 9], axis=1).sum()),
            full500_dominates_sorted_R3_count=int(np.all(a[:, 8] <= a[:, 10], axis=1).sum()),
            sort_beats_identity_R2_count=int(np.all(a[:, 9] <= a[:, 11], axis=1).sum()),
            sort_beats_random_R2_count=int(np.all(a[:, 9] <= a[:, 13], axis=1).sum()))
    body2 = (SWITCH*muls_rank(4)+(T-SWITCH)*muls_rank(2))/T
    body3 = (SWITCH*muls_rank(4)+(T-SWITCH)*muls_rank(3))/T
    summary = dict(phase="dev", cases=list(MU_BY_CASE), runs_per_case=5,
                   max_legacy_baseline_gap_db=max_legacy_gap,
                   max_original_result_gap_db=max_original_gap,
                   max_truncation_gap=max_trunc_gap,
                   max_physical_fir_gap=max_fir_gap,
                   max_cost_ratio_gap=max_cost_gap,
                   mean_initial_truncation=mean_truncation,
                   fixed_rank_cost_per_sample={f"R{r}": muls_rank(r) for r in range(1,9)},
                   full500_cost_per_sample=14_508,
                   hybrid_R2_body_per_sample=body2,
                   hybrid_R3_body_per_sample=body3,
                   hybrid_R2_conventional_per_sample=body2+(500*4+2*45+1_000_000)/T,
                   hybrid_R3_conventional_per_sample=body3+(500*4+3*45+1_000_000)/T,
                   case_stats=values,
                   replay_script_sha256=sha(HERE / "replay.py"),
                   analysis_script_sha256=sha(__file__),
                   original_script_sha256=sha(SOURCE / "run.py"))
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    lines = [
        "# 20k 因果排序压缩开发实验：独立复算",
        "",
        "**结论：排序分支的数值收益可重演，未发现特征排列、物理 FIR、",
        "顺序 MCC 更新或 ANR 计算错误；但它是预定时刻的 R4 热启动压缩，",
        "不构成在线项数决策。full500 在此成本口径下比 R2/R3 混合分支更省，",
        "且在全部 10 次运行的四段均有更低（更好）的 ANR。**",
        "所有数据均来自 E2/E3 的 dev 输入；未读取选择或确认。",
        "",
        "独立 `replay.py` 在保存的每份 `x,d,v,Ω,φ,A0,B0` 上重演",
        "固定 R1–R8、full500 和排序/恒等/固定随机排列的 R2/R3 压缩。",
        "压缩路径前 20000 点只训练固定 R4，随后用其当时的 500 个系数构造排列和",
        "截断 SVD；参考与误差的后缀、十万点段界和末窗统计都不进入排序。",
        "其他固定秩与 full500 从第 0 点按原论文初值训练，只作为基线。",
        "重演与原 JSON 比较的是四段末 5000 个逐点 ANR 的均值。",
        "",
        "| 验证项目 | 十次最大差异 |",
        "|---|---:|",
        f"| 原排序分支/固定 R4 的四段 ANR | {max_original_gap:.3g} dB |",
        f"| v12 保存的固定 R1/R2/R4/R8/full500 段 ANR | {max_legacy_gap:.3g} dB |",
        f"| 20k 截断相对误差 | {max_trunc_gap:.3g} |",
        f"| float64 执行器输出独立物理 FIR 回放 | {max_fir_gap:.3g} |",
        f"| 费用比 | {max_cost_gap:.3g} |",
        "",
        "排序使用 `np.argsort(w, kind='stable')` 的**带符号**系数值，",
        "对 `z` 和 filtered-x `q` 应用同一排列；先验扰动 `d` 不由这个",
        "500 维字典直接生成。R4 在样本 19999 完成因果更新后才取 `w`，",
        "下一样本 20000 开始压缩分支。各排列分支都继承同一个 R4 执行器",
        "FIR 历史和 ANR EWMA，且使用相同步长 `μ_A=μ_B=μ/2`；",
        "E2 的 `μ=0.2`，E3 的 `μ=0.1`，full500 用 `μ`。",
        "同一排列组的 R2/R3 唯一差别是保留奇异值的个数。",
        "原脚本的身份/随机对照只保存了 run0；审计在其他 run1–4",
        "上增加了同输入的对照重演，标为事后开发诊断。",
        "",
        "| 工况 | 排序 R2 相对 R4 的四段平均 Δ（dB） | R2 逐运行四段≤+0.5 dB |",
        "|---|---|---:|",
    ]
    for case, a in arrays.items():
        delta = (a[:, 9] - a[:, 3]).mean(axis=0)
        lines.append(f"| {case} | " + ", ".join(f"{v:+.3f}" for v in delta) +
                     f" | {values[case]['sorted_R2_pass_count']}/5 |")
    lines += [
        "",
        "固定秩与 full500 的段 ANR 比较采用**同一预设 μ 主尺度及相同输入/映射**",
        "（Kron 两因子各用 `μ/2`，full500 用 `μ`）；full500 权重从零起步，",
        "但固定低秩从零时刻各自训练，而排序分支先享有 20000 点 R4",
        "训练。这是对压缩方案有利的热启动，不是与固定 R2/R3 等初值",
        "的结构消融。恒等和固定随机排列分支才是同等热启动的排序对照。",
        "虽然排序 R2 的截断残差和 ANR 均可核查，不能把它相对固定 R2",
        "的优势全部归于排序；其大小还含 R4 预训练收益。",
        "相同 R4 热启动的排序、恒等和固定随机排列对照中，排序 R2",
        "在 E2/E3 全 10 次运行的四个段均比另两者更低 ANR。",
        "十次 20k 截断的平均相对 Frobenius 残差（排序/恒等/随机）",
        f"在 R2 为 {mean_truncation['signed_sort_R2']:.4f}/"
        f"{mean_truncation['identity_R2']:.4f}/"
        f"{mean_truncation['fixed_random_R2']:.4f}，",
        f"在 R3 为 {mean_truncation['signed_sort_R3']:.4f}/"
        f"{mean_truncation['identity_R3']:.4f}/"
        f"{mean_truncation['fixed_random_R3']:.4f}。",
        "",
        "| 方法 | 每点标量乘法 | 说明 |",
        "|---|---:|---|",
        "| 固定 R2 | 15,822 | 全程 R2 |",
        "| full500 | 14,508 | 全程密集 MCC |",
        f"| 20k R4→R2 主体 | {body2:,.2f} | 先 R4、后 R2；不含一次性分解 |",
        f"| 20k R4→R3 主体 | {body3:,.2f} | 先 R4、后 R3；不含一次性分解 |",
        "| 固定 R4 | 19,132 | `12512+1655R` |",
        "",
        "每次报告的 `conventional_cost_ratio` 另按原脚本加",
        "`(500×4+45R+1000000)/400000` 次乘法/点；其中",
        "1000000 次 SVD 是**约定预算**而非逐操作实测。排序比较、",
        "每点重排 500 维 `z/q` 的内存搬运及墙钟开销不由乘法比反映。",
        "模拟同时运行 R4 对照和所有分支是离线评估；假想已切换分支",
        "只把前 20k 的 R4 与后 380k 的低秩主体计入分子。这对一次",
        "**事先指定的切换**是自洽口径，却没有支付发现何时及是否",
        "切换的在线候选/验证费用，因此不能作为完整动态方法成本。",
        "",
        "以下是五次运行平均的**末段** ANR（dB）；各运行的四段",
        "原值与后验排序对照见 `summary.json` 和 `*_audit.json`。",
        "",
        "| 方法 | E2 | E3 | 每点乘法/方案口径 |",
        "|---|---:|---:|---:|",
    ]
    display = [(f"fixed_R{r}", r-1, muls_rank(r)) for r in range(1,9)] + [
        ("full500", 8, 14_508), ("signed_sort_R2", 9, body2),
        ("signed_sort_R3", 10, body3), ("identity_R2", 11, body2),
        ("fixed_random_R2", 13, body2)]
    for name, idx, cost in display:
        lines.append(f"| {name} | {arrays['E2'][:,idx,3].mean():.3f} | "
                     f"{arrays['E3'][:,idx,3].mean():.3f} | {cost:,.2f} |")
    lines += [
        "",
        "full500 的 14,508/点低于混合 R2 主体",
        f"{body2:,.2f}/点；在 E2/E3 各 5 次运行里，full500 对排序",
        "R2 和 R3 均在四个段上逐次更优。该结论是在现有开发集和",
        "原密集 RFF 映射内的描述性支配，不是独立确认。",
        "",
        "由于没有真实在线触发或候选门槛，结论只能定位为",
        "‘给定 R4 系数后，因果带符号排序改善了这个压缩重参数化’。",
        "不能由此宣称在线项数选择已经奏效、端到端成本优于 full500，",
        "或该方法满足新 N13 容量假设。",
        "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(max_original_gap=max_original_gap,
                          max_legacy_gap=max_legacy_gap,
                          full500_dominance={case: (values[case]["full500_dominates_sorted_R2_count"],
                                                     values[case]["full500_dominates_sorted_R3_count"])
                                             for case in values}), ensure_ascii=False))


if __name__ == "__main__":
    main()
