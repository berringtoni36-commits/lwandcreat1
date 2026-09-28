"""Post-hoc capacity envelope on E2/E3 development data only.

This diagnostic uses hindsight (segment boundaries, future losses and free
trained controller states). It is neither an implementable selector nor a
mathematical bound on every possible adaptive factor algorithm.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
V12 = HERE.parent
ROOT = V12.parent
SOURCE_DIR = V12 / "outputs" / "main_dev_round1"
sys.path.insert(0, str(SOURCE_DIR / "sources" / "experiments_v10"))
import anc_core as ac

MU = (.05, .1, .2, .4, .8)
FIXED_COST = np.array([ac.mults_kron(25, 20, r, 20, 4) + 4
                       for r in range(1, 9)], dtype=float)
COST_R4 = float(FIXED_COST[3])
BUDGET = .9 * COST_R4
SHARED_FEATURE_COST = 25 * 20 * (20 + 1 + 4)
OUTPUT_FIR_COST = 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_batched_update():
    """Verify batched step sizes against separate original controllers."""
    rng = np.random.default_rng(82631)
    for rank in (1, 5, 8):
        batched = ac.KronRFF("batch", 5, 25, 20, rank,
                             np.array(MU)/2, np.array(MU)/2, 2., 1e-8, rng)
        separate = []
        for j, mu in enumerate(MU):
            c = ac.KronRFF("single", 1, 25, 20, rank, mu/2, mu/2,
                           2., 1e-8, rng)
            c.A = batched.A[j:j+1].copy()
            c.B = batched.B[j:j+1].copy()
            separate.append(c)
        for _ in range(300):
            z, q, dv = rng.normal(size=(1, 500)), rng.normal(size=(1, 500)), rng.normal(size=1)
            got = batched.step(z, q, dv)
            expected = np.array([c.step(z, q, dv)[0] for c in separate])
            np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)
        for j, c in enumerate(separate):
            np.testing.assert_allclose(batched.A[j], c.A[0], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(batched.B[j], c.B[0], atol=1e-12, rtol=1e-12)


def supplement(spec):
    case, run = spec
    source = SOURCE_DIR / f"case{case}_run{run:02d}.npz"
    existing_path = source.with_name(source.stem + "_fixed_sweep.json")
    existing = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else None
    protocol = json.loads((source.parent / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["phase"] == "dev", "Only development data are permitted"
    present = set(existing["ranks"]) if existing else set()
    missing = sorted(set(range(1, 9)) - present)
    with np.load(source) as f:
        x, d, v = f["x"], f["d"], f["v"]
        Om, ph = f["Om"], f["ph"]
        A, B = f["initial_A"], f["initial_B"]
        meta = json.loads(str(f["meta"]))
        overhead = float(np.mean(f["total_mults"] - f["core_mults"])) - 4.
    T = len(x)
    assert case in ("E2", "E3") and meta["case"] == case
    if existing:
        assert existing["T"] == T and tuple(existing["mu"]) == MU
    controllers = []
    for rank in missing:
        c = ac.KronRFF(f"R{rank}", 5, 25, 20, rank,
                       np.array(MU)/2, np.array(MU)/2, 2., 1e-8, np.random.default_rng(0))
        c.A = np.repeat(A[:, :, :rank], 5, axis=0)
        c.B = np.repeat(B[:, :, :rank], 5, axis=0)
        controllers.append(c)
    sums = np.zeros((4, len(missing), 5))
    counts = np.zeros(4, dtype=int)
    Ae = np.zeros((len(missing), 5))
    Ad = 0.
    xb, zh = np.zeros((1, 20)), np.zeros((4, 1, 500))
    start = perf_counter()
    for n in range(T):
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = x[n]
        z = np.sqrt(2/500) * np.cos(np.einsum("rdm,rm->rd", Om, xb) + ph)
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2] + .5 * zh[3]
        dv = np.array([d[n] + v[n]])
        Ad = .999 * Ad + .001 * abs(d[n])
        segment = min(3, 4*n//T)
        end = (segment+1)*T//4
        tally = n >= max(segment*T//4, end-5000)
        if tally:
            counts[segment] += 1
        for j, c in enumerate(controllers):
            ys = c.step(z, q, dv)
            Ae[j] = .999 * Ae[j] + .001 * np.abs(dv[0]-ys)
            if tally:
                sums[segment, j] += 20*np.log10((Ae[j]+1e-12)/(Ad+1e-12))
    extra = sums / counts[:, None, None]
    table = np.full((4, 8, 5), np.nan)
    if existing:
        old = np.array(existing["segment_anr"])
        for j, name in enumerate(existing["names"]):
            rank, mu = name[1:].split("_mu")
            table[:, int(rank)-1, MU.index(float(mu))] = old[:, j]
    for j, rank in enumerate(missing):
        table[:, rank-1] = extra[:, j]
    assert np.isfinite(table).all()
    # Original runs retain default-step R1/R8 traces (float32). Cross-check the
    # supplemental scan against those independent saved controller executions.
    summary = json.loads(source.with_name(source.stem + "_summary.json").read_text(encoding="utf-8"))
    default_mu = .1 if case == "E3" else .2
    discrepancy = 0.
    for segment in range(4):
        for rank in (1, 2, 4, 8):
            expected = summary["segment_anr"][segment][f"fixed_R{rank}"]
            discrepancy = max(discrepancy, abs(table[segment, rank-1, MU.index(default_mu)] - expected))
    assert discrepancy < 2e-5, discrepancy
    return dict(case=case, run=run, source=str(source.relative_to(ROOT)),
                source_sha256=sha(source), existing_scan_sha256=sha(existing_path) if existing else None,
                T=T, added_ranks=missing, mu=list(MU), segment_anr=table.tolist(),
                measured_overhead_above_fixed_branch=overhead,
                saved_baseline_max_abs_difference_db=discrepancy,
                seconds=perf_counter()-start)


def selected_record(indices, table, reference, overhead=0.):
    ranks = np.asarray(indices, dtype=int)
    performance = table[np.arange(4), ranks]
    delta = performance - reference
    cost = float(FIXED_COST[ranks].mean()+overhead)
    return dict(ranks=(ranks+1).tolist(), segment_anr=performance.tolist(),
                segment_delta_db=delta.tolist(), mean_delta_db=float(delta.mean()),
                worst_segment_delta_db=float(delta.max()),
                mean_mults=cost, cost_ratio=cost/COST_R4,
                overhead_mults=overhead)


def compute_envelope(row):
    full = np.asarray(row["segment_anr"])
    # The strongest hindsight relaxation: even the step size may change by
    # segment, with a fully trained counterfactual branch available for free.
    table = full.min(axis=2)
    mu_indices = full.argmin(axis=2)
    r4_mu = int(full[:, 3, :].mean(axis=0).argmin())
    r4_reference = full[:, 3, r4_mu]
    combinations = np.array(list(itertools.product(range(8), repeat=4)))
    candidate_anr = table[np.arange(4)[None, :], combinations]
    deltas = candidate_anr-r4_reference
    mean_delta = deltas.mean(axis=1)
    worst_delta = deltas.max(axis=1)
    costs = FIXED_COST[combinations].mean(axis=1)
    result = dict(case=row["case"], run=row["run"], fixed_R4_mu=MU[r4_mu],
                  fixed_R4_segment_anr=r4_reference.tolist(),
                  hindsight_best_segment_anr_by_rank=table.tolist(),
                  hindsight_best_mu_by_segment_rank=[[MU[j] for j in s] for s in mu_indices],
                  fixed_rank_full_costs=FIXED_COST.tolist())
    per_segment=[]
    for segment in range(4):
        affordable_quality = np.flatnonzero(table[segment]-r4_reference[segment] <= .5)
        rank = affordable_quality[np.argmin(FIXED_COST[affordable_quality])]
        per_segment.append(dict(segment=segment+1, rank=int(rank+1),
                                mu=MU[mu_indices[segment,rank]],
                                delta_db=float(table[segment,rank]-r4_reference[segment]),
                                full_branch_mults=float(FIXED_COST[rank]),
                                cost_ratio=float(FIXED_COST[rank]/COST_R4)))
    result['per_segment_lowest_cost_meeting_margin'] = per_segment
    for label, overhead in (("zero_selection_overhead", 0.),
                            ("observed_round1_overhead", row["measured_overhead_above_fixed_branch"])):
        affordable = np.flatnonzero(costs+overhead <= BUDGET+1e-9)
        quality = np.flatnonzero(np.all(deltas <= .5, axis=1))
        best_mean = affordable[np.argmin(mean_delta[affordable])] if len(affordable) else None
        best_worst = affordable[np.argmin(worst_delta[affordable])] if len(affordable) else None
        cheapest = quality[np.argmin(costs[quality])] if len(quality) else None
        def record(j):
            if j is None:
                return None
            out = selected_record(combinations[j], table, r4_reference, overhead)
            out["mu_by_segment"] = [MU[mu_indices[s, combinations[j, s]]] for s in range(4)]
            return out
        result[label] = dict(best_mean_at_cost_target=record(best_mean),
                             best_worst_segment_at_cost_target=record(best_worst),
                             lowest_cost_meeting_every_segment_margin=record(cheapest),
                             feasible_joint=bool(np.any((costs+overhead <= BUDGET) & np.all(deltas <= .5, axis=1))))
    # A stricter baseline which also enjoys segment-wise hindsight tuning.
    strict_deltas = candidate_anr - table[:, 3]
    result["strict_segmentwise_tuned_R4_joint_feasible"] = bool(np.any(
        (costs <= BUDGET) & np.all(strict_deltas <= .5, axis=1)))
    # An implementation retaining all branch trajectories must also pay for
    # unselected branches. Sharing features does not make factor updates free.
    result["eight_branch_bank_cost_ratio"] = float((SHARED_FEATURE_COST+np.sum(FIXED_COST-SHARED_FEATURE_COST)+OUTPUT_FIR_COST)/COST_R4)
    result["forty_branch_rank_and_mu_bank_cost_ratio"] = float((SHARED_FEATURE_COST+5*np.sum(FIXED_COST-SHARED_FEATURE_COST)+OUTPUT_FIR_COST)/COST_R4)
    frontier=[]
    last = np.inf
    for cost in sorted(set(costs)):
        eligible = np.flatnonzero(costs == cost)
        j = eligible[np.argmin(mean_delta[eligible])]
        if mean_delta[j] < last-1e-12:
            frontier.append(selected_record(combinations[j], table, r4_reference))
            last=mean_delta[j]
    result["zero_overhead_mean_loss_pareto_frontier"] = frontier
    return result


def audit_and_enrich(scans):
    """Recheck cached development provenance and the measured cost identity."""
    protocol = json.loads((SOURCE_DIR/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['phase'] == 'dev'
    for row in scans:
        assert row['case'] in ('E2', 'E3') and row['run'] in range(5)
        source = ROOT / row['source']
        assert source.parent == SOURCE_DIR
        assert row['source_sha256'] == sha(source)
        existing = source.with_name(source.stem+'_fixed_sweep.json')
        assert row['existing_scan_sha256'] == (sha(existing) if existing.exists() else None)
        summary = json.loads(source.with_name(source.stem+'_summary.json').read_text(encoding='utf-8'))
        with np.load(source) as f:
            meta = json.loads(str(f['meta']))
            assert meta['case'] == row['case'] and meta['run'] == row['run']
            assert len(f['x']) == row['T'] == 400000
            core = float(np.mean(f['core_mults']))
            candidate = float(np.mean(f['candidate_mults']))
            management = float(np.mean(f['management_mults']))
            total = float(np.mean(f['total_mults']))
            assert np.all(f['total_mults'] == f['core_mults']+f['candidate_mults']+f['management_mults'])
            assert abs(total-summary['mean_mults']) < 1e-7
            assert abs(candidate-summary['mean_candidate_mults']) < 1e-7
            assert abs(management-summary['mean_management_mults']) < 1e-7
            assert abs(candidate+management-OUTPUT_FIR_COST-row['measured_overhead_above_fixed_branch']) < 1e-7
        full = np.asarray(row['segment_anr'])
        default_mu = .1 if row['case'] == 'E3' else .2
        index = MU.index(default_mu)
        discrepancy = max(abs(full[s, rank-1, index]-summary['segment_anr'][s][f'fixed_R{rank}'])
                          for s in range(4) for rank in (1, 2, 4, 8))
        assert discrepancy < 2e-5, discrepancy
        row['saved_baseline_max_abs_difference_db'] = float(discrepancy)
        row['cost_ledger'] = dict(measured_core_mults=core,
                                  measured_candidate_mults=candidate,
                                  measured_management_mults=management,
                                  measured_total_mults=total,
                                  fixed_branch_output_fir_mults=OUTPUT_FIR_COST,
                                  measured_overhead_above_fixed_branch=candidate+management-OUTPUT_FIR_COST)


def write_report(results, scans):
    lines=["# E2/E3 容量与完整成本的事后乐观包络", "",
           "本报告只使用 main_dev_round1 的 E2/E3 各五次开发运行，不使用 select 或 confirm 数据。",
           "E2/E3 各自 run0–1 已有的 R2/R3/R4 五档步长扫描保持原样；其余固定分支均在相同的已保存输入、映射和初值上补扫。", "",
           "**这是事后上界诊断，不是可实施算法，也不是正式论文优越性证据。**", "",
           "## 包络定义", "",
           "每次运行长 400000 点、四段各 100000 点；性能统计保持原协议，每段末 5000 点 ANR 平均。",
           "参考为该开发运行上四段等权 ANR 最佳的单一步长固定 R4；这是事后调优参考，未作显著性推断。",
           "乐观策略可预知每段边界和未来表现，在 R1–R8 与五档步长中自由挑选已经训练好的分支。",
           "穷举四段的 8^4=4096 种项数组合，逐段步长也选择最优。切换、预训练和挑选先按零成本处理，以故意放宽可行域。",
           "因此零开销曲线是该有限固定分支库中的乐观性能诊断，不是任意动态学习算法的普遍数学界。", "",
           f"固定 R4 完整分支成本（含输出 FIR）={COST_R4:.0f} 次乘法/点；0.90 门槛={BUDGET:.1f}。",
           "同时要求四段各自相对该固定 R4 的 ANR 增量 ≤0.5 dB。只通过四段平均值不算通过。", "",
           "## 固定分支成本账本", "",
           f"各分支共享特征生成与滤波 {SHARED_FEATURE_COST} 次乘法/点；每条分支另有独立更新及预测成本与 {OUTPUT_FIR_COST} 次输出 FIR。",
           "下表成本包含共享特征一次、该秩控制器一次和该分支输出 FIR 一次。", "",
           "|项数|控制器增量乘法/点|完整分支乘法/点|相对 R4|",
           "|---:|---:|---:|---:|"]
    for rank, cost in enumerate(FIXED_COST, start=1):
        lines.append(f"|R{rank}|{cost-SHARED_FEATURE_COST-OUTPUT_FIR_COST:.0f}|{cost:.0f}|{cost/COST_R4:.4f}|")
    lines += ["",
           "## 最乐观结果：结构选择和状态转移均免费", "",
           "|工况|运行|成本≤0.90 时最佳平均差 dB|对应最差段差 dB|项数轨迹|每段≤0.5 所需最低成本比|联合可行|",
           "|---|---:|---:|---:|---|---:|---|"]
    for r in results:
        a=r["zero_selection_overhead"]
        best=a["best_mean_at_cost_target"]
        cheapest=a["lowest_cost_meeting_every_segment_margin"]
        lines.append(f"|{r['case']}|{r['run']}|{best['mean_delta_db']:+.3f}|{best['worst_segment_delta_db']:+.3f}|{best['ranks']}|{cheapest['cost_ratio']:.4f}|{'是' if a['feasible_joint'] else '否'}|")
    lines += ["", "## 逐段质量约束所需最低乐观成本", "",
              "每一段单独选择满足相对固定 R4 ≤0.5 dB 的最低成本固定分支；段间状态、预训练及选择仍免费。",
              "四段的最低成本平均值等于上表质量约束所需成本。", "",
              "|工况|运行|逐段最低项数|逐段最低成本/R4|四段平均最低成本/R4|",
              "|---|---:|---|---|---:|"]
    for r in results:
        rows=r['per_segment_lowest_cost_meeting_margin']
        ranks=[x['rank'] for x in rows]
        ratios=[round(x['cost_ratio'],4) for x in rows]
        mean_ratio=np.mean([x['cost_ratio'] for x in rows])
        lines.append(f"|{r['case']}|{r['run']}|{ranks}|{ratios}|{mean_ratio:.4f}|")
    lowest_r1_r2_gap=min(float(np.min(np.asarray(r['hindsight_best_segment_anr_by_rank'])[:,:2]
                                       -np.asarray(r['fixed_R4_segment_anr'])[:,None])) for r in results)
    lowest_r3_gap=min(float(np.min(np.asarray(r['hindsight_best_segment_anr_by_rank'])[:,2]
                                  -np.asarray(r['fixed_R4_segment_anr']))) for r in results)
    lines += ["", f"全段均用 R3 的成本比为 {FIXED_COST[2]/COST_R4:.4f}，仍超过 0.90；因此合规的四段平均成本必须至少有一段采用 R1 或 R2。",
              f"跨全部 40 段，R1/R2 即使逐段择优步长，相对固定 R4 的最小 ANR 增量仍为 +{lowest_r1_r2_gap:.3f} dB；R3 的最小增量为 +{lowest_r3_gap:.3f} dB。",
              "这给出了上述有限分支库在 0.90 成本及逐段 0.5 dB 质量门槛下的直接不可行性证据。"]
    lines += ["", "## 实测开销账本与敏感性", "",
              "原运行的逐点账本满足 total = core + candidate + management；表中均为全程平均乘法/点。",
              "将 candidate + management − 4 加到事后选择分支成本；减去的 4 是已在每个固定分支内计入的输出 FIR。",
              "该加法仅为开销敏感性诊断，不能保证另一种项数轨迹会产生完全相同的开销。", "",
              "|工况|运行|原运行 core|candidate|management|total|移植的额外成本|≤0.90 最佳平均差 dB|每段≤0.5 最低成本比|联合可行|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r, s in zip(results, scans):
        a=r['observed_round1_overhead']; best=a['best_mean_at_cost_target']; cheap=a['lowest_cost_meeting_every_segment_margin']
        gap=f"{best['mean_delta_db']:+.3f}" if best else "无可负担组合"
        c=s['cost_ledger']
        lines.append(f"|{r['case']}|{r['run']}|{c['measured_core_mults']:.1f}|{c['measured_candidate_mults']:.1f}|{c['measured_management_mults']:.1f}|{c['measured_total_mults']:.1f}|{c['measured_overhead_above_fixed_branch']:.1f}|{gap}|{cheap['cost_ratio']:.4f}|{'是' if a['feasible_joint'] else '否'}|")
    bank=results[0]
    lines += ["", f"若真实在线维护 8 个秩的完整分支以保留这些训练状态，即使共享特征，成本也约为固定 R4 的 {bank['eight_branch_bank_cost_ratio']:.3f} 倍；",
              f"若五档步长共 40 个分支也同时运行，则约为 {bank['forty_branch_rank_and_mu_bank_cost_ratio']:.3f} 倍。这尚未加候选选择逻辑，不能把免费事后包络当作可交付算法。", "",
              "## 判断与边界", ""]
    if not any(r['zero_selection_overhead']['feasible_joint'] for r in results):
        lines += [f"{len(results)} 个开发运行均在免费选择、免费已训练状态、免费切换的宽松设定下未达到 0.5 dB/0.90 联合门槛。",
                  "因此现有固定控制器库在 E2/E3 中没有显示出足够的低项数容量余量；仅缩减候选占用或调节接受阈值，缺乏达到该门槛的实质空间。",
                  "这不是证明所有动态方法都不可能成功：改变表示、学习动力学或得到优于现有固定分支的低项数模型仍可能改变包络，但必须提供新的独立证据。"]
    else:
        lines += ["至少一个开发运行的零开销包络存在交集，需以因果算法和完整成本判断该空间是否可实现。"]
    lines += ["", "E2/E3 的多项式变化并不等于 RFF 权重矩阵秩变化；本结论不替代 E1 已知秩机制实验。",
              "不允许为得到下降项数曲线而强行删项，也不允许把包络中的按段/按运行选参写成在线选择。", "",
              "## 重建与审计", "",
              "运行 `python -B experiments_v12/diagnostics/capacity_envelope.py --workers 3`；逐运行补扫缓存会被验证后复用，`--refresh` 会重扫同一开发数据。",
              "完整逐段 8×5 数值、输入与原扫描 SHA256、补扫耗时在 capacity_envelope.sweeps.json；",
              "4096 组合的最优候选、逐段质量最低成本及 Pareto 前沿在 capacity_envelope.json。源码使用该轮保存的 v10 核心。",
              "批量五步长更新先与独立单控制器逐步核对；R1/R2/R4/R8 的默认步长再与原运行保存值核对。", ""]
    for s in scans:
        lines.append(f"- {s['case']} run{s['run']}: 默认步长基线最大差 {s['saved_baseline_max_abs_difference_db']:.3g} dB，补扫 {s['seconds']:.1f} 秒。")
    (HERE/'capacity_envelope.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--refresh', action='store_true')
    args=parser.parse_args()
    assert 1 <= args.workers <= 3
    check_batched_update()
    specs=list(itertools.product(('E2','E3'),range(5)))
    cache=HERE/'capacity_envelope.sweeps.json'
    known={}
    if cache.exists() and not args.refresh:
        saved=json.loads(cache.read_text(encoding='utf-8'))
        assert saved['phase'] == 'dev_only_hindsight_diagnostic'
        assert saved['core_sha256'] == sha(Path(ac.__file__))
        known={(row['case'],row['run']):row for row in saved['runs']}
        assert set(known).issubset(set(specs))
    if not args.refresh:
        for case,run in specs:
            checkpoint=HERE/f'capacity_envelope.scan_{case}_run{run:02d}.json'
            if checkpoint.exists():
                known[(case,run)]=json.loads(checkpoint.read_text(encoding='utf-8'))
    audit_and_enrich(list(known.values()))
    pending=[spec for spec in specs if spec not in known]
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(supplement,s):s for s in pending}
            failures=[]
            for future in as_completed(jobs):
                try:
                    row=future.result()
                except Exception as exc:
                    failures.append((jobs[future],exc))
                    continue
                audit_and_enrich([row])
                checkpoint=HERE/f"capacity_envelope.scan_{row['case']}_run{row['run']:02d}.json"
                temporary=checkpoint.with_suffix('.tmp')
                temporary.write_text(json.dumps(row,indent=2),encoding='utf-8')
                temporary.replace(checkpoint)
                known[(row['case'],row['run'])]=row
                print(json.dumps({k:row[k] for k in ('case','run','seconds','saved_baseline_max_abs_difference_db')}),flush=True)
            if failures:
                raise RuntimeError(f"Failed development scans: {failures}")
    scans=[known[spec] for spec in specs]
    audit_and_enrich(scans)
    cache.write_text(json.dumps(dict(phase='dev_only_hindsight_diagnostic',
                     core_sha256=sha(Path(ac.__file__)),runs=scans),indent=2),encoding='utf-8')
    results=[compute_envelope(row) for row in scans]
    (HERE/'capacity_envelope.json').write_text(json.dumps(dict(
        interpretation='Post-hoc optimistic diagnostic, not implementable and not confirmatory evidence',
        cost_r4=COST_R4,cost_target=BUDGET,performance_margin_db=.5,
        capacity_certificate=dict(all_R3_cost_ratio=float(FIXED_COST[2]/COST_R4),
                                  minimum_R1_or_R2_segment_delta_db=min(float(np.min(
                                      np.asarray(r['hindsight_best_segment_anr_by_rank'])[:,:2]
                                      -np.asarray(r['fixed_R4_segment_anr'])[:,None])) for r in results),
                                  minimum_R3_segment_delta_db=min(float(np.min(
                                      np.asarray(r['hindsight_best_segment_anr_by_rank'])[:,2]
                                      -np.asarray(r['fixed_R4_segment_anr']))) for r in results)),
        fixed_branch_cost_ledger=[dict(rank=r, shared_feature_mults=SHARED_FEATURE_COST,
                                       controller_mults=float(FIXED_COST[r-1]-SHARED_FEATURE_COST-OUTPUT_FIR_COST),
                                       output_fir_mults=OUTPUT_FIR_COST,
                                       full_branch_mults=float(FIXED_COST[r-1])) for r in range(1,9)],
        runs=results),indent=2),encoding='utf-8')
    write_report(results,scans)
    for r in results:
        oracle=r['zero_selection_overhead']
        print(json.dumps(dict(case=r['case'],run=r['run'],
                              best_mean_delta_db=oracle['best_mean_at_cost_target']['mean_delta_db'],
                              lowest_quality_cost_ratio=oracle['lowest_cost_meeting_every_segment_margin']['cost_ratio'],
                              feasible_joint=oracle['feasible_joint'])),flush=True)


if __name__=='__main__':
    main()
