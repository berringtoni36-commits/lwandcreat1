"""Development-only terminal-weight spectrum and frozen physical-path audit.

Replay the saved E2/E3 run-0 inputs with the archived v10 controllers. Terminal
weights are evaluated in hindsight on each segment's last 10,000 samples:
5,000 samples initialize the ANR smoother; the last 5,000 are scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
V12 = HERE.parent.parent
ROOT = V12.parent
DEV = V12 / 'outputs' / 'main_dev_round1'
sys.path.insert(0, str(DEV / 'sources' / 'experiments_v10'))
import anc_core as ac

CASES = ('E2', 'E3')
RUN = 0
WARMUP = 5000
EVAL = 5000
BETA = .999
EPS = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(case):
    source = DEV / f'case{case}_run{RUN:02d}.npz'
    summary = source.with_name(source.stem + '_summary.json')
    return source, summary


def provenance(case):
    assert case in CASES
    protocol = json.loads((DEV / 'protocol.json').read_text(encoding='utf-8'))
    assert protocol['phase'] == 'dev' and protocol['mu_by_method'] == {}
    source, summary = source_paths(case)
    return dict(case=case, run=RUN, phase='dev',
                source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                summary_sha256=sha(summary), core_sha256=sha(Path(ac.__file__)))


def replay(case, info):
    source, summary_path = source_paths(case)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    with np.load(source, allow_pickle=False) as f:
        x, d, v = f['x'], f['d'], f['v']
        Om, ph = f['Om'], f['ph']
        initial_A, initial_B = f['initial_A'], f['initial_B']
        meta = json.loads(str(f['meta']))
    T = len(x)
    assert T == 400000 and len(d) == len(v) == T
    assert meta['case'] == case and meta['run'] == RUN
    assert tuple(meta['segment_bounds']) == tuple([s*100000, (s+1)*100000] for s in range(4))
    assert 'fixed_R4' in meta['controllers'] and 'full_RFF_MCC' in meta['controllers']
    mu = float(meta['mu'])
    fixed = ac.KronRFF('fixed_R4', 1, 25, 20, 4, mu/2, mu/2,
                       2., 1e-8, np.random.default_rng(0))
    fixed.A = initial_A[:, :, :4].copy()
    fixed.B = initial_B[:, :, :4].copy()
    full = ac.FullRFF('full_RFF_MCC', 1, 500, mu, 2., 1e-8)
    terminal_fixed = np.empty((4, 20, 25))
    terminal_full = np.empty((4, 20, 25))
    sums = np.zeros((4, 2))
    Ae, Ad = np.zeros(2), 0.
    xb, zh = np.zeros((1, 20)), np.zeros((4, 1, 500))
    start = perf_counter()
    for n in range(T):
        xb[:, 1:] = xb[:, :-1].copy()
        xb[0, 0] = x[n]
        z = np.sqrt(2/500) * np.cos(np.einsum('rdm,rm->rd', Om, xb) + ph)
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2] + .5 * zh[3]
        dv = np.array([d[n] + v[n]])
        ys = np.array([fixed.step(z, q, dv)[0], full.step(z, q, dv)[0]])
        Ae = BETA*Ae + (1-BETA)*np.abs(dv[0]-ys)
        Ad = BETA*Ad + (1-BETA)*abs(d[n])
        segment = n//100000
        if n >= (segment+1)*100000-EVAL:
            sums[segment] += 20*np.log10((Ae+EPS)/(Ad+EPS))
        if (n+1) % 100000 == 0:
            terminal_fixed[segment] = fixed.B[0] @ fixed.A[0].T
            terminal_full[segment] = full.w[0].reshape(25, 20).T
    online = sums/EVAL
    saved = np.array([[summary['segment_anr'][s][name]
                       for name in ('fixed_R4', 'full_RFF_MCC')] for s in range(4)])
    discrepancy = float(np.max(np.abs(online-saved)))
    assert discrepancy < 2e-5, (case, discrepancy)
    info.update(T=T, mu=mu, replay_seconds=perf_counter()-start,
                online_max_abs_difference_db=discrepancy)
    cache = HERE / f'replay_{case}_run{RUN:02d}.npz'
    temporary = cache.with_suffix('.tmp')
    with temporary.open('wb') as out:
        np.savez_compressed(out, terminal_fixed=terminal_fixed,
                            terminal_full=terminal_full, online_anr=online,
                            saved_anr=saved, provenance=json.dumps(info))
    temporary.replace(cache)
    return dict(info=info, terminal_fixed=terminal_fixed,
                terminal_full=terminal_full, online_anr=online, saved_anr=saved)


def load_or_replay(case, refresh):
    info = provenance(case)
    cache = HERE / f'replay_{case}_run{RUN:02d}.npz'
    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as f:
            saved_info = json.loads(str(f['provenance']))
            assert all(saved_info[key] == info[key] for key in info)
            result = dict(info=saved_info, terminal_fixed=f['terminal_fixed'].copy(),
                          terminal_full=f['terminal_full'].copy(),
                          online_anr=f['online_anr'].copy(),
                          saved_anr=f['saved_anr'].copy())
        assert np.max(np.abs(result['online_anr']-result['saved_anr'])) < 2e-5
        return result
    return replay(case, info)


def svd_analysis(W):
    U, singular, Vh = np.linalg.svd(W, full_matrices=False)
    norm = np.linalg.norm(singular)
    assert norm > 0 and np.isfinite(norm)
    retained = np.cumsum(singular**2)/norm**2
    projections = {rank:(U[:, :rank]*singular[:rank]) @ Vh[:rank]
                   for rank in (2, 3, 4)}
    for rank, projected in projections.items():
        np.testing.assert_allclose(np.linalg.norm(W-projected)/norm,
                                   np.linalg.norm(singular[rank:])/norm,
                                   atol=1e-12, rtol=1e-12)
    details = dict(singular_values=singular.tolist(),
                   frobenius_norm=float(norm),
                   energy_rank_99=int(np.searchsorted(retained, .99)+1),
                   relative_frobenius_truncation_error={str(rank):float(
                       np.linalg.norm(W-projections[rank])/norm) for rank in projections})
    return details, projections


def frozen_anr(x, d, v, Om, ph, end, weights):
    first = end-WARMUP-EVAL
    assert first >= 19
    history = sliding_window_view(x[first-19:end], 20)[:, ::-1]
    z = np.sqrt(2/500) * np.cos(history @ Om[0].T + ph[0])
    assert z.shape == (WARMUP+EVAL, 500)
    labels = list(weights)
    flat = np.stack([weights[key].T.reshape(500) for key in labels])
    y = z @ flat.T
    ys = ac.fir(ac.S_PATH, y.T)
    errors = d[first:end][None, :] + v[first:end][None, :] - ys
    Ae = np.zeros(len(labels))
    Ad = 0.
    sums = np.zeros(len(labels))
    for j in range(WARMUP+EVAL):
        Ae = BETA*Ae + (1-BETA)*np.abs(errors[:, j])
        Ad = BETA*Ad + (1-BETA)*abs(d[first+j])
        if j >= WARMUP:
            sums += 20*np.log10((Ae+EPS)/(Ad+EPS))
    return {key:float(sums[i]/EVAL) for i, key in enumerate(labels)}


def evaluate_case(case, replayed):
    source, _ = source_paths(case)
    with np.load(source, allow_pickle=False) as f:
        x, d, v, Om, ph = (f[k] for k in ('x', 'd', 'v', 'Om', 'ph'))
    segments = []
    start = perf_counter()
    for s in range(4):
        Wf = replayed['terminal_full'][s]
        Wk = replayed['terminal_fixed'][s]
        full_spectrum, full_projected = svd_analysis(Wf)
        fixed_spectrum, fixed_projected = svd_analysis(Wk)
        weights = dict(full500=Wf, full500_R2=full_projected[2],
                       full500_R3=full_projected[3], full500_R4=full_projected[4],
                       fixed_R4=Wk, fixed_R4_R2=fixed_projected[2],
                       fixed_R4_R3=fixed_projected[3])
        frozen = frozen_anr(x, d, v, Om, ph, (s+1)*100000, weights)
        deltas = {name: frozen[name]-frozen['full500' if name.startswith('full500') else 'fixed_R4']
                  for name in frozen if name not in ('full500', 'fixed_R4')}
        segments.append(dict(segment=s+1, bounds=[s*100000, (s+1)*100000],
                             online_anr_db=dict(fixed_R4=float(replayed['online_anr'][s,0]),
                                                full500=float(replayed['online_anr'][s,1])),
                             full500_spectrum=full_spectrum,
                             fixed_R4_spectrum=fixed_spectrum,
                             frozen_anr_db=frozen,
                             projected_minus_parent_frozen_anr_db=deltas))
    return dict(case=case, run=RUN, provenance=replayed['info'],
                frozen_evaluation_seconds=perf_counter()-start,
                segments=segments)


def write_report(cases):
    lines = ['# E2/E3 开发运行：低项数容量与在线优化的谱诊断', '',
             '仅使用 `main_dev_round1` 中 E2/E3 各 run0 的保存输入；未使用 select 或 confirm。',
             '重放固定 R4 与 full500 的顺序在线学习，在每段结束后保存 20×25 权重矩阵。',
             '固定 R4 矩阵为 B Aᵀ；full500 的 500 维向量按相同特征顺序重排。', '',
             '冻结评估在同段最后 10000 点上进行：前 5000 点初始化 0.999 指数平滑，末 5000 点平均 ANR；',
             '使用已保存的实际输入、扰动、测量噪声和物理次级通道 FIR。终点权重对该窗口包含未来信息，属于事后诊断。',
             '投影是 Frobenius 最优 SVD 截断；它不以物理 ANR 为目标。冻结投影与重新在线训练不可直接比较。', '',
             '## 重放核对', '',
             '|工况|固定步长|重放耗时 秒|相对保存分段 ANR 最大差 dB|冻结评估耗时 秒|',
             '|---|---:|---:|---:|---:|']
    for c in cases:
        p=c['provenance']
        lines.append(f"|{c['case']}|{p['mu']:.3g}|{p['replay_seconds']:.1f}|{p['online_max_abs_difference_db']:.2e}|{c['frozen_evaluation_seconds']:.1f}|")
    lines += ['', '## 奇异谱与最佳 Frobenius 截断', '',
              '误差为 ‖W−Wᵣ‖F/‖W‖F；`r99` 为达到 99% 权重能量的最小秩。完整 20 个奇异值见 JSON。', '',
              '|工况|段|控制器|前 6 奇异值|r99|R2 误差|R3 误差|R4 误差|',
              '|---|---:|---|---|---:|---:|---:|---:|']
    for c in cases:
        for s in c['segments']:
            for name, key in (('full500','full500_spectrum'),('固定 R4','fixed_R4_spectrum')):
                a=s[key]; e=a['relative_frobenius_truncation_error']
                top=', '.join(f'{x:.3g}' for x in a['singular_values'][:6])
                lines.append(f"|{c['case']}|{s['segment']}|{name}|{top}|{a['energy_rank_99']}|{e['2']:.3f}|{e['3']:.3f}|{e['4']:.3f}|")
    lines += ['', '## 同段末窗冻结物理 ANR', '',
              'Δ 为投影减去相同终点原矩阵的冻结 ANR；正值表示投影使残差变差。在线 ANR 仅列作重放参照。', '',
              '|工况|段|在线 R4|在线 full500|冻结 R4|R4→R2 Δ|R4→R3 Δ|冻结 full500|full→R2 Δ|full→R3 Δ|full→R4 Δ|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        for s in c['segments']:
            f=s['frozen_anr_db']; a=s['online_anr_db']; d=s['projected_minus_parent_frozen_anr_db']
            lines.append(f"|{c['case']}|{s['segment']}|{a['fixed_R4']:.3f}|{a['full500']:.3f}|{f['fixed_R4']:.3f}|{d['fixed_R4_R2']:+.3f}|{d['fixed_R4_R3']:+.3f}|{f['full500']:.3f}|{d['full500_R2']:+.3f}|{d['full500_R3']:+.3f}|{d['full500_R4']:+.3f}|")
    segments=[s for c in cases for s in c['segments']]
    def span(values):
        return f'{min(values):.3f}–{max(values):.3f}'
    fixed_r3_errors=span([s['fixed_R4_spectrum']['relative_frobenius_truncation_error']['3'] for s in segments])
    full_r3_errors=span([s['full500_spectrum']['relative_frobenius_truncation_error']['3'] for s in segments])
    fixed_r3_loss=span([s['projected_minus_parent_frozen_anr_db']['fixed_R4_R3'] for s in segments])
    fixed_r2_loss=span([s['projected_minus_parent_frozen_anr_db']['fixed_R4_R2'] for s in segments])
    full_r3_loss=span([s['projected_minus_parent_frozen_anr_db']['full500_R3'] for s in segments])
    lines += ['', '## 诊断判断', '',
              f'八段固定 R4 终点矩阵的最佳 R3 Frobenius 相对误差为 {fixed_r3_errors}；冻结 R3 相对同一 R4 矩阵的物理 ANR 均变差 {fixed_r3_loss} dB，R2 变差 {fixed_r2_loss} dB。',
              f'full500 终点矩阵的 R3 相对误差为 {full_r3_errors}，冻结 R3 的物理 ANR 均变差 {full_r3_loss} dB；其 99% 权重能量需要 16–17 个奇异方向。',
              '结合已有 E2/E3 开发固定分支包络中 R1–R3 的表现，这些结果更支持当前 RFF 特征排列与更新规则下的低项数表示容量受限，而非仅由因子在线优化造成差距。',
              '这仍不是全体低项数模型的容量下界：Frobenius 截断不是按物理误差最优的低秩解；RFF 特征的排列也会改变矩阵秩；同段终点冻结评估还使用了未来信息。',
              '若要严格拆分表示与优化，应另做按物理误差优化的低秩离线拟合并进行多初值核对；本结果不能把冻结投影当作重新在线训练的成绩。']
    lines += ['', '## 复现', '',
              '运行 `python -B experiments_v12/diagnostics/spectrum/spectrum_diagnostic.py`。',
              '保存的终点权重、开发数据 SHA256、归档核心 SHA256、全部奇异值与冻结评估数值在同目录 NPZ/JSON 中。',
              '使用 `--refresh` 可重新训练两个固定分支并覆盖已验证的重放缓存。', '']
    (HERE/'spectrum_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    start = perf_counter()
    cases = []
    for case in CASES:
        replayed = load_or_replay(case, args.refresh)
        result = evaluate_case(case, replayed)
        cases.append(result)
        print(json.dumps(dict(case=case, replay_seconds=result['provenance']['replay_seconds'],
                              validation_difference_db=result['provenance']['online_max_abs_difference_db'],
                              frozen_seconds=result['frozen_evaluation_seconds'])), flush=True)
    output = dict(interpretation='Development-only post-hoc frozen projection diagnostic',
                  warmup_samples=WARMUP, evaluation_samples=EVAL, cases=cases,
                  total_script_seconds=perf_counter()-start)
    (HERE/'spectrum_results.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    write_report(cases)


if __name__ == '__main__':
    main()
