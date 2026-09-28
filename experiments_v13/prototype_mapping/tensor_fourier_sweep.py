"""Development-only tensor Fourier map and fixed-rank capacity sweep.

Uses only saved E2/E3 run-0 inputs from experiments_v12/main_dev_round1.
The v10 archived controller is reused without modification so the intervention
is precisely the feature map. All R1..R8 branches share that new map.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEV = ROOT / 'experiments_v12' / 'outputs' / 'main_dev_round1'
sys.path.insert(0, str(DEV / 'sources' / 'experiments_v10'))
import anc_core as ac

CASES = ('E2', 'E3')
RUN = 0
MU = (.05, .1, .2, .4, .8)
ELL = 3.9
SEED = 2026092801
FEATURE_PROJECTION_MULTS = 24*10 + 19*10
FEATURE_SCALE_MULTS = 24 + 19
FEATURE_OUTER_MULTS = 25*20
FEATURE_MULTS = FEATURE_PROJECTION_MULTS + FEATURE_SCALE_MULTS + FEATURE_OUTER_MULTS
FILTER_SPARSE_MULTS = 500
FILTER_NOMINAL_MULTS = 500*len(ac.S_PATH)
OLD_SHARED = 500*(20+1+len(ac.S_PATH))
CONTROLLER_MULTS = np.array([ac.mults_kron(25,20,r,20,4)-OLD_SHARED for r in range(1,9)])
OUTPUT_FIR_MULTS = len(ac.S_PATH)
COST_NOMINAL = FEATURE_MULTS + FILTER_NOMINAL_MULTS + CONTROLLER_MULTS + OUTPUT_FIR_MULTS
COST_SPARSE = FEATURE_MULTS + FILTER_SPARSE_MULTS + CONTROLLER_MULTS + OUTPUT_FIR_MULTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(case):
    source = DEV / f'case{case}_run{RUN:02d}.npz'
    return source, source.with_name(source.stem+'_fixed_sweep.json')


def map_parameters(case):
    assert case in CASES
    case_id = 1 if case == 'E2' else 2
    rng = np.random.default_rng([SEED, case_id, RUN, 13])
    omega_a = rng.normal(0., 1./ELL, (24, 10))
    phase_a = rng.uniform(0., 2*np.pi, 24)
    omega_b = rng.normal(0., 1./ELL, (19, 10))
    phase_b = rng.uniform(0., 2*np.pi, 19)
    return omega_a, phase_a, omega_b, phase_b


def save_map_parameters(case):
    params=map_parameters(case)
    path=HERE/f'tensor_map_{case}_run{RUN:02d}.npz'
    keys=('omega_a','phase_a','omega_b','phase_b')
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            for key,value in zip(keys,params):
                np.testing.assert_array_equal(f[key],value)
    else:
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as out:
            np.savez_compressed(out,**dict(zip(keys,params)))
        temporary.replace(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))


def tensor_features(history, params):
    omega_a, phase_a, omega_b, phase_b = params
    a = np.empty(25)
    b = np.empty(20)
    a[0] = b[0] = 1/np.sqrt(2)
    a[1:] = np.cos(omega_a @ history[::2] + phase_a)/np.sqrt(24)
    b[1:] = np.cos(omega_b @ history[1::2] + phase_b)/np.sqrt(19)
    return np.outer(a,b).reshape(1,500)


def check_batched_controllers():
    rng = np.random.default_rng(60291)
    params=map_parameters('E2')
    history=rng.normal(size=20)
    z=tensor_features(history,params)
    omega_a,phase_a,omega_b,phase_b=params
    a=np.r_[1/np.sqrt(2),np.cos(omega_a@history[::2]+phase_a)/np.sqrt(24)]
    b=np.r_[1/np.sqrt(2),np.cos(omega_b@history[1::2]+phase_b)/np.sqrt(19)]
    np.testing.assert_allclose(z.reshape(25,20).T,np.outer(b,a),atol=1e-14)
    A=rng.normal(size=(25,4));B=rng.normal(size=(20,4))
    W=B@A.T
    np.testing.assert_allclose((z@W.T.reshape(500)).item(),float(np.sum(W*np.outer(b,a))),atol=1e-12)
    batch = ac.KronRFF('batch',5,25,20,4,np.array(MU)/2,np.array(MU)/2,
                       2.,1e-8,rng)
    singles=[]
    for j,mu in enumerate(MU):
        c=ac.KronRFF('single',1,25,20,4,mu/2,mu/2,2.,1e-8,rng)
        c.A=batch.A[j:j+1].copy();c.B=batch.B[j:j+1].copy()
        singles.append(c)
    for _ in range(100):
        z,q,dv=rng.normal(size=(1,500)),rng.normal(size=(1,500)),rng.normal(size=1)
        got=batch.step(z,q,dv)
        want=np.array([c.step(z,q,dv)[0] for c in singles])
        np.testing.assert_allclose(got,want,rtol=1e-12,atol=1e-12)


def provenance(case):
    protocol = json.loads((DEV/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['phase']=='dev' and case in protocol['cases']
    source, old_sweep = source_paths(case)
    return dict(case=case,run=RUN,phase='dev',
                source=str(source.relative_to(ROOT)),source_sha256=sha(source),
                old_fixed_sweep_sha256=sha(old_sweep),
                archived_core_sha256=sha(Path(ac.__file__)),
                map_seed=[SEED,1 if case=='E2' else 2,RUN,13],
                ell=ELL,mu=list(MU))


def scan(case, info):
    source, _ = source_paths(case)
    with np.load(source,allow_pickle=False) as f:
        x,d,v=f['x'],f['d'],f['v']
        A,B=f['initial_A'],f['initial_B']
        meta=json.loads(str(f['meta']))
    T=len(x)
    assert T==400000 and len(d)==len(v)==T
    assert meta['case']==case and meta['run']==RUN
    params=map_parameters(case)
    controllers=[]
    for rank in range(1,9):
        c=ac.KronRFF(f'R{rank}',5,25,20,rank,np.array(MU)/2,np.array(MU)/2,
                     2.,1e-8,np.random.default_rng(0))
        c.A=np.repeat(A[:,:,:rank],5,axis=0)
        c.B=np.repeat(B[:,:,:rank],5,axis=0)
        controllers.append(c)
    sums=np.zeros((4,8,5))
    Ae=np.zeros((8,5))
    Ad=0.
    history=np.zeros(20)
    zh=np.zeros((4,1,500))
    feature_norm_sum=0.
    start=perf_counter()
    for n in range(T):
        history[1:]=history[:-1].copy()
        history[0]=x[n]
        z=tensor_features(history,params)
        feature_norm_sum+=float(np.sum(z*z))
        zh[1:]=zh[:-1].copy()
        zh[0]=z
        q=zh[2]+.5*zh[3]
        dv=np.array([d[n]+v[n]])
        Ad=.999*Ad+.001*abs(d[n])
        segment=n//100000
        tally=n >= (segment+1)*100000-5000
        for j,c in enumerate(controllers):
            ys=c.step(z,q,dv)
            Ae[j]=.999*Ae[j]+.001*np.abs(dv[0]-ys)
            if tally:
                sums[segment,j]+=20*np.log10((Ae[j]+1e-12)/(Ad+1e-12))
    table=sums/5000
    assert np.isfinite(table).all()
    info.update(samples=T,seconds=perf_counter()-start,
                mean_feature_norm_squared=feature_norm_sum/T)
    return dict(provenance=info,segment_anr=table.tolist())


def load_or_scan(case,refresh):
    info=provenance(case)
    cache=HERE/f'tensor_{case}_run{RUN:02d}.json'
    if cache.exists() and not refresh:
        row=json.loads(cache.read_text(encoding='utf-8'))
        assert all(row['provenance'][key]==info[key] for key in info)
        assert np.asarray(row['segment_anr']).shape==(4,8,5)
        return row
    row=scan(case,info)
    temporary=cache.with_suffix('.tmp')
    temporary.write_text(json.dumps(row,indent=2),encoding='utf-8')
    temporary.replace(cache)
    return row


def old_reference(case):
    _, path=source_paths(case)
    sweep=json.loads(path.read_text(encoding='utf-8'))
    table=np.array(sweep['segment_anr'])
    choices=[j for j,name in enumerate(sweep['names']) if name.startswith('R4_mu')]
    best=min(choices,key=lambda j:table[:,j].mean())
    return dict(mu=float(sweep['names'][best].split('_mu')[1]),
                segment_anr=table[:,best].tolist())


def analyze(row):
    case=row['provenance']['case']
    full=np.asarray(row['segment_anr'])
    r4_mu=int(full[:,3,:].mean(axis=0).argmin())
    ref=full[:,3,r4_mu]
    oracle=full.min(axis=2)
    best_mu=full.argmin(axis=2)
    combos=np.array(list(itertools.product(range(8),repeat=4)))
    scores=oracle[np.arange(4)[None,:],combos]
    delta=scores-ref
    mean_delta=delta.mean(axis=1)
    worst_delta=delta.max(axis=1)
    costs=COST_NOMINAL[combos].mean(axis=1)
    affordable=np.flatnonzero(costs <= .9*COST_NOMINAL[3]+1e-9)
    quality=np.flatnonzero(np.all(delta<=.5,axis=1))
    best=int(affordable[np.argmin(mean_delta[affordable])])
    cheapest=int(quality[np.argmin(costs[quality])])
    def selection(index):
        ranks=combos[index]
        return dict(ranks=(ranks+1).tolist(),
                    mu_by_segment=[MU[best_mu[s,ranks[s]]] for s in range(4)],
                    segment_delta_db=delta[index].tolist(),
                    mean_delta_db=float(mean_delta[index]),
                    worst_segment_delta_db=float(worst_delta[index]),
                    mean_mults_nominal=float(costs[index]),
                    cost_ratio_nominal=float(costs[index]/COST_NOMINAL[3]),
                    cost_ratio_sparse=float(COST_SPARSE[ranks].mean()/COST_SPARSE[3]))
    frontier=[]
    threshold=np.inf
    for cost in sorted(set(costs)):
        eligible=np.flatnonzero(costs==cost)
        j=eligible[np.argmin(mean_delta[eligible])]
        if mean_delta[j]<threshold-1e-12:
            frontier.append(selection(j));threshold=mean_delta[j]
    fixed=[]
    for rank in range(8):
        j=int(full[:,rank,:].mean(axis=0).argmin())
        fixed.append(dict(rank=rank+1,mu=MU[j],segment_anr=full[:,rank,j].tolist(),
                          mean_anr=float(full[:,rank,j].mean()),
                          mean_delta_vs_new_R4_db=float(full[:,rank,j].mean()-ref.mean()),
                          cost_ratio_nominal=float(COST_NOMINAL[rank]/COST_NOMINAL[3]),
                          cost_ratio_sparse=float(COST_SPARSE[rank]/COST_SPARSE[3])))
    per_segment=[]
    for s in range(4):
        qual=np.flatnonzero(oracle[s]-ref[s]<=.5)
        rank=int(qual[np.argmin(COST_NOMINAL[qual])])
        per_segment.append(dict(segment=s+1,rank=rank+1,
                                mu=MU[best_mu[s,rank]],
                                delta_db=float(oracle[s,rank]-ref[s]),
                                cost_ratio_nominal=float(COST_NOMINAL[rank]/COST_NOMINAL[3])))
    return dict(case=case,run=RUN,reference_new_R4_mu=MU[r4_mu],
                reference_new_R4_segment_anr=ref.tolist(),
                old_mapping_R4_context=old_reference(case),
                fixed_rank_frontier=fixed,
                hindsight_best_anr_by_segment_rank=oracle.tolist(),
                hindsight_best_mu_by_segment_rank=[[MU[j] for j in row] for row in best_mu],
                per_segment_lowest_quality_cost=per_segment,
                oracle_best_mean_at_0_9=selection(best),
                oracle_lowest_cost_meeting_margin=selection(cheapest),
                nominal_budget_headroom_after_quality_oracle=float(.9*COST_NOMINAL[3]-costs[cheapest]),
                oracle_joint_feasible=bool(np.any((costs<=.9*COST_NOMINAL[3]) & np.all(delta<=.5,axis=1))),
                zero_overhead_mean_loss_pareto_frontier=frontier)


def write_report(scans,analyses):
    lines=['# 张量 Fourier 映射的低项数容量试验（E2/E3 开发数据）','',
           '只使用 `main_dev_round1` 的 E2/E3 各 run0 已保存输入、扰动、噪声和嵌套初值；不使用 select/confirm。',
           '这是一个独立原型与事后固定分支容量诊断，尚非动态方法，也不是论文确认性结果。','',
           '## 映射与更新公式','',
           '令 20 个当前至历史样本按偶/奇滞后分成两个 10 维向量 hₐ 与 hᵦ。',
           'a₀=b₀=1/√2；对 i=1…24，aᵢ=cos(ωₐᵢᵀhₐ+βₐᵢ)/√24；对 j=1…19，bⱼ=cos(ωᵦⱼᵀhᵦ+βᵦⱼ)/√19。',
           '频率各自从 N(0,3.9⁻² I₁₀) 采样、相位从 U[0,2π) 采样；固定的种子与参数记录在结果文件。',
           '按原 25×20 特征顺序，z₂₀ᵢ₊ⱼ=aᵢbⱼ，重排矩阵 Z=b aᵀ。固定 R 分支 W=B Aᵀ、y=tr(Bᵀ Z A)；',
           'filtered-x 为 qₙ=zₙ₋₂+0.5zₙ₋₃，故矩阵 Q=bₙ₋₂aₙ₋₂ᵀ+0.5bₙ₋₃aₙ₋₃ᵀ，秩至多 2；物理输出 FIR 仍为原次级通道 [0,0,1,0.5]。',
           '常数坐标保留单侧非线性；全部 R1–R8 及五档步长共用同一新映射和保存初值前缀。控制器更新采用归档 v10 顺序 MCC 核心。','',
           '## 每点乘法成本口径','',
           f'频率投影 {FEATURE_PROJECTION_MULTS}，余弦后缩放 {FEATURE_SCALE_MULTS}，张量外积 {FEATURE_OUTER_MULTS}，生成特征合计 {FEATURE_MULTS}；另有 43 次余弦求值，未折为乘法。',
           f'实际稀疏 filtered-x 需 {FILTER_SPARSE_MULTS} 次乘法（乘 0.5）；与 v10 同口径的保守四抽头计数为 {FILTER_NOMINAL_MULTS}。',
           '两套成本均给每个秩使用同一实现口径；主门槛采用保守四抽头计数。',
           '沿用 v10 顺序更新的秩依赖账本 1655R+8，并计输出 FIR 4 次；未额外计入候选维护、选择与切换。','',
           '|项数|完整成本：保守|相对新 R4|完整成本：稀疏滤波|相对新 R4|',
           '|---:|---:|---:|---:|---:|']
    for rank in range(8):
        lines.append(f'|R{rank+1}|{COST_NOMINAL[rank]:.0f}|{COST_NOMINAL[rank]/COST_NOMINAL[3]:.4f}|{COST_SPARSE[rank]:.0f}|{COST_SPARSE[rank]/COST_SPARSE[3]:.4f}|')
    lines += ['','## 同一新映射下的固定秩与乐观前沿','',
              '每段 ANR 为该段末 5000 点的原协议平均。固定秩步长以该运行四段等权平均事后选定。',
              'R4 参照也使用同一新映射和同一五档步长优化口径；旧映射 R4 仅在后文作绝对性能背景。','',
              '|工况|项数|最佳步长|平均 ANR dB|相对新 R4 平均差 dB|保守成本/R4|',
              '|---|---:|---:|---:|---:|---:|']
    for a in analyses:
        for f in a['fixed_rank_frontier']:
            lines.append(f"|{a['case']}|R{f['rank']}|{f['mu']:.2g}|{f['mean_anr']:.3f}|{f['mean_delta_vs_new_R4_db']:+.3f}|{f['cost_ratio_nominal']:.4f}|")
    lines += ['','乐观策略可在每段预知未来并选择已训练好的秩/步长，训练状态、选择和切换全部免费；它只诊断潜在容量。','',
              '|工况|≤0.9 成本最佳平均差 dB|对应最差段差 dB|秩轨迹|逐段≤0.5 最低成本/R4|余下开销空间 乘法/点|联合可行|',
              '|---|---:|---:|---|---:|---:|---|']
    for a in analyses:
        best=a['oracle_best_mean_at_0_9'];cheap=a['oracle_lowest_cost_meeting_margin']
        lines.append(f"|{a['case']}|{best['mean_delta_db']:+.3f}|{best['worst_segment_delta_db']:+.3f}|{best['ranks']}|{cheap['cost_ratio_nominal']:.4f}|{a['nominal_budget_headroom_after_quality_oracle']:.1f}|{'是' if a['oracle_joint_feasible'] else '否'}|")
    lines += ['','|工况|逐段≤0.5 所需最低项数|逐段最低保守成本/R4|',
              '|---|---|---|']
    for a in analyses:
        rows=a['per_segment_lowest_quality_cost']
        lines.append(f"|{a['case']}|{[x['rank'] for x in rows]}|{[round(x['cost_ratio_nominal'],4) for x in rows]}|")
    lines += ['','## 绝对性能背景与边界','',
              '旧映射 R4 同样从五档步长中选该开发运行的平均最优；这仅检查新映射是否以整体性能退化换得低项数空间。',
              '|工况|旧映射 R4 平均 ANR|新映射 R4 平均 ANR|新减旧 dB|新特征平均平方范数|扫描耗时 秒|',
              '|---|---:|---:|---:|---:|---:|']
    for a,s in zip(analyses,scans):
        old=np.mean(a['old_mapping_R4_context']['segment_anr'])
        new=np.mean(a['reference_new_R4_segment_anr'])
        p=s['provenance']
        lines.append(f"|{a['case']}|{old:.3f}|{new:.3f}|{new-old:+.3f}|{p['mean_feature_norm_squared']:.3f}|{p['seconds']:.1f}|")
    full_path=HERE/'full_tensor_results.json'
    if full_path.exists():
        full=json.loads(full_path.read_text(encoding='utf-8'))['rows']
        assert [x['case'] for x in full]==list(CASES)
        lines += ['','另以同一新映射重放 full500，并在同一五档步长中选各开发运行四段平均最优：','',
                  '|工况|新映射 full500 最佳步长|平均 ANR dB|相比新 R4 dB|相比旧映射 R4 dB|扫描耗时 秒|',
                  '|---|---:|---:|---:|---:|---:|']
        for r in full:
            lines.append(f"|{r['case']}|{r['best_mu']:.2g}|{r['mean_anr']:.3f}|{r['full_minus_new_R4_db']:+.3f}|{r['full_minus_old_R4_db']:+.3f}|{r['seconds']:.1f}|")
        lines += ['','**判断：当前候选映射不宜推进为论文动态项数方案。** 它在新映射内部产生了低项数/成本交集，但 full500 仍明显弱于旧映射 R4，说明绝对抑噪损失主要来自新映射本身。',
                  '新映射固定 R1 相对新 R4 的四段平均差在 E2/E3 分别为 +5.173/+4.306 dB，故 R1 不支配这一内部动态潜力；但内部潜力无法弥补映射级退化。',
                  '新映射下 E2 固定 R3 的平均差虽仅 +0.124 dB，第四段仍差 +0.619 dB；E3 固定 R3 的平均差为 +0.776 dB。事后按段换秩/步长的收益尚无在线因果实现与候选/切换成本。']
    lines += ['','以上乐观前沿不含在线维持多个分支的成本，不得写成已实现的动态节省。',
              '映射本身改变了特征空间；任何绝对 ANR 改善不能归因于项数选择。',
              '两个开发运行仅用于筛查方案，不能作统计确认。','',
              '## 复现','',
              '先运行 `python -B experiments_v13/prototype_mapping/tensor_fourier_sweep.py`，再运行 `python -B experiments_v13/prototype_mapping/full_tensor_check.py`，最后运行 `python -B experiments_v13/prototype_mapping/verify_tensor.py`；已完成的单工况 JSON 会验证来源 SHA256 后复用，首个脚本 `--refresh` 重扫。',
              '两张 4×8×5 完整分段表、实际映射参数 NPZ、输入/旧扫描/核心 SHA256、4096 轨迹前沿与全部成本记录在同目录 JSON。','']
    (HERE/'tensor_fourier_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    check_batched_controllers()
    scans=[]
    map_files=[]
    for case in CASES:
        map_files.append(save_map_parameters(case))
        row=load_or_scan(case,args.refresh)
        scans.append(row)
        print(json.dumps(dict(case=case,seconds=row['provenance']['seconds'],
                              mean_feature_norm_squared=row['provenance']['mean_feature_norm_squared'])),flush=True)
    analyses=[analyze(row) for row in scans]
    result=dict(interpretation='Development-only post-hoc fixed-branch capacity diagnostic',
                feature_cost=dict(projection=FEATURE_PROJECTION_MULTS,
                                  scale=FEATURE_SCALE_MULTS,outer=FEATURE_OUTER_MULTS,
                                  total=FEATURE_MULTS,cosine_evaluations=43,
                                  filtered_x_sparse=FILTER_SPARSE_MULTS,
                                  filtered_x_nominal=FILTER_NOMINAL_MULTS),
                cost_nominal=COST_NOMINAL.tolist(),cost_sparse=COST_SPARSE.tolist(),
                cost_target_ratio=.9,performance_margin_db=.5,
                map_parameter_files=map_files,analyses=analyses)
    (HERE/'tensor_fourier_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    write_report(scans,analyses)
    for a in analyses:
        print(json.dumps(dict(case=a['case'],
                              new_R4_mu=a['reference_new_R4_mu'],
                              best_at_0_9=a['oracle_best_mean_at_0_9']['mean_delta_db'],
                              lowest_quality_cost_ratio=a['oracle_lowest_cost_meeting_margin']['cost_ratio_nominal'],
                              feasible_joint=a['oracle_joint_feasible'])),flush=True)


if __name__=='__main__':
    main()
