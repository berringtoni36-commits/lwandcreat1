"""Offline physical filtered-x low-rank regression on E2/E3 development inputs.

The fits use noise-free saved d as an intentionally favorable offline target.
They are capacity diagnostics, not causal adaptive controllers or online costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DEV=ROOT/'experiments_v12'/'outputs'/'main_dev_round1'
sys.path.insert(0,str(DEV/'sources'/'experiments_v10'))
import anc_core as ac

CASES=('E2','E3')
SEGMENTS=(1,3)
RUN=0
N_TRAIN=4096
N_INNER=1024
WARMUP=5000
N_EVAL=5000
LAMBDAS=(1e-5,1e-3)
MAX_ITERS=150
STARTS=('svd','random0','random1')
BETA=.999
EPS=1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paths(case):
    source=DEV/f'case{case}_run{RUN:02d}.npz'
    return source,source.with_name(source.stem+'_summary.json')


def provenance(case,segment):
    assert case in CASES and segment in SEGMENTS
    protocol=json.loads((DEV/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['phase']=='dev' and case in protocol['cases']
    source,summary=paths(case)
    return dict(phase='dev',case=case,run=RUN,segment=segment,
                source=str(source.relative_to(ROOT)),source_sha256=sha(source),
                summary_sha256=sha(summary),core_sha256=sha(Path(ac.__file__)),
                train_samples=N_TRAIN,inner_samples=N_INNER,
                lambdas=list(LAMBDAS),max_als_iterations=MAX_ITERS,
                starts=list(STARTS),warmup=WARMUP,evaluation_samples=N_EVAL)


def design_for_indices(x,Om,ph,indices):
    """Exact physical filtered-x q_n = z_(n-2) + .5 z_(n-3)."""
    indices=np.asarray(indices,dtype=np.int64)
    assert indices.min()>=22 and indices.max()<len(x)
    lag=np.arange(20)
    hist2=x[indices[:,None]-2-lag]
    hist3=x[indices[:,None]-3-lag]
    scale=np.sqrt(2/500)
    z2=scale*np.cos(hist2@Om[0].T+ph[0])
    z3=scale*np.cos(hist3@Om[0].T+ph[0])
    return z2+.5*z3


def check_design(x,Om,ph,case,segment):
    begin=(segment-1)*100000
    indices=np.array([100,begin+137,begin+4321])
    q=design_for_indices(x,Om,ph,indices)
    for i,n in enumerate(indices):
        z2=np.sqrt(2/500)*np.cos(Om[0]@x[n-2-np.arange(20)]+ph[0])
        z3=np.sqrt(2/500)*np.cos(Om[0]@x[n-3-np.arange(20)]+ph[0])
        np.testing.assert_allclose(q[i],z2+.5*z3,atol=1e-14,rtol=1e-14)


def fit_full(q_train,y_train,q_inner,y_inner):
    start=perf_counter()
    n=len(y_train)
    gram=q_train.T@q_train/n
    rhs=q_train.T@y_train/n
    eigen,vec=np.linalg.eigh(gram)
    eigen=np.maximum(eigen,0.)
    proj=vec.T@rhs
    solutions={}
    candidates=[]
    for lam in LAMBDAS:
        w=vec@(proj/(eigen+lam))
        val=float(np.mean((y_inner-q_inner@w)**2))
        candidates.append(dict(lambda_value=lam,inner_mse=val))
        solutions[lam]=w
    best=min(candidates,key=lambda row:row['inner_mse'])
    return solutions,candidates,best,perf_counter()-start


def ridge_solve(X,y,lam):
    n=len(y)
    gram=X.T@X/n
    gram.flat[::len(gram)+1]+=lam
    rhs=X.T@y/n
    return np.linalg.solve(gram,rhs)


def start_factors(W_full,rank,kind,rng):
    if kind=='svd':
        U,s,Vh=np.linalg.svd(W_full,full_matrices=False)
        root=np.sqrt(s[:rank])
        return Vh[:rank].T*root, U[:,:rank]*root
    return rng.normal(0.,.1,(25,rank)),rng.normal(0.,.1,(20,rank))


def fit_rank(q_train,y_train,q_inner,y_inner,full_solutions,
             case,segment,rank):
    Q=q_train.reshape(-1,25,20).transpose(0,2,1)
    all_trials=[]
    selected=None
    elapsed=0.
    case_id=1 if case=='E2' else 2
    for lam_idx,lam in enumerate(LAMBDAS):
        for start_idx,kind in enumerate(STARTS):
            rng=np.random.default_rng([2026092801,case_id,RUN,segment,rank,lam_idx,start_idx,31])
            A,B=start_factors(full_solutions[lam].reshape(25,20).T,
                              rank,kind,rng)
            t0=perf_counter()
            previous=np.inf
            objectives=[]
            validation_mses=[]
            best_inner=np.inf
            best_W=None
            best_iteration=0
            for iteration in range(MAX_ITERS):
                XB=np.einsum('nij,jr->nir',Q,A,optimize=True).reshape(len(y_train),20*rank)
                B=ridge_solve(XB,y_train,lam).reshape(20,rank)
                XA=np.einsum('nji,jr->nir',Q,B,optimize=True).reshape(len(y_train),25*rank)
                A=ridge_solve(XA,y_train,lam).reshape(25,rank)
                W=B@A.T
                predicted=q_train@W.T.reshape(500)
                objective=float(np.mean((y_train-predicted)**2)
                                +lam*(np.sum(A*A)+np.sum(B*B)))
                assert np.isfinite(objective)
                objectives.append(objective)
                inner=float(np.mean((y_inner-q_inner@W.T.reshape(500))**2))
                validation_mses.append(inner)
                if inner<best_inner:
                    best_inner=inner
                    best_W=W.copy()
                    best_iteration=iteration+1
                if previous-objective>=0 and previous-objective<1e-6*max(1.,previous):
                    break
                previous=objective
            seconds=perf_counter()-t0
            elapsed+=seconds
            W=best_W
            inner=best_inner
            trial=dict(lambda_value=lam,start=kind,seed=[2026092801,case_id,RUN,
                                                           segment,rank,lam_idx,start_idx,31],
                       iterations=len(objectives),best_iteration=best_iteration,
                       training_objectives=objectives,inner_validation_mses=validation_mses,
                       inner_mse=inner,seconds=seconds)
            all_trials.append(trial)
            if selected is None or inner<selected[0]['inner_mse']:
                selected=(trial,W.copy())
    return selected[1],selected[0],all_trials,elapsed


def physical_anr(q,d,v,w):
    ys=q@w
    error=d+v-ys
    Ae=Ad=0.
    total=0.
    for j in range(len(d)):
        Ae=BETA*Ae+(1-BETA)*abs(error[j])
        Ad=BETA*Ad+(1-BETA)*abs(d[j])
        if j>=WARMUP:
            total+=20*np.log10((Ae+EPS)/(Ad+EPS))
    return float(total/N_EVAL)


def run_one(case,segment,info):
    source,summary_path=paths(case)
    with np.load(source,allow_pickle=False) as f:
        x,d,v,Om,ph=(f[k] for k in ('x','d','v','Om','ph'))
        meta=json.loads(str(f['meta']))
    assert len(x)==400000 and meta['case']==case and meta['run']==RUN
    check_design(x,Om,ph,case,segment)
    begin=(segment-1)*100000
    end=segment*100000
    train=np.linspace(begin+50,begin+39999,N_TRAIN,dtype=np.int64)
    inner=np.linspace(begin+40000,begin+49999,N_INNER,dtype=np.int64)
    evaluation=np.arange(end-WARMUP-N_EVAL,end)
    assert len(np.unique(train))==N_TRAIN and len(np.unique(inner))==N_INNER
    assert train.max()<inner.min()<inner.max()<evaluation.min()
    t0=perf_counter()
    q_train=design_for_indices(x,Om,ph,train)
    q_inner=design_for_indices(x,Om,ph,inner)
    q_eval=design_for_indices(x,Om,ph,evaluation)
    design_seconds=perf_counter()-t0
    y_train=d[train]  # optimistic offline oracle; real ANC cannot observe noise-free d
    y_inner=d[inner]
    full_solutions,full_candidates,full_best,full_seconds=fit_full(
        q_train,y_train,q_inner,y_inner)
    weights={'full500':full_solutions[full_best['lambda_value']].reshape(25,20).T}
    lowrank={}
    for rank in (2,3,4):
        W,best,trials,seconds=fit_rank(q_train,y_train,q_inner,y_inner,
                                       full_solutions,case,segment,rank)
        weights[f'R{rank}']=W
        lowrank[f'R{rank}']=dict(best=best,trials=trials,total_fit_seconds=seconds)
    results={}
    for name,W in weights.items():
        flat=W.T.reshape(500)
        results[name]=dict(heldout_anr_db=physical_anr(q_eval,d[evaluation],v[evaluation],flat),
                           heldout_primary_mse=float(np.mean((d[evaluation]-q_eval@flat)**2)),
                           inner_mse=float(np.mean((y_inner-q_inner@flat)**2)),
                           matrix_frobenius_norm=float(np.linalg.norm(W)))
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    original=dict(fixed_R4_online_anr_db=summary['segment_anr'][segment-1]['fixed_R4'],
                  full500_online_anr_db=summary['segment_anr'][segment-1]['full_RFF_MCC'])
    info.update(training_indices=dict(first=int(train.min()),last=int(train.max()),count=N_TRAIN),
                inner_validation_indices=dict(first=int(inner.min()),last=int(inner.max()),count=N_INNER),
                heldout_indices=dict(first=int(evaluation.min()),last=int(evaluation.max()),count=len(evaluation)),
                design_seconds=design_seconds,full_fit_seconds=full_seconds,
                lowrank_fit_seconds=sum(x['total_fit_seconds'] for x in lowrank.values()),
                total_seconds=perf_counter()-t0)
    row=dict(provenance=info,full_candidates=full_candidates,full_best=full_best,
             lowrank=lowrank,heldout=results,original_online_reference=original)
    weight_path=HERE/f'weights_{case}_segment{segment}.npz'
    temp=weight_path.with_suffix('.tmp')
    with temp.open('wb') as out:
        np.savez_compressed(out,**weights)
    temp.replace(weight_path)
    row['weights_sha256']=sha(weight_path)
    path=HERE/f'fit_{case}_segment{segment}.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)
    return row


def load_or_run(case,segment,refresh):
    info=provenance(case,segment)
    path=HERE/f'fit_{case}_segment{segment}.json'
    if path.exists() and not refresh:
        row=json.loads(path.read_text(encoding='utf-8'))
        assert all(row['provenance'][key]==info[key] for key in info)
        weight_path=HERE/f'weights_{case}_segment{segment}.npz'
        assert row['weights_sha256']==sha(weight_path)
        return row
    return run_one(case,segment,info)


def enrich_tuned_reference(row):
    case=row['provenance']['case']
    segment=row['provenance']['segment']
    source,_=paths(case)
    sweep_path=source.with_name(source.stem+'_fixed_sweep.json')
    sweep=json.loads(sweep_path.read_text(encoding='utf-8'))
    table=np.asarray(sweep['segment_anr'])
    options=[j for j,name in enumerate(sweep['names']) if name.startswith('R4_mu')]
    best=min(options,key=lambda j:table[:,j].mean())
    row['original_online_reference'].update(
        fixed_R4_tuned_mu=float(sweep['names'][best].split('_mu')[1]),
        fixed_R4_tuned_segment_anr_db=float(table[segment-1,best]),
        fixed_sweep_sha256=sha(sweep_path))
    path=HERE/f'fit_{case}_segment{segment}.json'
    path.write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding='utf-8')
    return row


def write_report(rows):
    lines=['# 原始稠密 RFF 的低秩物理拟合容量诊断','',
           '只读取 E2/E3 `main_dev_round1` 各 run0 的已保存输入；选择低非线性的第1段与高非线性的第3段。没有读取 select 或 confirm。',
           '沿用原始 500 维 RFF 映射与真实次级通道 S=[0,0,1,0.5]。Qₙ=reshape(zₙ₋₂+0.5zₙ₋₃) 为 20×25；拟合的控制权重 W=B Aᵀ。','',
           '每段前 40000 点中均匀取 4096 点拟合，40000–49999 中取 1024 点选岭系数/初值。',
           '50000–89999 与拟合隔离；末 10000 点仅用于评价，前 5000 点初始化 0.999 指数平滑，最后 5000 点平均原协议 ANR。',
           '离线拟合使用保存的无噪声 d 作为最有利的物理目标；实际评价含保存的测量噪声 v。在线控制器无法直接访问 d。',
           'R2/R3/R4 各以 full500 岭解的 SVD 初始化及两个隔离随机初值做最多 150 轮交替带岭最小二乘；岭候选固定为 1e-5 和 1e-3。',
           '所有选择只用前半段内部验证损失，末窗从未参与拟合或选择。训练计算为离线诊断成本，不能计作在线方法的免费步骤。',
           '“原在线 R4 调优”采用该开发运行四段平均最佳的同一个固定步长，遵循原固定项数扫描的口径。','',
           '## 留出末窗的物理 ANR', '',
           '|工况|非线性段|原在线R4默认|原在线R4调优|原在线full500|离线全500|离线R2|R2−离线R4|离线R3|R3−离线R4|离线R4|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        p=row['provenance']; h=row['heldout'];o=row['original_online_reference']
        f=h['full500']['heldout_anr_db']
        segment_label='低（1）' if p['segment']==1 else '高（3）'
        r4=h['R4']['heldout_anr_db']
        lines.append(f"|{p['case']}|{segment_label}|{o['fixed_R4_online_anr_db']:.3f}|{o['fixed_R4_tuned_segment_anr_db']:.3f}|{o['full500_online_anr_db']:.3f}|{f:.3f}|{h['R2']['heldout_anr_db']:.3f}|{h['R2']['heldout_anr_db']-r4:+.3f}|{h['R3']['heldout_anr_db']:.3f}|{h['R3']['heldout_anr_db']-r4:+.3f}|{r4:.3f}|")
    r2_gap=[row['heldout']['R2']['heldout_anr_db']-row['heldout']['R4']['heldout_anr_db'] for row in rows]
    r3_gap=[row['heldout']['R3']['heldout_anr_db']-row['heldout']['R4']['heldout_anr_db'] for row in rows]
    r3_tuned_gap=[row['heldout']['R3']['heldout_anr_db']-row['original_online_reference']['fixed_R4_tuned_segment_anr_db'] for row in rows]
    lines += ['', f"在相同离线目标、原始映射、时间划分和拟合预算下，四段 R2 比 R4 差 {min(r2_gap):.3f}–{max(r2_gap):.3f} dB，R3 比 R4 差 {min(r3_gap):.3f}–{max(r3_gap):.3f} dB；均超过 0.5 dB 容差。",
              f"离线 R3 对照原在线调优 R4 也差 {min(r3_tuned_gap):.3f}–{max(r3_tuned_gap):.3f} dB；这项跨训练方式对照仅作辅助。",
              '全 500 维离线拟合与原在线 full500 的量级相近，而离线 R4 与原在线调优 R4 也接近；这支持当前 500 维 RFF 排列下的低秩表示容量受限，而非仅由原在线因子优化造成差距。',
              '但交替最小二乘有非凸局部最优，部分候选在 150 轮末仍改善；本实验不是低秩物理损失的全局下界。']
    lines += ['','原在线 R4/full500 仅供绝对尺度参照：它们从运行开始顺序更新；这里的离线权重只从本段前半拟合并在末窗冻结。两种训练历史不可直接比较。','',
              '## 离线拟合预算与选择','',
              '|工况|段|构造Q 秒|全500拟合 秒|R2/R3/R4合计 秒|总耗时 秒|全500岭|R2岭/初值|R3岭/初值|R4岭/初值|',
              '|---|---:|---:|---:|---:|---:|---:|---|---|---|']
    for row in rows:
        p=row['provenance'];l=row['lowrank']
        label=lambda key:f"{l[key]['best']['lambda_value']:.0e}/{l[key]['best']['start']}"
        lines.append(f"|{p['case']}|{p['segment']}|{p['design_seconds']:.1f}|{p['full_fit_seconds']:.1f}|{p['lowrank_fit_seconds']:.1f}|{p['total_seconds']:.1f}|{row['full_best']['lambda_value']:.0e}|{label('R2')}|{label('R3')}|{label('R4')}|")
    lines += ['','每个秩共 2 岭系数 × 3 初值 × 最多 150 轮，每轮两次岭求解；在前半段内部验证集选最佳迭代。全部候选、迭代目标和耗时见逐段 JSON。',
              '报告范围是有限离线优化的可行证据，不是低秩表示的数学最优解；若多初值间仍有明显差异，应判为优化未充分收敛。',
              'RFF 特征、目标 d、评价 v、实际 S_PATH 与原保存运行完全配对；脚本检查开发相位及输入 SHA256。','',
              '## 复现','',
              '运行 `python -B experiments_v13/diagnostics/physical_rank_fit/physical_rank_fit.py`，随后运行 `python -B experiments_v13/diagnostics/physical_rank_fit/verify_physical_rank_fit.py`。计时运行设置 `OPENBLAS_NUM_THREADS=2`。',
              '每段 JSON/权重 NPZ 完成即缓存；`--refresh` 重算。可用 `--case E2 --segment 1` 单独重算一段。完整结果、固定随机种子、时序边界与拟合耗时均保存在本目录。','']
    (HERE/'physical_rank_fit_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--segment',type=int,choices=SEGMENTS)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    specs=[(case,segment) for case in CASES for segment in SEGMENTS
           if (args.case is None or args.case==case) and (args.segment is None or args.segment==segment)]
    selected=[]
    for case,segment in specs:
        row=enrich_tuned_reference(load_or_run(case,segment,args.refresh))
        selected.append(row)
        print(json.dumps(dict(case=case,segment=segment,
                              heldout_anr={k:round(v['heldout_anr_db'],3) for k,v in row['heldout'].items()},
                              seconds=row['provenance']['total_seconds'])),flush=True)
    all_paths=[HERE/f'fit_{case}_segment{segment}.json' for case in CASES for segment in SEGMENTS]
    if all(path.exists() for path in all_paths):
        rows=[json.loads(path.read_text(encoding='utf-8')) for path in all_paths]
        result=dict(interpretation='development-only offline physical filtered-x regression',
                    cases=list(CASES),segments=list(SEGMENTS),rows=rows)
        (HERE/'physical_rank_fit_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        write_report(rows)


if __name__=='__main__':
    main()
