"""Paired per-run analysis; no development run is relabeled confirmatory."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'.deps'))
sys.path.append(str(HERE.parent/'experiments_v11'/'.deps'))
from scipy.stats import t


def case_seed(case,rank):
    return 31*sum(ord(ch) for ch in case)+rank


def bound(values,alpha):
    x=np.asarray(values,dtype=float)
    if len(x)<2:
        return float('nan')
    return float(x.mean()+t.ppf(1-alpha,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))


def bootstrap_bound(values,alpha,seed=8,reps=10000):
    x=np.asarray(values,dtype=float)
    if len(x)<2:
        return float('nan')
    rng=np.random.default_rng(seed)
    index=rng.integers(0,len(x),size=(reps,len(x)))
    return float(np.quantile(x[index].mean(axis=1),1-alpha))


def analyze(folder):
    folder=Path(folder).resolve()
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    summaries=json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
    alpha=protocol['confirm_alpha'] if protocol['phase']=='confirm' else .025
    fixed4_cost=19132.0
    lines=['# v12 实验报告','',
           f"阶段：{protocol['phase']}；轮次：{protocol['round']}；每工况 {protocol['runs']} 次；单侧 α={alpha:.6f}。",'',
           'ANR 越负越好。下表中的差值为动态方法减固定 R=4；成本比包含搜索、验证、剪枝预算费用和切换。', '',
           '| 工况 | 段 | 动态 ANR | 固定 R4 ANR | 差值均值 | 差值上界 | 成本比均值 | 成本比上界 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    rows=[];status={}
    for case in protocol['cases']:
        group=sorted((r for r in summaries if r['case']==case),key=lambda r:r['run'])
        if len(group)!=protocol['runs']:
            raise ValueError(f'{case}: incomplete runs')
        q=np.array([r['mean_mults']/fixed4_cost for r in group])
        q_upper=bound(q,alpha)
        q_boot=bootstrap_bound(q,alpha,seed=protocol['seed']+1000+len(case))
        case_status=[]
        for seg in range(len(group[0]['segment_anr'])):
            dynamic=np.array([r['segment_anr'][seg]['adaptive'] for r in group])
            fixed=np.array([r['segment_anr'][seg]['fixed_R4'] for r in group])
            delta=dynamic-fixed
            up=bound(delta,alpha)
            boot=bootstrap_bound(delta,alpha,seed=protocol['seed']+seg+len(case))
            passed=bool(up<=.5 and boot<=.5 and q_upper<=.9 and q_boot<=.9)
            case_status.append(passed)
            row=dict(case=case,segment=seg+1,n=len(group),dynamic_anr=float(dynamic.mean()),
                     fixed4_anr=float(fixed.mean()),delta_db=float(delta.mean()),
                     delta_ucb=up,delta_boot_ucb=boot,cost_ratio=float(q.mean()),
                     cost_ucb=q_upper,cost_boot_ucb=q_boot,passed=passed)
            rows.append(row)
            lines.append(f"| {case} | {seg+1} | {row['dynamic_anr']:.3f} | "
                         f"{row['fixed4_anr']:.3f} | {row['delta_db']:+.3f} | "
                         f"{up:+.3f} | {q.mean():.3f} | {q_upper:.3f} |")
        status[case]=all(case_status)
    with (folder/'paired_stats.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    lines += ['', '## 性能—成本对照', '',
              '每个固定 R 的成本按相同理论口径计输出 FIR。以下为逐工况、全部段等权的描述性对照；正式优越性还需要锁定优势维度与独立检验。', '',
              '| 工况 | 方法 | ANR dB | 总乘法/采样 | 该方法在开发均值上是否同时超过动态方法 |',
              '|---|---|---:|---:|---|']
    from adaptive_core import ac
    for case in protocol['cases']:
        group=[r for r in summaries if r['case']==case]
        methods=list(group[0]['segment_anr'][0])
        values={name:float(np.mean([np.mean([seg[name] for seg in r['segment_anr']])
                                    for r in group])) for name in methods}
        dynamic_cost=float(np.mean([r['mean_mults'] for r in group]))
        for name in methods:
            if name.startswith('fixed_R'):
                R=int(name.removeprefix('fixed_R'))
                cost=float(ac.mults_kron(25,20,R,20,4)+4)
            elif name=='adaptive':
                cost=dynamic_cost
            elif name=='full180':
                cost=float(ac.mults_full(180,20,4)+4)
            elif name=='full_RFF_MCC':
                cost=float(ac.mults_full(500,20,4)+4)
            else:
                cost=float('nan')
            dominated=(name!='adaptive' and np.isfinite(cost) and cost<=dynamic_cost and
                       values[name]<=values['adaptive'])
            lines.append(f'| {case} | {name} | {values[name]:.3f} | {cost:.0f} | '
                         f'{"是" if dominated else "否"} |')
    lines += ['', '## 固定项数前沿的配对检验', '',
              '正式验证时，检验维度须由独立选择集事先冻结。开发阶段下表只诊断可行性。', '',
              '| 工况 | 固定 R | 锁定/诊断维度 | 动态减固定的均值 | 单侧上界 | 通过 |',
              '|---|---:|---|---:|---:|---|']
    frontier_status={}
    from adaptive_core import ac
    for case in protocol['cases']:
        group=[r for r in summaries if r['case']==case]
        locked=protocol.get('frontier_witness_by_case',{}).get(case,{})
        checks=[]
        for rank in range(1,9):
            name=f'fixed_R{rank}'
            if name not in group[0]['segment_anr'][0]:
                checks.append(False)
                continue
            fixed_cost=ac.mults_kron(25,20,rank,20,4)+4
            cost_diff=np.array([r['mean_mults']-fixed_cost for r in group])
            perf_diff=np.array([np.mean([seg['adaptive']-seg[name]
                                         for seg in r['segment_anr']]) for r in group])
            witness=locked.get(str(rank))
            if witness not in ('cost','anr') and protocol['phase']=='confirm':
                checks.append(False)
                lines.append(f'| {case} | {rank} | 未锁定 | — | — | 否 |')
                continue
            if witness not in ('cost','anr'):
                witness='cost' if cost_diff.mean()<0 else 'anr'
            difference=cost_diff if witness=='cost' else perf_diff
            upper=bound(difference,alpha)
            boot=bootstrap_bound(difference,alpha,
                                 seed=protocol['seed']+case_seed(case,rank))
            passed=bool(upper<0 and boot<0)
            checks.append(passed)
            lines.append(f'| {case} | {rank} | {witness} | '
                         f'{difference.mean():+.4f} | {upper:+.4f} | '
                         f'{"是" if passed else "否"} |')
        frontier_status[case]=all(checks)
    lines += ['', '## 结构事件与成本', '',
              '| 工况 | 接受增长 | 接受剪枝 | 平均主体 R | 平均候选成本 | 平均管理成本 |',
              '|---|---:|---:|---:|---:|---:|']
    for case in protocol['cases']:
        group=[r for r in summaries if r['case']==case]
        lines.append(f"| {case} | {sum(r['accepted_growth'] for r in group)} | "
                     f"{sum(r['accepted_prune'] for r in group)} | "
                     f"{np.mean([r['mean_R'] for r in group]):.2f} | "
                     f"{np.mean([r['mean_candidate_mults'] for r in group]):.1f} | "
                     f"{np.mean([r['mean_management_mults'] for r in group]):.1f} |")
    if protocol['config']['prune_mode']=='counted':
        lines += ['', '剪枝使用显式 QR/最多 32 轮 Jacobi SVD；账本记录该内核实际执行的标量乘法。'
                  '除法、开方、加法与硬件指令另列，不能把理论乘法直接解释为运行时间。']
    else:
        lines += ['', '当前 SVD 采用库函数，每次剪枝候选在账本中收取 1,000,000 次乘法的公开预算费用。'
                  '它不是库调用的精确乘法数；正式精确计数结论需要显式可计数内核。']
    lines += [
              '所有图表依据逐次文件生成；开发阶段的统计上界只用于判断下一轮方案，不能当作独立论文验证。', '',
              '主门槛必须同时通过两个非平稳工况的全部段、总成本、固定项数前沿、机制检查和 MATLAB 复核。', '',
              '## 暂定门槛状态', '']
    for case,passed in status.items():
        lines.append(f"- {case}: {'ANR/成本数值门槛通过' if passed else 'ANR/成本数值门槛未通过'}；"
                     f"{'固定项数前沿通过' if frontier_status[case] else '固定项数前沿未通过或未完整评估'}。")
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (folder/'analysis_status.json').write_text(json.dumps({
        'anr_and_cost':status,'fixed_frontier':frontier_status,
        'joint_numeric':{c:bool(status[c] and frontier_status[c]) for c in status}
    },indent=2),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for case in protocol['cases']:
            first=next(r for r in summaries if r['case']==case and r['run']==0)
            data=np.load(folder/f'case{case}_run00.npz')
            names=list(first['ss_anr']);T=first['segment_bounds'][-1][1]
            n=np.arange(0,T,100)
            fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
            for name in ('fixed_R2','fixed_R4','fixed_R8','adaptive','full_RFF_MCC'):
                if name in names:
                    i=names.index(name)
                    ax[0,0].plot(n,data['anr'][i,::100],label=name,lw=1)
            ax[0,0].legend(fontsize=7)
            ax[0,0].set(ylabel='Physical ANR (dB)')
            ax[0,1].step(n,data['R'][::100],where='post')
            ax[0,1].set(ylabel='Active Kronecker terms',ylim=(.5,8.5))
            ax[1,0].plot(n,np.cumsum(data['total_mults'])[::100]/(n+1))
            ax[1,0].axhline(fixed4_cost,ls='--',color='gray')
            ax[1,0].set(ylabel='Cumulative modeled multiplies/sample')
            ax[1,1].plot(n,data['candidate_mults'][::100])
            ax[1,1].set(ylabel='Candidate multiplies/sample')
            for panel in ax.flat:
                panel.set_xlabel('Sample');panel.grid(alpha=.2)
                for start,end in first['segment_bounds'][:-1]:
                    panel.axvline(end,color='gray',ls=':',lw=.8)
            fig.suptitle(f'{case}: run 0; {protocol["phase"]}')
            fig.savefig(folder/f'{case}_overview.png',dpi=150)
            plt.close(fig)
    except ImportError:
        lines.append('绘图库未安装；数值报告已生成。')
    return status


if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python analyze.py OUTPUT_DIRECTORY')
    print(analyze(sys.argv[1]))
