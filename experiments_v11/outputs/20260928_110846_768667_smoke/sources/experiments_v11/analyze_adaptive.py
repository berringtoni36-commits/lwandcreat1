"""Read-only analysis of a named v11 run directory; writes its derived artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

# Optional project-local plotting dependencies; existing NumPy is loaded first
# by adaptive_core to preserve the runtime used by the simulation.
from adaptive_core import ac
sys.path.append(str(Path(__file__).resolve().parent / '.deps'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def analyze(dest):
    dest = Path(dest).resolve()
    protocol = json.loads((dest/'protocol.json').read_text(encoding='utf-8'))
    rows=[]; records={}
    fixed4_cost = ac.mults_kron(25,20,4,20,4)+4
    for file in sorted(dest.glob('case*_run*.npz')):
        with np.load(file) as data:
            meta=json.loads(str(data['meta'])); names=meta['controllers']
            events=json.loads(file.with_name(file.stem+'_events.json').read_text(encoding='utf-8'))
            for j,name in enumerate(names):
                dynamic=name=='adaptive'
                R=int(name[-1]) if name.startswith('fixed_R') else None
                modeled_cost=float(data['total_mults'].mean()) if dynamic else (
                    ac.mults_kron(25,20,R,20,4)+4 if R else ac.mults_full(500,20,4)+4)
                rows.append(dict(case=meta['case'],run=meta['run'],controller=name,
                    steady_anr_db=float(data['anr'][j,-5000:].mean()),
                    mean_selected_R=float(data['R'].mean()) if dynamic else R,
                    final_selected_R=int(data['next_R'][-1]) if dynamic else R,
                    mean_reference_mults=modeled_cost,
                    mult_ratio_fixed4=modeled_cost/fixed4_cost,
                    mean_candidate_mults=float(data['candidate_mults'].mean()) if dynamic else 0,
                    mean_management_mults=float(data['management_mults'].mean()) if dynamic else 4,
                    mean_resident_factor_coeffs=float(data['resident_factor_coeffs'].mean()) if dynamic else (45*R if R else 500),
                    peak_resident_factor_coeffs=int(data['resident_factor_coeffs'].max()) if dynamic else (45*R if R else 500),
                    completed_switches=sum(e['event']=='switch_complete' for e in events) if dynamic else 0,
                    growth_accepts=sum(e['event']=='accept' and e['new_R']>e['old_R'] for e in events) if dynamic else 0,
                    prune_accepts=sum(e['event']=='accept' and e['new_R']<e['old_R'] for e in events) if dynamic else 0,
                    controller_wall_seconds=meta['controller_seconds'][name]))
            records.setdefault(meta['case'],[]).append({k:data[k].copy() for k in
                ['anr','R','total_mults','candidate_R','gamma']})
    with (dest/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    colors=['#8c9daa','#607d8b','#277da8','#7353a6','#273444','#d5652b']
    for case,runs in records.items():
        anr=np.stack([r['anr'] for r in runs]);T=anr.shape[-1];n=np.arange(T)
        fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for j,name in enumerate(names):
            axs[0,0].plot(n,anr[:,j].mean(0),label=name,lw=1.2,color=colors[j])
        axs[0,0].set(ylabel='Physical ANR (dB)',ylim=(-18,3))
        axs[0,0].legend(fontsize=7,ncol=2)
        for j,r in enumerate(runs):
            axs[0,1].step(n,r['R'],where='post',lw=1,label=f'run {j}')
        axs[0,1].set(ylabel='Selected R (both branches run during fade)',ylim=(.7,8.3))
        axs[0,1].legend(fontsize=7)
        for j,r in enumerate(runs):
            cumulative=np.cumsum(r['total_mults'])/(n+1)
            axs[1,0].plot(n,cumulative,label=f'run {j}',lw=1)
            axs[1,1].plot(n,cumulative/fixed4_cost,lw=1)
        axs[1,0].axhline(fixed4_cost,color='#277da8',ls='--',label='fixed R=4')
        axs[1,0].set(ylabel='Cumulative mean modeled multiplies/sample')
        axs[1,0].legend(fontsize=7)
        axs[1,1].axhline(1,color='#277da8',ls='--')
        axs[1,1].set(ylabel='Total modeled cost / fixed R=4')
        for ax in axs.flat:
            ax.set_xlabel('Sample');ax.grid(alpha=.2)
            if case==2:
                for boundary in [T//4,T//2,3*T//4]:ax.axvline(boundary,color='gray',ls=':',lw=.8)
                ax.axvspan(3*T//8,3*T//8+100,color='red',alpha=.15)
        fig.suptitle(protocol['case_names'][case]+' | development pilot, includes candidate overhead')
        fig.savefig(dest/f'case{case}_overview.png',dpi=160)
        plt.close(fig)
    lines=['# 动态 Kronecker 项数：开发验证结果','',
        f"运行配置：{protocol['profile']}；每种工况 {protocol['runs']} 次独立运行。",'',
        '本次结果用于检验实现和初始参数，不作为正式论文显著性或优越性结论。', '',
        '| 工况 | 动态稳态 ANR | 固定 R=2 | 固定 R=4 | 固定 R=8 | 平均所选 R | 总计数/固定 R=4 | 接受增长/剪枝 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for case in records:
        group=[r for r in rows if r['case']==case]
        dyn=[r for r in group if r['controller']=='adaptive']
        means={name:np.mean([r['steady_anr_db'] for r in group if r['controller']==name]) for name in names}
        lines.append(f"| {protocol['case_names'][case]} | {means['adaptive']:.3f} | {means['fixed_R2']:.3f} | "
                     f"{means['fixed_R4']:.3f} | {means['fixed_R8']:.3f} | {np.mean([r['mean_selected_R'] for r in dyn]):.2f} | "
                     f"{np.mean([r['mult_ratio_fixed4'] for r in dyn]):.3f} | "
                     f"{sum(r['growth_accepts'] for r in dyn)}/{sum(r['prune_accepts'] for r in dyn)} |")
    lines += ['', '稳态：每次运行最后 5000 个采样点的 ANR 均值；单位 dB，越负越好。', '',
        '乘法计数沿用 v10 的理论口径，加计候选控制器、筛选、验证、混合及输出通道滤波；共享 RFF 与特征滤波只计一次。'
        '这不是硬件指令实测，除法、指数、内存复制、随机数生成不折算成乘法；逐点指数次数另存于 NPZ。', '',
        'controller_wall_seconds 是同一 Python 运行环境下逐调用计时，排除共享特征生成，不能直接解释为硬件实时能力。', '',
        '模型选择的 J 比较未来单控制器主体成本，最终报告的总成本另加实际发生的搜索和切换开销；因此 J 改善不保证总成本节省。', '',
        'R 曲线表示当前主体结构；切换时两套分支同时运行，candidate_R、gamma、live_terms 和 resident_factor_coeffs 记录额外活动与存储。', '',
        '所有结构决策仅使用当时可用数据。候选验证使用自己的输出历史经过 S(z) 后的物理残差，不能用冻结参数预测残差替代。', '',
        '完整协议：protocol.json；逐次输入、映射、误差与成本：case*_run*.npz；结构事件：case*_events.json；统计：summary.csv。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'REPORT={dest / "REPORT.md"}',flush=True)


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python analyze_adaptive.py OUTPUT_DIRECTORY')
    analyze(sys.argv[1])
