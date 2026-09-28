"""Rebuild development figures; neither panel is formal confirmation evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'.deps'))
sys.path.append(str(HERE.parent/'experiments_v11'/'.deps'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from adaptive_core import ac


OUT=HERE/'figures'
OUT.mkdir(exist_ok=True)
ROOT=HERE/'outputs'


def frontier():
    folder=ROOT/'selection_e1_v500'
    step=json.loads((folder/'fixed_step_selection.json').read_text(encoding='utf-8'))
    data=json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
    ranks=np.arange(1,9)
    cost=np.array([ac.mults_kron(25,20,int(r),20,4)+4 for r in ranks])/19132
    anr=np.array([step['choice'][f'fixed_R{r}']['selection_anr'] for r in ranks])
    dynamic_cost=np.mean([r['mean_mults'] for r in data])/19132
    dynamic_anr=np.mean([np.mean([s['adaptive'] for s in r['segment_anr']]) for r in data])
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    ax.plot(cost,anr,'o-',label='Fixed Kronecker R=1..8')
    for r,x,y in zip(ranks,cost,anr):
        ax.annotate(str(r),(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.scatter([dynamic_cost],[dynamic_anr],s=95,marker='D',color='#c44e52',
               label='Adaptive (selection, not confirmed)',zorder=4)
    ax.axvline(.9,color='gray',ls='--',lw=1,label='10% cost saving target')
    ax.set(xlabel='Modeled total multiplication ratio vs fixed R=4',
           ylabel='Equal-segment mean physical ANR (dB)',
           title='E1 known-rank teacher: independent selection set (8 runs)')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'E1_selection_frontier.{ext}',dpi=180)
    plt.close(fig)


def cost_breakdown():
    folders={'Initial selector':'main_dev_round1',
             'Aggressive growth':'main_dev_aggressive_growth',
             'Continue on accept':'main_dev_continue'}
    rows=[]
    for label,name in folders.items():
        runs=json.loads((ROOT/name/'run_summary.json').read_text(encoding='utf-8'))
        for case in ('E2','E3'):
            group=[r for r in runs if r['case']==case]
            total=np.mean([r['mean_mults'] for r in group])
            candidate=np.mean([r['mean_candidate_mults'] for r in group])
            management=np.mean([r['mean_management_mults'] for r in group])
            rows.append((case,label,total-candidate-management,candidate,management))
    x=np.arange(len(rows));width=.7
    core=np.array([r[2] for r in rows])/19132
    cand=np.array([r[3] for r in rows])/19132
    management=np.array([r[4] for r in rows])/19132
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.bar(x,core,width,label='Active controller + shared RFF')
    ax.bar(x,cand,width,bottom=core,label='Shadow candidate')
    ax.bar(x,management,width,bottom=core+cand,label='Management + transitions')
    ax.axhline(.9,color='#c44e52',ls='--',label='10% saving target')
    ax.axhline(1,color='black',ls=':',label='Fixed R=4')
    ax.set_xticks(x,[f'{r[0]}\n{r[1]}' for r in rows],fontsize=8)
    ax.set(ylabel='Mean full modeled multiplication ratio vs fixed R=4',
           title='Nonstationary ANC: quality/cost design tradeoff (development)')
    ax.legend(fontsize=8,ncol=2);ax.grid(axis='y',alpha=.2)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'nonstationary_cost_breakdown.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    frontier();cost_breakdown()
