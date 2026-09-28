"""N14 new development-only dual-rate prototype and paired audit output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import controller

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
INPUTS=ROOT/'experiments_v14/outputs/n14_dev_inputs'
FRONT=ROOT/'experiments_v14/fixed_frontier'
FROZEN=ROOT/'experiments_v14/dev_eval_frozen'
R4_COST=19132


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def run(case,run,post_mu,bridge=False,target_rank=2):
    assert case in ('E2','E3') and run in range(5)
    source=INPUTS/f'case{case}_run{run:02d}.npz'
    manifest=json.loads((INPUTS/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['phase']=='dev' and manifest['status']=='inputs_only'
    entry=next(e for e in manifest['entries'] if e['case']==case and e['run']==run)
    assert entry['phase']=='dev' and sha(source)==entry['file_sha256']
    with np.load(source,allow_pickle=False) as f:
        x,d,v,om,ph,A,B=[f[k].copy() for k in
                         ('x','d','v','omega','rff_phase','initial_A','initial_B')]
    config={'arm':'once','proposal':'signed_sort','ramp_samples':100,
            'ablate_spectral_veto':False,'mu_after':post_mu,
            'target_rank':target_rank}
    result=controller.simulate(x,d,v,om[None],ph[None],A[None],B[None],.05,config)
    fixed=json.loads((FRONT/f'case{case}_run{run:02d}'/'result.json').read_text(encoding='utf-8'))
    baseline=np.array(fixed['segment_anr_db'][fixed['method_names'].index('R4_mu0.05')])
    segments=[]
    for j in range(4):
        lo,hi=j*100000,(j+1)*100000
        anr=float(result['anr'][hi-5000:hi].mean())
        cost=float(result['cost']['total'][lo:hi].mean())
        segments.append({'segment':j+1,'anr_db':anr,'fixed_R4_mu0.05_anr_db':float(baseline[j]),
                         'gap_db':anr-float(baseline[j]),'mean_mults':cost,
                         'cost_ratio_to_R4':cost/R4_COST})
    summary={'phase':'dev','case':case,'run':run,'mu_before':.05,'mu_after':post_mu,
             'target_rank':target_rank,
             'config':config,'input_sha256':sha(source),
             'controller_sha256':sha(HERE/'controller.py'),
             'source_core_sha256':'60F6FA3F0AFE5B9FFC03EF4B610BC446EFD7A9E5ED391D621FCE6F9B1CC78846',
             'segments':segments,'events':result['events'],
             'max_gap_db':max(s['gap_db'] for s in segments),
             'max_cost_ratio':max(s['cost_ratio_to_R4'] for s in segments),
             'mean_cost_ratio':float(result['cost']['total'].mean()/R4_COST),
             'physical_fir_max_abs':result['physical_fir_max_abs'],
             'post_ramp_max_abs':result['post_ramp_max_abs'],
             'seconds':result['seconds']}
    if bridge:
        assert post_mu==.05
        with np.load(FROZEN/f'case{case}_run{run:02d}_mu0.05.npz') as f:
            checks={k:float(np.max(np.abs(result[k]-f[k]))) for k in ('drive','error','anr','R')}
            checks['cost_total']=float(np.max(np.abs(result['cost']['total']-f['cost_total'])))
        assert checks['drive']<1e-10 and checks['error']<1e-10 and checks['anr']<1e-6
        assert checks['R']==checks['cost_total']==0
        summary['frozen_60f6_bridge_max_abs']=checks
    stem=f'case{case}_run{run:02d}_pre0.05_post{post_mu:g}_targetR{target_rank}'
    if bridge:stem+='_bridge'
    (HERE/f'{stem}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    np.savez_compressed(HERE/f'{stem}.npz',anr=result['anr'],error=result['error'],
                        drive=result['drive'],R=result['R'],candidate_R=result['candidate_R'],
                        retiring_R=result['retiring_R'],gamma=result['gamma'],
                        **{f'cost_{k}':v for k,v in result['cost'].items()})
    print(case,run,'post',post_mu,'gap',[round(s['gap_db'],3) for s in segments],
          'cost',round(summary['mean_cost_ratio'],4),'accepted',
          sum(e.get('accepted',False) for e in result['events'] if e['type']=='prune'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=['E2','E3'],required=True)
    p.add_argument('--runs',type=int,nargs='+',default=list(range(5)))
    p.add_argument('--post-mu',type=float,default=.1)
    p.add_argument('--bridge',action='store_true')
    p.add_argument('--target-rank',type=int,choices=[2,3],default=2)
    a=p.parse_args()
    for i in a.runs:run(a.case,i,a.post_mu,a.bridge,a.target_rank)
