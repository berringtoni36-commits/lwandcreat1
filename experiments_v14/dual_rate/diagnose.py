"""Read only N14 development residuals and fixed-rank capacity references."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FROZEN=ROOT/'experiments_v14/dev_eval_frozen'
FRONT=ROOT/'experiments_v14/fixed_frontier'


def main():
    rows=[]
    for case in ('E2','E3'):
        for run in range(5):
            ds=json.loads((FROZEN/f'case{case}_run{run:02d}_mu0.05_summary.json').read_text(encoding='utf-8'))
            fr=FRONT/f'case{case}_run{run:02d}'
            fs=json.loads((fr/'result.json').read_text(encoding='utf-8'))
            names=fs['method_names']
            mat=np.asarray(fs['segment_anr_db'],float)
            r4=mat[names.index('R4_mu0.05')]
            r2_grid=np.array([mat[names.index(f'R2_mu{mu:.2f}')] for mu in (.05,.1,.2,.4,.8)])
            with np.load(FROZEN/f'case{case}_run{run:02d}_mu0.05.npz') as dt, np.load(fr/'traces.npz') as ft:
                de=dt['error'].copy();fe=ft['physical_error'][names.index('R4_mu0.05')].copy()
            for j,segment in enumerate(ds['segments']):
                lo,hi=j*100000,(j+1)*100000
                first=slice(lo+5000,lo+10000) if j==0 else slice(lo,lo+5000)
                tail=slice(hi-5000,hi)
                ratio=lambda sl:float(np.mean(np.abs(de[sl]))/np.mean(np.abs(fe[sl])))
                clip=lambda sl:float(np.mean(np.minimum(np.abs(de[sl]),3*np.median(np.abs(fe[sl]))))/
                                     np.mean(np.minimum(np.abs(fe[sl]),3*np.median(np.abs(fe[sl])))))
                rows.append({'case':case,'run':run,'segment':j+1,
                             'dynamic_gap_to_R4_db':float(segment['anr_db']-r4[j]),
                             'fixed_R2_mu0.05_gap_db':float(r2_grid[0,j]-r4[j]),
                             'fixed_R2_optimistic_per_segment_gap_db':float(r2_grid[:,j].min()-r4[j]),
                             'early_abs_residual_ratio':ratio(first),
                             'tail_abs_residual_ratio':ratio(tail),
                             'early_clipped_ratio':clip(first),
                             'tail_clipped_ratio':clip(tail)})
    result={'phase':'dev','baseline':'paired fixed R4 mu=.05','rows':rows}
    (HERE/'diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('E2 failures (>+.5 dB):')
    for r in rows:
        if r['case']=='E2' and r['dynamic_gap_to_R4_db']>.5:
            print(r)
    print('fixed R2 optimistic min gap across all 40 segments',min(r['fixed_R2_optimistic_per_segment_gap_db'] for r in rows))


if __name__=='__main__':main()
