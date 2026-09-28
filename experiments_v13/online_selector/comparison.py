"""Fair development-only fixed-rank and full-500 comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'experiments_v12/outputs/main_dev_round1'
sys.path.insert(0,str(SOURCE/'sources/experiments_v10'))
import anc_core as ac

MU=[.05,.1,.2,.4,.8]
SWEEP=json.loads((ROOT/'experiments_v12/diagnostics/capacity_envelope.sweeps.json').read_text(encoding='utf-8'))
R4C=ac.mults_kron(25,20,4,20,4)+4


def main():
    cases={}
    for case in ('E2','E3'):
        records=sorted((r for r in SWEEP['runs'] if r['case']==case),key=lambda r:r['run'])
        assert len(records)==5
        arr=np.array([r['segment_anr'] for r in records]) # run, segment, rank, mu
        assert arr.shape==(5,4,8,5)
        chosen=np.argmin(arr.mean(axis=(0,1)),axis=1)
        assert MU[chosen[3]]=={'E2':.1,'E3':.05}[case]
        fixed4=arr[:,:,3,chosen[3]]
        rows=[]
        for rank in range(1,9):
            mu_idx=int(chosen[rank-1]);val=arr[:,:,rank-1,mu_idx]
            gap=val-fixed4
            cost=ac.mults_kron(25,20,rank,20,4)+4
            rows.append({'method':f'fixed_R{rank}','mu':MU[mu_idx],
                         'factor_coeffs':45*rank,'cost_mults':cost,
                         'cost_ratio_to_R4':cost/R4C,
                         'worst_segment_gap_to_tuned_R4_db':float(gap.max()),
                         'mean_segment_gap_to_tuned_R4_db':float(gap.mean()),
                         'segments_within_half_db':int(np.sum(gap<=.5))})
        full=np.array([[json.loads((SOURCE/f'case{case}_run{run:02d}_summary.json').read_text(encoding='utf-8'))['segment_anr'][j]['full_RFF_MCC'] for j in range(4)] for run in range(5)])
        full_gap=full-fixed4
        full_cost=ac.mults_full(500,20,4)+4
        rows.append({'method':'full_RFF_MCC_original_default_mu',
                     'mu':{'E2':.2,'E3':.1}[case], 'factor_coeffs':500,
                     'cost_mults':full_cost,'cost_ratio_to_R4':full_cost/R4C,
                     'worst_segment_gap_to_tuned_R4_db':float(full_gap.max()),
                     'mean_segment_gap_to_tuned_R4_db':float(full_gap.mean()),
                     'segments_within_half_db':int(np.sum(full_gap<=.5))})
        for arm in ('once','gated_growth'):
            summaries=[json.loads((HERE/f'{case}_run{r:02d}_tuned_{arm}_summary.json').read_text(encoding='utf-8')) for r in range(5)]
            aa=np.array([[seg['selector_anr_db'] for seg in s['segments']] for s in summaries])
            cc=np.array([[seg['cost_ratio_to_R4'] for seg in s['segments']] for s in summaries])
            rows.append({'method':'online_'+arm,'mu':{'E2':.1,'E3':.05}[case],
                         'factor_coeffs_active_after_accept':90,
                         'max_resident_factor_coeffs':max(s['max_resident_factor_coeffs'] for s in summaries),
                         'mean_cost_mults':float(np.mean([s['mean_mults'] for s in summaries])),
                         'mean_cost_ratio_to_R4':float(cc.mean()),
                         'worst_segment_cost_ratio_to_R4':float(cc.max()),
                         'worst_segment_gap_to_tuned_R4_db':float((aa-fixed4).max()),
                         'mean_segment_gap_to_tuned_R4_db':float((aa-fixed4).mean()),
                         'segments_within_half_db':int(np.sum(aa-fixed4<=.5)),
                         'worst_segment_gap_to_default_full500_db':float((aa-full).max()),
                         'mean_segment_gap_to_default_full500_db':float((aa-full).mean()),
                         'segments_better_than_default_full500':int(np.sum(aa<=full))})
        cases[case]={'chosen_fixed_mu_from_5x4_dev_mean':{str(r):MU[chosen[r-1]] for r in range(1,9)},'rows':rows}
    result={'phase':'dev','mu_grid':MU,'selection_rule':'one mu per case and fixed rank minimizing 5 runs x 4 segments development mean ANR','cases':cases}
    (HERE/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for case,obj in cases.items():
        print(case)
        for row in obj['rows']:
            print(row['method'],'mu',row['mu'],'worst_gap',round(row['worst_segment_gap_to_tuned_R4_db'],3),'mean_gap',round(row['mean_segment_gap_to_tuned_R4_db'],3),'cost_ratio',round(row.get('worst_segment_cost_ratio_to_R4',row.get('cost_ratio_to_R4')),3))


if __name__=='__main__':main()
