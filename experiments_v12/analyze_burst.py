"""Paired development-only burst-trigger audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE=Path(__file__).resolve().parent
FOLDER=HERE/'outputs'/'bridge_burst_dev_aggressive'


def main():
    rows=[]
    for run in range(5):
        a=FOLDER/f'caseFclean_run{run:02d}.npz'
        b=FOLDER/f'caseFburst_run{run:02d}.npz'
        with np.load(a) as fa,np.load(b) as fb:
            for key in ('x','d','Om','ph','initial_A','initial_B'):
                if not np.array_equal(fa[key],fb[key]):
                    raise ValueError(f'Non-paired {key} run {run}')
            mid=len(fa['x'])//2
            activity=float(np.sum(fb['R'][mid:]-fa['R'][mid:]))
            candidate=float(np.sum(fb['candidate_R'][mid:]-fa['candidate_R'][mid:]))
            cost=float(np.mean(fb['total_mults'][mid:]-fa['total_mults'][mid:]))
            r_equal=bool(np.array_equal(fa['R'],fb['R']))
            candidate_equal=bool(np.array_equal(fa['candidate_R'],fb['candidate_R']))
        def growth(path):
            events=json.loads(path.read_text(encoding='utf-8'))
            return sum(e['event']=='accept' and e['new_R']>e['old_R'] and
                       e['sample']>=mid for e in events)
        extra_growth=growth(b.with_name(b.stem+'_events.json'))-growth(
            a.with_name(a.stem+'_events.json'))
        rows.append({'run':run,'extra_growth_after_burst':extra_growth,
                     'extra_R_time_integral':activity,
                     'extra_candidate_R_time_integral':candidate,
                     'mean_extra_mults_after_burst':cost,
                     'R_trace_identical':r_equal,
                     'candidate_R_trace_identical':candidate_equal})
    result={'phase':'dev','paired_runs':rows,
            'all_no_extra_growth':all(r['extra_growth_after_burst']==0 for r in rows),
            'all_rank_traces_identical':all(r['R_trace_identical'] for r in rows)}
    (HERE/'BURST_DEV_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
