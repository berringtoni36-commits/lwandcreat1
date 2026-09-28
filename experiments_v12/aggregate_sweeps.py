"""Freeze fair fixed-rank step sizes using only the independent selection set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze import bound, bootstrap_bound


def aggregate(folder):
    folder=Path(folder)
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    if protocol['phase']!='select' or len(protocol['cases'])!=1:
        raise ValueError('Expected one-case selection folder')
    case=protocol['cases'][0]
    runs=[]
    for i in range(protocol['runs']):
        path=folder/f'case{case}_run{i:02d}_fixed_sweep.json'
        runs.append(json.loads(path.read_text(encoding='utf-8')))
    names=runs[0]['names']
    if any(r['names']!=names for r in runs):
        raise ValueError('Sweep methods differ across runs')
    table=np.asarray([r['segment_anr'] for r in runs])
    if table.shape[1] != (3 if case=='E1' else 4 if case in ('E2','E3') else 1):
        raise ValueError('Sweep segmentation does not match case')
    choice={}
    for rank in range(1,9):
        indices=[j for j,name in enumerate(names) if name.startswith(f'R{rank}_')]
        if len(indices)!=5:
            raise ValueError(f'Incomplete sweep for R={rank}')
        j=min(indices,key=lambda j:float(table[:,:,j].mean()))
        choice[f'fixed_R{rank}']={'mu':float(names[j].split('mu')[1]),
                                  'selection_anr':float(table[:,:,j].mean()),
                                  'per_run_segment_anr':table[:,:,j].tolist()}
    summaries=json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
    summaries=sorted(summaries,key=lambda r:r['run'])
    dynamic=np.asarray([[seg['adaptive'] for seg in r['segment_anr']] for r in summaries])
    fixed=np.asarray(choice['fixed_R4']['per_run_segment_anr'])
    delta=dynamic-fixed
    report={'folder':str(folder),'phase':'select','case':case,
            'choice':choice,'dynamic_minus_tuned_fixed_R4':{
                'mean_by_segment':delta.mean(axis=0).tolist(),
                'ucb_by_segment':[bound(delta[:,j],.025) for j in range(delta.shape[1])],
                'bootstrap_ucb_by_segment':[bootstrap_bound(delta[:,j],.025,
                    seed=protocol['seed']+j) for j in range(delta.shape[1])]}}
    (folder/'fixed_step_selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'mu_by_fixed_method':{k:v['mu'] for k,v in choice.items()},
                      'dynamic_minus_tuned_fixed_R4':report['dynamic_minus_tuned_fixed_R4']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    aggregate(p.parse_args().folder)
