"""Summarize the immutable v11 pilot as development evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT=HERE.parent/'experiments_v11'/'outputs'/'20260928_110901_119741_pilot'


def diagnose(folder):
    folder=Path(folder)
    records=[]
    for file in sorted(folder.glob('case*_events.json')):
        events=json.loads(file.read_text(encoding='utf-8'))
        for event in events:
            if event['event'] not in ('accept','reject'):
                continue
            event=dict(event)
            event['source']=file.name
            event['direction']='grow' if event['new_R']>event['old_R'] else 'prune'
            event['loss_change']=event['new_loss']-event['old_loss']
            records.append(event)
    out={}
    for direction in ('grow','prune'):
        e=[r for r in records if r['direction']==direction]
        out[direction]=dict(decisions=len(e),accepted=sum(r['event']=='accept' for r in e),
                            loss_change_min=float(min(r['loss_change'] for r in e)),
                            loss_change_max=float(max(r['loss_change'] for r in e)))
    occupancies=[];cost_ratios=[]
    for file in sorted(folder.glob('case*_run*.npz')):
        with np.load(file) as data:
            occupancies.append(float(np.mean(data['candidate_R']>0)))
            cost_ratios.append(float(np.mean(data['total_mults'])/19132))
    out['pilot']=dict(runs=len(occupancies),mean_candidate_occupancy=float(np.mean(occupancies)),
                      mean_cost_ratio_fixed4=float(np.mean(cost_ratios)))
    (HERE/'V11_DIAGNOSIS.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    diagnose(Path(sys.argv[1]) if len(sys.argv)>1 else DEFAULT)
