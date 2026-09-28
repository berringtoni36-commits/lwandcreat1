"""One bounded full500 capacity check for the same development tensor map."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

import tensor_fourier_sweep as tf

HERE = Path(__file__).resolve().parent


def check_batched_full_update():
    rng=np.random.default_rng(36117)
    batch=tf.ac.FullRFF('batch',5,500,np.array(tf.MU),2.,1e-8)
    singles=[tf.ac.FullRFF('single',1,500,mu,2.,1e-8) for mu in tf.MU]
    for _ in range(100):
        z,q,dv=rng.normal(size=(1,500)),rng.normal(size=(1,500)),rng.normal(size=1)
        got=batch.step(np.repeat(z,5,axis=0),np.repeat(q,5,axis=0),np.repeat(dv,5))
        want=np.array([c.step(z,q,dv)[0] for c in singles])
        np.testing.assert_allclose(got,want,rtol=1e-12,atol=1e-12)
    for j,c in enumerate(singles):
        np.testing.assert_allclose(batch.w[j],c.w[0],rtol=1e-12,atol=1e-12)


def run_case(case):
    info=tf.provenance(case)
    cache=HERE/f'full_tensor_{case}_run00.json'
    if cache.exists():
        row=json.loads(cache.read_text(encoding='utf-8'))
        assert all(row['provenance'][key]==info[key] for key in info)
        return row
    source,_=tf.source_paths(case)
    with np.load(source,allow_pickle=False) as f:
        x,d,v=f['x'],f['d'],f['v']
    assert len(x)==400000
    params=tf.map_parameters(case)
    ctrl=tf.ac.FullRFF('full500',5,500,np.array(tf.MU),2.,1e-8)
    history=np.zeros(20)
    zh=np.zeros((4,1,500))
    Ae=np.zeros(5)
    Ad=0.
    sums=np.zeros((4,5))
    start=perf_counter()
    for n in range(len(x)):
        history[1:]=history[:-1].copy();history[0]=x[n]
        z=tf.tensor_features(history,params)
        zh[1:]=zh[:-1].copy();zh[0]=z
        q=zh[2]+.5*zh[3]
        dv=d[n]+v[n]
        ys=ctrl.step(np.repeat(z,5,axis=0),np.repeat(q,5,axis=0),np.full(5,dv))
        Ae=.999*Ae+.001*np.abs(dv-ys)
        Ad=.999*Ad+.001*abs(d[n])
        segment=n//100000
        if n>=(segment+1)*100000-5000:
            sums[segment]+=20*np.log10((Ae+1e-12)/(Ad+1e-12))
    table=sums/5000
    assert np.isfinite(table).all()
    info.update(seconds=perf_counter()-start)
    row=dict(provenance=info,segment_anr=table.tolist())
    temporary=cache.with_suffix('.tmp')
    temporary.write_text(json.dumps(row,indent=2),encoding='utf-8')
    temporary.replace(cache)
    return row


def main():
    check_batched_full_update()
    base=json.loads((HERE/'tensor_fourier_results.json').read_text(encoding='utf-8'))
    rows=[]
    for case in tf.CASES:
        row=run_case(case)
        a=next(x for x in base['analyses'] if x['case']==case)
        full=np.asarray(row['segment_anr'])
        index=int(full.mean(axis=0).argmin())
        new_r4=np.mean(a['reference_new_R4_segment_anr'])
        old_r4=np.mean(a['old_mapping_R4_context']['segment_anr'])
        rows.append(dict(case=case,best_mu=tf.MU[index],
                         segment_anr=full[:,index].tolist(),
                         mean_anr=float(full[:,index].mean()),
                         new_R4_mean_anr=float(new_r4),
                         old_R4_mean_anr=float(old_r4),
                         full_minus_new_R4_db=float(full[:,index].mean()-new_r4),
                         full_minus_old_R4_db=float(full[:,index].mean()-old_r4),
                         seconds=row['provenance']['seconds'],
                         provenance=row['provenance']))
        print(json.dumps({k:rows[-1][k] for k in ('case','best_mu','mean_anr','full_minus_new_R4_db','full_minus_old_R4_db','seconds')}),flush=True)
    (HERE/'full_tensor_results.json').write_text(json.dumps(dict(rows=rows),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
