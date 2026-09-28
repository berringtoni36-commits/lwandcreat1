"""Independent artifact audit for the bounded development physical fits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import physical_rank_fit as pf

HERE=Path(__file__).resolve().parent


def main():
    output=json.loads((HERE/'physical_rank_fit_results.json').read_text(encoding='utf-8'))
    assert len(output['rows'])==4
    for row in output['rows']:
        p=row['provenance']
        case,segment=p['case'],p['segment']
        known=pf.provenance(case,segment)
        assert all(p[key]==known[key] for key in known)
        assert p['training_indices']['last']<p['inner_validation_indices']['first']
        assert p['inner_validation_indices']['last']<p['heldout_indices']['first']
        source,_=pf.paths(case)
        with np.load(source,allow_pickle=False) as f:
            x,d,v,Om,ph=(f[k] for k in ('x','d','v','Om','ph'))
        pf.check_design(x,Om,ph,case,segment)
        eval_index=np.arange(segment*100000-pf.WARMUP-pf.N_EVAL,segment*100000)
        q=pf.design_for_indices(x,Om,ph,eval_index)
        weight_path=HERE/f'weights_{case}_segment{segment}.npz'
        assert row['weights_sha256']==pf.sha(weight_path)
        with np.load(weight_path,allow_pickle=False) as f:
            for name in ('full500','R2','R3','R4'):
                W=f[name]
                assert W.shape==(20,25)
                if name!='full500':
                    rank=int(name[1:])
                    singular=np.linalg.svd(W,compute_uv=False)
                    assert singular[rank]<1e-10*singular[0]
                got=pf.physical_anr(q,d[eval_index],v[eval_index],W.T.reshape(500))
                np.testing.assert_allclose(got,row['heldout'][name]['heldout_anr_db'],atol=1e-10)
        for name,fit in row['lowrank'].items():
            assert len(fit['trials'])==len(pf.LAMBDAS)*len(pf.STARTS)
            assert fit['best']['inner_mse']==min(t['inner_mse'] for t in fit['trials'])
            assert all(t['best_iteration']<=t['iterations']<=pf.MAX_ITERS for t in fit['trials'])
        gap=row['heldout']['R3']['heldout_anr_db']-row['heldout']['R4']['heldout_anr_db']
        print(f'{case} segment{segment}: R3 minus R4 = {gap:+.3f} dB')
    print('development-only physical rank fit verification passed')


if __name__=='__main__':
    main()
