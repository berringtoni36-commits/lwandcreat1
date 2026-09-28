"""At the same causal 20k proposal, audit identity and random permutations.

The control is stopped at the veto: after rejection its deployed controller
remains the original tuned R4 for all later samples. No future signal is read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from online_selector import (D,D1,D2,HERE,M,ROOT,SOURCE,MU,
                             SPECTRAL_THRESHOLD,make_ctrl,prune)


def one(case,run):
    path=SOURCE/f'case{case}_run{run:02d}.npz'
    with np.load(path) as src:
        x=src['x'][:20001].copy();d=src['d'][:20001].copy();v=src['v'][:20001].copy()
        om=src['Om'][0].copy();ph=src['ph'][0].copy()
        A=src['initial_A'][:,:,:4].copy();B=src['initial_B'][:,:,:4].copy()
    ctrl=make_ctrl(4,MU[case],A,B)
    xb=np.zeros(M);zh=np.zeros((4,D))
    for n in range(20001):
        xb[1:]=xb[:-1];xb[0]=x[n]
        z=np.sqrt(2/D)*np.cos(om@xb+ph)
        zh[1:]=zh[:-1];zh[0]=z
        q=zh[2]+.5*zh[3]
        ctrl.step(z[None,:],q[None,:],np.array([d[n]+v[n]]))
    w=(ctrl.B[0]@ctrl.A[0].T).T.reshape(D)
    out=[]
    for method in ('signed_sort','identity','fixed_random'):
        candidate,perm,cost,sweeps,trunc=prune(ctrl,2,MU[case],np.arange(D),method)
        # Library SVD is a post-hoc diagnostic for failed Jacobi only; never
        # used in candidate formation, selection, or charged runtime.
        p=perm
        Wp=w[p].reshape(D1,D2).T
        singular=np.linalg.svd(Wp,compute_uv=False)
        library_trunc=float(np.sqrt(np.sum(singular[2:]**2)/np.sum(singular**2)))
        decision='reject_jacobi_nonconvergence' if candidate is None else (
            'reject_spectral_veto' if trunc>SPECTRAL_THRESHOLD else 'send_to_physical_candidate_validation')
        out.append({'method':method,'decision':decision,'counted_jacobi_sweeps':sweeps,
                    'counted_reorder_mults':cost,
                    'counted_truncation_relative_frobenius':None if not np.isfinite(trunc) else trunc,
                    'posthoc_library_truncation_for_audit_only':library_trunc})
    return {'case':case,'run':run,'phase':'dev','proposal_n':20000,'mu':MU[case],
            'spectral_threshold':SPECTRAL_THRESHOLD,'controls':out}


def main():
    protocol=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['phase']=='dev' and protocol['cases']==['E2','E3']
    rows=[one(case,run) for case in ('E2','E3') for run in range(5)]
    result={'phase':'dev','rows':rows,
            'threshold_sensitivity':{str(th):{
                method:sum(c['posthoc_library_truncation_for_audit_only']<=th
                           for row in rows for c in row['controls'] if c['method']==method)
                for method in ('signed_sort','identity','fixed_random')}
                for th in (.10,.15,.20,.30)},
            'important':'Library SVD above is only post-hoc control audit; deployed factorization is counted QR/Jacobi and nonconvergence rejects.'}
    (HERE/'negative_controls.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['threshold_sensitivity'],indent=2))
    for method in ('signed_sort','identity','fixed_random'):
        selected=[c for row in rows for c in row['controls'] if c['method']==method]
        print(method,[c['decision'] for c in selected],
              'truncation range',min(c['posthoc_library_truncation_for_audit_only'] for c in selected),
              max(c['posthoc_library_truncation_for_audit_only'] for c in selected))


if __name__=='__main__':main()
