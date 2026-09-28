"""Quantify the old Jacobi count omission without altering saved trials."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import importlib.util

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'.deps'))
import numpy as np
from scipy.stats import t


def upper(values, alpha=.025):
    x = np.asarray(values, float)
    return float(x.mean() + t.isf(alpha, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))


def load_counter(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_formula():
    old = load_counter(HERE.parent/'outputs'/'selection_e1_v500'/'sources'/
                       'experiments_v12'/'counted_linalg.py','legacy_counted_linalg')
    new = load_counter(HERE.parent/'counted_linalg.py','current_counted_linalg')
    checks = []
    for rank in (2,4,8):
        square = np.random.default_rng(rank).normal(size=(rank,rank))
        a,b = old.Count(),new.Count()
        _,sa,_,sweeps_a = old.jacobi_svd(square,a)
        _,sb,_,sweeps_b = new.jacobi_svd(square,b)
        missing = sweeps_a*rank*(rank-1)//2
        assert sweeps_a==sweeps_b and np.array_equal(sa,sb)
        assert b.multiplies-a.multiplies == missing
        checks.append(dict(rank=rank,sweeps=sweeps_a,legacy=a.multiplies,
                           current=b.multiplies,missing=missing))
    return checks


def main():
    source = json.loads((HERE/'recomputed.json').read_text(encoding='utf-8'))
    formula_checks = verify_formula()
    stats = {(r['folder'],r['case'],r['segment']): r for r in source['stats']}
    result = []
    for folder in sorted({r['folder'] for r in source['details']}):
        cases = sorted({r['case'] for r in source['details'] if r['folder']==folder})
        for case in cases:
            runs = sorted((r for r in source['details']
                           if r['folder']==folder and r['case']==case),key=lambda r:r['run'])
            old = np.array([r['mean_mults']/19132 for r in runs])
            add = np.array([r['omitted_jacobi_multiplications']/r['T']/19132
                            for r in runs])
            corrected = old+add
            row = dict(folder=folder,case=case,n=len(runs),
                       omitted_multiplications=int(sum(r['omitted_jacobi_multiplications'] for r in runs)),
                       omitted_per_run=[r['omitted_jacobi_multiplications'] for r in runs],
                       mean_legacy_ratio=float(old.mean()),
                       mean_ratio_correction=float(add.mean()),
                       mean_corrected_ratio=float(corrected.mean()),
                       legacy_ratio_upper=upper(old),corrected_ratio_upper=upper(corrected),
                       old_cost_gate=upper(old)<=.9,corrected_cost_gate=upper(corrected)<=.9,
                       anr_gates_all=all(r['delta_ucb']<=.5 and r['delta_boot_ucb']<=.5
                           for (f,c,_),r in stats.items() if f==folder and c==case))
            result.append(row)
    assert all(r['old_cost_gate']==r['corrected_cost_gate'] for r in result)
    (HERE/'legacy_ledger_impact.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (HERE/'jacobi_count_checks.json').write_text(json.dumps(formula_checks,indent=2),encoding='utf-8')
    print(json.dumps([dict(folder=r['folder'],case=r['case'],
                           omitted=r['omitted_multiplications'],
                           mean_ratio_correction=r['mean_ratio_correction'],
                           corrected_ratio_upper=r['corrected_ratio_upper'])
                      for r in result]))


if __name__=='__main__':
    main()
