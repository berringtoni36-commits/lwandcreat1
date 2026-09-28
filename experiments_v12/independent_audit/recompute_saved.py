"""Independent read-only recomputation of saved v12 development/selection data.

Only this directory is written. This script does not import the experiment's
analysis helpers, does not simulate new trials and refuses confirmation data.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
V12 = HERE.parent
sys.path.insert(0, str(V12 / '.deps'))
import numpy as np
from scipy.signal import lfilter
from scipy.stats import t as student_t

TARGETS = [
    'selection_e1_v500', 'selection_e1_growth', 'selection_e1_t1000_v500',
    'main_dev_round1', 'main_dev_aggressive_growth', 'main_dev_continue',
    'main_dev_mid_penalty',
]


def upper(values, alpha=.025):
    x = np.asarray(values, float)
    return float(x.mean() + student_t.isf(alpha, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))


def boot(values, seed, alpha=.025):
    x = np.asarray(values, float)
    indices = np.random.default_rng(seed).integers(len(x), size=(10000, len(x)))
    return float(np.quantile(x[indices].mean(axis=1), 1-alpha))


def audit_run(file, protocol):
    events = json.loads(file.with_name(file.stem+'_events.json').read_text(encoding='utf-8'))
    cfg = protocol['config']
    with np.load(file) as f:
        meta = json.loads(str(f['meta']))
        names = meta['controllers']; j = names.index('adaptive')
        T = len(f['d']); saved = f['anr']; d = f['d']; v = f['v']
        # lfilter implements the causal scalar recursion independently of v12.
        y = f['actuator_output']
        ys = lfilter([0., 0., 1., .5], [1.], y)
        e = d + v - ys
        ad = lfilter([.001], [1., -.999], np.abs(d))
        ae = lfilter([.001], [1., -.999], np.abs(e))
        anr = 20*np.log10((ae+1e-12)/(ad+1e-12))
        # The fixed-controller curves are audited from their saved physical
        # secondary outputs, rather than trusting either ANR or JSON summaries.
        all_errors = d[None, :] + v[None, :] - f['secondary_output']
        all_ae = lfilter([.001], [1., -.999], np.abs(all_errors), axis=-1)
        all_anr = 20*np.log10((all_ae+1e-12)/(ad[None, :]+1e-12))
        R=f['R']; cR=f['candidate_R']; phase=f['phase']; gamma=f['gamma']
        core = 12508 + 1655*R
        shadow = np.where(cR>0, 8+1655*cR, 0)
        management = 4*(2+(cR>0).astype(int)).astype(float)
        management += 4*(phase==2) + 2*(gamma>0)
        if cfg['prune_mode']=='column':
            screen=(phase==0)&(np.arange(T)%cfg['screen_every']==0)&cfg['enabled']
            management += screen*(R*500+R*20+2*(R+1)+2*R)
        if cfg.get('growth_mode','random')!='random':
            raise ValueError('This audit targets random growth only')
        decisions=0; event_mismatches=[]; omitted_jacobi_multiplications=0
        proposal=None; validate_start=None
        for ev in events:
            n=ev['sample']
            if ev['event']=='proposal':
                proposal=ev; validate_start=None
                management[n] += ev.get('decomposition_charge',0)
                if ev['new_R']>ev['old_R']:
                    management[n] += 25*(ev['old_R']+2)
                elif ev.get('prune_mode')=='counted':
                    # Existing kernel misses tol * sqrt(app*aqq) once per pair.
                    r=ev['old_R']
                    omitted_jacobi_multiplications += ev['decomposition_sweeps']*r*(r-1)//2
            elif ev['event']=='validation_start':
                validate_start=n
                if proposal is None or n-proposal['sample']!=cfg['train_samples']:
                    event_mismatches.append('training_window')
            elif ev['event'] in ('accept','reject'):
                decisions+=1;management[n]+=2
                if validate_start is None or n-validate_start!=cfg['validation_samples']:
                    event_mismatches.append('validation_window')
                s0=ev['old_loss']+cfg['lambda_cost']*(12508+1655*ev['old_R'])/14504
                s1=ev['new_loss']+cfg['lambda_cost']*(12508+1655*ev['new_R'])/14504
                if abs(s0-ev['old_score'])>1e-12 or abs(s1-ev['new_score'])>1e-12:
                    event_mismatches.append('score')
                if (s1<s0-cfg['margin']) != (ev['event']=='accept'):
                    event_mismatches.append('acceptance')
                if not np.all(phase[validate_start+1:n+1]==2):
                    event_mismatches.append('phase')
        ledger = core+shadow+management
        segs=[]; summary_discrepancy=0.
        # Dynamic metrics are recomputed from double-precision actuator history.
        # Fixed metrics are recomputed from raw saved ANR, not JSON summaries.
        for k,(start,end) in enumerate(meta['segment_bounds']):
            start=max(start,end-5000)
            metrics={name:float(saved[i,start:end].mean(dtype=np.float64)) for i,name in enumerate(names)}
            metrics['adaptive']=float(anr[start:end].mean())
            for name,val in metrics.items():
                summary_discrepancy=max(summary_discrepancy,abs(val-meta['segment_anr'][k][name]))
            segs.append(metrics)
        detail=dict(file=str(file.relative_to(V12)),case=meta['case'],run=meta['run'],T=T,
                    fir_max_abs=float(np.max(np.abs(ys-f['secondary_output'][j]))),
                    physical_error_max_abs=float(np.max(np.abs(e-f['physical_error'][j]))),
                    anr_max_abs=float(np.max(np.abs(anr-saved[j]))),
                    all_controller_anr_max_abs=float(np.max(np.abs(all_anr-saved))),
                    summary_max_abs=summary_discrepancy,
                    core_max_abs=float(np.max(np.abs(core-f['core_mults']))),
                    candidate_max_abs=float(np.max(np.abs(shadow-f['candidate_mults']))),
                    management_max_abs=float(np.max(np.abs(management-f['management_mults']))),
                    total_max_abs=float(np.max(np.abs(ledger-f['total_mults']))),
                    decisions=decisions,event_mismatches=event_mismatches,
                    omitted_jacobi_multiplications=int(omitted_jacobi_multiplications),
                    omitted_jacobi_cost_ratio=float(omitted_jacobi_multiplications/T/19132),
                    mean_mults=float(ledger.mean()),segment_anr=segs,
                    growth=sum(ev['event']=='accept' and ev['new_R']>ev['old_R'] for ev in events),
                    prune=sum(ev['event']=='accept' and ev['new_R']<ev['old_R'] for ev in events))
        return detail


def main():
    test=subprocess.run([sys.executable,'-B',str(V12/'test_adaptive.py')],
                        text=True,capture_output=True)
    (HERE/'existing_tests.txt').write_text(test.stdout+test.stderr,encoding='utf-8')
    assert test.returncode==0
    details=[];rows=[];hash_checks=[]
    for name in TARGETS:
        folder=V12/'outputs'/name
        protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
        assert protocol['phase'] in ('dev','select')
        for relative, expected in protocol['sources_sha256'].items():
            path=folder/'sources'/Path(relative)
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            hash_checks.append(dict(file=str(path.relative_to(V12)),match=actual==expected))
        group=[]
        for file in sorted(folder.glob('case*_run[0-9][0-9].npz')):
            detail=audit_run(file,protocol);detail['folder']=name
            group.append(detail);details.append(detail)
        for case in protocol['cases']:
            runs=sorted((r for r in group if r['case']==case),key=lambda r:r['run'])
            assert len(runs)==protocol['runs']
            q=np.array([r['mean_mults']/19132 for r in runs])
            for seg in range(len(runs[0]['segment_anr'])):
                a=np.array([r['segment_anr'][seg]['adaptive'] for r in runs])
                fixed=np.array([r['segment_anr'][seg]['fixed_R4'] for r in runs])
                delta=a-fixed
                row=dict(folder=name,case=case,segment=seg+1,n=len(runs),
                         dynamic_anr=float(a.mean()),fixed4_anr=float(fixed.mean()),
                         delta_db=float(delta.mean()),delta_ucb=upper(delta),
                         delta_boot_ucb=boot(delta,protocol['seed']+seg+len(case)),
                         cost_ratio=float(q.mean()),cost_ucb=upper(q),
                         cost_boot_ucb=boot(q,protocol['seed']+1000+len(case)))
                row['passed']=all((row['delta_ucb']<=.5,row['delta_boot_ucb']<=.5,
                                   row['cost_ucb']<=.9,row['cost_boot_ucb']<=.9))
                rows.append(row)
        print(name, len(group), 'saved runs audited',flush=True)
    with (HERE/'paired_statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    payload=dict(test_exit_code=test.returncode,runs=len(details),source_hashes=hash_checks,
                 details=details,stats=rows)
    (HERE/'recomputed.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    compact=dict(runs=len(details),source_hash_mismatches=sum(not h['match'] for h in hash_checks),
                 max_fir=max(r['fir_max_abs'] for r in details),
                 max_anr=max(r['anr_max_abs'] for r in details),
                 max_all_controller_anr=max(r['all_controller_anr_max_abs'] for r in details),
                 max_summary=max(r['summary_max_abs'] for r in details),
                 max_ledger=max(r['total_max_abs'] for r in details),
                 event_mismatches=sum(len(r['event_mismatches']) for r in details),
                 max_missing_jacobi_cost_ratio=max(r['omitted_jacobi_cost_ratio'] for r in details))
    (HERE/'summary.json').write_text(json.dumps(compact,indent=2),encoding='utf-8')
    print(json.dumps(compact),flush=True)


if __name__=='__main__':
    main()
