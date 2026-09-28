"""Replay saved output and verify the complete development-only cost ledger."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'experiments_v12/outputs/main_dev_round1'
sys.path.insert(0,str(SOURCE/'sources/experiments_v10'))
import anc_core as ac

R4_COST=ac.mults_kron(25,20,4,20,4)+4


def check(case,run,arm):
    stem=f'{case}_run{run:02d}_tuned_{arm}'
    summary=json.loads((HERE/f'{stem}_summary.json').read_text(encoding='utf-8'))
    with np.load(HERE/f'{stem}_trace.npz') as tf, np.load(SOURCE/f'case{case}_run{run:02d}.npz') as sf:
        t={k:tf[k] for k in tf.files}
        src={k:sf[k] for k in ('x','d','v','Om','ph','initial_A','initial_B')}
        assert summary['phase']=='dev' and summary['case']==case and summary['run']==run
        assert len(t['anr'])==400000
        assert np.array_equal(t['total'],t['core']+t['candidate']+t['retiring']+t['management']+t['reorder']+4)
        core_expected=np.zeros_like(t['core'])
        for r in np.unique(t['R']):
            core_expected[t['R']==r]=ac.mults_kron(25,20,int(r),20,4)
        assert np.array_equal(t['core'],core_expected)
        assert np.array_equal(t['resident_factor_coeffs'],45*(t['R'].astype(int)+t['candidate_R'].astype(int)+t['retiring_R'].astype(int)))
        assert abs(t['total'].mean()-summary['mean_mults'])<1e-9
        # Independent tuned-R4 replay up through the causal proposal sample.
        ctrl=ac.KronRFF('audit_R4',1,25,20,4,summary['mu']/2,summary['mu']/2,
                        2.,1e-8,np.random.default_rng(0))
        ctrl.A=src['initial_A'][:,:,:4].copy()
        ctrl.B=src['initial_B'][:,:,:4].copy()
        xb=np.zeros((1,20));zh=np.zeros((4,1,500))
        prefix_error=0.
        om=src['Om'];ph=src['ph'];x=src['x'];d=src['d'];v=src['v']
        for n in range(20001):
            xb[:,1:]=xb[:,:-1].copy();xb[0,0]=x[n]
            z=np.sqrt(2/500)*np.cos(np.einsum('rdm,rm->rd',om,xb)+ph)
            zh[1:]=zh[:-1].copy();zh[0]=z
            q=zh[2]+.5*zh[3]
            dv=np.array([d[n]+v[n]])
            ys=ctrl.step(z,q,dv)[0]
            prefix_error=max(prefix_error,abs(float(dv[0]-ys)-float(t['error'][n])))
        assert prefix_error<1e-5,prefix_error
        ys=ac.fir(ac.S_PATH,t['drive'][None,:])[0]
        physical=float(np.max(np.abs(ys-(src['d']+src['v']-t['error']))))
        assert physical<1e-5 and abs(physical-summary['physical_fir_max_abs'])<1e-10
        assert np.max(np.abs(ys-t['ys_actual']))<3e-6
        assert abs(float(np.mean(t['exp_evals']))-summary['mean_exp_evals'])<1e-9
        ramp=np.flatnonzero(t['retiring_R'])
        accepted=[e for e in summary['events'] if e.get('accepted')]
        assert len(ramp)==100*len(accepted)
        for event in accepted:
            rr=np.arange(event['ramp_start'],event['ramp_end']+1)
            assert len(rr)==100 and np.array_equal(ramp[(ramp>=rr[0])&(ramp<=rr[-1])],rr)
            expected=np.arange(1,101)/100
            assert np.max(np.abs(t['gamma'][rr]-expected))<1e-7
            assert np.min(t['retiring'][rr])>0
            assert np.isfinite(t['ys_active'][rr]).all()
            assert np.isfinite(t['ys_retiring'][rr]).all()
            post=np.arange(event['ramp_end']+1,event['ramp_end']+4)
            assert np.max(np.abs(t['post_ramp_output_error'][post]))<1e-10
            assert np.all(t['management'][rr]>=18)
        # Independent ANR recomputation from saved physical error.
        ae=ad=0.;maxdiff=0.
        for n in range(400000):
            ae=.999*ae+.001*abs(float(t['error'][n]))
            ad=.999*ad+.001*abs(float(src['d'][n]))
            if n%1000==0 or n>=395000:
                expected=20*np.log10((ae+1e-12)/(ad+1e-12))
                maxdiff=max(maxdiff,abs(expected-float(t['anr'][n])))
        assert maxdiff<2e-4,maxdiff
        for j,s in enumerate(summary['segments']):
            lo=j*100000;hi=(j+1)*100000
            assert abs(s['selector_anr_db']-float(t['anr'][hi-5000:hi].mean()))<1e-6
            assert abs(s['mean_mults']-float(t['total'][lo:hi].mean()))<1e-9
            assert abs(s['cost_ratio_to_R4']-s['mean_mults']/R4_COST)<1e-12
        sweep=json.loads((ROOT/'experiments_v12/diagnostics/capacity_envelope.sweeps.json').read_text(encoding='utf-8'))
        scan=next(item for item in sweep['runs'] if item['case']==case and item['run']==run)
        mu_index=scan['mu'].index(summary['mu'])
        for j,s in enumerate(summary['segments']):
            assert abs(s['tuned_fixed_R4_anr_db']-scan['segment_anr'][j][3][mu_index])<1e-9
        proposals=[e for e in summary['events'] if e['type'].startswith('proposal_')]
        decisions=[e for e in summary['events'] if not e['type'].startswith('proposal_')]
        assert len(proposals)==len(decisions)
        assert proposals[0]['n']==20000 and proposals[0]['from_R']==4 and proposals[0]['to_R']==2
        for proposal,decision in zip(proposals,decisions):
            assert decision['n']==proposal['n']+4000
            assert decision['type']==proposal['type'].removeprefix('proposal_')
            assert int(t['reorder'][proposal['n']])==proposal['reorder_mults']
            assert int(t['candidate'][proposal['n']+1:decision['n']+1].min())>0
        if arm=='once':
            assert len(proposals)==1 and proposals[0]['type']=='proposal_prune'
        assert np.max(t['resident_factor_coeffs'])==summary['max_resident_factor_coeffs']
        return {'case':case,'run':run,'arm':arm,'physical_fir_max_abs':physical,
                'tuned_fixed_R4_prefix_max_abs':prefix_error,
                'anr_replay_max_abs_db':maxdiff,
                'max_segment_gap_db':max(s['anr_gap_db'] for s in summary['segments']),
                'max_segment_cost_ratio':max(s['cost_ratio_to_R4'] for s in summary['segments']),
                'accepted_prune':summary['accepted_prune'],
                'accepted_growth':summary['accepted_growth'],
                'rejected_proposals':summary['rejected_proposals'],
                'candidate_mean_mults':summary['ledger_mean']['candidate'],
                'reorder_mean_mults':summary['ledger_mean']['reorder']}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=['once','gated_growth','both'],default='both')
    args=ap.parse_args()
    arms=['once','gated_growth'] if args.arm=='both' else [args.arm]
    rows=[check(case,run,arm) for arm in arms for case in ('E2','E3') for run in range(5)]
    aggregate={}
    for arm in arms:
        rr=[r for r in rows if r['arm']==arm]
        aggregate[arm]={
            'n_runs':len(rr),'n_segments':len(rr)*4,
            'worst_segment_gap_db':max(r['max_segment_gap_db'] for r in rr),
            'worst_segment_cost_ratio':max(r['max_segment_cost_ratio'] for r in rr),
            'joint_gate_all_segments':all(r['max_segment_gap_db']<=.5 and r['max_segment_cost_ratio']<=.9 for r in rr),
            'accepted_prune':sum(r['accepted_prune'] for r in rr),
            'accepted_growth':sum(r['accepted_growth'] for r in rr),
            'rejected_proposals':sum(r['rejected_proposals'] for r in rr),
            'max_physical_fir_abs':max(r['physical_fir_max_abs'] for r in rr),
            'max_anr_replay_abs_db':max(r['anr_replay_max_abs_db'] for r in rr),
        }
    output={'phase':'dev','rows':rows,'aggregate':aggregate,
            'cost_reference':{'fixed_R4_mults_per_sample':R4_COST,
                              'fixed_R1_to_R8':{str(r):ac.mults_kron(25,20,r,20,4)+4 for r in range(1,9)},
                              'full_RFF_MCC':ac.mults_full(500,20,4)+4}}
    (HERE/'audit_results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
