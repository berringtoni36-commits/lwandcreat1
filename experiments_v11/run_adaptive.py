"""Isolated reproducible pilot. Run: python run_adaptive.py --profile pilot

All inputs, maps, traces and events go into a NEW timestamped output directory.
The unchanged v10 simulation files and historical results are never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np
from adaptive_core import ac, AdaptiveKron, SelectionConfig

HERE = Path(__file__).resolve().parent
PHASES = {"idle": 0, "train": 1, "validate": 2, "ramp": 3}


def make_inputs(case, run, T, seed):
    # Per-run generators make run 0 invariant to the requested total run count.
    rng = np.random.default_rng([seed, case, run, 1])
    x = ac.logistic_delay6(rng, 1, T)[0] if case == 0 else rng.standard_normal(T)
    g = ac.fir(ac.P_PATH, x)
    if case == 2:
        # A diagnostic change in primary-path nonlinearity; no target R trajectory
        # and no segment-wise normalization using future observations.
        a2 = np.repeat([.02, .08, .16, .02], (T+3)//4)[:T]
        a3 = np.repeat([.01, .04, .08, .01], (T+3)//4)[:T]
        d = np.zeros(T)
        d[2:] = g[:-2] + a2[2:]*g[:-2]**2 - a3[2:]*g[1:-1]**3
    else:
        d = ac.primary_disturbance(x)
    v = np.zeros(T) if case == 0 else .01*rng.standard_normal(T)
    if case in (1, 2):
        if case == 1:
            v = .05*ac.sas_cms(rng, 1.6, T)
        else:
            start = 3*T//8
            v[start:start+100] += .5*ac.sas_cms(rng, 1.6, 100)
    Om, ph = ac.draw_rff(np.random.default_rng([seed, case, run, 2]), 1, 500, 20, 3.9)
    return x,d,v,Om,ph


def make_controllers(case, run, seed, cfg):
    mu = .1 if case == 1 else .2
    rng = np.random.default_rng([seed, case, run, 3])
    # Nested initialization: all fixed and dynamic branches share factor prefixes.
    A = .01*rng.standard_normal((1,25,8))
    B = .01*rng.standard_normal((1,20,8))
    controllers = {}
    for R in (1,2,4,8):
        ctrl = ac.KronRFF(f"fixed_R{R}",1,25,20,R,mu/2,mu/2,2.,1e-8,np.random.default_rng(0))
        ctrl.A, ctrl.B = A[:,:,:R].copy(), B[:,:,:R].copy()
        controllers[ctrl.name] = ctrl
    controllers['full_RFF_MCC'] = ac.FullRFF('full_RFF_MCC',1,500,mu,2.,1e-8)
    controllers['adaptive'] = AdaptiveKron(controllers['fixed_R2'],
                                           np.random.default_rng([seed,case,run,4]),cfg)
    return controllers,A,B,mu


def simulate(case, run, T, seed, cfg, dest):
    x,d,v,Om,ph = make_inputs(case,run,T,seed)
    cs,A0,B0,mu = make_controllers(case,run,seed,cfg)
    names = list(cs)
    anr = np.zeros((len(cs),T),np.float32)
    errors = np.zeros_like(anr)
    ys_record = np.zeros_like(anr)
    y_adaptive = np.zeros(T)
    fields = ['R','next_R','candidate_R','gamma','core_mults','candidate_mults',
              'management_mults','total_mults','exp_evals','resident_factor_coeffs','live_terms']
    traces = {k: np.zeros(T) for k in fields}
    traces['phase'] = np.zeros(T,np.int8)
    timings = dict.fromkeys(names,0.)
    xb,zh = np.zeros((1,20)),np.zeros((4,1,500))
    Ad = 0.; Ae = np.zeros(len(cs)); start = perf_counter()
    for n in range(T):
        xb[:,1:] = xb[:,:-1].copy(); xb[0,0] = x[n]
        z = np.sqrt(2/500)*np.cos(np.einsum('rdm,rm->rd',Om,xb)+ph)
        zh[1:] = zh[:-1].copy(); zh[0] = z
        q = zh[2] + .5*zh[3]
        dv = np.array([d[n]+v[n]])
        Ad = .999*Ad+.001*abs(d[n])
        for j,(name,ctrl) in enumerate(cs.items()):
            t0 = perf_counter(); ys = ctrl.step(z,q,dv)[0]; timings[name] += perf_counter()-t0
            error = dv[0]-ys
            errors[j,n] = error; ys_record[j,n] = ys
            Ae[j] = .999*Ae[j]+.001*abs(error)
            anr[j,n] = 20*np.log10((Ae[j]+1e-12)/(Ad+1e-12))
        for k in fields:
            traces[k][n] = cs['adaptive'].last[k]
        traces['phase'][n] = PHASES[cs['adaptive'].last['phase']]
        y_adaptive[n] = cs['adaptive'].last['y']
    assert np.isfinite(anr).all() and np.isfinite(errors).all()
    # End-to-end physical path audit, including every structural transition.
    error_fir = np.max(np.abs(ac.fir(ac.S_PATH,y_adaptive)-ys_record[-1]))
    assert error_fir < 1e-5
    meta = dict(case=case,run=run,T=T,seed=seed,mu=mu,controllers=names,
                seconds=perf_counter()-start,controller_seconds=timings,
                selection=asdict(cfg),phase_codes=PHASES,physical_fir_max_abs=float(error_fir))
    np.savez_compressed(dest/f'case{case}_run{run:02d}.npz',x=x,d=d,v=v,Om=Om,ph=ph,
                        initial_A=A0,initial_B=B0,anr=anr,physical_error=errors,
                        secondary_output=ys_record,actuator_output=y_adaptive,
                        **traces,meta=json.dumps(meta))
    (dest/f'case{case}_run{run:02d}_events.json').write_text(
        json.dumps(cs['adaptive'].events,indent=2,ensure_ascii=False),encoding='utf-8')
    summary = dict(case=case,run=run,ss_anr={name:round(float(anr[j,-5000:].mean()),3)
                      for j,name in enumerate(names)},mean_R=round(float(traces['R'].mean()),3),
                   final_R=cs['adaptive'].active.R,
                   accepted=sum(e['event']=='accept' for e in cs['adaptive'].events),
                   mean_mults=round(float(traces['total_mults'].mean()),1),
                   seconds=round(meta['seconds'],2))
    print(json.dumps(summary),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile',choices=['smoke','pilot'],default='smoke')
    parser.add_argument('--seed',type=int,default=20260928)
    parser.add_argument('--runs',type=int)
    args = parser.parse_args()
    runs = args.runs if args.runs is not None else (1 if args.profile=='smoke' else 3)
    if runs < 1:
        parser.error('--runs must be positive')
    cfg = SelectionConfig()
    lengths = [8000]*3 if args.profile=='smoke' else [20000,20000,32000]
    dest = HERE/'outputs'/f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.profile}"
    dest.mkdir(parents=True,exist_ok=False)
    sources = [HERE/'adaptive_core.py',HERE/'run_adaptive.py',HERE/'analyze_adaptive.py',
               HERE/'test_adaptive.py',
               HERE.parent/'experiments_v10'/'anc_core.py']
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in sources:
        target = dest/'sources'/p.parent.name/p.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    protocol = dict(profile=args.profile,runs=runs,lengths=lengths,seed=args.seed,
                    config=asdict(cfg),python=sys.version,numpy=np.__version__,platform=platform.platform(),
                    sources_sha256=hashes,case_names=['chaotic_stationary','gaussian_impulsive_noise',
                    'gaussian_piecewise_primary_with_burst'],
                    interpretation='Development pilot, not confirmatory publication evidence. No target R trajectory.')
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(f'OUTPUT_DIR={dest}',flush=True)
    summaries=[]
    for case,T in enumerate(lengths):
        for run in range(runs):
            summaries.append(simulate(case,run,T,args.seed,cfg,dest))
    (dest/'run_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    from analyze_adaptive import analyze
    analyze(dest)


if __name__=='__main__':
    main()
