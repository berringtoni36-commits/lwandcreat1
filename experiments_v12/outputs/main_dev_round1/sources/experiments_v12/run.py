"""Independent, resumable v12 experiments; v10/v11 history is unchanged."""
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
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import numpy as np
from adaptive_core import ac, AdaptiveKron, AdaptiveKronV12, SelectionConfig, V12Config

HERE = Path(__file__).resolve().parent
PHASES = {"idle": 0, "train": 1, "validate": 2, "ramp": 3}
CASES = {'E1':0,'E2':1,'E3':2,'C1':3,'C2':4,'C3':5,'C4':6,'Fclean':7,'Fburst':8}
STATIC_LENGTHS = {'C1':20000,'C2':20000,'C3':30000,'C4':30000,
                  'Fclean':100000,'Fburst':100000}
SEEDS = {'dev':2026092801,'select':2026092802,
         'confirm':2026092803,'ablation':2026092805}


def stream(seed, case, run, part):
    paired_case = 7 if case in ('Fclean','Fburst') else CASES[case]
    return np.random.default_rng([seed, paired_case, run, part])


def default_length(case, cfg):
    if case in STATIC_LENGTHS:
        return STATIC_LENGTHS[case]
    segment = 5000 * int(np.ceil(max(40000, 5*(cfg.train_samples +
        cfg.validation_samples + cfg.ramp_samples + cfg.max_backoff)+5000)/5000))
    return segment*(3 if case=='E1' else 4)


def make_inputs(case, run, T, seed):
    rng = stream(seed,case,run,1)
    x = (ac.logistic_delay6(rng,1,T)[0] if case=='C1' else
         ac.alpha_stable_reference(rng,1,T)[0] if case=='C2' else rng.standard_normal(T))
    Om, ph = ac.draw_rff(stream(seed,case,run,2),1,500,20,3.9)
    g = ac.fir(ac.P_PATH, x)
    if case in ('E2','E3'):
        a2 = np.repeat([.02, .08, .16, .02], (T+3)//4)[:T]
        a3 = np.repeat([.01, .04, .08, .01], (T+3)//4)[:T]
        d = np.zeros(T)
        d[2:] = g[:-2] + a2[2:]*g[:-2]**2 - a3[2:]*g[1:-1]**3
    elif case=='E1':
        target_rng=stream(seed,case,run,5)
        U,_=np.linalg.qr(target_rng.standard_normal((20,4)))
        V,_=np.linalg.qr(target_rng.standard_normal((25,4)))
        low=10*np.outer(U[:,0],V[:,0]); high=5*(U@V.T)
        raw=np.zeros(T); xb=np.zeros(20)
        for n in range(T):
            xb[1:]=xb[:-1]; xb[0]=x[n]
            z=np.sqrt(2/500)*np.cos(Om[0]@xb+ph[0])
            W=low if n<T//3 or n>=2*T//3 else high
            raw[n]=np.sum(W*z.reshape(25,20).T)
        d=ac.fir(ac.S_PATH,raw)
    else:
        d = ac.primary_disturbance(x)
    v=(np.zeros(T) if case in ('C1','C2') else
       .05*ac.sas_cms(stream(seed,case,run,4),1.6,T) if case in ('C4','E3') else
       .01*stream(seed,case,run,4).standard_normal(T))
    if case=='Fburst':
        mid=T//2;v[mid:mid+100]+=.5*ac.sas_cms(stream(seed,case,run,9),1.6,100)
    return x,d,v,Om,ph


def make_controllers(case, run, seed, cfg, baselines='all', mu_map=None):
    mu = .1 if case in ('C2','C4','E3') else .2
    mu_map=mu_map or {}
    rng = stream(seed,case,run,3)
    # Nested initialization: all fixed and dynamic branches share factor prefixes.
    A = .01*rng.standard_normal((1,25,8))
    B = .01*rng.standard_normal((1,20,8))
    controllers = {}
    ranks=range(1,9) if baselines=='all' else (1,2,4,8)
    for R in ranks:
        rmu=float(mu_map.get(f'fixed_R{R}',mu))
        ctrl = ac.KronRFF(f"fixed_R{R}",1,25,20,R,rmu/2,rmu/2,2.,1e-8,np.random.default_rng(0))
        ctrl.A, ctrl.B = A[:,:,:R].copy(), B[:,:,:R].copy()
        controllers[ctrl.name] = ctrl
    controllers['full_RFF_MCC'] = ac.FullRFF('full_RFF_MCC',1,500,
                                             float(mu_map.get('full_RFF_MCC',mu)),2.,1e-8)
    if baselines=='all':
        controllers['full180'] = ac.FullRFF('full180',1,180,
                                            float(mu_map.get('full180',mu)),2.,1e-8)
        controllers['v11'] = AdaptiveKron(controllers['fixed_R2'],
                                          stream(seed,case,run,40),SelectionConfig())
    amu=float(mu_map.get('adaptive',mu))
    base=ac.KronRFF('adaptive_base',1,25,20,2,amu/2,amu/2,2.,1e-8,np.random.default_rng(0))
    base.A,base.B=A[:,:,:2].copy(),B[:,:,:2].copy()
    controllers['adaptive'] = AdaptiveKronV12(base,stream(seed,case,run,4),cfg)
    return controllers,A,B,mu


def simulate(case, run, T, seed, cfg, dest, baselines='all', mu_map=None):
    x,d,v,Om,ph = make_inputs(case,run,T,seed)
    cs,A0,B0,mu = make_controllers(case,run,seed,cfg,baselines,mu_map)
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
        z180=np.sqrt(500/180)*z[:,:180]
        q180=np.sqrt(500/180)*q[:,:180]
        dv = np.array([d[n]+v[n]])
        Ad = .999*Ad+.001*abs(d[n])
        for j,(name,ctrl) in enumerate(cs.items()):
            t0 = perf_counter()
            ys = ctrl.step(z180,q180,dv)[0] if name=='full180' else ctrl.step(z,q,dv)[0]
            timings[name] += perf_counter()-t0
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
    segments=3 if case=='E1' else 4 if case in ('E2','E3') else 1
    segment_anr=[];segment_mults=[];bounds=[]
    for j in range(segments):
        first=j*T//segments;end=(j+1)*T//segments
        bounds.append([first,end])
        segment_anr.append({name:float(anr[k,max(first,end-5000):end].mean())
                            for k,name in enumerate(names)})
        segment_mults.append(float(traces['total_mults'][first:end].mean()))
    meta = dict(case=case,run=run,T=T,seed=seed,mu=mu,controllers=names,
                seconds=perf_counter()-start,controller_seconds=timings,
                selection=asdict(cfg),phase_codes=PHASES,physical_fir_max_abs=float(error_fir),
                segment_bounds=bounds,segment_anr=segment_anr,segment_mults=segment_mults)
    np.savez_compressed(dest/f'case{case}_run{run:02d}.npz',x=x,d=d,v=v,Om=Om,ph=ph,
                        initial_A=A0,initial_B=B0,anr=anr,physical_error=errors,
                        secondary_output=ys_record,actuator_output=y_adaptive,
                        **traces,meta=json.dumps(meta))
    (dest/f'case{case}_run{run:02d}_events.json').write_text(
        json.dumps(cs['adaptive'].events,indent=2,ensure_ascii=False),encoding='utf-8')
    summary = dict(case=case,run=run,ss_anr={name:float(anr[j,-5000:].mean())
                      for j,name in enumerate(names)},segment_anr=segment_anr,
                   segment_mults=segment_mults,segment_bounds=bounds,
                   mean_R=float(traces['R'].mean()),
                   final_R=cs['adaptive'].active.R,
                   accepted_growth=sum(e['event']=='accept' and e['new_R']>e['old_R']
                                       for e in cs['adaptive'].events),
                   accepted_prune=sum(e['event']=='accept' and e['new_R']<e['old_R']
                                      for e in cs['adaptive'].events),
                   mean_mults=float(traces['total_mults'].mean()),
                   mean_candidate_mults=float(traces['candidate_mults'].mean()),
                   mean_management_mults=float(traces['management_mults'].mean()),
                   decomposition_calls=cs['adaptive'].decomposition_calls,
                   decomposition_seconds=cs['adaptive'].decomposition_seconds,
                   physical_fir_max_abs=float(error_fir),
                   seconds=meta['seconds'])
    print(json.dumps(summary),flush=True)
    return summary


def run_one(spec):
    case,run,T,seed,config,baselines,mu_map,destination=spec
    cfg=V12Config(**config);dest=Path(destination)
    marker=dest/f'case{case}_run{run:02d}_summary.json'
    if marker.exists():
        return json.loads(marker.read_text(encoding='utf-8'))
    data=dest/f'case{case}_run{run:02d}.npz'
    if data.exists():
        archived=data.with_suffix(f'.interrupted_{datetime.now():%Y%m%d_%H%M%S_%f}.npz')
        data.rename(archived)
    try:
        result=simulate(case,run,T,seed,cfg,dest,baselines,mu_map)
        marker.write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    except BaseException:
        log=dest/f'case{case}_run{run:02d}_failure_{datetime.now():%Y%m%d_%H%M%S_%f}.txt'
        log.write_text(traceback.format_exc(),encoding='utf-8')
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=list(SEEDS),required=True)
    parser.add_argument('--cases',nargs='+',choices=list(CASES),required=True)
    parser.add_argument('--runs',type=int)
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--T',type=int,help='short diagnostic only; development data')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--baselines',choices=['essential','all'],default='all')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.phase=='confirm' and (args.config is None or args.T is not None or
                                  args.runs not in (None,20) or args.baselines!='all'):
        parser.error('Confirmation requires a frozen config, full lengths, 20 runs and all baselines')
    if args.T is not None and args.phase!='dev':
        parser.error('--T is only allowed in development')
    if args.T is not None and args.T<1000:
        parser.error('--T must be at least 1000')
    runs=args.runs or {'dev':5,'select':8,'confirm':20,'ablation':5}[args.phase]
    if runs<1 or args.workers not in (1,2,3,4) or args.round<1:
        parser.error('Invalid run count, workers or round')
    frozen=json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    cfg=V12Config(**frozen.get('selection',{}))
    mu_map=frozen.get('mu_by_method',{})
    seed=SEEDS[args.phase]+100*(args.round-1)
    lengths={case:args.T or default_length(case,cfg) for case in args.cases}
    dest=(args.output or HERE/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_{args.phase}').resolve()
    dest.mkdir(parents=True,exist_ok=True)
    sources=[HERE/'adaptive_core.py',HERE/'run.py',HERE/'test_adaptive.py',
             HERE.parent/'experiments_v10'/'anc_core.py']
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=HERE.parent,
                        text=True,capture_output=True,check=False).stdout.strip()
    protocol=dict(phase=args.phase,round=args.round,runs=runs,cases=args.cases,
                  lengths=lengths,seed=seed,config=asdict(cfg),mu_by_method=mu_map,
                  baselines=args.baselines,git_head=head,python=sys.version,
                  numpy=np.__version__,platform=platform.platform(),
                  sources_sha256={str(p.relative_to(HERE.parent)):
                                  hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  performance_margin_db=.5,total_cost_ratio_target=.9,
                  confirm_alpha=.05/(args.round*(args.round+1)),
                  interpretation='Only phase=confirm with frozen inputs is independent confirmation')
    pfile=dest/'protocol.json'
    if pfile.exists():
        if json.loads(pfile.read_text(encoding='utf-8'))!=protocol:
            parser.error('Resume protocol differs from this directory; create a new run')
    else:
        pfile.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        for p in sources:
            target=dest/'sources'/p.relative_to(HERE.parent)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
    print(f'OUTPUT_DIR={dest}',flush=True)
    specs=[(case,run,lengths[case],seed,asdict(cfg),args.baselines,mu_map,str(dest))
           for case in args.cases for run in range(runs)]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_one,spec):spec[:2] for spec in specs}
        for future in as_completed(futures):
            result=future.result()
            results.append(result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
    results.sort(key=lambda r:(CASES[r['case']],r['run']))
    (dest/'run_summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
