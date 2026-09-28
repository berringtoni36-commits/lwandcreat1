"""Causal rank selector using the original 500-dimensional RFF.

The simulated plant uses d and v to produce measured physical error. With
the exact secondary path, the controller reconstructs dv from that error and
the known actuator output. No segment labels, future data, or ANR enter the
selector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / 'experiments_v12/outputs/main_dev_round1/sources/experiments_v10'
sys.path.insert(0, str(CORE))
import anc_core as ac
sys.path.insert(0, str(ROOT / 'experiments_v12'))
import counted_linalg as cl

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments_v12/outputs/main_dev_round1'
D1, D2, D, M = 25, 20, 500, 20
SHARED = D*M + D + D*len(ac.S_PATH)
MU = {'E2': .1, 'E3': .05}  # development-wide fixed-R4 mean winners
TRAIN = 2000
VALIDATE = 2000
PROPOSE_INTERVAL = 40000
FIRST_PROPOSE = 20000
COOLDOWN = 12000
DETECT_REL = .28
DETECT_HOLD = 250
SPECTRAL_THRESHOLD = .15
RAMP_SAMPLES = 100


def make_ctrl(rank: int, mu: float, A: np.ndarray, B: np.ndarray):
    c = ac.KronRFF('selector', 1, D1, D2, rank, mu/2, mu/2,
                   2., 1e-8, np.random.default_rng(0))
    c.A = A.copy()
    c.B = B.copy()
    return c


def branch_output(ctrl,z,want_secondary=False):
    """Produce y_n from pre-update factors, then advance private FIR state."""
    Z=ctrl.mat(z[None,:])
    y=float(np.einsum('rjk,rjk->r',ctrl.B,Z@ctrl.A)[0])
    ctrl.ybuf[:,1:]=ctrl.ybuf[:,:-1]
    ctrl.ybuf[:,0]=y
    ys=float((ctrl.ybuf@ac.S_PATH)[0]) if want_secondary else np.nan
    return y,ys


def branch_update(ctrl,q,dv):
    """Exact v10 sequential B-then-A update after measuring physical error."""
    A,B=ctrl.A,ctrl.B
    Q=ctrl.mat(q[None,:])
    UB=Q@A
    e1=np.array([dv])-np.einsum('rjk,rjk->r',B,UB)
    psi1,_=ac.influence(e1,ctrl.sigma)
    nb=np.einsum('rjk,rjk->r',UB,UB)
    B=B+(ctrl.mu_b*psi1/(ctrl.eps+nb))[:,None,None]*UB
    UA=Q.transpose(0,2,1)@B
    e2=np.array([dv])-np.einsum('rjk,rjk->r',A,UA)
    psi2,_=ac.influence(e2,ctrl.sigma)
    na=np.einsum('rjk,rjk->r',UA,UA)
    A=A+(ctrl.mu_a*psi2/(ctrl.eps+na))[:,None,None]*UA
    ctrl.A,ctrl.B=A,B


def prune(ctrl, target: int, mu: float, old_perm: np.ndarray, method: str):
    """Sort current signed weights and count a QR/Jacobi dense SVD."""
    if method in ('thin_qr','clone'):
        # Exact rank-2 gauge transform: W = B A^T = (B R^T) Q^T,
        # with A = Q R. This tests same-rank refactorization without
        # changing the feature order or representable coefficient matrix.
        assert ctrl.R == target == 2
        count = cl.Count()
        if method == 'thin_qr':
            Q,R = cl.qr_mgs2(ctrl.A[0],count)
            B = cl.product(ctrl.B[0],R.T,count)
            A = Q
            assert np.allclose(B@A.T, ctrl.B[0]@ctrl.A[0].T, atol=1e-10)
        else:
            A = ctrl.A[0].copy(); B = ctrl.B[0].copy()
        out = make_ctrl(2,mu,A[None,:,:],B[None,:,:])
        out.ybuf = ctrl.ybuf.copy()
        return out,old_perm.copy(),count.multiplies,0,0.0
    W = ctrl.B[0] @ ctrl.A[0].T
    w = W.T.reshape(D)
    if method=='signed_sort':
        p=np.argsort(w,kind='stable')
    elif method=='identity':
        p=np.arange(D)
    elif method=='fixed_random':
        p=np.random.default_rng(2026092815).permutation(D)
    else:
        raise ValueError(method)
    new_perm = old_perm[p]
    Wp = w[p].reshape(D1,D2).T
    count=cl.Count()
    Q,R=cl.qr_mgs2(Wp.T,count)
    try:
        U,s,V,sweeps=cl.jacobi_svd(R,count)
    except ValueError:
        # Bounded Jacobi failed. Reject without an uncounted library fallback.
        count.multiplies += D*ctrl.R
        return None,new_perm,count.multiplies,32,float('inf')
    # Wp = V diag(s) (Q U)^T, since Wp.T = Q U diag(s) V^T.
    left=V[:,:target]
    right=cl.product(Q,U[:,:target],count)
    t = np.sqrt(s[:target])
    count.square_roots += target
    A = (right*t).reshape(1,D1,target)
    B = (left*t).reshape(1,D2,target)
    count.multiplies += (D1+D2)*target
    out = make_ctrl(target, mu, A, B)
    out.ybuf = ctrl.ybuf.copy()
    count.multiplies += D*ctrl.R  # assemble current 500 coefficients
    # Stable sort uses comparisons, not scalar multiplications.
    assert np.allclose(Wp, V@np.diag(s)@(Q@U).T,atol=1e-8)
    truncation=float(np.sqrt(np.sum(s[target:]**2)/np.sum(s**2)))
    count.multiplies += len(s) # squared singular values for truncation audit
    return out, new_perm, count.multiplies, sweeps, truncation


def grow(ctrl, target: int, mu: float, Q_hist, e_hist):
    """Add novel singular directions of a causal robust residual gradient."""
    qs = np.asarray(Q_hist)
    es = np.asarray(e_hist)
    if len(qs) == 0:
        G = np.zeros((D2,D1))
    else:
        cap = max(float(np.median(np.abs(es))) * 3, 1e-3)
        es = np.clip(es, -cap, cap)
        G = np.einsum('n,nij->ij', es, qs.reshape(-1,D1,D2).transpose(0,2,1)) / len(es)
    U0,_ = np.linalg.qr(ctrl.B[0]); V0,_ = np.linalg.qr(ctrl.A[0])
    G = G-U0@(U0.T@G)
    G = G-(G@V0)@V0.T
    U,s,Vt = np.linalg.svd(G, full_matrices=False)
    k = target-ctrl.R
    amp = np.sqrt(np.maximum(s[:k],1e-10))*.02
    A = np.concatenate([ctrl.A[0], Vt[:k].T*amp],axis=1)[None,:,:]
    B = np.concatenate([ctrl.B[0], U[:,:k]*amp],axis=1)[None,:,:]
    out = make_ctrl(target,mu,A,B)
    out.ybuf = ctrl.ybuf.copy()
    # 500 multiplies per stored sample for gradient; projections and SVD.
    count = len(es)*D + 2*D*ctrl.R + 20*D2*D1*D2
    return out, count


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def simulate(x,d,v,Om,ph,A0,B0,mu,config=None):
    """Run a causal selector; inputs contain no case, phase or segment boundary.

    Returns all physical and cost traces, without reading files or references.
    d and v form the plant measurement; the controller update receives only
    same-sample reconstructed dv. d separately enters the offline ANR
    denominator, never a structure decision.
    """
    t0 = perf_counter()
    config=config or {}
    arm=config.get('arm','once')
    proposal=config.get('proposal','signed_sort')
    ablate_spectral_veto=bool(config.get('ablate_spectral_veto',False))
    ramp_samples=int(config.get('ramp_samples',RAMP_SAMPLES))
    # Development-only diagnostic: periodically re-sort the current R2
    # coefficient vector without consulting a known segment boundary.
    resort_interval=int(config.get('resort_interval',0))
    assert resort_interval==0 or resort_interval>=TRAIN+VALIDATE+ramp_samples
    T=len(x)
    assert T==len(d)==len(v) and Om.shape==(1,D,M) and ph.shape==(1,D)
    assert ramp_samples>=0
    active = make_ctrl(4,mu,A0[:,:,:4],B0[:,:,:4])
    candidate = None
    retiring=None
    retiring_perm=None
    ramp_left=0
    active_perm=np.arange(D)
    candidate_perm=None
    phase='idle'; phase_end=-1; pending=None
    last_event=-COOLDOWN
    next_prune=T if arm=='fixed' else FIRST_PROPOSE
    events=[]
    cost = {k:np.zeros(T,dtype=np.int64) for k in
            ('core','candidate','retiring','management','reorder','total')}
    rank=np.zeros(T,dtype=np.int8)
    candidate_rank=np.zeros(T,dtype=np.int8)
    retiring_rank=np.zeros(T,dtype=np.int8)
    coeffs=np.zeros(T,dtype=np.int16)
    anr=np.zeros(T,dtype=np.float32)
    error=np.zeros(T,dtype=np.float64)
    drive=np.zeros(T,dtype=np.float64)
    ys_active=np.zeros(T,dtype=np.float64)
    ys_candidate=np.full(T,np.nan,dtype=np.float64)
    ys_retiring=np.full(T,np.nan,dtype=np.float64)
    ys_actual=np.zeros(T,dtype=np.float64)
    gamma=np.ones(T,dtype=np.float64)
    post_ramp_output_error=np.full(T,np.nan,dtype=np.float64)
    post_ramp_check=0
    exp_evals=np.zeros(T,dtype=np.int8)
    # The actual physical actuator has its own FIR history across switches.
    yactual=np.zeros(4)
    xb=np.zeros(20); zh=np.zeros((4,500))
    Ad=Ae=0.
    # Observable robust residual detector; slow baseline freezes on alarm.
    slow=fast=0.; hold=0
    q_history=[];e_history=[]
    check_active=check_candidate=0.; check_count=0
    cooldown_until=0
    next_growth=0
    cos_scale=np.sqrt(2/500)
    for n in range(T):
        xb[1:]=xb[:-1]; xb[0]=x[n]
        z=cos_scale*np.cos(Om[0]@xb+ph[0])
        zh[1:]=zh[:-1];zh[0]=z
        q=zh[2]+.5*zh[3]
        executed_rank=active.R
        executed_candidate_rank=candidate.R if candidate is not None else 0
        executed_retiring_rank=retiring.R if retiring is not None else 0
        ya,ys_a=branch_output(active,z[active_perm],
                              candidate is not None or retiring is not None or post_ramp_check>0)
        ys_active[n]=ys_a
        ys_c=0.;yc=0.
        cost['core'][n]=ac.mults_kron(D1,D2,active.R,M,4)
        if candidate is not None:
            yc,ys_c=branch_output(candidate,z[candidate_perm],True)
            ys_candidate[n]=ys_c
            cost['candidate'][n]=ac.mults_kron(D1,D2,candidate.R,M,4)-SHARED+4
        yr=0.
        if retiring is not None:
            yr,ys_r=branch_output(retiring,z[retiring_perm],True)
            ys_retiring[n]=ys_r
            cost['retiring'][n]=ac.mults_kron(D1,D2,retiring.R,M,4)-SHARED+4
        # Both branches keep private hypothetical secondary histories. The
        # true plant filters the mixed actuator drive using its own history.
        mix=1. if retiring is None else (ramp_samples-ramp_left+1)/ramp_samples
        gamma[n]=mix
        yout=ya if retiring is None else mix*ya+(1-mix)*yr
        yactual[1:]=yactual[:-1];yactual[0]=yout
        ys=float(yactual@ac.S_PATH)
        ys_actual[n]=ys
        # The plant exposes measured e=d-S*y+v. The exact known secondary
        # model reconstructs dv=e+S*y on this SAME sample, after actuation.
        e=float(d[n]-ys+v[n])
        dv=e+ys
        assert abs(dv-float(d[n]+v[n]))<1e-10
        if post_ramp_check>0:
            post_ramp_output_error[n]=ys_a-ys
            post_ramp_check-=1
        branch_update(active,q[active_perm],dv)
        exp_evals[n]+=2
        if candidate is not None:
            branch_update(candidate,q[candidate_perm],dv)
            exp_evals[n]+=2
        if retiring is not None:
            branch_update(retiring,q[retiring_perm],dv)
            exp_evals[n]+=2
        drive[n]=yout;error[n]=e
        Ad=.999*Ad+.001*abs(d[n]);Ae=.999*Ae+.001*abs(e)
        anr[n]=20*np.log10((Ae+1e-12)/(Ad+1e-12))
        if arm!='fixed':
            # 2 EWMA multiplies each plus clipping/ratio bookkeeping.
            robust=min(abs(e),3*max(slow,1e-3)) if slow>0 else abs(e)
            fast=.98*fast+.02*robust
            if fast>slow*(1+DETECT_REL) and n>1000:
                hold+=1
            else:
                hold=0
            if hold<DETECT_HOLD:
                slow=.9995*slow+.0005*robust
            if arm=='gated_growth' and n%8==0:
                q_history.append(q.copy());e_history.append(e)
                if len(q_history)>256:
                    q_history.pop(0);e_history.pop(0)
        # Sparse buffer copy and detector/phase bookkeeping are charged.
        cost['management'][n]=0 if arm=='fixed' else 12+(D if arm=='gated_growth' and n%8==0 else 0)
        if candidate is not None or retiring is not None or post_ramp_output_error[n]==post_ramp_output_error[n]:
            cost['management'][n]+=4 # active private FIR for branch comparison
        if retiring is not None:
            cost['management'][n]+=2 # actuator crossfade multiplications
            ramp_left-=1
            if ramp_left==0:
                active.ybuf[0]=yactual
                post_ramp_check=3
                retiring=None;retiring_perm=None
        if candidate is not None:
            # Hypothetical candidate residual is observable through the exact S.
            # Robust clipped validation loss uses only samples AFTER proposal.
            if phase=='validate':
                cap=max(3*slow,1e-3)
                check_active+=min(abs(dv-ys_a),cap)
                check_candidate+=min(abs(dv-ys_c),cap)
                check_count+=1
                cost['management'][n]+=4
            if n>=phase_end:
                if phase=='train':
                    phase='validate';phase_end=n+VALIDATE
                    check_active=check_candidate=0.;check_count=0
                else:
                    ratio=check_candidate/max(check_active,1e-12)
                    # 0.5 dB physical amplitude margin is 10**(.5/20).
                    accepted=(ratio<=10**(.5/20) if pending=='prune'
                              else ratio<=10**(-.5/20))
                    events.append({'n':n,'type':pending,'from_R':active.R,
                                   'to_R':candidate.R,'accepted':accepted,
                                   'validation_ratio':ratio,
                                   'validation_samples':check_count})
                    if accepted:
                        if ramp_samples>0:
                            retiring=active
                            retiring_perm=active_perm.copy()
                            ramp_left=ramp_samples
                            events[-1]['ramp_start']=n+1
                            events[-1]['ramp_end']=n+ramp_samples
                        active=candidate
                        active_perm=candidate_perm
                        # Controller hypothesis may have differed in last three
                        # outputs. Preserve actual actuator FIR at the switch.
                        active.ybuf[0]=yactual
                        if pending=='prune':next_prune=n+(resort_interval if resort_interval>0 else PROPOSE_INTERVAL)
                        else:next_prune=n+COOLDOWN
                    if pending=='grow':next_growth=n+100000
                    candidate=None;candidate_perm=None;pending=None;phase='idle'
                    last_event=n;cooldown_until=n+COOLDOWN;hold=0
        if arm!='fixed' and candidate is None and retiring is None and n>=cooldown_until:
            alarm=hold>=DETECT_HOLD
            propose=None
            # Grow only after a successful capacity reduction. Otherwise the
            # startup transient would promote R4 to R8 before any prune test.
            if arm=='gated_growth' and alarm and active.R<4 and n>=next_growth:
                propose='grow'
            elif n>=next_prune and (active.R>2 or (resort_interval>0 and active.R==2)):
                propose='prune'
            if propose is not None:
                old=active.R
                if propose=='grow':
                    qh=[row[active_perm] for row in q_history]
                    candidate,rc=grow(active,min(4,old+2),mu,qh,e_history)
                    candidate_perm=active_perm.copy()
                else:
                    # Audit-only ablation: preserve the initial signed-sort
                    # reduction, but change subsequent R2-to-R2 maintenance.
                    maintenance = proposal if old>2 else config.get('maintenance_proposal',proposal)
                    candidate,candidate_perm,rc,sweeps,truncation=prune(active,2,mu,active_perm,maintenance)
                    next_prune=(T if arm=='once' else n+PROPOSE_INTERVAL)
                cost['reorder'][n]+=rc
                cost['management'][n]+=2*old*(D1+D2)
                events.append({'n':n,'type':'proposal_'+propose,
                               'from_R':old,'to_R':candidate.R if candidate is not None else 2,
                               'reorder_mults':rc,
                               'truncation_relative_frobenius':truncation if propose=='prune' else None,
                               'jacobi_sweeps':sweeps if propose=='prune' else None})
                if propose=='prune' and (candidate is None or
                    (not ablate_spectral_veto and truncation>SPECTRAL_THRESHOLD)):
                    events[-1]['to_R']=2
                    events.append({'n':n,'type':'prune','from_R':old,'to_R':2,
                                   'accepted':False,'reason':('jacobi_nonconvergence' if candidate is None else 'spectral_veto'),
                                   'validation_ratio':None,'validation_samples':0})
                    candidate=None;candidate_perm=None
                    cooldown_until=n+COOLDOWN
                else:
                    pending=propose;phase='train';phase_end=n+TRAIN
                hold=0
        rank[n]=executed_rank
        candidate_rank[n]=executed_candidate_rank
        retiring_rank[n]=executed_retiring_rank
        coeffs[n]=(executed_rank+executed_candidate_rank+executed_retiring_rank)*(D1+D2)
        cost['total'][n]=sum(cost[k][n] for k in ('core','candidate','retiring','management','reorder'))+4
    # The last +4 counts the true secondary FIR; core excludes it.
    physical=float(np.max(np.abs(ac.fir(ac.S_PATH,drive[None,:])[0]-(d+v-error))))
    post_ramp_max=float(np.nanmax(np.abs(post_ramp_output_error))) if np.isfinite(post_ramp_output_error).any() else 0.
    assert post_ramp_max<1e-10
    return {'anr':anr,'error':error,'drive':drive,'R':rank,
            'candidate_R':candidate_rank,'retiring_R':retiring_rank,
            'resident_factor_coeffs':coeffs,'ys_active':ys_active,
            'ys_candidate':ys_candidate,'ys_retiring':ys_retiring,
            'ys_actual':ys_actual,'gamma':gamma,'cost':cost,
            'post_ramp_output_error':post_ramp_output_error,
            'post_ramp_max_abs':post_ramp_max,'exp_evals':exp_evals,
            'events':events,'physical_fir_max_abs':physical,
            'seconds':perf_counter()-t0}


def main(case, run, arm, proposal, ablate_spectral_veto=False):
    path = SOURCE / f'case{case}_run{run:02d}.npz'
    with np.load(path) as src:
        x,d,v,Om,ph = [src[k].copy() for k in ('x','d','v','Om','ph')]
        A0,B0 = src['initial_A'].copy(),src['initial_B'].copy()
        meta=json.loads(str(src['meta']))
    protocol=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['phase']=='dev' and case in protocol['cases']
    assert meta['case']==case and meta['run']==run and meta['seed']==protocol['seed']
    sweep_path=ROOT/'experiments_v12/diagnostics/capacity_envelope.sweeps.json'
    sweep=json.loads(sweep_path.read_text(encoding='utf-8'))
    scan=next(item for item in sweep['runs'] if item['case']==case and item['run']==run)
    assert scan['mu']==[.05,.1,.2,.4,.8] and scan['source_sha256']==hash_file(path)
    mu=MU[case]
    mu_index=scan['mu'].index(mu)
    fixed4_segments=[float(row[3][mu_index]) for row in scan['segment_anr']]
    result=simulate(x,d,v,Om,ph,A0,B0,mu,{
        'arm':arm,'proposal':proposal,'ablate_spectral_veto':ablate_spectral_veto,
        'ramp_samples':RAMP_SAMPLES})
    anr=result['anr'];cost=result['cost'];rank=result['R']
    candidate_rank=result['candidate_R'];retiring_rank=result['retiring_R']
    coeffs=result['resident_factor_coeffs'];events=result['events']
    T=len(x);physical=result['physical_fir_max_abs']
    bounds=[[j*T//4,(j+1)*T//4] for j in range(4)]
    seg=[]
    for lo,hi in bounds:
        sl=slice(hi-5000,hi)
        seg.append({'selector_anr_db':float(anr[sl].mean()),
                    'tuned_fixed_R4_anr_db':fixed4_segments[len(seg)],
                    'anr_gap_db':float(anr[sl].mean()-fixed4_segments[len(seg)]),
                    'mean_mults':float(cost['total'][lo:hi].mean()),
                    'cost_ratio_to_R4':float(cost['total'][lo:hi].mean()/(ac.mults_kron(D1,D2,4,M,4)+4)),
                    'mean_R':float(rank[lo:hi].mean())})
    source_hash=hash_file(path)
    summary={'case':case,'run':run,'arm':arm,'proposal':proposal,'phase':'dev','source':str(path.relative_to(ROOT)),
             'fixed_R4_source':str(sweep_path.relative_to(ROOT)),
             'source_sha256':source_hash,'T':T,'mu':mu,'configuration':{
                 'train':TRAIN,'validate':VALIDATE,'first_propose':FIRST_PROPOSE,
                 'propose_interval':PROPOSE_INTERVAL,'cooldown':COOLDOWN,
                 'detect_rel':DETECT_REL,'detect_hold':DETECT_HOLD,
                 'spectral_threshold':None if ablate_spectral_veto else SPECTRAL_THRESHOLD,
                 'ramp_samples':RAMP_SAMPLES},
             'segments':seg,'events':events,
             'mean_mults':float(cost['total'].mean()),
             'mean_cost_ratio_to_R4':float(cost['total'].mean()/(ac.mults_kron(D1,D2,4,M,4)+4)),
             'ledger_mean':{k:float(a.mean()) for k,a in cost.items()},
             'mean_exp_evals':float(result['exp_evals'].mean()),
             'post_ramp_max_abs':result['post_ramp_max_abs'],
             'max_resident_factor_coeffs':int(coeffs.max()),
             'accepted_prune':sum(e['accepted'] for e in events if e['type']=='prune'),
             'accepted_growth':sum(e['accepted'] for e in events if e['type']=='grow'),
             'rejected_proposals':sum(not e['accepted'] for e in events if e['type'] in ('prune','grow')),
             'physical_fir_max_abs':physical,'seconds':result['seconds']}
    HERE.mkdir(parents=True,exist_ok=True)
    stem=f'{case}_run{run:02d}_tuned_{arm}'+('' if proposal=='signed_sort' else f'_{proposal}')
    if ablate_spectral_veto:stem+='_no_spectral_veto'
    np.savez_compressed(HERE/f'{stem}_trace.npz',anr=anr,error=result['error'],drive=result['drive'],
                        R=rank,candidate_R=candidate_rank,retiring_R=retiring_rank,
                        resident_factor_coeffs=coeffs,ys_active=result['ys_active'],
                        ys_candidate=result['ys_candidate'],ys_retiring=result['ys_retiring'],
                        ys_actual=result['ys_actual'],gamma=result['gamma'],
                        post_ramp_output_error=result['post_ramp_output_error'],
                        exp_evals=result['exp_evals'],
                        **cost)
    (HERE/f'{stem}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'case':case,'run':run,'arm':arm,'proposal':proposal,'seconds':summary['seconds'],
                      'segments':seg,'events':len(events),
                      'accepted_prune':summary['accepted_prune'],
                      'accepted_growth':summary['accepted_growth']},ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--case',choices=['E2','E3'],required=True)
    ap.add_argument('--run',type=int,choices=range(5),default=0)
    ap.add_argument('--arm',choices=['once','gated_growth'],default='once')
    ap.add_argument('--proposal',choices=['signed_sort','identity','fixed_random'],default='signed_sort')
    ap.add_argument('--ablate-spectral-veto',action='store_true')
    args=ap.parse_args()
    main(args.case,args.run,args.arm,args.proposal,args.ablate_spectral_veto)
