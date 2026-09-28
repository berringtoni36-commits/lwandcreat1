"""Generate deterministic short dynamic-controller replay cases (development only)."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_core import AdaptiveKronV12, V12Config, ac
sys.path.insert(0, str(ROOT / '.deps'))
from scipy.io import savemat


def run_case(teacher_gain: float, mu: float, lam: float, seed: int = 42, T: int = 120):
    d1, d2, D, M = 3, 2, 6, 4
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    Om = rng.standard_normal((1, D, M)) * .5
    ph = rng.uniform(0, 2*np.pi, (1, D))
    A0 = np.array([[[1.0], [.2], [-.3]]])
    B0 = np.array([[[.8], [-.3]]])
    growth_seed = 314159
    a = np.random.default_rng(growth_seed).standard_normal((1, d1, 1))
    a *= np.sqrt(np.sum(A0*A0) / (np.sum(a*a) + 1e-8))
    xb = np.zeros((1, M)); zh = np.zeros((4, 1, D))
    Z = np.zeros((T, D)); Q = np.zeros((T, D))
    for n in range(T):
        xb[:, 1:] = xb[:, :-1].copy(); xb[0, 0] = x[n]
        z = np.sqrt(2/D)*np.cos(np.einsum('rdm,rm->rd',Om,xb)+ph)
        zh[1:] = zh[:-1].copy(); zh[0] = z
        q = zh[2] + .5*zh[3]
        Z[n] = z[0]; Q[n] = q[0]
    teacher_B2 = np.array([[.25], [1.]]) * teacher_gain
    teacher_W = B0[0] @ A0[0].T + teacher_B2 @ a[0].T
    d = np.array([np.sum(teacher_W * Q[n].reshape(d1,d2).T) for n in range(T)])
    v = np.zeros(T)
    ctrl = ac.KronRFF('base',1,d1,d2,1,mu/2,mu/2,2.0,1e-8,np.random.default_rng(1))
    ctrl.A = A0.copy(); ctrl.B = B0.copy()
    cfg = V12Config(r_min=1,r_max=2,warmup=10,train_samples=30,
        validation_samples=40,ramp_samples=5,cooldown=200,screen_every=20,
        lambda_cost=lam,margin=0.,prune_mode='column')
    dyn = AdaptiveKronV12(ctrl,np.random.default_rng(growth_seed),cfg,M=M)
    traces = []
    candidate_a = None
    phase_code = {'idle': 0, 'train': 1, 'validate': 2, 'ramp': 3}
    for n in range(T):
        ys = dyn.step(Z[n:n+1],Q[n:n+1],np.array([d[n]+v[n]]))[0]
        if dyn.candidate is not None and candidate_a is None:
            candidate_a = dyn.candidate.A[0, :, -1].copy()
        traces.append((dyn.last['R'],dyn.last['candidate_R'],dyn.last['y'],ys,
                       dyn.last['gamma'],phase_code[dyn.last['phase']],
                       float(dyn.active.A.sum()),float(dyn.active.B.sum()),
                       float(dyn.candidate.A.sum()) if dyn.candidate is not None else np.nan,
                       float(dyn.candidate.B.sum()) if dyn.candidate is not None else np.nan))
    return dict(x=x,Om=Om[0],ph=ph[0],A0=A0[0],B0=B0[0],growth_a=a[0,:,0],d=d,v=v,
                growth_a_at_proposal=candidate_a,Z=Z,Q=Q,traces=np.asarray(traces),
                events=dyn.events,config=cfg,
                A_final=dyn.active.A[0],B_final=dyn.active.B[0],teacher_gain=teacher_gain,mu=mu,lam=lam)


if __name__ == '__main__':
    outdir = Path(__file__).resolve().parent
    for name, gain in [('growth_accept', 1.), ('growth_reject', 0.)]:
        r=run_case(gain,.8,.001)
        decisions=[e for e in r['events'] if e['event'] in ('accept','reject')]
        assert len(decisions)==1 and decisions[0]['event']==name.split('_')[1]
        event=decisions[0]
        proposal=next(e for e in r['events'] if e['event']=='proposal')
        validation_start=next(e for e in r['events'] if e['event']=='validation_start')
        switches=[e for e in r['events'] if e['event']=='switch_complete']
        matpath=outdir / f'{name}.mat'
        savemat(matpath, dict(x=r['x'],d=r['d'],v=r['v'],Om=r['Om'],ph=r['ph'],
            A0=r['A0'],B0=r['B0'],growth_a_at_proposal=r['growth_a_at_proposal'],
            z_expected=r['Z'],q_expected=r['Q'],trace_expected=r['traces'],
            A_final=r['A_final'],B_final=r['B_final'],
            mu=r['mu'],lambda_cost=r['lam'],sigma=2.,eps=1e-8,
            warmup=r['config'].warmup,train_samples=r['config'].train_samples,
            validation_samples=r['config'].validation_samples,
            ramp_samples=r['config'].ramp_samples,cooldown=r['config'].cooldown,
            screen_every=r['config'].screen_every,margin=r['config'].margin,
            proposal_sample=proposal['sample'],
            validation_start_sample=validation_start['sample'],
            switch_sample=switches[0]['sample'] if switches else -1,
            decision_sample=event['sample'],decision_accept=int(event['event']=='accept'),
            old_loss=event['old_loss'],new_loss=event['new_loss'],
            old_score=event['old_score'],new_score=event['new_score']),do_compression=True)
        print(matpath, event)
