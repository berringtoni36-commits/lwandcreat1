"""C1-C4 development-only bridge from frozen resort controller to v10.

Inputs are generated exclusively by protocol.generator.make_case(phase='dev').
No selection/confirmation path or materializer is imported or read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GEN_PATH = ROOT / "experiments_v14/protocol/generator.py"
SEL_PATH = ROOT / "experiments_v14/resort_diagnostic/selector_resort.py"
CORE_PATH = ROOT / "experiments_v10/anc_core.py"
MU_GRID = (.05, .1, .2, .4, .8)
WINDOW = 5000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


gen = import_file(GEN_PATH, "n14_bridge_dev_generator")
selector = import_file(SEL_PATH, "n14_bridge_dev_selector")
ac = import_file(CORE_PATH, "n14_bridge_dev_anc_core")
assert sha(CORE_PATH) == sha(ROOT / "experiments_v12/outputs/main_dev_round1/sources/experiments_v10/anc_core.py")


def v10_baselines(data):
    """Run original v10 controller step on the same RFF and physical plant."""
    T = len(data.x)
    ctrls = {}
    for mu in MU_GRID:
        r4 = ac.KronRFF(f"R4_mu{mu:g}", 1, 25, 20, 4,
                        mu/2, mu/2, 2., 1e-8, np.random.default_rng(0))
        r4.A = data.initial_A[:, :4][None].copy()
        r4.B = data.initial_B[:, :4][None].copy()
        full = ac.FullRFF(f"full500_mu{mu:g}", 1, 500, mu, 2., 1e-8)
        ctrls[r4.name] = r4
        ctrls[full.name] = full
    anr = {name: np.zeros(T, dtype=np.float32) for name in ctrls}
    ae = {name: 0. for name in ctrls}
    drive = {name: np.zeros(T) for name in ("R4_mu0.05", "full500_mu0.05")}
    error = {name: np.zeros(T) for name in drive}
    xb = np.zeros(20)
    zh = np.zeros((4, 500))
    ad = 0.
    scale = np.sqrt(2./500.)
    for n in range(T):
        xb[1:] = xb[:-1]
        xb[0] = data.x[n]
        z = scale*np.cos(data.omega@xb + data.rff_phase)
        zh[1:] = zh[:-1]
        zh[0] = z
        q = zh[2] + .5*zh[3]
        dv = np.array([data.d[n]+data.v[n]])
        ad = .999*ad + .001*abs(data.d[n])
        for name,c in ctrls.items():
            ys = float(c.step(z[None,:],q[None,:],dv)[0])
            e = float(dv[0]-ys)
            ae[name] = .999*ae[name] + .001*abs(e)
            anr[name][n] = 20*np.log10((ae[name]+1e-12)/(ad+1e-12))
            if name in drive:
                drive[name][n] = c.ybuf[0,0]
                error[name][n] = e
    fir_check = {name: float(np.max(np.abs(
        ac.fir(ac.S_PATH, drive[name][None,:])[0]-(data.d+data.v-error[name]))))
        for name in drive}
    return anr,drive,error,fir_check


def run(case: str, index: int, smoke_length: int | None):
    if case not in ("C1","C2","C3","C4") or index not in range(5):
        raise ValueError("Only C1-C4 development run0..4")
    t0 = perf_counter()
    data = gen.make_case(case,index,phase="dev",smoke_length=smoke_length)
    validated = gen.validate_case(data,require_full=smoke_length is None)
    T = len(data.x)
    tag = f"case{case}_run{index:02d}" + ("" if smoke_length is None else f"_smoke{smoke_length}")
    inputs = HERE / "inputs"
    inputs.mkdir(parents=True,exist_ok=True)
    input_file = inputs / f"{tag}.npz"
    np.savez_compressed(input_file,x=data.x,d=data.d,v=data.v,
                        omega=data.omega,rff_phase=data.rff_phase,
                        initial_A=data.initial_A,initial_B=data.initial_B)
    base_anr,base_drive,base_error,base_fir = v10_baselines(data)
    common_args = (data.x,data.d,data.v,data.omega[None],data.rff_phase[None],
                   data.initial_A[None],data.initial_B[None],.05)
    fixed = selector.simulate(*common_args,config={"arm":"fixed"})
    bridge = {
        "drive_max_abs":float(np.max(np.abs(fixed["drive"]-base_drive["R4_mu0.05"]))),
        "physical_error_max_abs":float(np.max(np.abs(fixed["error"]-base_error["R4_mu0.05"]))),
        "anr_max_abs_db":float(np.max(np.abs(fixed["anr"]-base_anr["R4_mu0.05"]))),
        "v10_fir_max_abs":base_fir["R4_mu0.05"],
        "selector_fir_max_abs":fixed["physical_fir_max_abs"],
    }
    assert bridge["drive_max_abs"] < 1e-10
    assert bridge["physical_error_max_abs"] < 1e-10
    assert bridge["anr_max_abs_db"] < 1e-5
    assert base_fir["R4_mu0.05"] < 1e-10
    assert np.all(fixed["cost"]["total"]==19132)
    assert not fixed["events"]
    periodic = selector.simulate(*common_args,config={
        "arm":"once","proposal":"signed_sort","ramp_samples":100,
        "ablate_spectral_veto":False,"resort_interval":50000})
    assert periodic["physical_fir_max_abs"] < 1e-10
    assert periodic["post_ramp_max_abs"] < 1e-10
    if case in ("C1","C2") or T<20001:
        assert not periodic["events"]
    tail = min(WINDOW,T)
    final = {name:float(curve[-tail:].mean()) for name,curve in base_anr.items()}
    periodic_anr=float(periodic["anr"][-tail:].mean())
    result = dict(phase="dev",case=case,run=index,smoke_only=smoke_length is not None,
                  T=T,window=tail,seed_keys=data.seed_keys,
                  source_sha256={"generator":sha(GEN_PATH),"selector":sha(SEL_PATH),
                                 "v10_core":sha(CORE_PATH),"runner":sha(Path(__file__))},
                  input_file=str(input_file.relative_to(ROOT)),
                  input_sha256=sha(input_file),generator_validation=validated,
                  mu_grid=MU_GRID,fixed_tail_anr_db=final,
                  periodic_tail_anr_db=periodic_anr,
                  gap_to_R4_mu005_db=periodic_anr-final["R4_mu0.05"],
                  gap_to_full500_mu005_db=periodic_anr-final["full500_mu0.05"],
                  baseline_cost_mults={"R4":19132,"full500":ac.mults_full(500,20,4)+4},
                  periodic_cost_mean_mults=float(periodic["cost"]["total"].mean()),
                  periodic_cost_ratio_to_R4=float(periodic["cost"]["total"].mean()/19132),
                  periodic_cost_ratio_to_full500=float(periodic["cost"]["total"].mean()/(ac.mults_full(500,20,4)+4)),
                  periodic_ledger_mean={k:float(v.mean()) for k,v in periodic["cost"].items()},
                  fixed_bridge=bridge,full500_fir_max_abs=base_fir["full500_mu0.05"],
                  periodic_physical_fir_max_abs=periodic["physical_fir_max_abs"],
                  periodic_post_ramp_max_abs=periodic["post_ramp_max_abs"],
                  events=periodic["events"],elapsed_seconds=perf_counter()-t0)
    output = HERE / "outputs"
    output.mkdir(parents=True,exist_ok=True)
    (output/f"{tag}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    np.savez_compressed(output/f"{tag}.npz",
                        **{f"anr_{name}":curve for name,curve in base_anr.items()},
                        fixed_drive=fixed["drive"],fixed_error=fixed["error"],
                        full500_drive=base_drive["full500_mu0.05"],
                        full500_error=base_error["full500_mu0.05"],
                        periodic_drive=periodic["drive"],periodic_error=periodic["error"],
                        periodic_anr=periodic["anr"],periodic_R=periodic["R"],
                        periodic_candidate_R=periodic["candidate_R"],
                        periodic_retiring_R=periodic["retiring_R"],
                        **{f"cost_{k}":v for k,v in periodic["cost"].items()})
    print(json.dumps({"case":case,"run":index,"T":T,"bridge":bridge,
                      "periodic_tail":periodic_anr,"events":result["events"],
                      "cost_ratio_R4":result["periodic_cost_ratio_to_R4"],
                      "seconds":result["elapsed_seconds"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--case",choices=("C1","C2","C3","C4"),required=True)
    p.add_argument("--runs",type=int,nargs="+",default=list(range(5)))
    p.add_argument("--smoke-length",type=int)
    a=p.parse_args()
    for i in a.runs:
        run(a.case,i,a.smoke_length)
