"""Export an N14 *development* short numerical replay fixture for MATLAB.

This script pins the actual saved input arrays and selected Python selector
source by SHA. It independently runs the R4 prefix, calls the selector's
current counted prune kernel once, then follows both R4 and R2 shadow factors.
The forced actuator ramp is a numerical unit probe, not a selector decision.
Re-export after the parent freezes the selector; never treat provisional
events or policy outcomes here as N14 results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV = ROOT / "experiments_v14" / "outputs" / "n14_dev_inputs"
SELECTOR = ROOT / "experiments_v13" / "online_selector" / "online_selector.py"
sys.path.insert(0, str(ROOT / "experiments_v12" / ".deps"))
from scipy.io import savemat  # noqa: E402

TAIL_CHECKS = (0, 1, 2, 3, 100, 19999, 20000, 20001, 22000, 22001,
               24000, 24001, 24100, 24101)
D, D1, D2, M = 500, 25, 20, 20
TRAIN, VALIDATE, RAMP = 2000, 2000, 100
S_PATH = np.array([0., 0., 1., .5])


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arr_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_input(case: str, run: int):
    assert case in ("E2", "E3") and 0 <= run <= 4
    manifest_path = DEV / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["phase"] == "dev" and manifest["status"] == "inputs_only"
    assert manifest["cases"] == ["E2", "E3"]
    row = next(e for e in manifest["entries"] if e["case"] == case and e["run"] == run)
    assert row["phase"] == "dev" and not row["smoke_only"]
    path = DEV / f"case{case}_run{run:02d}.npz"
    meta_path = DEV / f"case{case}_run{run:02d}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta == row and sha(path) == row["file_sha256"]
    with np.load(path, allow_pickle=False) as file:
        arrays = {key:file[key].copy() for key in
                  ("x", "d", "v", "omega", "rff_phase", "initial_A", "initial_B")}
        bounds = file["segment_bounds"].copy()
    assert np.array_equal(bounds, np.array(row["segment_bounds"]))
    assert len(arrays["x"]) == len(arrays["d"]) == len(arrays["v"]) == 400000
    assert arrays["omega"].shape == (D, M) and arrays["rff_phase"].shape == (D,)
    assert arrays["initial_A"].shape == (D1, 8)
    assert arrays["initial_B"].shape == (D2, 8)
    for key in row["array_sha256"]:
        assert arr_sha(arrays[key]) == row["array_sha256"][key]
    return arrays, dict(source=str(path.relative_to(ROOT)).replace("\\", "/"),
                        source_sha256=sha(path), source_metadata_sha256=sha(meta_path),
                        manifest_sha256=sha(manifest_path), phase="dev", case=case,
                        run=run, seed_keys=row["seed_keys"])


def load_selector():
    before = sha(SELECTOR)
    spec = importlib.util.spec_from_file_location("n14_selector_fixture_source", SELECTOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert sha(SELECTOR) == before
    return module, before


def drive_before_update(ctrl, z):
    Z = z.reshape(D1, D2).T
    return float(np.sum(ctrl.B[0]*(Z@ctrl.A[0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("E2", "E3"), default="E2")
    parser.add_argument("--run", type=int, choices=range(5), default=0)
    parser.add_argument("--mu", type=float, default=.1)
    parser.add_argument("--proposal-index", type=int, default=20000,
                        help="Zero-based index; candidate begins on following sample")
    parser.add_argument("--length", type=int, default=24200)
    parser.add_argument("--tag", default="provisional")
    parser.add_argument("--frozen-selector-sha", default=None,
                        help="Required with --tag frozen after the parent freezes the selector")
    args = parser.parse_args()
    assert 0 < args.mu < 2 and args.length <= 400000
    assert args.proposal_index + TRAIN + VALIDATE + RAMP + 2 < args.length
    arrays, provenance = load_input(args.case, args.run)
    selector, selector_sha = load_selector()
    if args.tag == "frozen":
        if not args.frozen_selector_sha or selector_sha != args.frozen_selector_sha:
            raise RuntimeError("A matching frozen selector SHA is required")
    assert selector.D == D and selector.D1 == D1 and selector.D2 == D2
    assert selector.TRAIN == TRAIN and selector.VALIDATE == VALIDATE
    assert selector.RAMP_SAMPLES == RAMP
    assert selector.SPECTRAL_THRESHOLD == .15
    assert np.array_equal(selector.ac.S_PATH, S_PATH)
    N = args.length
    x, d, v = (arrays[key][:N] for key in ("x", "d", "v"))
    omega, phase = arrays["omega"], arrays["rff_phase"]
    A0, B0 = (arrays[key][:, :4].copy() for key in ("initial_A", "initial_B"))
    active = selector.make_ctrl(4, args.mu, A0[None, :, :], B0[None, :, :])
    candidate = None
    perm = None
    xb = np.zeros(M)
    zh = np.zeros((len(S_PATH), D))
    actual_ybuf = np.zeros(len(S_PATH))
    active_drive = np.zeros(N)
    active_ys = np.zeros(N)
    candidate_drive = np.full(N, np.nan)
    candidate_ys = np.full(N, np.nan)
    actual_drive = np.zeros(N)
    actual_ys = np.zeros(N)
    actual_error = np.zeros(N)
    actual_anr = np.zeros(N)
    gamma = np.zeros(N)
    z_checks = []
    q_checks = []
    factor_checks = []
    check_indices = []
    ae = ad = 0.
    ramp_begin = args.proposal_index + 1 + TRAIN + VALIDATE
    proposal = {}
    candidate_A0 = candidate_B0 = None
    for n in range(N):
        xb[1:] = xb[:-1].copy()
        xb[0] = x[n]
        z = np.sqrt(2/D)*np.cos(omega@xb+phase)
        zh[1:] = zh[:-1].copy()
        zh[0] = z
        q = zh[2]+.5*zh[3]
        dv = float(d[n]+v[n])
        ya = drive_before_update(active, z)
        ysa = float(active.step(z[None, :], q[None, :], np.array([dv]))[0])
        active_drive[n], active_ys[n] = ya, ysa
        if candidate is not None:
            yc = drive_before_update(candidate, z[perm])
            ysc = float(candidate.step(z[None, perm], q[None, perm],
                                       np.array([dv]))[0])
            candidate_drive[n], candidate_ys[n] = yc, ysc
        if n == args.proposal_index:
            frozen_A, frozen_B = active.A[0].copy(), active.B[0].copy()
            frozen_ybuf = active.ybuf[0].copy()
            candidate, perm, counted_mults, sweeps, truncation = selector.prune(
                active, 2, args.mu, np.arange(D), "signed_sort")
            if candidate is None:
                raise RuntimeError("Bounded Jacobi did not converge at this proposal; choose another dev probe")
            candidate_A0 = candidate.A[0].copy()
            candidate_B0 = candidate.B[0].copy()
            w = (frozen_B@frozen_A.T).T.reshape(D)
            np.testing.assert_array_equal(np.sort(perm), np.arange(D))
            output_invariance = max(abs(z@w-z[perm]@w[perm]),
                                    abs(q@w-q[perm]@w[perm]))
            assert output_invariance < 1e-12
            proj = candidate.B[0]@candidate.A[0].T
            target = w[perm].reshape(D1, D2).T
            recon = float(np.linalg.norm(target-proj)/np.linalg.norm(target))
            assert abs(recon-truncation) < 1e-8
            proposal = dict(index_zero_based=n, candidate_first_index=n+1,
                            permutation_zero_based=perm.tolist(),
                            output_invariance_abs=float(output_invariance),
                            truncation_relative_frobenius=truncation,
                            numpy_reconstruction_relative_frobenius=recon,
                            counted_mults=int(counted_mults), jacobi_sweeps=int(sweeps),
                            spectral_veto_would_reject=bool(truncation > .15))
        mix = (0. if n < ramp_begin else
               min(1., (n-ramp_begin+1)/RAMP))
        gamma[n] = mix
        y = ya if n < ramp_begin else (1-mix)*ya+mix*candidate_drive[n]
        actual_ybuf[1:] = actual_ybuf[:-1].copy()
        actual_ybuf[0] = y
        ys = float(actual_ybuf@S_PATH)
        error = dv-ys
        actual_drive[n], actual_ys[n], actual_error[n] = y, ys, error
        ad=.999*ad+.001*abs(d[n])
        ae=.999*ae+.001*abs(error)
        actual_anr[n]=20*np.log10((ae+1e-12)/(ad+1e-12))
        if n in TAIL_CHECKS and n < N:
            check_indices.append(n+1)
            z_checks.append(z.copy())
            q_checks.append(q.copy())
            factor_checks.append((active.A[0].copy(), active.B[0].copy()))
    assert proposal
    assert max(abs(active_ys[:args.proposal_index+1]-actual_ys[:args.proposal_index+1])) < 1e-12
    assert np.max(np.abs(np.convolve(actual_drive, S_PATH)[:N]-actual_ys)) < 1e-12
    HERE.mkdir(parents=True, exist_ok=True)
    stem = f"{args.case}_run{args.run:02d}_{args.tag}_short"
    fixture_path = HERE / f"{stem}.mat"
    savemat(fixture_path, dict(x=x, d=d, v=v, omega=omega, rff_phase=phase,
        initial_A=A0, initial_B=B0, mu=args.mu, sigma=2., eps=1e-8,
        proposal_index_zero_based=args.proposal_index, train_samples=TRAIN,
        validation_samples=VALIDATE, ramp_samples=RAMP,
        spectral_threshold=.15, S_PATH=S_PATH,
        candidate_permutation_one_based=perm+1,
        R4_A_at_proposal=frozen_A, R4_B_at_proposal=frozen_B,
        R4_ybuf_at_proposal=frozen_ybuf,
        candidate_A_at_proposal=candidate_A0,
        candidate_B_at_proposal=candidate_B0,
        active_drive_expected=active_drive, active_ys_expected=active_ys,
        candidate_drive_expected=candidate_drive,
        candidate_ys_expected=candidate_ys,
        actual_drive_expected=actual_drive, actual_ys_expected=actual_ys,
        actual_error_expected=actual_error, actual_anr_expected=actual_anr,
        gamma_forced_expected=gamma,
        check_indices_one_based=np.asarray(check_indices),
        z_checks_expected=np.asarray(z_checks), q_checks_expected=np.asarray(q_checks),
        R4_A_checks_expected=np.asarray([a for a,_ in factor_checks]),
        R4_B_checks_expected=np.asarray([b for _,b in factor_checks]),
        prune_mults_expected=proposal["counted_mults"],
        jacobi_sweeps_expected=proposal["jacobi_sweeps"],
        truncation_expected=proposal["truncation_relative_frobenius"]),
        do_compression=True)
    metadata = dict(provenance=provenance, selector_source=str(SELECTOR.relative_to(ROOT)).replace("\\", "/"),
                    selector_sha256=selector_sha, frozen_selector=args.tag == "frozen",
                    fixture=fixture_path.name, fixture_sha256=sha(fixture_path),
                    mu=args.mu, short_length=N, proposal=proposal,
                    forced_ramp_begin_zero_based=ramp_begin,
                    note="Forced ramp is a numerical probe, not an N14 acceptance event")
    (HERE / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(dict(fixture=str(fixture_path), selector_sha256=selector_sha,
                          counted_mults=proposal["counted_mults"],
                          truncation=proposal["truncation_relative_frobenius"],
                          spectral_veto_would_reject=proposal["spectral_veto_would_reject"])))


if __name__ == "__main__":
    main()
