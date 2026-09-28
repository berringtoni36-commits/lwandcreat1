"""Static audit of all saved periodic-resort N14 development results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "experiments_v14/outputs/n14_dev_inputs"
PARENT = ROOT / "experiments_v14/resort_diagnostic"
FIXED = ROOT / "experiments_v14/fixed_frontier"
DEST = Path(__file__).resolve().parent / "resort_saved_audit.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(INPUTS / "manifest.json")
assert manifest["phase"] == "dev" and manifest["runs_per_case"] == 5
entries = {(r["case"], r["run"]): r for r in manifest["entries"]}
assert len(entries) == 10
source_hash = sha(PARENT / "selector_resort.py")
base = read(Path(__file__).resolve().parent / "resort_audit_E2_run00_signed_sort.json")
assert base["segment_cost_ratio"] and len(base["segment_cost_ratio"]) == 4
base_reorders = [e for e in base["events"] if e["type"] == "proposal_prune"]
rows = []
for case in ("E2", "E3"):
    for i in range(5):
        stem = f"case{case}_run{i:02d}_mu0.05_interval50000"
        r = read(PARENT / f"{stem}.json")
        inp = INPUTS / f"case{case}_run{i:02d}.npz"
        fixed = read(FIXED / f"case{case}_run{i:02d}" / "result.json")
        assert r["phase"] == "dev_diagnostic"
        assert r["input_sha256"] == sha(inp) == entries[case,i]["file_sha256"] == fixed["input_sha256"]
        assert r["simulator_sha256"] == source_hash
        assert r["config"] == dict(arm="once", proposal="signed_sort", ramp_samples=100,
                                    ablate_spectral_veto=False, resort_interval=50000)
        assert fixed["method_names"][15] == "R4_mu0.05"
        fixed4 = fixed["segment_anr_db"][15]
        gaps = [a-b for a,b in zip(r["segment_anr_db"],fixed4)]
        events = r["events"]
        p = [e for e in events if e["type"] == "proposal_prune"]
        a = [e for e in events if e["type"] == "prune"]
        assert len(p) == 8 and len(a) == 7 and all(e["accepted"] for e in a)
        assert [e["n"] for e in p] == [20000,74000,128000,182000,236000,290000,344000,398000]
        assert [e["n"] for e in a] == [24000,78000,132000,186000,240000,294000,348000]
        assert r["physical_fir_max_abs"] < 1e-10
        # All runs have identical proposal/accept times and rank timelines.
        # Under the source ledger, only QR/Jacobi reorder counts then vary.
        segment_cost = [base["segment_cost_ratio"][s] +
                        sum(e["reorder_mults"]-b["reorder_mults"]
                            for e,b in zip(p,base_reorders)
                            if s*100000 <= e["n"] < (s+1)*100000) / (100000*19132)
                        for s in range(4)]
        rows.append(dict(case=case,run=i,segment_gap_to_fixed4_db=gaps,
                         max_gap_db=max(gaps),cost_ratio=r["total_cost_ratio"],
                         segment_cost_ratio=segment_cost,
                         max_truncation=max(e["truncation_relative_frobenius"] for e in p),
                         max_validation_ratio=max(e["validation_ratio"] for e in a),
                         physical_fir_max_abs=r["physical_fir_max_abs"]))

out = dict(phase="dev_audit", parent_controller_sha256=source_hash, rows=rows,
           max_segment_gap_db=max(v for r in rows for v in r["segment_gap_to_fixed4_db"]),
           max_segment_cost_ratio=max(v for r in rows for v in r["segment_cost_ratio"]),
           min_cost_ratio=min(r["cost_ratio"] for r in rows),
           max_cost_ratio=max(r["cost_ratio"] for r in rows))
DEST.write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(dict(max_segment_gap_db=out["max_segment_gap_db"],
                      max_segment_cost_ratio=out["max_segment_cost_ratio"],
                      cost_ratio_range=[out["min_cost_ratio"],out["max_cost_ratio"]],
                      source_hash=source_hash)))
