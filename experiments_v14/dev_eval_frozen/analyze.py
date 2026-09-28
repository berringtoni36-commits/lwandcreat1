"""Compare NEW N14 development controller results with paired fixed R4.

All values here are exploratory; fixed μ is finally selected on new select
streams, not these five development runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FRONT = ROOT / "experiments_v14" / "fixed_frontier"


def analyze(case: str, mu: float, fixed_mu: float):
    rows = []
    for run in range(5):
        dyn_path = HERE / f"case{case}_run{run:02d}_mu{mu:g}_summary.json"
        if not dyn_path.exists():
            continue
        dyn = json.loads(dyn_path.read_text(encoding="utf-8"))
        fixed = json.loads((FRONT / f"case{case}_run{run:02d}" / "result.json").read_text(encoding="utf-8"))
        name = f"R4_mu{fixed_mu:.2f}"
        baseline = fixed["segment_anr_db"][fixed["method_names"].index(name)]
        got = [x["anr_db"] for x in dyn["segments"]]
        delta = np.asarray(got)-np.asarray(baseline)
        rows.append(dict(run=run, delta_db=delta.tolist(),
                         cost_ratio=dyn["mean_cost_ratio_to_R4"],
                         accepted_prune=dyn["accepted_prune"],
                         quality_pass=bool(np.max(delta) <= .5),
                         full_pass=bool(np.max(delta) <= .5 and
                                        dyn["mean_cost_ratio_to_R4"] <= .9 and
                                        dyn["accepted_prune"] == 1)))
    report = dict(case=case, dynamic_mu=mu, fixed_R4_mu=fixed_mu,
                  runs=rows,
                  complete=len(rows)==5,
                  pass_count=sum(r["full_pass"] for r in rows),
                  max_segment_delta_db=max((max(r["delta_db"]) for r in rows), default=None))
    print(case, "dyn μ", mu, "fixed μ", fixed_mu,
          "pass", report["pass_count"], "/", len(rows),
          "max Δ", round(report["max_segment_delta_db"], 3) if rows else None)
    for row in rows:
        print(" ", row["run"], [round(x, 3) for x in row["delta_db"]],
              "cost", round(row["cost_ratio"], 4), "pass", row["full_pass"])
    return report


if __name__ == "__main__":
    analyze("E2", .05, .05)
    analyze("E3", .05, .05)
