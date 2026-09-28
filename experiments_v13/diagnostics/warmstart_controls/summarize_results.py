"""Print paired, segment-level summaries from the saved development streams."""
import json

import numpy as np

import warmstart_controls as wc


def load(case):
    return [json.loads((wc.HERE / f"{case}_run{run:02d}.json").read_text(encoding="utf-8"))
            for run in wc.RUNS]


def matrix(results, arm):
    return np.array([[s["arm_ANR_db"][arm] for s in r["segments"]] for r in results])


def main():
    for case in wc.CASES:
        results = load(case)
        base = matrix(results, "continuous_R4")
        print(f"\n{case}: five runs, four segment tails per run")
        print("arm | s1 | s2 | s3 | s4 | mean delta vs R4 | better segments/20 | better runs/5")
        for arm in wc.ARMS:
            values = matrix(results, arm)
            delta = values-base
            cells = [f"{x:.3f}" for x in values.mean(axis=0)]
            print(" | ".join([arm, *cells, f"{delta.mean():+.3f}",
                              str(int(np.sum(delta < -1e-9))),
                              str(int(np.sum(delta.mean(axis=1) < -1e-9)))]))
        for left, right in (("sorted_svd_R2", "sorted_random_R2"),
                            ("sorted_svd_R2", "identity_svd_R2"),
                            ("sorted_svd_R2", "sorted_frozen_R2")):
            diff = matrix(results, left)-matrix(results, right)
            print(f"{left} minus {right}: mean {diff.mean():+.3f} dB, "
                  f"run means {[round(x, 3) for x in diff.mean(axis=1)]}")
        for label in ("sorted_R2", "sorted_R4", "identity_R2"):
            errs = [r["initialization"][label]["relative_frobenius_error"]
                    for r in results]
            print(f"{label} relative Frobenius error: "
                  f"mean {np.mean(errs):.6f}, range {min(errs):.6f}..{max(errs):.6f}")
    print("\nRun00 paired segment ANR (dB):")
    for case in wc.CASES:
        result = load(case)[0]
        for arm in wc.ARMS:
            print(case, arm, " ".join(f"{s['arm_ANR_db'][arm]:.3f}" for s in result["segments"]))


if __name__ == "__main__":
    main()
