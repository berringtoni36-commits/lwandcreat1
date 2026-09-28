"""Print five-run paired physical ANR and rank-geometry tables."""
import json

import numpy as np

import e1_capacity as ec


def read(suffix=""):
    return [json.loads((ec.HERE / f"E1_run{run:02d}{suffix}.json").read_text(encoding="utf-8"))
            for run in ec.RUNS]


def main():
    base, realign, delayed = read(), read("_realign"), read("_delayed")
    print("Five-run mean physical segment-tail ANR (dB):")
    for arm in ec.ARMS:
        print(arm, [round(float(np.mean([r["segments"][j]["arm_ANR_db"][arm]
                                         for r in base])), 3) for j in range(3)])
    print("boundary_realign_200k", [round(float(np.mean([
        r["segments"][j]["boundary_realign_ANR_db"] for r in realign])), 3)
                                    for j in range(3)])
    for arm in ("prune_220k", "prune_250k"):
        print(arm, [round(float(np.mean([r["segments"][j]["arm_ANR_db"][arm]
                                         for r in delayed])), 3) for j in range(3)])
    print("High-stage original-R4 minus realigned-R4 per run:", [round(
        r["segments"][1]["boundary_realign_ANR_db"]-
        r["segments"][1]["original_R4_ANR_db"], 6) for r in realign])
    print("Final low-stage delayed ANR per run:")
    for run,r in enumerate(delayed):
        print(run, {arm:round(r["segments"][2]["arm_ANR_db"][arm], 3)
                    for arm in ("original_R4", "prune_200k", "prune_220k", "prune_250k")})
    for stage in ("low", "high"):
        for rank in (2, 4):
            vals = [r["switch"]["teacher_geometry_sorted"][stage][f"relative_best_R{rank}"]
                    for r in base]
            print(f"initial-sort true-{stage} best-R{rank} tail:",
                  round(float(np.mean(vals)), 4),
                  [round(float(min(vals)), 4), round(float(max(vals)), 4)])
    for at in (200000, 220000, 250000):
        arm = f"prune_{at//1000}k"
        vals = [r["transitions"][arm]["true_low_teacher_geometry_under_current_sort"]["relative_best_R2"]
                for r in delayed]
        projection = [r["transitions"][arm]["R4_to_R2_projection_relative_error"]
                      for r in delayed]
        print(arm, "teacher low R2 tail", round(float(np.mean(vals)), 4),
              "current-weight truncation", round(float(np.mean(projection)), 4))


if __name__ == "__main__":
    main()
