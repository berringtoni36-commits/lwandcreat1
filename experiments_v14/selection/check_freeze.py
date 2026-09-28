"""Read-only N14 selection freeze template and preflight validator.

Neither mode creates output folders, random samples, or controller results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selection_runner import (CONFIG, MU, SOURCE_RELATIVE,
                              check_freeze, model_config_keys, source_hashes)


def draft_template():
    hashes = source_hashes()
    config = {**CONFIG, **{key: "REVIEW_REQUIRED" for key in
                          model_config_keys() - set(CONFIG)}}
    return dict(status="draft_only", phase="select", version="N14",
                grid_kind="one_signed_sort_periodic_resort_spectral_physical_strategy",
                cases=["E2", "E3"], runs_per_case=8,
                segment_length=100_000, tail_samples=5_000,
                mu_grid=MU.tolist(), dynamic_config=config,
                cost_rule="C_R=12512+1655R;full500=14508;physical_FIR=4",
                selection_rule="fixed_8x4_mean_ANR;dynamic_25_joint_J_ge7of8",
                protocol_sha256=hashes[SOURCE_RELATIVE[0]],
                source_sha256=hashes,
                dev_audit_relative_path="FILL_WITH_REVIEWED_RESORT_AUDIT_PATH",
                freeze_authorization="NOT_REVIEWED",
                evidence_note="Fill after development tests, causal and cost audits; "
                "change status and authorization only after parent review")


def draft_seal_template():
    return dict(status="draft_only", decision="do_not_run_select",
                grid_sha256="FILL_WITH_FINAL_GRID_FILE_SHA256",
                dev_audit_sha256="FILL_WITH_REVIEWED_RESORT_AUDIT_SHA256",
                reason_after_dev_review="NOT_REVIEWED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--print-draft-template", action="store_true")
    group.add_argument("--print-draft-seal-template", action="store_true")
    group.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.print_draft_template:
        print(json.dumps(draft_template(), ensure_ascii=False, indent=2))
    elif args.print_draft_seal_template:
        print(json.dumps(draft_seal_template(), ensure_ascii=False, indent=2))
    else:
        frozen = check_freeze(args.freeze)
        print(json.dumps(dict(status="ready_for_select", freeze_sha256=frozen.digest,
                              source_count=len(frozen.document["source_sha256"])),
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
