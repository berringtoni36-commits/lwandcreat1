"""Read-only N14 confirmation preflight and explicitly non-unlocking templates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import confirmation_runner as cr


def draft_template():
    hashes = cr.source_hashes()
    return dict(status="draft_only", phase="confirm", version="N14",
                primary_cases=list(cr.PRIMARY), bridge_cases=list(cr.BRIDGE),
                paired_runs=20, k_threshold=17, null_success_probability=.5,
                alpha_round=.05/12,
                decision_rule="single_paired_J;K_ge_17_of_20;one_sided_exact_binomial",
                bridge_mu=cr.BRIDGE_MU, environment=cr.environment_fingerprint(),
                dynamic_config="FILL_FROM_REVIEWED_SELECTION_GRID",
                selection_freeze_sha256="FILL_AFTER_ALL_16_SELECTION_RUNS",
                selection_output_manifest_sha256="FILL_AFTER_SELECTION",
                selection_analysis_sha256="FILL_AFTER_SELECTION",
                selection_result_sha256="FILL_WITH_ALL_16_RESULT_SHA256",
                locked_fixed_mu="FILL_FROM_SELECTION_8x4_MEAN_ANR",
                locked_dynamic_mu="FILL_FROM_UNIQUE_25_PAIR_J_WINNER",
                source_sha256=hashes,
                protocol_sha256=hashes["experiments_v14/protocol/PREREGISTRATION.md"],
                freeze_authorization="NOT_REVIEWED")


def draft_seal_template():
    return dict(status="draft_only", decision="do_not_run_confirm",
                confirmation_freeze_sha256="FILL_WITH_FINAL_CONFIRM_FREEZE_SHA256",
                selection_analysis_sha256="FILL_WITH_LOCKED_SELECTION_ANALYSIS_SHA256",
                reason_after_selection_review="NOT_REVIEWED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--print-draft-template", action="store_true")
    group.add_argument("--print-draft-seal-template", action="store_true")
    group.add_argument("--freeze", type=Path)
    parser.add_argument("--select-freeze", type=Path)
    args = parser.parse_args()
    if args.print_draft_template:
        print(json.dumps(draft_template(), ensure_ascii=False, indent=2))
    elif args.print_draft_seal_template:
        print(json.dumps(draft_seal_template(), ensure_ascii=False, indent=2))
    else:
        frozen = cr.check_confirmation_ready(args.freeze, args.select_freeze)
        print(json.dumps(dict(status="ready_for_one_locked_confirmation",
                              confirmation_freeze_sha256=frozen.digest,
                              selection_analysis_sha256=frozen.selection_analysis_digest,
                              locked_dynamic_mu=frozen.document["locked_dynamic_mu"]),
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
