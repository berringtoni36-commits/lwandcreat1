"""Read-only audit of saved inputs, RNG namespaces, and E1 fixed-step choices.

Writes only to independent_audit/. No new experiment or confirmation trial.
"""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
V12 = HERE.parent
OUTPUTS = V12 / 'outputs'
FOLDERS = ('selection_e1_v500', 'selection_e1_growth',
           'selection_e1_t1000_v500', 'main_dev_round1',
           'main_dev_aggressive_growth', 'main_dev_continue',
           'main_dev_mid_penalty')
CASE_INDEX = {'E1': 0, 'E2': 1, 'E3': 2}
FIELDS = ('x', 'd', 'v', 'Om', 'ph', 'initial_A', 'initial_B')


def run_data(folder: str, case: str, run: int):
    path = OUTPUTS / folder / f'case{case}_run{run:02d}.npz'
    with np.load(path) as f:
        return {key: f[key] for key in FIELDS}


def expected_noise(seed: int, case: str, run: int, length: int):
    rng = np.random.default_rng([seed, CASE_INDEX[case], run, 4])
    if case != 'E3':
        return .01 * rng.standard_normal(length)
    # Independent transcription of the CMS draw order used by anc_core:
    # all uniform draws precede all exponential draws, so length matters.
    U = rng.uniform(-np.pi / 2, np.pi / 2, size=length)
    W = rng.exponential(1., size=length)
    alpha = 1.6
    return .05 * (np.sin(alpha * U) / np.cos(U)**(1. / alpha)
                  * (np.cos((1.-alpha)*U) / W)**((1.-alpha)/alpha))


def main():
    protocols = {name: json.loads((OUTPUTS/name/'protocol.json').read_text(encoding='utf-8'))
                 for name in FOLDERS}
    assert all(p['phase'] in ('dev', 'select') for p in protocols.values())
    trials = {}
    for name, p in protocols.items():
        for case in p['cases']:
            for run in range(p['runs']):
                trials.setdefault((case, run), []).append(name)

    pair_checks = []
    for (case, run), names in sorted(trials.items()):
        for left, right in combinations(names, 2):
            a, b = run_data(left, case, run), run_data(right, case, run)
            match = {}
            for field in FIELDS:
                if field in ('x', 'd', 'v'):
                    length = min(len(a[field]), len(b[field]))
                    match[field] = bool(np.array_equal(a[field][:length], b[field][:length]))
                else:
                    match[field] = bool(np.array_equal(a[field], b[field]))
            pair_checks.append(dict(case=case, run=run, left=left, right=right,
                                    lengths=[len(a['x']), len(b['x'])], match=match))

    noise_checks = []
    for (case, run), names in sorted(trials.items()):
        for name in names:
            p = protocols[name]
            with np.load(OUTPUTS/name/f'case{case}_run{run:02d}.npz') as f:
                actual = f['v']
            expected = expected_noise(p['seed'], case, run, len(actual))
            rng_growth_old = np.random.default_rng([p['seed'], CASE_INDEX[case], run, 4])
            rng_growth_new = np.random.default_rng([p['seed'], CASE_INDEX[case], run, 41])
            old_first = rng_growth_old.standard_normal(25)
            new_first = rng_growth_new.standard_normal(25)
            noise_checks.append(dict(folder=name, case=case, run=run,
                                     noise_replay_max_abs=float(np.max(np.abs(actual-expected))),
                                     gaussian_noise_equals_old_growth_prefix=(
                                         bool(np.array_equal(actual[:25], .01*old_first))
                                         if case != 'E3' else None),
                                     new_growth_prefix_differs=not np.array_equal(old_first,new_first)))

    # Rebuild the saved E1 rank-wise step selection from the eight per-run
    # sweep files, with no import of sweep_fixed.py or aggregate_sweeps.py.
    e1_folder = OUTPUTS/'selection_e1_v500'
    sweeps = [json.loads((e1_folder/f'caseE1_run{run:02d}_fixed_sweep.json')
                         .read_text(encoding='utf-8')) for run in range(8)]
    chosen = json.loads((e1_folder/'fixed_step_selection.json').read_text(encoding='utf-8'))['choice']
    step_checks = []
    for rank in range(1, 9):
        options = []
        for mu in sweeps[0]['mu']:
            key = f'R{rank}_mu{mu}'
            score = float(np.mean([np.mean(np.asarray(s['segment_anr'])[:, s['names'].index(key)])
                                   for s in sweeps]))
            options.append((score, mu))
        score, best_mu = min(options)
        saved = chosen[f'fixed_R{rank}']
        step_checks.append(dict(rank=rank, best_mu=best_mu, pooled_segment_anr=score,
                                saved_mu=saved['mu'], saved_score=saved['selection_anr'],
                                score_abs_diff=abs(score-saved['selection_anr'])))
    fixed_r4_segment_diffs = []
    for run, sweep in enumerate(sweeps):
        j = sweep['names'].index('R4_mu0.2')
        with np.load(e1_folder/f'caseE1_run{run:02d}.npz') as f:
            meta = json.loads(str(f['meta']))
        fixed_r4_segment_diffs.extend(abs(sweep['segment_anr'][seg][j]
                                      -meta['segment_anr'][seg]['fixed_R4'])
                                      for seg in range(3))

    assert all(c['noise_replay_max_abs'] < 1e-12 for c in noise_checks)
    assert all(c['new_growth_prefix_differs'] for c in noise_checks)
    assert all(c['gaussian_noise_equals_old_growth_prefix'] for c in noise_checks
               if c['case'] != 'E3')
    assert all(c['best_mu'] == c['saved_mu'] and c['score_abs_diff'] < 1e-12
               for c in step_checks)
    assert max(fixed_r4_segment_diffs) < 1e-5
    result = dict(pair_checks=pair_checks, noise_checks=noise_checks,
                  e1_step_checks=step_checks,
                  e1_fixed_r4_sweep_vs_saved_max_abs=max(fixed_r4_segment_diffs),
                  e1_saved_runs=24, e1_distinct_case_run_streams=8,
                  interpretation='Audited snapshots use growth part=4. Current source uses part=41.')
    (HERE/'streams_and_baselines.json').write_text(json.dumps(result, indent=2),encoding='utf-8')
    compact = dict(paired_config_comparisons=len(pair_checks),
                   exact_noise_replays=len(noise_checks),
                   same_length_all_input_matches=sum(all(p['match'].values())
                       for p in pair_checks if p['lengths'][0] == p['lengths'][1]),
                   different_length_e3_noise_mismatches=sum(
                       p['case']=='E3' and p['lengths'][0]!=p['lengths'][1]
                       and not p['match']['v'] for p in pair_checks),
                   e1_rank_step_choices_rebuilt=len(step_checks),
                   e1_fixed_r4_sweep_vs_saved_max_abs=max(fixed_r4_segment_diffs))
    (HERE/'streams_summary.json').write_text(json.dumps(compact,indent=2),encoding='utf-8')
    print(json.dumps(compact))


if __name__ == '__main__':
    main()
