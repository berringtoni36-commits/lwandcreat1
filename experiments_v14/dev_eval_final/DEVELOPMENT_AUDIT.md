# N14 new-stream development audit (not selection or confirmation)

The input arrays are the ten new E2/E3 `dev` streams in
`../outputs/n14_dev_inputs/`. Every result below is paired with the same-run
fixed R4 controller at `mu=0.05` from `../fixed_frontier/`. All four 5000-sample
segment tail ANRs must differ by at most +0.5 dB, and the online cost ratio
must be at most 0.90. The final baseline `mu` remains to be chosen on a new
selection set, so this is an exploratory development check only.

| Case | Dynamic `mu` | Five-run all-segment passes | Mean of 20 segment differences (dB) | Worst segment difference (dB) | Cost / fixed R4 | Accepted prunes |
|---|---:|---:|---:|---:|---:|---:|
| E2 | 0.05 | 2/5 | +0.235 | +0.774 | 0.8399 | 5/5 |
| E2 | 0.10 | 2/5 | +0.154 | +0.877 | 0.8399 | 5/5 |
| E2 | 0.20 | 0/5 | +0.286 | +0.740 | 0.8399 | 5/5 |
| E3 | 0.05 | 4/5 | +0.027 | +0.582 | 0.8399 | 5/5 |
| E3 | 0.10 | 4/5 | +0.122 | +0.720 | 0.8399 | 5/5 |

E2 run 0 is `+0.774` dB in the highest-nonlinearity third stage at dynamic
`mu=0.05`; increasing to `mu=0.10` improves that difference to `+0.505` dB,
still narrowly outside the gate. E2 run 1 reverses the pattern: `mu=0.05`
passes all stages, whereas `mu=0.10` has a `+0.877` dB worst difference. A
single larger global step size is therefore not a reliable fix. The old v12
development inputs had all 40 stages pass at their then-chosen settings, but
that result does not transfer to these new streams.

The `mu=0.05` runs live under `../dev_eval_frozen/`; `mu=0.10/0.20` live here.
The former controller SHA differs only from the later 60F6 source by
non-numerical comments/docstrings. Raw JSON, traces, cost ledgers, and input
SHA are retained for every run. The `.npz` filename suffix bug in the first
wrapper was corrected before these runs; existing files were renamed, not
silently overwritten. No selection or confirmation random streams were read.

**Decision:** current one-step-size N14 candidate fails its development
quality gate and must not be frozen or sent to selection. The next comparison
is a preselected, separately named development change that allows distinct
R4 warmup and R2 continuation rates while preserving the same causal
candidate/physical validation and exact cost ledger. Any failure will remain
recorded; the quality threshold is unchanged.
