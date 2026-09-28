# Periodic signed reordering of a compressed Kronecker controller

**Status: N14 development only.** This report does not use or imply any
selection/confirmation results. The candidate is separate from the original
`online_selector.py` and its earlier one-prune development failures. It is a
causal periodic coordinate update **after one R4→R2 acceptance**, not evidence
of bidirectional rank adaptation. Its claim, if later confirmed, would be
adaptive Kronecker *structure/feature ordering* plus a retained low term
count, not online growth and pruning in response to arbitrary capacity change.

## Mechanism and timing

The first candidate is the previously audited signed-sort R4→R2 proposal at
sample 20000, with 2000 future training samples, 2000 later physical
validation samples, and a 100-sample actuator crossfade if accepted. Thereafter
the manager proposes an **R2→R2 reordering** 50000 samples after the previous
acceptance. It sorts only the coefficients available at that instant, composes
the new feature permutation with the current one, computes a counted
QR/Jacobi rank-2 factorization, and applies the same future physical
training/validation gate and crossfade. It never reads the generator's case or
segment boundaries. All current candidates, including the terminal proposal
whose validation cannot finish within the 400000-sample file, are charged.

On the ten new N14 dev streams, the first prune was accepted at n=24000.
Later equal-R reorders were accepted at n=78000, 132000, 186000, 240000,
294000, and 348000; the n=398000 proposal was still training at file end.
These times are consequences of the fixed interval and prior decisions, not
inputs to the controller. Candidate validation loss and physical branch
histories are in each JSON/trace run of the simulator; the companion
`test_resort.py` changed all future signals after n=110000 and verified that
all preceding outputs, events and costs remain bitwise identical.

## Paired 5-run development result

All methods in this table used the same actual inputs, RFF map and initial
factors per run. The fixed R4 comparison here uses `mu=0.05`, the best mean
R4 setting on these five development streams. Formal baseline and dynamic
step-size selection must use new reserved select streams.

| Case | All-segment passes (Δ≤+0.5 dB) | Mean Δ to fixed R4, 20 segments | Worst Δ | Online cost / R4 | Actual first R4→R2 acceptance |
|---|---:|---:|---:|---:|---:|
| E2 | 5/5 runs, 20/20 segments | −0.813 dB | −0.427 dB | 0.8518 | 5/5 |
| E3 | 5/5 runs, 20/20 segments | −0.994 dB | −0.420 dB | 0.8518 | 5/5 |

Thus the periodic update recovers the persistent third-stage failures in the
one-prune candidate, rather than merely changing a transient after the
first switch. The relative ANR gain to a same-timing, **same-R factor
rebalancing without reordering** control was −1.028 dB (E2) and −1.011 dB
(E3) averaged over 20 segments each; signed reordering improved all 40
matched segments, with its smallest improvement 0.359 dB. The control used
counted QR/Jacobi on the current two-column factors, preserved their exact
coefficient matrix and feature indexing, and underwent the same 2000+2000
future candidate gate and 100-sample crossfade. This isolates an effect of
changing coordinate order from factor balancing or the scheduled intervention
alone. The control implementation and all run outputs are in
`../reparam_control/`.

The full 500-feature unstructured MCC remains a strong counterexample: at
the same development step size it beat this new method by mean 2.175 dB
(E2) and 1.832 dB (E3), while its counted multiplication cost is
14508/sample versus about 16297/sample for the periodic method. No general
compute/performance Pareto claim relative to full500 is justified.

The deliberately harder E1 teacher tests a **true** low-rank/high-rank/low-
rank change. On its new development run 0, the periodic R2 method trailed
fixed R4 by +2.889 dB in the first tail, +24.417 dB in the high-rank stage,
and +23.460 dB after the teacher returned to low rank. The same simulator
reported a 0.8561 cost ratio. This is a severe migration failure, retained in
`E1_run00_boundary.json`, and explicitly prevents a general rank-adaptive or
automatic safety claim. The E2/E3 benefit cannot be generalized to settings
that need new directions absent from the compressed/reordered coordinates.

## Integrity and selection decision

The full ten-run result, paired same-R control, source SHA, input SHA, physical
FIR reconstruction, and cost ratio are retained in `case*.json` files.
[Independent code and full ten-run replay](../dual_rate/REPORT.md) found the
same 40 segment ANRs, proposal/accept events, and total costs; it checked
the source delta from 60F6, no segment-boundary input, the terminal incomplete
candidate, and the severe E1 migration failure. [Hidden MATLAB full-length
replay](../matlab_replay/REPORT_RESORT.md) independently matched the E2/E3
run00 natural events, physical output and every per-sample cost category.
The same-prefix/different-future test also passed. The development integrity
gate is therefore satisfied for this narrow candidate. The sole selected
development configuration is a 50000-sample interval, `mu=0.05` diagnostic
operating point, 2000+2000 candidate windows and 100-sample crossfade; the
formal selection protocol still scans all five predeclared step sizes.

Proceed to the **first independent selection stage** only after the source,
protocol, this report and selection analyzer are SHA-frozen in the explicit
selection grid and seal. Preserve the original ≥7/8 paired selection gate,
≤+0.5 dB/≤0.90 cost gates, and E1 boundary. A 5/5 development result is
encouraging but insufficient for a paper conclusion.
