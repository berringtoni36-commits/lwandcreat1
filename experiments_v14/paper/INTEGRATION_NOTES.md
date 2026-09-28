# N14 methods fragment: integration notes

`methods_draft.tex` is an English LaTeX **methods-only** fragment. It uses the notation, packages, equations (1)–(12), and Proposition 1 of `稿件/当前阶段稿/overleaf_stage_update/main_stage_draft.tex`; it supplies no selection or confirmation performance result. It is based on the frozen `experiments_v14/selection/freeze_candidate_grid.json`, `experiments_v14/protocol/PREREGISTRATION.md`, the frozen `experiments_v14/resort_diagnostic/selector_resort.py`, and `experiments_v14/prior_art/REPORT.md`. This directory is independent of the frozen source and original manuscript.

## Where to insert

Insert `\input{.../methods_draft.tex}` **after** the existing `\subsection{Adaptive state and arithmetic}` and its table/explanatory paragraph, immediately **before** `\section{Simulation results}`. The fragment starts its own `\section` and has no preamble. If copying the text into Overleaf, omit the two opening comment lines. The original paper's `\tag{1}`–`\tag{16}` numbering remains intact; the new equations use labels and automatic numbering. Existing `amsmath`, `amssymb`, and `bm` suffice.

The old manuscript's arithmetic table counts the controller kernel **without** the physical output FIR. Its fixed-R4 value is 19,128 and full500 value is 14,504. N14's complete online ledger adds 4 physical-FIR multiplications, giving R4 = 19,132, full500 = 14,508, and R2 = 15,822. Keep these scopes explicit when editing the old table/caption. The original statement about Proposition 1 concerns each fixed-layout sequential factor update; do not apply it to SVD truncation, candidate acceptance, or actuator crossfade.

For a coherent later revision, update the title, abstract, highlights, Introduction contribution list, experiment setup, results, and conclusion **only after** the frozen selection/confirmation analyses are authorized and verified. In particular, the current paper's C1–C4 fixed-structure results cannot be reused as evidence for N14 switching, nor should E2/E3 development traces be substituted for independent results. The present fragment makes no Pareto, all-baseline, real-time, automatic-growth, or first-in-literature claim. It retains all 500 RFFs; only adaptive factor count changes once. The subsequent R2→R2 events change the coordinate layout.

## Citation keys to add to the existing manual bibliography

The manuscript already defines `ref11` for Ye et al. (2025). The fragment's four new keys can be added as `\bibitem`s in its current `thebibliography` environment; check house-style metadata when assembling the submission. These are primary-paper/DOI links, also reviewed in [`../prior_art/REPORT.md`](../prior_art/REPORT.md).

```latex
\bibitem{n14_nkp_fxnsaf_mcc} J. Ye, H. Zhao,
Nearest Kronecker Product Decomposition Based Subband Adaptive Filter:
Algorithms and Applications, IEEE/ACM Trans. Audio Speech Lang. Process. (2026),
doi:10.1109/TASLPRO.2025.3649394.
\bibitem{n14_spiriti} S. Spiriti, D. Morici, L. Piroddi,
A gradient-free adaptation method for nonlinear active noise control,
J. Sound Vib. 333 (2014) 13--30, doi:10.1016/j.jsv.2013.09.006.
\bibitem{n14_einsort} T. Koike-Akino, J. Liu, Y. Wang,
EinSort: Sorting is All We Need for Tensorizing LLM,
arXiv:2606.08565 (2026), doi:10.48550/arXiv.2606.08565.
\bibitem{n14_neukron} T. Kwon, J. Ko, J. Jung, K. Shin,
NeuKron: Constant-Size Lossy Compression
of Sparse Reorderable Matrices and Tensors, Proc. WWW (2023),
doi:10.1145/3543507.3583226.
```

Source links: [Ye/Zhao](https://doi.org/10.1109/TASLPRO.2025.3649394), [Spiriti et al.](https://doi.org/10.1016/j.jsv.2013.09.006), [EinSort](https://doi.org/10.48550/arXiv.2606.08565), [NeuKron](https://doi.org/10.1145/3543507.3583226). The author lists for Ye/Zhao, EinSort, and NeuKron were checked against their primary arXiv records; check final journal volume and page metadata before submission. The mechanism comparisons in the fragment rely on the primary texts linked in the prior-art report, not on an assertion of priority.

## Technical alignment checks for merger

- Proposal is made after the R4 update on zero-based sample 20,000. A live candidate is trained on the next 2,000 samples and validated on the following 2,000; acceptance starts a 100-sample actuator mixture. Subsequent accepted proposals are scheduled 50,000 samples after the preceding acceptance and are R2→R2.
- The sorting key is the current **signed** full coefficient vector in the active branch's coordinates, with stable ascending sort. Both `z` and filtered `q` are permuted; the full-vector identity holds only before rank-two SVD truncation.
- Branch virtual residuals use `e_actual + S*y_actual` and each branch's private filtered output. These are model-based counterfactual residuals, not two microphone measurements. The candidate gate accumulates clipped absolute virtual residuals; the MCC influence applies to online B-then-A factor updates.
- The known and exact secondary-path model is a simulation assumption. A later robustness study of path-model mismatch would be separate work.
- The multiplication ledger includes live shadow and retiring branches, vetoed/rejected proposals, an unfinished final candidate, QR/Jacobi decomposition, physical FIR, and transition management. Comparison sorting, permutation traffic/storage, transcendental operations, and wall-clock time have not been converted to scalar multiplications.
