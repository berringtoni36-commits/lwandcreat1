> Markdown editing copy converted from the current Overleaf main.tex on 2026-09-26. Use this file for prose polishing. Keep equations, numeric values, citation keys, labels, table data, figure assets, and author-decision items unchanged unless explicitly requested. After polishing, merge approved text back into the Overleaf LaTeX source and compile it; this Markdown file is not the submission source.

<!-- REVISION 2026-09-26 (Claude): structural revision for Signal Processing. See 修改说明_期刊版.md for the change list and the AUTHOR CHECK items. New citation keys are descriptive (e.g., kuo1999); existing keys ref1–ref17 are unchanged. -->

# Kronecker-structured adaptation of random Fourier features for robust nonlinear active noise control

**Authors:** [Author names]

**Corresponding author:** [Name and email].

**Affiliation:** [Department, institution, city, postal code, country]

## Highlights

- Kronecker-structured coefficients for random Fourier feature ANC controllers
- Normalized correntropy factor updates driven by secondary-path-filtered features
- Factor recursion acts as a preconditioned update with step bound μa + μb < 2
- Four Kronecker terms cut adaptive coefficients from 500 to 180
- Faster initial attenuation under chaotic, impulsive, and Gaussian references

## Abstract

Random Fourier features (RFFs) provide nonlinear active noise control (ANC) with a fixed nonlinear map of predetermined dimension, but accurate kernel approximation requires many features, so that a long output-coefficient vector must be adapted online. This paper represents the RFF coefficient vector as a sum of Kronecker products of short factor pairs and adapts the factors with normalized maximum-correntropy updates driven by secondary-path-filtered features. The factor recursion is shown to be equivalent, to first order, to a full-vector filtered-x update with a data-dependent positive semidefinite preconditioner, which yields the step-size condition $\mu_a + \mu_b < 2$ for a contracting a posteriori residual. Storage and multiplication counts and a bound on each factor increment are also derived. With 500 random features, four Kronecker terms require 180 adaptive coefficients instead of 500. In simulations with a benchmark nonlinear primary path, the structured controller reaches $-20$ dB roughly 5,000 iterations earlier than the full correntropy controller under logistic-chaotic excitation and roughly 2,900 iterations earlier under Gaussian excitation with measurement noise. Under $\alpha$-stable impulsive excitation, it enters the attenuating regime earlier, whereas a non-robust RFF controller loses attenuation. The late-stage levels of the two correntropy controllers remain within about 1 dB of each other.

**Keywords:** nonlinear active noise control · random Fourier features · Kronecker product decomposition · maximum correntropy criterion · filtered-x normalized least mean square

# Introduction

Active noise control (ANC) attenuates unwanted acoustic noise by superposing an anti-noise signal radiated by a secondary source \cite{kuo1999,lu2021part1}. Linear filtered-x least-mean-square (FxLMS) controllers perform well when the primary and secondary paths are linear, but saturating actuators, nonlinear primary paths, and chaotic or impulsive noise sources create nonlinear relationships between the reference signal and the disturbance that a linear controller cannot represent \cite{george2013,ref1}. Nonlinear controllers based on Volterra series \cite{tan2001}, functional-link expansions \cite{das2004}, spline adaptive filters \cite{scarpiniti2013}, and kernel methods \cite{liu2008klms} have therefore been developed; a recent survey is given in \cite{ref1}. Once a representation has been chosen, the parameterization of its adaptive coefficients and the update rule determine how quickly the controller reaches an attenuating operating point.

Kernel adaptive filters offer universal nonlinear approximation with a convex adaptation problem in a reproducing kernel Hilbert space \cite{liu2008klms}, but their dictionaries grow with every new sample unless sparsification is applied. Random Fourier features (RFFs) remove this growth by approximating a shift-invariant kernel with an explicit finite-dimensional map \cite{ref2}, so that a kernel filter becomes a linear filter in a fixed feature space. RFF-based adaptive filters have been developed for robust \cite{wang2018rffmcc}, distributed \cite{bouboulis2018}, and multikernel \cite{ref3} learning, and RFF controllers have been introduced to nonlinear ANC. Deb et al. \cite{ref4} formulated RFF-based FxLMS controllers for multichannel narrowband control, Xiao et al. \cite{ref10} combined RFFs with a generalized hyperbolic tangent criterion and conjugate-gradient optimization, and Ye et al. \cite{ref11} fused several RFF maps through a projection matrix under generalized correntropy. In all of these controllers, the $D$ output coefficients of the feature map are adapted as a single unstructured vector. Because an accurate kernel approximation requires a large $D$, the controller must adapt a long coefficient vector, and the convergence time of normalized gradient updates grows with the number of adaptive coefficients \cite{haykin2014}.

Non-Gaussian and impulsive disturbances pose a second difficulty. The maximum correntropy criterion (MCC) weights each error by a Gaussian kernel of its magnitude, so that excursions much larger than the kernel width have a vanishing influence on the update \cite{ref5,chen2016gmcc}. Correntropy-based criteria have been applied to ANC in \cite{kurian2017,zhu2020gmcc}, and recent work in this field includes a variable kernel width \cite{ref6}, a Euclidean-direction-search MCC update \cite{ref7}, a projection FxLMS framework for impulsive environments \cite{ref8}, and a fractional-order generalized complex correntropy algorithm \cite{ref9}. These methods modify the cost function or the optimization strategy, while the coefficient parameterization of the controller remains unchanged.

Kronecker product decomposition (KPD) shortens the adaptive parameter vector without shortening the model. Paleologu et al. \cite{paleologu2018} showed that a long impulse response can be approximated by a sum of a few Kronecker products of short factors, which can be identified with lower-dimensional adaptive filters. The approach has been extended to robust adaptive filters \cite{ref15}, acoustic feedback cancellation \cite{bhattacharjee2021afc}, and multichannel filtered-x ANC \cite{ref16,ref17}, and related low-rank structures have been used in nonlinear spline \cite{ref13} and Volterra \cite{ref14} controllers. Structural modifications have also been used to accelerate nonlinear ANC, for instance the delayless multi-sampled subband functional-link network of \cite{ref12}. In these works, the decomposition acts on the taps of a linear filter or of a polynomial expansion, where the reshaped coefficient matrix inherits the temporal structure of an impulse response. The output coefficients of an RFF map have no such ordering, and, to the best of our knowledge, the effect of a Kronecker structure on the transient behavior of a robust RFF controller has not been examined. Here the factorization is used as an adaptation structure rather than as a physical model of the coefficients: Section \ref{sec:analysis} shows that it acts as a data-dependent preconditioner on the full coefficient update, and Section \ref{sec:sim} examines how much late-stage attenuation is retained with a small number of Kronecker terms.

This paper proposes a Kronecker-structured RFF controller with normalized correntropy adaptation. The main contributions are as follows:

1. The RFF output-coefficient vector is represented by $R$ Kronecker products of short factors, and normalized correntropy recursions for the factors are derived with explicit projected regressors computed from secondary-path-filtered features (Section \ref{sec:method}).
2. The factor recursion is shown to be equivalent, to first order, to a filtered-x update of the full coefficient vector with a positive semidefinite, factor-dependent preconditioner. This yields the step-size condition $\mu_a + \mu_b < 2$ for a contracting a posteriori residual, together with a bound on each factor increment and closed-form storage and multiplication counts (Section \ref{sec:analysis}).
3. Simulations with a benchmark nonlinear primary path under logistic-chaotic, $\alpha$-stable impulsive, and Gaussian references show that the structured controller enters the attenuating regime earlier than the full correntropy controller in all three conditions, with 64% fewer adaptive coefficients and late-stage levels within about 1 dB (Section \ref{sec:sim}).

The remainder of this paper is organized as follows. Section \ref{sec:model} introduces the ANC model, the RFF map, and the correntropy baseline. Section \ref{sec:method} derives the structured recursion, and Section \ref{sec:analysis} analyzes its storage, arithmetic cost, update bound, and step-size range. Section \ref{sec:sim} presents the simulation results, and Section \ref{sec:concl} concludes the paper.

# Nonlinear ANC model and random Fourier features
<!-- LaTeX label: sec:model -->

*Notation.* Scalars are denoted by italic letters, column vectors by bold lowercase letters, and matrices by bold uppercase letters. The superscript $T$ denotes transposition, $\otimes$ the Kronecker product, $\mathrm{vec}(\cdot)$ columnwise vectorization, $\mathrm{tr}(\cdot)$ the trace, $\|\cdot\|_2$ the Euclidean norm, and $\mathbf{I}_M$ the $M \times M$ identity matrix.

## Signal model

Let the reference vector be

$$
\mathbf{x}_n = [x(n), x(n-1), \ldots, x(n-M+1)]^T \in \mathbb{R}^M .
$$

Let $d(n)$ denote the primary disturbance at the error microphone and $y(n)$ the controller output. The secondary-path output and the error signal are

$$
y_s(n) = \sum_{\ell=0}^{L_s-1} s_\ell\, y(n-\ell), \qquad e(n) = d(n) - y_s(n) + v(n),
$$
<!-- LaTeX label: eq:err -->

where $s_\ell$ are the secondary-path coefficients and $v(n)$ denotes measurement noise, when present. The primary and secondary paths are modeled as finite impulse response (FIR) filters:

$$
P(z) = z^{-3} - 0.3z^{-4} + 0.2z^{-5}, \qquad S(z) = z^{-2} + 0.5z^{-3}.
$$
<!-- LaTeX label: eq:paths -->

Including leading zeros, $P(z)$ has six coefficients and $S(z)$ has $L_s=4$. With $g(n)$ denoting the output of $P(z)$ driven by $x(n)$, the nonlinear primary disturbance is

$$
d(n) = g(n-2) + 0.08\,g^2(n-2) - 0.04\,g^3(n-1).
$$
<!-- LaTeX label: eq:nl -->

This model combines temporal memory with second- and third-order nonlinear terms; the paths in \eqref{eq:paths} and the nonlinearity in \eqref{eq:nl} follow a nonlinear ANC benchmark widely used in the literature \cite{tan2001,das2004}. The frequency responses of the linear paths are shown in Fig. \ref{fig:paths}; $S(z)$ is minimum phase. The simulations use an exact secondary-path model, so no secondary-path modeling error is introduced.

> **Figure 1. Frequency responses of the primary and secondary paths: (a) magnitude and (b) phase. The frequency axis is normalized by $\pi$ rad/sample.**

<!-- image: fig1.pdf; LaTeX label: fig:paths -->

## Fixed nonlinear feature map

For a Gaussian kernel with length scale $\ell_r$, the frequency vectors $\boldsymbol{\omega}_j \sim \mathcal{N}(\mathbf{0}, \ell_r^{-2}\mathbf{I}_M)$ and phases $b_j \sim \mathcal{U}[0, 2\pi)$, $j = 1, \ldots, D$, are drawn independently. The RFF map is

$$
\mathbf{z}_n = \sqrt{\frac{2}{D}}
\begin{bmatrix}
\cos(\boldsymbol{\omega}_1^T \mathbf{x}_n + b_1) \\ \vdots \\ \cos(\boldsymbol{\omega}_D^T \mathbf{x}_n + b_D)
\end{bmatrix}.
$$
<!-- LaTeX label: eq:rff -->

In the reference configuration, $\ell_r = 3.9$. The frequencies and phases remain fixed during adaptation, and the expected inner product of two feature vectors equals the Gaussian kernel \cite{ref2}. A full RFF controller computes

$$
y(n) = \mathbf{w}_n^T \mathbf{z}_n, \qquad \mathbf{w}_n \in \mathbb{R}^D .
$$

The secondary-path-filtered feature vector is

$$
\mathbf{q}_n = \sum_{\ell=0}^{L_s-1} s_\ell\, \mathbf{z}_{n-\ell},
$$
<!-- LaTeX label: eq:q -->

that is, each feature sequence is filtered in time by the secondary path. Under the usual assumption that the coefficients vary slowly (the frozen-coefficient approximation), the secondary-path output is approximated by $\mathbf{w}_n^T \mathbf{q}_n$. This regression form is shared by the full and factorized controllers.

## Normalized correntropy baseline

For a correntropy width $\sigma_c > 0$, the scaled instantaneous loss and its derivative (the influence function) are defined as

$$
\rho_{\sigma_c}(e) = \sigma_c^2 \left[ 1 - \exp\!\left( -\frac{e^2}{2\sigma_c^2} \right) \right], \qquad
\psi_{\sigma_c}(e) = e \exp\!\left( -\frac{e^2}{2\sigma_c^2} \right).
$$
<!-- LaTeX label: eq:mcc -->

The factor $\sigma_c^2$ in $\rho_{\sigma_c}$ is chosen so that $\psi_{\sigma_c}(e) \approx e$ for $|e| \ll \sigma_c$, which keeps the step-size convention consistent with FxNLMS; for $|e| \gg \sigma_c$, $\psi_{\sigma_c}(e)$ tends to zero. The normalized correntropy update of the full coefficient vector is

$$
\mathbf{w}_{n+1} = \mathbf{w}_n + \mu \frac{\psi_{\sigma_c}(e_n)}{\varepsilon + \|\mathbf{q}_n\|_2^2}\, \mathbf{q}_n ,
$$
<!-- LaTeX label: eq:fullupd -->

where $\mu > 0$ is the step size, $\varepsilon > 0$ is a small regularization constant, and $e_n$ is the adaptation residual specified in Section 3.2. This baseline combines RFFs, filtered-x normalized least-mean-square (FxNLMS) adaptation, and MCC, and is therefore denoted RFF-FxNLMS-MCC, or RFF-MCC for short. Replacing $\psi_{\sigma_c}(e)$ by $e$ yields the RFF-FxNLMS comparator, abbreviated RFF-NLMS.

# Kronecker-structured correntropy adaptation
<!-- LaTeX label: sec:method -->

## Coefficient factorization

Let $D = D_1 D_2$. The coefficient vector is represented as

$$
\mathbf{w}_n = \sum_{r=1}^{R} \mathbf{a}_{r,n} \otimes \mathbf{b}_{r,n}, \qquad \mathbf{a}_{r,n} \in \mathbb{R}^{D_1}, \quad \mathbf{b}_{r,n} \in \mathbb{R}^{D_2},
$$
<!-- LaTeX label: eq:kron -->

where $R$ is the number of Kronecker terms. Equivalently, the $D_2 \times D_1$ matrix obtained by reshaping $\mathbf{w}_n$ has rank at most $R$. Define $\mathbf{A}_n = [\mathbf{a}_{1,n}, \ldots, \mathbf{a}_{R,n}]$ and $\mathbf{B}_n = [\mathbf{b}_{1,n}, \ldots, \mathbf{b}_{R,n}]$, and let $\mathbf{Q}_n \in \mathbb{R}^{D_2 \times D_1}$ denote the columnwise reshaping of $\mathbf{q}_n$, i.e., $\mathbf{q}_n = \mathrm{vec}(\mathbf{Q}_n)$. Then

$$
\mathbf{w}_n = \mathrm{vec}(\mathbf{B}_n \mathbf{A}_n^T), \qquad
\widehat{d}_n = \sum_{r=1}^{R} \mathbf{b}_{r,n}^T \mathbf{Q}_n \mathbf{a}_{r,n}.
$$
<!-- LaTeX label: eq:vec -->

The output before secondary-path filtering is obtained in the same way by replacing $\mathbf{Q}_n$ with the reshaped unfiltered feature vector. The factorization therefore changes only the adaptive coefficient model and leaves the random features unchanged. The number of adaptive coefficients is reduced from $D_1 D_2$ to $R(D_1 + D_2)$, while every feature still contributes to the controller output.

## Projected regressors and factor updates

Stacking the factors as $\mathbf{a}_n = \mathrm{vec}(\mathbf{A}_n)$ and $\mathbf{b}_n = \mathrm{vec}(\mathbf{B}_n)$, the projected regressors are defined as

$$
\mathbf{u}_{b,n} = \mathrm{vec}(\mathbf{Q}_n \mathbf{A}_n) \in \mathbb{R}^{D_2 R}, \qquad
\mathbf{u}_{a,n} = \mathrm{vec}(\mathbf{Q}_n^T \mathbf{B}_n) \in \mathbb{R}^{D_1 R}.
$$
<!-- LaTeX label: eq:reg -->

Both regressors yield the same predicted disturbance:

$$
\widehat{d}_n = \mathbf{b}_n^T \mathbf{u}_{b,n} = \mathbf{a}_n^T \mathbf{u}_{a,n}.
$$

The normalized factor updates are

$$
\begin{aligned}
\mathbf{b}_{n+1} &= \mathbf{b}_n + \mu_b \frac{\psi_{\sigma_c}(e_n)}{\varepsilon + \|\mathbf{u}_{b,n}\|_2^2}\, \mathbf{u}_{b,n}, \\
\mathbf{a}_{n+1} &= \mathbf{a}_n + \mu_a \frac{\psi_{\sigma_c}(e_n)}{\varepsilon + \|\mathbf{u}_{a,n}\|_2^2}\, \mathbf{u}_{a,n}.
\end{aligned}
$$
<!-- LaTeX label: eq:facupd -->

Both regressors in \eqref{eq:reg} are computed from the factors available at the beginning of iteration $n$ and are used in both updates; hence, the two factor updates can be executed sequentially without recomputing either regressor. Each denominator normalizes the complete stacked regressor over all $R$ terms. Under the frozen-coefficient approximation of Section 2.2, \eqref{eq:facupd} is a normalized correntropy-gradient update with respect to each factor. Equal step sizes $\mu_a = \mu_b$ are used in the experiments; separate symbols are retained to keep the roles of the two factors explicit.

The resulting controller is denoted NKP-RFF-FxNLMS-MCC, or NKP-RFF-MCC for short, where NKP refers to the nearest Kronecker product representation underlying \eqref{eq:kron}. The factors are updated directly during adaptation, so no singular value decomposition is required at each sample, and the number of Kronecker terms $R$ becomes a parameter of the adaptive model itself.

The adaptation residual used in the correntropy influence function is evaluated in the frozen-coefficient form

$$
e_n = d(n) - \mathbf{b}_n^T \mathbf{u}_{b,n} + v(n),
$$
<!-- LaTeX label: eq:res -->

where $v(n)$ is omitted in the noise-free conditions; the same residual is used for both factor updates at iteration $n$. For RFF-MCC, the corresponding residual is $d(n) - \mathbf{w}_n^T \mathbf{q}_n + v(n)$.
<!-- AUTHOR CHECK: confirm that the RFF-MCC and RFF-NLMS simulations used this frozen-coefficient residual (and not the physical error e(n)). If the baseline used e(n), this sentence must be changed and the difference discussed. -->
Although $d(n)$ is not measured directly, $d(n) + v(n) = e(n) + y_s(n)$ can be reconstructed from the measured error and the controller output filtered by the secondary-path model, as in the modified filtered-x structure \cite{bjarnason1992}. With the exact secondary-path model used in the simulations, this reconstruction is exact, and \eqref{eq:res} is implementable from measured signals.

## Initialization and implementation

The full RFF coefficient vector can be initialized to zero. The factorized controller, however, requires nonzero initial factors: if both $\mathbf{A}$ and $\mathbf{B}$ were zero, both regressors in \eqref{eq:reg} would vanish and the factors would never be updated. The factors are therefore initialized with small independent Gaussian values (0.01 times standard normal draws). The seeds of the random map and of the factor initialization are part of the experimental configuration. The complete procedure is summarized in Algorithm 1.

**Algorithm 1.** Normalized Kronecker-structured RFF correntropy adaptation

1. *Initialization:* choose $D_1$, $D_2$, $R$, $\sigma_c$, $\mu_a$, $\mu_b$, and $\varepsilon$; draw and fix the random frequencies and phases; initialize $\mathbf{A}_0$ and $\mathbf{B}_0$ with small nonzero values.
2. For $n = 0, 1, 2, \ldots$, form $\mathbf{x}_n$ and compute $\mathbf{z}_n$ using \eqref{eq:rff}.
3. Filter the feature sequences through the secondary-path model to obtain $\mathbf{q}_n$, and reshape it into $\mathbf{Q}_n$.
4. Compute the projected regressors $\mathbf{u}_{b,n}$ and $\mathbf{u}_{a,n}$ from the current factors using \eqref{eq:reg}.
5. Compute the residual $e_n$ in \eqref{eq:res}, with $d(n) + v(n)$ reconstructed from the measured error, and the influence $\psi_{\sigma_c}(e_n)$ in \eqref{eq:mcc}.
6. Update $\mathbf{b}_n$ and $\mathbf{a}_n$ using \eqref{eq:facupd} with the regressors from Step 4, and reshape them into $\mathbf{B}_{n+1}$ and $\mathbf{A}_{n+1}$.
7. Shift the feature and path buffers and proceed to the next iteration.

The correntropy weight and the normalized factor directions play complementary roles: the correntropy weight regulates the effect of the residual magnitude, whereas the Kronecker factors determine the directions along which the feature-space coefficient vector is modified. Section \ref{sec:analysis} makes the second role precise.

# Structural and computational analysis
<!-- LaTeX label: sec:analysis -->

## Adaptive coefficient storage

The unrestricted controller stores $D$ adaptive coefficients, whereas the Kronecker representation stores $R(D_1 + D_2)$. The relative reduction is

$$
\eta_{\mathrm{store}} = 1 - \frac{R(D_1 + D_2)}{D_1 D_2}.
$$

Representative values are listed in Table \ref{tab:storage}.

**Adaptive coefficient counts for representative factorizations.**

<!-- LaTeX label: tab:storage -->
| $D$ | $D_1 \times D_2$ | $R$ | Full coefficients | Factor coefficients | Reduction |
| --- | --- | --- | --- | --- | --- |
| 200 | $20 \times 10$ | 4 | 200 | 120 | 40% |
| 200 | $20 \times 10$ | 6 | 200 | 180 | 10% |
| 500 | $25 \times 20$ | 4 | 500 | 180 | 64% |
| 500 | $25 \times 20$ | 6 | 500 | 270 | 46% |

Storage is reduced whenever $R < D_1 D_2 / (D_1 + D_2)$. For a fixed product $D_1 D_2$, the sum $D_1 + D_2$ is minimized by nearly balanced factor dimensions. The RFF frequency matrix and phase vector are fixed parameters shared by both controllers and are not included in these counts; they can be regenerated from a stored random seed.

Table \ref{tab:storage} shows how $R$ controls the adaptive model size: four pairs of factors of lengths 25 and 20 describe a 500-dimensional coefficient vector with 180 adaptive values. The experiments use $R = 4$, which provides a 64% storage reduction at a moderate increase in per-iteration arithmetic (Fig. \ref{fig:mults}); $R = 6$ is an alternative with a 46% reduction.

## Arithmetic cost

The arithmetic cost is measured by the number of real multiplications per iteration in the filtered-feature learning loop. Random-feature initialization, signal generation, plotting, and metric evaluation are excluded; cosine, exponential, and division operations are counted separately, and common scalar constants are precomputed. The same convention is applied to both controllers.

Both controllers require $D$ cosine evaluations. A literal two-factor implementation uses two correntropy evaluations and two normalized scalar divisions, whereas the baseline uses one of each. Because both factor updates use the same residual \eqref{eq:res} and kernel width, the exponential can in fact be reused; Table \ref{tab:mults} conservatively retains the literal two-evaluation convention. The reconstruction of $d(n) + v(n)$ adds the same $L_s$ multiplications to both controllers and is not listed.

For $D = 500$, $M = 20$, $L_s = 4$, $D_1 = 25$, $D_2 = 20$, and $R = 4$, the two controllers require 14,004 and 17,048 multiplications per iteration, respectively. The factorized controller thus trades additional projected-regressor computations for a smaller set of adaptive parameters, so that coefficient storage, transient behavior, and arithmetic cost should be considered together. As shown in Fig. \ref{fig:mults}, increasing $R$ enlarges both the factor model and its per-iteration arithmetic cost.

**Multiplications per iteration in the filtered-feature learning loop.**

<!-- LaTeX label: tab:mults -->
| Operation | RFF-MCC | NKP-RFF-MCC |
| --- | --- | --- |
| Random-feature projection and scaling | $DM + D$ | $DM + D$ |
| Secondary-path filtering of features | $DL_s$ | $DL_s$ |
| Two factor-regressor contractions | $0$ | $2RD$ |
| Prediction dot products | $D$ | $R(D_1 + D_2)$ |
| Regressor squared norms | $D$ | $R(D_1 + D_2)$ |
| Weight-vector increments | $D$ | $R(D_1 + D_2)$ |
| Scalar correntropy/update operations | $4$ | $8$ |
| Total | $DM + DL_s + 4D + 4$ | $DM + DL_s + D + 2RD + 3R(D_1 + D_2) + 8$ |

> **Figure 2. Analytical multiplication counts per iteration of the filtered-feature learning loop for $M = 20$ and $L_s = 4$: (a) $D = 200$ ($20 \times 10$) and (b) $D = 500$ ($25 \times 20$). Solid lines: NKP-RFF-MCC; dashed lines: RFF-MCC. All counts follow Table \ref{tab:mults} and include the random-feature mapping.**

<!-- image: fig2.pdf; LaTeX label: fig:mults -->

## Bound on normalized factor increments

The correntropy influence function satisfies

$$
\sup_{e} |\psi_{\sigma_c}(e)| = \sigma_c \exp(-1/2),
$$
<!-- LaTeX label: eq:sup -->

with the supremum attained at $|e| = \sigma_c$. For any regressor $\mathbf{u}$ and $\varepsilon > 0$, the inequality $\varepsilon + \|\mathbf{u}\|_2^2 \geq 2\sqrt{\varepsilon}\,\|\mathbf{u}\|_2$ gives

$$
\frac{\|\mathbf{u}\|_2}{\varepsilon + \|\mathbf{u}\|_2^2} \leq \frac{1}{2\sqrt{\varepsilon}}.
$$
<!-- LaTeX label: eq:normbound -->

Combining \eqref{eq:facupd}, \eqref{eq:sup}, and \eqref{eq:normbound} yields

$$
\|\Delta \mathbf{a}_n\|_2 \leq \frac{\mu_a \sigma_c \exp(-1/2)}{2\sqrt{\varepsilon}}, \qquad
\|\Delta \mathbf{b}_n\|_2 \leq \frac{\mu_b \sigma_c \exp(-1/2)}{2\sqrt{\varepsilon}}.
$$

This result bounds each factor increment for a fixed kernel width without any distributional assumption on the instantaneous error, and it makes explicit the roles of correntropy weighting and normalization in limiting the update magnitude.

## Equivalent preconditioned update and step-size range
<!-- LaTeX label: sec:precond -->

The factor recursion can be related directly to the full update \eqref{eq:fullupd}. Let $c_{a,n} = \mu_a / (\varepsilon + \|\mathbf{u}_{a,n}\|_2^2)$ and $c_{b,n} = \mu_b / (\varepsilon + \|\mathbf{u}_{b,n}\|_2^2)$. In matrix form, \eqref{eq:facupd} reads $\Delta\mathbf{B}_n = c_{b,n}\psi_{\sigma_c}(e_n)\,\mathbf{Q}_n\mathbf{A}_n$ and $\Delta\mathbf{A}_n = c_{a,n}\psi_{\sigma_c}(e_n)\,\mathbf{Q}_n^T\mathbf{B}_n$. Substituting these increments into $\mathbf{w}_{n+1} = \mathrm{vec}\big((\mathbf{B}_n + \Delta\mathbf{B}_n)(\mathbf{A}_n + \Delta\mathbf{A}_n)^T\big)$ and using $\mathrm{vec}(\mathbf{X}\mathbf{Y}\mathbf{Z}) = (\mathbf{Z}^T \otimes \mathbf{X})\,\mathrm{vec}(\mathbf{Y})$ gives

$$
\mathbf{w}_{n+1} = \mathbf{w}_n + \psi_{\sigma_c}(e_n)\,\mathbf{P}_n \mathbf{q}_n + \mathrm{vec}(\Delta\mathbf{B}_n \Delta\mathbf{A}_n^T),
$$
<!-- LaTeX label: eq:precond -->

with

$$
\mathbf{P}_n = c_{b,n}\,(\mathbf{A}_n \mathbf{A}_n^T \otimes \mathbf{I}_{D_2}) + c_{a,n}\,(\mathbf{I}_{D_1} \otimes \mathbf{B}_n \mathbf{B}_n^T).
$$
<!-- LaTeX label: eq:Pmat -->

The matrix $\mathbf{P}_n$ is symmetric and positive semidefinite. To first order in the step sizes, the factor recursion is therefore a filtered-x correntropy update of the full coefficient vector in which the scalar normalization $\mu / (\varepsilon + \|\mathbf{q}_n\|_2^2)$ of \eqref{eq:fullupd} is replaced by the factor-dependent preconditioner $\mathbf{P}_n$. The structured controller thus uses the same features and the same residual as the full controller, but it distributes each correction along directions determined by its current factors; this is the mechanism through which the coefficient structure changes the transient behavior.

The preconditioned form also yields a step-size range. Let $g_n = \exp(-e_n^2 / (2\sigma_c^2)) \in (0, 1]$, so that $\psi_{\sigma_c}(e_n) = g_n e_n$, and let $\rho_{a,n} = \|\mathbf{u}_{a,n}\|_2^2 / (\varepsilon + \|\mathbf{u}_{a,n}\|_2^2)$ and $\rho_{b,n} = \|\mathbf{u}_{b,n}\|_2^2 / (\varepsilon + \|\mathbf{u}_{b,n}\|_2^2)$, both in $[0, 1)$. Since $\mathbf{q}_n^T \mathbf{P}_n \mathbf{q}_n = c_{b,n}\|\mathbf{u}_{b,n}\|_2^2 + c_{a,n}\|\mathbf{u}_{a,n}\|_2^2$, the residual re-evaluated with the updated factors and the same data is

$$
e_n^{+} = \big[1 - g_n(\mu_a \rho_{a,n} + \mu_b \rho_{b,n})\big]\, e_n - \mathrm{tr}(\Delta\mathbf{B}_n^T \mathbf{Q}_n \Delta\mathbf{A}_n).
$$
<!-- LaTeX label: eq:post -->

For the full update \eqref{eq:fullupd}, the corresponding factor is $1 - g_n \mu \rho_n$ with $\rho_n = \|\mathbf{q}_n\|_2^2 / (\varepsilon + \|\mathbf{q}_n\|_2^2)$. Neglecting the last term in \eqref{eq:post}, which is proportional to $\mu_a \mu_b \psi_{\sigma_c}^2(e_n)$ and is negligible for small step sizes, the a posteriori residual satisfies $|e_n^{+}| < |e_n|$ for nonzero regressors whenever

$$
0 < \mu_a + \mu_b < 2.
$$
<!-- LaTeX label: eq:stepcond -->

Hence, the sum $\mu_a + \mu_b$ plays the role of the step size $\mu$ in \eqref{eq:fullupd}, for which $0 < \mu < 2$ is the familiar normalized LMS range \cite{haykin2014}. In the experiments, $\mu_a + \mu_b = 0.16$ for the chaotic and impulsive references and $0.24$ for the Gaussian reference, well within \eqref{eq:stepcond}. Because $g_n \leq 1$, the correntropy weight can only reduce the effective step and therefore does not affect this range.

The number of Kronecker terms $R$ and the correntropy width $\sigma_c$ govern different aspects of the adaptation: $R$ determines the representational structure, the coefficient count, and the rank of the preconditioner, whereas $\sigma_c$ determines the residual scale emphasized by the update. Their effects are therefore examined separately: the kernel-parameter sweep in Section \ref{sec:sweep} addresses error weighting, whereas $R$ is selected from the storage–arithmetic trade-off in Section \ref{sec:analysis} (Fig. \ref{fig:mults}).

# Simulation results
<!-- LaTeX label: sec:sim -->

## Simulation setup and performance measures

The controllers are evaluated on the benchmark of Section \ref{sec:model} with three reference signals: a logistic-chaotic sequence, an $\alpha$-stable impulsive sequence generated by the Chambers–Mallows–Stuck method \cite{chambers1976}, and a Gaussian sequence with measurement noise added to the error signal. RFF-MCC and NKP-RFF-MCC are compared in all three conditions, and RFF-NLMS is additionally included in the impulsive and Gaussian comparisons to isolate the effect of correntropy weighting. The chaotic and impulsive simulations run for 20,000 iterations and the Gaussian simulation for 30,000 iterations. All parameters are listed in Table \ref{tab:config}.

The average noise reduction (ANR) is computed from exponentially smoothed absolute amplitudes:

$$
\begin{aligned}
A_d(n) &= \beta A_d(n-1) + (1-\beta)|d(n)|, \\
A_e(n) &= \beta A_e(n-1) + (1-\beta)|e(n)|, \\
\mathrm{ANR}(n) &= 20 \log_{10} \frac{A_e(n) + \epsilon_a}{A_d(n) + \epsilon_a}.
\end{aligned}
$$

Negative values indicate attenuation. The reference configuration uses $\beta = 0.999$, 20 Monte Carlo runs, and random seed 871, and all ANR curves are averages over the runs. The initial learning period and the late-stage residual level are evaluated separately. To quantify the transient, the first threshold-crossing index is defined as

$$
t_\tau = \min\{ n : \mathrm{ANR}(n) \leq \tau \}.
$$
<!-- LaTeX label: eq:cross -->

The numerical values reported in this section are rounded readings of the averaged ANR curves. The coefficient counts refer to $D = 500$ with a $25 \times 20$ factorization and $R = 4$.

**Simulation parameters and evaluation intervals.**

<!-- LaTeX label: tab:config -->
| Quantity | Description and value |
| --- | --- |
| $M$ | Reference-vector length; 20 |
| $D$ | Number of fixed random features; 500 |
| $D_1$, $D_2$ | Factor dimensions; 25 and 20 ($D = D_1 D_2$) |
| $R$ | Number of Kronecker terms; 4 ($R = 6$ as an alternative model size) |
| $\ell_r$ | RFF kernel length scale; 3.9 |
| $\sigma_c$ | Correntropy width in the comparisons; 2 |
| Step sizes | Logistic-chaotic and $\alpha$-stable: $\mu = 0.05$ (baseline, \eqref{eq:fullupd}), $\mu_a = \mu_b = 0.08$ (factors, \eqref{eq:facupd}); Gaussian: $\mu = 0.30$, $\mu_a = \mu_b = 0.12$ |
| Factor initialization | Independent $0.01 \times$ standard normal values |
| Logistic initialization | First six states set to 0.37; centered and normalized to unit sample standard deviation |
| $\alpha$-stable generator | Symmetric CMS generator with $\alpha = 1.6$, skewness 0, scale 1, location 0; median-amplitude normalization and clipping to $\pm 5$ |
| Gaussian reference | Standard normal; measurement noise added to the error signal |
| Run protocol | 20 Monte Carlo runs; seed 871 |
| $P(z)$, $S(z)$ | Primary and secondary paths in \eqref{eq:paths} |
| Primary nonlinearity | Delayed polynomial in \eqref{eq:nl} |
| $\beta$ | Smoothing factor of the absolute amplitudes; 0.999 |
| $\varepsilon$ | Regularization in \eqref{eq:fullupd} and \eqref{eq:facupd}; $10^{-8}$ |
| $\epsilon_a$ | ANR regularization; $10^{-12}$ |
| Chaotic and impulsive runs | 20,000 iterations |
| Gaussian runs | 30,000 iterations |
| Transient thresholds | $-20$ dB (chaotic and Gaussian); $-2$ dB (impulsive) |

## Sensitivity to correntropy weighting
<!-- LaTeX label: sec:sweep -->

The correntropy width determines which errors contribute strongly to the update. To examine this effect separately from the coefficient structure, the width of RFF-MCC is swept under logistic-chaotic excitation using

$$
g_k(e) = \exp\!\left( -\frac{k e^2}{2\sigma_0^2} \right), \qquad \sigma_{c,\mathrm{eff}} = \frac{\sigma_0}{\sqrt{k}}.
$$
<!-- LaTeX label: eq:k -->

With $\sigma_0 = 2.0$, $k = 1$ corresponds to the reference width $\sigma_c = 2$, and increasing $k$ narrows the effective kernel; $k$ is unrelated to the number of Kronecker terms $R$. The sweep uses a fixed step size $\mu = $ [AUTHOR: value used in the sweep], so its absolute levels are not directly comparable with those in Figs. \ref{fig:chaos}–\ref{fig:gauss}. Fig. \ref{fig:ksweep} shows the learning curves, and Fig. \ref{fig:kss} summarizes the steady-state levels over the final evaluation window.
<!-- AUTHOR CHECK: the sweep script in the code folder (FIG5_sweepK_NKP_RFFxMCC_Logistic.m) uses mu_mcc = 0.8, gamma_rff = 1.0 and a final window of 5,000 samples. Confirm which settings produced Figs. 3-4 before filling in the value. -->

> **Figure 3. ANR learning curves of RFF-MCC for different correntropy parameters $k$ under logistic-chaotic excitation, with $k \in \{1, 1.5, 2, 2.8, 4, 5.6, 6, 8, 12, 15, 20\}$. Larger $k$ corresponds to a narrower effective kernel in \eqref{eq:k}.**

<!-- image: fig3.pdf; LaTeX label: fig:ksweep -->

> **Figure 4. Steady-state ANR over the final evaluation window versus the correntropy parameter $k$. The ANR decreases from approximately $-15$ dB to approximately $-22$ dB over the swept range, with diminishing changes at larger $k$.**

<!-- image: fig4.pdf; LaTeX label: fig:kss -->

The steady-state ANR decreases (i.e., the attenuation deepens) as $k$ increases. Across the sweep, it changes by approximately 7 dB, and the incremental improvement diminishes toward larger $k$. A narrower effective kernel reduces the contribution of large residual excursions while retaining the updates driven by small errors, so the error-weighting scale is an important tuning parameter. The comparisons in Sections 5.3–5.5 nevertheless use the same width $\sigma_c = 2$ ($k = 1$) for both correntropy controllers. This common, untuned width ensures that the observed differences are attributable to the coefficient parameterization rather than to kernel tuning; tuning the width is complementary to the structure studied here.

## Logistic-chaotic reference

The chaotic reference provides a nonlinear, non-Gaussian excitation for comparing the initial learning behavior of the two correntropy controllers. It is generated by the delayed logistic recursion $u(n) = 4u(n-6)[1 - u(n-6)]$, followed by centering and amplitude normalization. Fig. \ref{fig:chaos} compares the ANR over 20,000 iterations.

> **Figure 5. ANR of RFF-MCC and NKP-RFF-MCC under logistic-chaotic excitation. The dotted line marks the $-20$ dB threshold used in Table \ref{tab:cross}. The structured controller reaches this threshold earlier and maintains a lower ANR throughout the displayed interval.**

<!-- image: fig5.pdf; LaTeX label: fig:chaos -->

The two controllers separate clearly during the initial learning period. After about 2,000 iterations, NKP-RFF-MCC reaches approximately $-20$ dB, whereas RFF-MCC reaches approximately $-13$ dB. This 7 dB difference corresponds to an earlier entry of the structured controller into the attenuation region: its ANR first reaches $-20$ dB at around 1,600–1,700 iterations, whereas the baseline approaches this level only at around 6,500–7,000 iterations, i.e., roughly 5,000 iterations later.

As adaptation continues, the two curves gradually converge. At 20,000 iterations, the baseline and the structured controller reach approximately $-24$ dB and $-25$ dB, respectively. The rank-4 coefficient model therefore does not degrade the late-stage attenuation in this condition, and the principal benefit is faster access to a low-residual operating region. In this comparison, the effective step of the structured controller, $\mu_a + \mu_b = 0.16$ by \eqref{eq:stepcond}, exceeds the baseline step $\mu = 0.05$, so the transient gain reflects both the preconditioning of Section \ref{sec:precond} and a larger effective step. The Gaussian comparison in Section 5.5 reverses this ordering of the effective steps.

## Impulsive-noise condition

The $\alpha$-stable comparison examines adaptation in the presence of large noise excursions. Fig. \ref{fig:impulse} includes RFF-NLMS, which uses no correntropy weighting, together with the two MCC-based controllers, so that the effect of robust error weighting can be separated from the transient behavior associated with the coefficient structure.

> **Figure 6. ANR under $\alpha$-stable impulsive excitation. The dotted line marks the $-2$ dB threshold used in Table \ref{tab:cross}. RFF-NLMS rises above the displayed range ($+2$ dB). Both correntropy-based controllers maintain a negative ANR over the later interval, and NKP-RFF-MCC reaches $-2$ dB earlier.**

<!-- image: fig6.pdf; LaTeX label: fig:impulse -->

RFF-NLMS initially reduces the residual, but its ANR then rises above 0 dB and leaves the displayed range at $+2$ dB. In contrast, both correntropy-based controllers maintain attenuation throughout the remainder of the experiment. This difference is consistent with the error weighting in \eqref{eq:mcc}: residuals much larger than the kernel width have a vanishing influence on the update.

The structured controller first crosses $-2$ dB within approximately 600–800 iterations, compared with 2,600–2,900 iterations for RFF-MCC; it therefore establishes attenuation earlier under impulsive excitation as well. At 20,000 iterations, RFF-MCC and NKP-RFF-MCC reach approximately $-4.9$ dB and $-4.7$ dB, respectively, i.e., similar late-stage levels after different initial transients. The distinction supported by this experiment is therefore the faster onset of attenuation, combined with the sustained attenuation provided by correntropy weighting.

## Gaussian reference with measurement noise

The Gaussian comparison evaluates the controllers under a conventional reference distribution with measurement noise in the error signal. As shown in Fig. \ref{fig:gauss}, all three controllers converge to a stable residual level. The two unstructured RFF controllers follow closely spaced trajectories, whereas the structured controller descends more rapidly during the initial learning period.

> **Figure 7. ANR under Gaussian excitation with measurement noise in the error signal over 30,000 iterations. The dotted line marks the $-20$ dB threshold used in Table \ref{tab:cross}. NKP-RFF-MCC reaches this threshold earlier, and the three controllers converge to closely spaced late-stage levels.**

<!-- image: fig7.pdf; LaTeX label: fig:gauss -->

NKP-RFF-MCC crosses $-20$ dB within approximately 5,700–6,000 iterations, whereas RFF-MCC reaches this level within 8,500–8,900 iterations and RFF-NLMS at around 8,000 iterations. The structured controller thus crosses $-20$ dB about 2,500–3,200 iterations earlier than RFF-MCC, with a central estimate of approximately 2,900 iterations. In this comparison, its effective step $\mu_a + \mu_b = 0.24$ is smaller than the baseline step $\mu = 0.30$, so the earlier crossing cannot be attributed to a larger step and is consistent with the preconditioning mechanism of Section \ref{sec:precond}. In the absence of impulsive excursions, correntropy weighting slightly slows the unstructured controller, yet the structured controller still achieves the earliest crossing.

At 30,000 iterations, the two full controllers reach approximately $-27.8$ dB and the structured controller approximately $-28.4$ dB, a late-stage difference of about 0.6 dB. The main result is again the earlier attenuation, here accompanied by a slightly lower final residual. This comparison also shows that the transient benefit persists when impulsive excursions do not dominate the reference signal.

## Comparison across conditions

Table \ref{tab:cross} summarizes the transient readings of the three experiments. The thresholds differ across conditions because the attainable attenuation levels differ. The ranges are read from the averaged curves using the first-crossing criterion in \eqref{eq:cross} and do not represent statistical intervals across runs. Within each condition, the same threshold is applied to both correntropy-based controllers.

**Approximate first threshold-crossing iterations read from the averaged ANR curves (Gaussian: with measurement noise).**

<!-- LaTeX label: tab:cross -->
| Condition | Threshold | RFF-MCC | NKP-RFF-MCC | Difference |
| --- | --- | --- | --- | --- |
| Chaotic | $-20$ dB | 6,500–7,000 | 1,600–1,700 | $\approx$5,000 |
| Impulsive | $-2$ dB | 2,600–2,900 | 600–800 | $\approx$1,800–2,300 |
| Gaussian | $-20$ dB | 8,500–8,900 | 5,700–6,000 | $\approx$2,500–3,200 |

The structured controller approaches the selected attenuation level earlier in every condition. In the chaotic and Gaussian comparisons, it additionally attains lower late-stage curves, by about 1 dB and 0.6 dB, respectively. In the impulsive comparison, the two correntropy-based controllers achieve similar final levels, whereas the controller without correntropy weighting leaves the attenuating regime. These results separate the roles of the three components of the method: the RFF map provides the nonlinear features, the Kronecker structure preconditions the coefficient adaptation, and correntropy limits the response to large errors. With 64% fewer adaptive coefficients and a 22% increase in multiplications per iteration, the structured controller shortens the transient without degrading the late-stage attenuation by more than 0.2 dB in any of the three conditions.

# Conclusion
<!-- LaTeX label: sec:concl -->

This paper proposed a Kronecker-structured RFF controller with normalized correntropy adaptation for nonlinear ANC. The long output-coefficient vector of a fixed RFF map is represented by a few Kronecker products of short factors, whose normalized correntropy updates are driven by secondary-path-filtered features. The factor recursion was shown to act, to first order, as a full-vector filtered-x update with a factor-dependent positive semidefinite preconditioner, which yields the step-size condition $\mu_a + \mu_b < 2$; storage and multiplication counts and a bound on each factor increment were also derived. A four-term $25 \times 20$ factorization reduces the number of adaptive coefficients from 500 to 180.

In simulations with a benchmark nonlinear primary path, the structured controller reached $-20$ dB roughly 5,000 iterations earlier than the full correntropy controller under logistic-chaotic excitation and roughly 2,900 iterations earlier under Gaussian excitation with measurement noise, and it entered the attenuating regime earlier under $\alpha$-stable impulsive excitation, where a non-robust RFF controller lost attenuation. The late-stage levels of the two correntropy controllers remained within about 1 dB. Structured coefficient adaptation is therefore a useful design choice when fast initial noise reduction and a compact adaptive model are both required. The present study assumes an exact secondary-path model and synthetic FIR paths. Future work will examine secondary-path modeling errors, measured acoustic paths, and multichannel configurations, as well as the adaptive selection of $R$ and $\sigma_c$.

# CRediT authorship contribution statement

[Author-specific contributions to be supplied: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Writing – original draft, Writing – review & editing, Supervision, and Funding acquisition, as applicable.]

# Declaration of competing interest

[Competing-interest declaration to be completed by the authors.]

# Declaration of generative AI and AI-assisted technologies in the writing process

[To be completed by the authors if applicable, e.g.: During the preparation of this work the author(s) used [NAME TOOL / SERVICE] in order to [REASON]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article.]

# Data availability

[Data and code availability statement to be completed by the authors.]

# Acknowledgements

[Funding and acknowledgements, if applicable.]

## References

<!-- Numbered in order of first citation. Existing keys ref1–ref17 keep their original entries; new entries were added with descriptive keys. PENDING VERIFICATION marks entries still to be checked against the publisher record. -->

- **[kuo1999]** PENDING
- **[lu2021part1]** PENDING
- **[george2013]** PENDING
- **[ref1]** L. Lu, K.-L. Yin, R.C. de Lamare, Z. Zheng, Y. Yu, X. Yang, B. Chen, A survey on active noise control in the past decade–Part II: Nonlinear systems, Signal Process. 181 (2021) 107929. https://doi.org/10.1016/j.sigpro.2020.107929.
