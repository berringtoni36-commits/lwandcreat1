> Markdown editing copy converted from the current Overleaf main.tex on 2026-09-26. Use this file for prose polishing. Keep equations, numeric values, citation keys, labels, table data, figure assets, and author-decision items unchanged unless explicitly requested. After polishing, merge approved text back into the Overleaf LaTeX source and compile it; this Markdown file is not the submission source.

# Kronecker-structured adaptation of random Fourier features for robust nonlinear active noise control

**Authors:** [Author names]

**Corresponding author:** [Name and email].

**Affiliation:** [Department, institution, city, postal code, country]

## Abstract

Nonlinear active noise control (ANC) requires controllers that can model nonlinear primary paths and attain useful attenuation quickly under non-Gaussian disturbances. Random Fourier features (RFFs) provide a fixed nonlinear representation, yet the parameterization of their output coefficients remains a free design choice. This paper proposes a Kronecker-structured RFF controller adapted by normalized maximum-correntropy updates. The long coefficient vector is represented by a sum of Kronecker products of short factor pairs, which are adapted through secondary-path-filtered features. The formulation decouples the feature dimension, the number of Kronecker terms, and the correntropy width, and it yields explicit storage and arithmetic counts together with a bound on each factor increment. With 500 random features, a four-term factorization requires only 180 adaptive coefficients, a 64% reduction relative to the full vector. Under logistic-chaotic excitation, the structured controller reaches approximately $-20$ dB after about 2,000 iterations, whereas the full correntropy controller reaches approximately $-13$ dB; the final difference is about 1 dB. Under $\alpha$-stable impulsive excitation, both correntropy controllers maintain attenuation, and the structured controller enters the attenuating regime earlier. Under Gaussian excitation with measurement noise, it crosses $-20$ dB roughly 2,900 iterations earlier and finishes with an average noise reduction (ANR) about 0.6 dB lower. These results identify coefficient structure as a practical design variable for initial convergence speed and adaptive model size.

**Keywords:** nonlinear active noise control · random Fourier features · Kronecker decomposition · maximum correntropy criterion · normalized adaptive filtering
# Introduction

Nonlinear active noise control (ANC) requires a controller that can represent the nonlinear relationship between the reference signal and the disturbance, and that learns this relationship quickly enough to provide useful attenuation. Polynomial, functional-link, spline, neural-network, and kernel representations have been developed for different types of nonlinearity \cite{ref1}. Once a representation has been chosen, the parameterization of its coefficients and the update rule determine how quickly the controller approaches an attenuating operating point. This paper studies this adaptation problem in a fixed random-feature space.

Random Fourier features (RFFs) approximate a shift-invariant kernel by an explicit, finite set of nonlinear features \cite{ref2}. Because their dimension is fixed in advance, they avoid the growing dictionary of conventional kernel adaptive filters. Shen et al. \cite{ref3} developed random-feature approximations for multikernel adaptive filtering. In ANC, Deb et al. \cite{ref4} formulated RFF-based filtered-x least-mean-square (FxLMS) controllers and extended them to multichannel narrowband control. These studies establish the fixed nonlinear map as a practical controller representation, but they leave open how its output coefficients should be parameterized and adapted.

Robust adaptation limits the influence of large residuals. The maximum correntropy criterion (MCC) weights each error with a Gaussian kernel evaluated at the error magnitude, thereby suppressing excursions that are large relative to the kernel width \cite{ref5}. Huang et al. \cite{ref6} proposed a variable kernel width for MCC adaptation. Recent robust ANC algorithms include a Euclidean-direction-search MCC update \cite{ref7}, a projection FxLMS framework for impulsive environments \cite{ref8}, and a fractional-order generalized complex correntropy algorithm \cite{ref9}. Xiao et al. \cite{ref10} combined RFFs with a generalized hyperbolic tangent criterion and conjugate-gradient optimization. These methods modify the cost function or the optimization strategy; the present study instead introduces a structured parameterization of the feature-space coefficients.

Feature design and robust adaptation have also been combined directly. Ye et al. \cite{ref11} proposed an enhanced multiple-RFF ANC model that fuses several feature maps through a projection matrix and adapts it with generalized correntropy. Whereas that approach changes how multiple feature maps are combined, the present work retains a single fixed map and represents its long output-coefficient vector by pairs of shorter factors. This coefficient-space construction decouples the nonlinear dictionary from the number and organization of the adaptive parameters. Zhang et al. \cite{ref12} accelerated nonlinear ANC by arranging functional-link features in a delayless multi-sampled subband structure and derived stability and complexity results. The present construction addresses a different aspect, namely the coefficient structure within a single RFF map.

Kronecker product decomposition (KPD) provides the required factorization. In nonlinear ANC, low-rank structures have been exploited in spline adaptive filters \cite{ref13} and tap-decomposed robust Volterra filters \cite{ref14}. KPD has also been used to develop robust adaptive filters and filtered-x ANC algorithms \cite{ref15,ref16,ref17}, including correntropy-based variants; these works, however, apply the decomposition to linear filters. In contrast, this paper structures the output coefficients of a fixed RFF map inside a nonlinear ANC loop. Normalized correntropy weighting is applied to the resulting factor regressors, and the initial and late-stage attenuation are examined under three excitation conditions. The experimental focus is the time required to enter a low-residual operating region, considered jointly with the adaptive coefficient storage and the per-iteration arithmetic cost.

The main contributions of this paper are summarized as follows:

1. A normalized correntropy recursion is derived for a Kronecker-structured RFF controller, with explicit projected regressors and a specified factor-update order (Section \ref{sec:method}).
2. The adaptive coefficient storage and the per-iteration arithmetic cost are expressed in terms of the feature dimension and the number of Kronecker terms, and a bound on each normalized factor increment is established (Section \ref{sec:analysis}).
3. Comparisons under logistic-chaotic, $\alpha$-stable impulsive, and Gaussian excitation distinguish the initial attenuation from the late-stage residual. Under chaotic excitation, the structured controller is approximately 7 dB lower than the full controller after about 2,000 iterations; under Gaussian excitation, it crosses $-20$ dB roughly 2,900 iterations earlier (Section \ref{sec:sim}).

The remainder of this paper is organized as follows. Section \ref{sec:model} introduces the ANC model, the RFF map, and the correntropy baseline. Section \ref{sec:method} derives the structured recursion, and Section \ref{sec:analysis} analyzes its storage, arithmetic cost, and update bound. Section \ref{sec:sim} presents the simulation results, and Section \ref{sec:concl} concludes the paper.

# Nonlinear ANC model and random Fourier features
<!-- LaTeX label: sec:model -->

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

This model combines temporal memory with second- and third-order nonlinear terms. The frequency responses of the linear paths are shown in Fig. \ref{fig:paths}. The filtered-feature formulation below uses the true secondary-path response; hence, no secondary-path modeling error is introduced in the simulations.

> **Figure 1. Frequency responses of the primary and secondary paths: (a) magnitude and (b) phase. The frequency axis is normalized by $\pi$ rad/sample.**

<!-- image: fig1.pdf; LaTeX label: fig:paths -->

## Fixed nonlinear feature map

For a Gaussian kernel with length scale $\ell_r$, the frequency vectors $\boldsymbol{\omega}_j \sim \mathcal{N}(\mathbf{0}, \ell_r^{-2}\mathbf{I}_M)$ and phases $b_j \sim \mathcal{U}[0, 2\pi)$, $j = 1, \ldots, D$, are drawn independently, where $\mathbf{I}_M$ is the $M \times M$ identity matrix. The RFF map is

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

i.e., each feature sequence is filtered in time by the secondary path. Under the usual assumption that the coefficients vary slowly (the frozen-coefficient approximation), the secondary-path output is approximated by $\mathbf{w}_n^T \mathbf{q}_n$. This regression form is shared by the full and factorized controllers.

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

where $\mu > 0$ is the step size and $\varepsilon > 0$ is a small regularization constant. This baseline combines RFFs, filtered-x normalized least-mean-square (FxNLMS) adaptation, and MCC, and is therefore denoted RFF-FxNLMS-MCC, or RFF-MCC for short. Replacing $\psi_{\sigma_c}(e)$ by $e$ yields the RFF-FxNLMS comparator, abbreviated RFF-NLMS.

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

The output before secondary-path filtering is obtained in the same way by replacing $\mathbf{Q}_n$ with the reshaped unfiltered feature vector. The factorization therefore changes only the adaptive coefficient model and leaves the random features unchanged.

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

In the implementation, the residual used in the correntropy influence function is

$$
e_n = d(n) - \mathbf{b}_n^T \mathbf{u}_{b,n} + v(n),
$$
<!-- LaTeX label: eq:res -->

where $v(n)$ is omitted in the noise-free conditions. The same residual is used for both factor updates at iteration $n$.

## Initialization and implementation

The full RFF coefficient vector can be initialized to zero. The factorized controller, however, requires nonzero initial factors: if both $\mathbf{A}$ and $\mathbf{B}$ were zero, both regressors in \eqref{eq:reg} would vanish and the factors would never be updated. The factors are therefore initialized with small independent Gaussian values (0.01 times standard normal draws). The seeds of the random map and of the factor initialization are part of the experimental configuration. The complete procedure is summarized in Algorithm 1.

**Algorithm 1.** Normalized Kronecker-structured RFF correntropy adaptation

1. *Initialization:* choose $D_1$, $D_2$, $R$, $\sigma_c$, $\mu_a$, $\mu_b$, and $\varepsilon$; draw and fix the random frequencies and phases; initialize $\mathbf{A}_0$ and $\mathbf{B}_0$ with small nonzero values.
2. For $n = 0, 1, 2, \ldots$, form $\mathbf{x}_n$ and compute $\mathbf{z}_n$ using \eqref{eq:rff}.
3. Filter the feature sequences through the secondary-path model to obtain $\mathbf{q}_n$, and reshape it into $\mathbf{Q}_n$.
4. Compute the projected regressors $\mathbf{u}_{b,n}$ and $\mathbf{u}_{a,n}$ from the current factors using \eqref{eq:reg}.
5. Compute the residual $e_n$ in \eqref{eq:res} and the influence $\psi_{\sigma_c}(e_n)$ in \eqref{eq:mcc}.
6. Update $\mathbf{b}_n$ and $\mathbf{a}_n$ using \eqref{eq:facupd} with the regressors from Step 4, and reshape them into $\mathbf{B}_{n+1}$ and $\mathbf{A}_{n+1}$.
7. Shift the feature and path buffers and proceed to the next iteration.

The correntropy weight and the normalized factor directions play complementary roles: the correntropy weight regulates the effect of the residual magnitude, whereas the Kronecker factors determine the directions along which the feature-space coefficient vector is modified. Together, they yield a structured nonlinear controller with a bounded instantaneous error influence.

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

Storage is reduced whenever $R < D_1 D_2 / (D_1 + D_2)$. For a fixed product $D_1 D_2$, the sum $D_1 + D_2$ is minimized by nearly balanced factor dimensions. The RFF frequency matrix and phase vector are fixed parameters shared by both controllers and are not included in these counts.

Table \ref{tab:storage} shows how $R$ controls the adaptive model size: four pairs of factors of lengths 25 and 20 describe a 500-dimensional coefficient vector with 180 adaptive values. The experiments use $R = 4$, which provides a 64% storage reduction at a moderate increase in per-iteration arithmetic (Fig. \ref{fig:mults}); $R = 6$ is an alternative with a 46% reduction.

## Arithmetic cost

The arithmetic cost is measured by the number of real multiplications per iteration in the filtered-feature learning loop. Random-feature initialization, signal generation, plotting, and metric evaluation are excluded; cosine, exponential, and division operations are counted separately, and common scalar constants are precomputed. The same convention is applied to both controllers.

Both controllers require $D$ cosine evaluations. A literal two-factor implementation uses two correntropy evaluations and two normalized scalar divisions, whereas the baseline uses one of each. Because both factor updates use the same residual \eqref{eq:res} and kernel width, the exponential can in fact be reused; Table \ref{tab:mults} conservatively retains the literal two-evaluation convention.

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

## Interpretation of the factor adaptation

Each projected regressor depends on the companion factor. Consequently, the same filtered feature vector produces different adaptation directions as $\mathbf{A}$ and $\mathbf{B}$ evolve. A change in one short factor modifies a coordinated group of entries in the implied long coefficient vector, which provides a mechanism through which the controller can organize its initial adaptation within the selected factor model.

The number of Kronecker terms $R$ and the correntropy width $\sigma_c$ govern different aspects of this process: $R$ determines the representational structure and the coefficient count, whereas $\sigma_c$ determines the residual scale emphasized by the update. Their effects are therefore examined separately: the kernel-parameter sweep in Section \ref{sec:sweep} addresses error weighting, whereas $R$ is selected from the storage–arithmetic trade-off in Section \ref{sec:analysis} (Fig. \ref{fig:mults}).

# Simulation results
<!-- LaTeX label: sec:sim -->

## Evaluation setup and attenuation measure

The simulations compare RFF-MCC and NKP-RFF-MCC under logistic-chaotic, $\alpha$-stable impulsive, and Gaussian excitation. RFF-NLMS is additionally included in the impulsive and Gaussian comparisons to isolate the effect of correntropy weighting. The chaotic and impulsive simulations run for 20,000 iterations, and the Gaussian simulation runs for 30,000 iterations with measurement noise added to the error signal.

The average noise reduction (ANR) is computed from exponentially smoothed absolute amplitudes:

$$
\begin{aligned}
A_d(n) &= \beta A_d(n-1) + (1-\beta)|d(n)|, \\
A_e(n) &= \beta A_e(n-1) + (1-\beta)|e(n)|, \\
\mathrm{ANR}(n) &= 20 \log_{10} \frac{A_e(n) + \epsilon_a}{A_d(n) + \epsilon_a}.
\end{aligned}
$$

Negative values indicate attenuation. The reference configuration uses $\beta = 0.999$, 20 Monte Carlo runs, and random seed 871; Table \ref{tab:config} lists all settings and evaluation intervals. The initial learning period and the late-stage residual level are evaluated separately. To quantify the transient, the first threshold-crossing index is defined as

$$
t_\tau = \min\{ n : \mathrm{ANR}(n) \leq \tau \}.
$$
<!-- LaTeX label: eq:cross -->

The numerical values reported in this section are rounded readings of the displayed ANR curves. The coefficient counts refer to $D = 500$ with a $25 \times 20$ factorization and $R = 4$.

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

The correntropy width determines which errors contribute strongly to the update. The sweep is parameterized as

$$
g_k(e) = \exp\!\left( -\frac{k e^2}{2\sigma_0^2} \right), \qquad \sigma_{c,\mathrm{eff}} = \frac{\sigma_0}{\sqrt{k}}.
$$
<!-- LaTeX label: eq:k -->

With $\sigma_0 = 2.0$, $k = 1$ corresponds to the reference width $\sigma_c = 2$, and increasing $k$ narrows the effective kernel. Note that $k$ is unrelated to the number of Kronecker terms $R$. Fig. \ref{fig:ksweep} shows the RFF-MCC learning curves over the swept range, and Fig. \ref{fig:kss} summarizes the corresponding steady-state levels over the final evaluation window.

> **Figure 3. ANR learning curves of RFF-MCC for different correntropy parameters $k$ under logistic-chaotic excitation, with $k \in \{1, 1.5, 2, 2.8, 4, 5.6, 6, 8, 12, 15, 20\}$. Larger $k$ corresponds to a narrower effective kernel in \eqref{eq:k}.**

<!-- image: fig3.pdf; LaTeX label: fig:ksweep -->

> **Figure 4. Steady-state ANR over the final evaluation window versus the correntropy parameter $k$. The ANR decreases from approximately $-15$ dB to approximately $-22$ dB over the swept range, with diminishing changes at larger $k$.**

<!-- image: fig4.pdf; LaTeX label: fig:kss -->

The steady-state ANR decreases (i.e., the attenuation deepens) as $k$ increases. Across the sweep, it changes by approximately 7 dB, and the incremental improvement diminishes toward larger $k$. This trend indicates that the error-weighting scale is an important tuning parameter: a narrower effective kernel reduces the contribution of large residual excursions while retaining the updates driven by small errors. The useful operating range of $k$ should therefore be selected from the resulting attenuation trajectories; it is not tied to the number of Kronecker terms.

## Logistic-chaotic reference

The chaotic reference provides a nonlinear, non-Gaussian excitation for comparing the initial learning behavior of the two correntropy controllers. It is generated by the delayed logistic recursion $u(n) = 4u(n-6)[1 - u(n-6)]$, followed by centering and amplitude normalization. Fig. \ref{fig:chaos} compares the ANR over 20,000 iterations.

> **Figure 5. ANR of RFF-MCC and NKP-RFF-MCC under logistic-chaotic excitation. The structured controller reaches the $-20$ dB region earlier and maintains a lower ANR throughout the displayed interval.**

<!-- image: fig5.pdf; LaTeX label: fig:chaos -->

The two controllers separate clearly during the initial learning period. After about 2,000 iterations, NKP-RFF-MCC reaches approximately $-20$ dB, whereas RFF-MCC reaches approximately $-13$ dB. This 7 dB difference corresponds to an earlier entry of the structured controller into the attenuation region: its ANR first reaches $-20$ dB at around 1,600–1,700 iterations, whereas the baseline approaches this level only at around 6,500–7,000 iterations, i.e., roughly 5,000 iterations later.

As adaptation continues, the two curves gradually converge. At 20,000 iterations, the baseline and the structured controller reach approximately $-24$ dB and $-25$ dB, respectively. Compared with the pronounced transient improvement, the final difference of approximately 1 dB is modest; under this condition, the principal benefit is thus faster access to a low-residual operating region.

## Impulsive-noise condition

The $\alpha$-stable comparison examines adaptation in the presence of large noise excursions. Fig. \ref{fig:impulse} includes RFF-NLMS, which uses no correntropy weighting, together with the two MCC-based controllers, so that the effect of robust error weighting can be separated from the transient behavior associated with the coefficient structure.

> **Figure 6. ANR under $\alpha$-stable impulsive excitation. RFF-NLMS rises above the displayed range ($+2$ dB). Both correntropy-based controllers maintain a negative ANR over the later interval, and NKP-RFF-MCC reaches $-2$ dB earlier.**

<!-- image: fig6.pdf; LaTeX label: fig:impulse -->

RFF-NLMS initially reduces the residual, but its ANR then rises above 0 dB and leaves the displayed range at $+2$ dB. In contrast, both correntropy-based controllers maintain attenuation throughout the remainder of the experiment. This difference is consistent with the error weighting in \eqref{eq:mcc}: residuals much larger than the kernel width have a vanishing influence on the update.

The structured controller first crosses $-2$ dB within approximately 600–800 iterations, compared with 2,600–2,900 iterations for RFF-MCC; it therefore establishes attenuation earlier under impulsive excitation as well. At 20,000 iterations, RFF-MCC and NKP-RFF-MCC reach approximately $-4.9$ dB and $-4.7$ dB, respectively, i.e., similar late-stage levels after different initial transients. The distinction supported by this experiment is therefore the faster onset of attenuation, combined with the sustained attenuation provided by correntropy weighting.

## Gaussian reference with measurement noise

The Gaussian comparison evaluates the controllers under a conventional reference distribution with measurement noise in the error signal. As shown in Fig. \ref{fig:gauss}, all three controllers converge to a stable residual level. The two unstructured RFF controllers follow closely spaced trajectories, whereas the structured controller descends more rapidly during the initial learning period.

> **Figure 7. ANR under Gaussian excitation with measurement noise in the error signal over 30,000 iterations. NKP-RFF-MCC reaches the $-20$ dB region earlier, and the three controllers converge to closely spaced late-stage levels.**

<!-- image: fig7.pdf; LaTeX label: fig:gauss -->

NKP-RFF-MCC crosses $-20$ dB within approximately 5,700–6,000 iterations, whereas RFF-MCC reaches this level within 8,500–8,900 iterations and RFF-NLMS at around 8,000 iterations. The structured controller thus crosses $-20$ dB about 2,500–3,200 iterations earlier than RFF-MCC, with a central estimate of approximately 2,900 iterations. In the absence of impulsive excursions, correntropy weighting slightly slows the unstructured controller, yet the structured controller still achieves the earliest crossing.

At 30,000 iterations, the two full controllers reach approximately $-27.8$ dB and the structured controller approximately $-28.4$ dB, a late-stage difference of about 0.6 dB. The main result is again the earlier attenuation, here accompanied by a slightly lower final residual. This comparison also shows that the transient benefit persists when impulsive excursions do not dominate the reference signal.

## Comparison across conditions

Table \ref{tab:cross} summarizes the transient readings of the three experiments. The thresholds differ across conditions because the attainable attenuation levels differ. The ranges are read from the digitized curves using the first-crossing criterion in \eqref{eq:cross} and do not represent statistical intervals across runs. Within each condition, the same threshold is applied to both correntropy-based controllers.

**Approximate first threshold-crossing iterations read from the ANR curves.**

<!-- LaTeX label: tab:cross -->
| Condition | Threshold | RFF-MCC | NKP-RFF-MCC | Difference |
| --- | --- | --- | --- | --- |
| Logistic-chaotic | $-20$ dB | 6,500–7,000 | 1,600–1,700 | $\approx$5,000 |
| $\alpha$-stable impulsive | $-2$ dB | 2,600–2,900 | 600–800 | $\approx$1,800–2,300 |
| Gaussian with measurement noise | $-20$ dB | 8,500–8,900 | 5,700–6,000 | $\approx$2,500–3,200 |

The consistent observation is that the structured controller approaches the selected attenuation level earlier. In the chaotic and Gaussian comparisons, it additionally attains lower late-stage curves. In the impulsive comparison, the two correntropy-based controllers achieve similar final levels, whereas the controller without correntropy weighting leaves the attenuating regime. These results clarify the roles of the three components of the method: the RFF map provides the nonlinear features, the Kronecker model organizes the coefficient adaptation, and correntropy limits the response to large errors. Overall, the evidence indicates that the structured controller improves the transient speed while reducing the adaptive model size.

# Conclusion
<!-- LaTeX label: sec:concl -->

This paper formulated a Kronecker-structured RFF controller with normalized correntropy adaptation for nonlinear ANC. Its factor updates operate on secondary-path-filtered features and decouple the nonlinear dictionary from the adaptive coefficient structure. A four-term $25 \times 20$ factorization reduces the number of adaptive coefficients from 500 to 180, and explicit multiplication counts quantify the additional cost of the factor-regressor computations.

The simulations demonstrate faster initial attenuation. Under logistic-chaotic excitation, the structured controller is approximately 7 dB lower than the full correntropy controller after about 2,000 iterations and retains an advantage of about 1 dB at the end of the run. Under Gaussian excitation, it reaches $-20$ dB roughly 2,900 iterations earlier. Under impulsive excitation, correntropy weighting maintains attenuation, and the structured controller enters the attenuating regime earlier. These results indicate that structured coefficient adaptation is a useful design choice when fast initial noise reduction and a compact adaptive model are both required. Future work will examine the robustness of the method to secondary-path modeling errors under the same factor and feature budgets.

# CRediT authorship contribution statement

[Author-specific contributions to be supplied: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Writing – original draft, Writing – review & editing, Supervision, and Funding acquisition, as applicable.]

# Declaration of competing interest

[Competing-interest declaration to be completed by the authors.]

# Data availability

[Data and code availability statement to be completed by the authors.]

# Acknowledgements

[Funding and acknowledgements, if applicable.]

## References

- **[ref1]** L. Lu, K.-L. Yin, R.C. de Lamare, Z. Zheng, Y. Yu, X. Yang, B. Chen, A survey on active noise control in the past decade–Part II: Nonlinear systems, Signal Process. 181 (2021) 107929. https://doi.org/10.1016/j.sigpro.2020.107929.
- **[ref2]** A. Rahimi, B. Recht, Random features for large-scale kernel machines, Adv. Neural Inf. Process. Syst. 20 (2007). https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html.
- **[ref3]** M. Shen, K. Xiong, S. Wang, Multikernel adaptive filtering based on random features approximation, Signal Process. 176 (2020) 107712. https://doi.org/10.1016/j.sigpro.2020.107712.
- **[ref4]** T. Deb, D. Ray, N.V. George, A reduced complexity random Fourier filter based nonlinear multichannel narrowband active noise control system, IEEE Trans. Circuits Syst. II Express Briefs 68 (1) (2021) 516–520. https://doi.org/10.1109/TCSII.2020.3007999.
- **[ref5]** W. Liu, P.P. Pokharel, J.C. Principe, Correntropy: Properties and applications in non-Gaussian signal processing, IEEE Trans. Signal Process. 55 (11) (2007) 5286–5298. https://doi.org/10.1109/TSP.2007.896065.
- **[ref6]** W. Huang, H. Shan, J. Xu, X. Yao, Robust variable kernel width for maximum correntropy criterion algorithm, Signal Process. 182 (2021) 107948. https://doi.org/10.1016/j.sigpro.2020.107948.
- **[ref7]** J. Wang, L. Lu, Z. Zheng, K.-L. Yin, Y. Yu, L. Shi, Euclidean direction search algorithm with maximum correntropy criterion for active noise control system, Signal Process. 229 (2025) 109759. https://doi.org/10.1016/j.sigpro.2024.109759.
- **[ref8]** P. Feng, Z.-Y. Wang, H.C. So, Projection FxLMS framework of active noise control against impulsive noise environments, Signal Process. 225 (2024) 109624. https://doi.org/10.1016/j.sigpro.2024.109624.
- **[ref9]** Y. Wang, B. Lin, Y. Guan, J. Qian, Y.-R. Chien, G. Qian, Fractional-order generalized complex correntropy algorithm for robust active noise control, Signal Process. 235 (2025) 110024. https://doi.org/10.1016/j.sigpro.2025.110024.
- **[ref10]** Y. Xiao, S. Chen, Q. Zhang, D. Lin, M. Shen, J. Qian, S. Wang, Generalized hyperbolic tangent based random Fourier conjugate gradient filter for nonlinear active noise control, IEEE/ACM Trans. Audio Speech Lang. Process. 31 (2023) 619–632. https://doi.org/10.1109/TASLP.2022.3230545.
- **[ref11]** Z. Ye, Y. Zheng, T. Zhou, C. Gao, S. Wang, Enhanced multiple random Fourier features based generalized maximum correntropy algorithm for active noise control, Appl. Acoust. 239 (2025) 110807. https://doi.org/10.1016/j.apacoust.2025.110807.
- **[ref12]** S. Zhang, W.X. Zheng, H. Han, Design of delayless multi-sampled subband functional link neural network with application to active noise control, Signal Process. 202 (2023) 108757. https://doi.org/10.1016/j.sigpro.2022.108757.
- **[ref13]** S.S. Bhattacharjee, V. Patel, N.V. George, Nonlinear spline adaptive filters based on a low rank approximation, Signal Process. 201 (2022) 108726. https://doi.org/10.1016/j.sigpro.2022.108726.
- **[ref14]** K.-L. Yin, H.-R. Zhao, Y.-F. Pu, L. Lu, Nonlinear active noise control with tap-decomposed robust Volterra filter, Mech. Syst. Signal Process. 206 (2024) 110887. https://doi.org/10.1016/j.ymssp.2023.110887.
- **[ref15]** S.S. Bhattacharjee, K. Kumar, N.V. George, Nearest Kronecker product decomposition based generalized maximum correntropy and generalized hyperbolic secant robust adaptive filters, IEEE Signal Process. Lett. 27 (2020) 1525–1529. https://doi.org/10.1109/LSP.2020.3017106.
- **[ref16]** L. Li, S. Wang, S.S. Bhattacharjee, J.R. Jensen, M.G. Christensen, Nearest Kronecker product decomposition based multichannel filtered-x affine projection algorithm for active noise control, Mech. Syst. Signal Process. 224 (2025) 112055. https://doi.org/10.1016/j.ymssp.2024.112055.
- **[ref17]** H. Lee, Y. Park, Control filter estimation for multichannel active noise control using Kronecker product decomposition, IET Signal Process. 2025 (2025) 2128989. https://doi.org/10.1049/sil2/2128989.
