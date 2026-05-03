# Chapter 3 — Robust Federated Tensor Learning for Heterogeneous Imaging Data Under Sample and User Contamination

> **Draft, v0.1 — methodology stub.** This document establishes the formulation, the three robust scatter losses, and the modified federated estimation results (Propositions 1′ and 2′) needed for Chapter 3. Experimental matrix and the empirical motivation pilot are deferred to a separate document once the simulation/real data are available.

---

## 3.1 Motivation and Problem Statement

Chapter 2 established a federated tensor learning model that fuses heterogeneous imaging data from $M$ organizations through local projections $\{V_n^m\}_{n=1,m=1}^{3,M}$ and a globally shared projection $\{U_n\}_{n=1}^3$, yielding low-dimensional features $\hat{\mathcal{Z}}_j^m \in \mathbb{R}^{P_1 \times P_2 \times P_3}$ for downstream LLS regression. The estimation criterion maximizes the total scatter

$$
\Psi^{\mathrm{all}} \;=\; \sum_{m=1}^{M}\sum_{j=1}^{J_m}\, \big\| \mathcal{X}_j^m \times_1 V_1^m \times_2 V_2^m \times_3 V_3^m \times_1 U_1 \times_2 U_2 \times_3 U_3 \big\|_F^2 \tag{2}
$$

over $\{V_n^m, U_n\}$.

This formulation rests on two implicit assumptions that are rarely satisfied in industrial practice:

1. **Sample-level cleanliness.** Every degradation image stream $\mathcal{X}_j^m$ is assumed to faithfully represent the asset's true degradation. In practice, individual samples are corrupted by transient sensor saturation, dust or condensation on the lens, partial occlusion by maintenance personnel, mislabeled time-to-failure values, or single frames affected by lighting or vibration.
2. **User-level cleanliness.** Every organization is assumed to contribute equally informative data. In practice, organizations differ in calibration quality, environmental control, and labeling rigor; one badly miscalibrated user can dominate the global scatter.

Because the squared-Frobenius criterion in (2) penalizes deviations *quadratically*, even a small fraction of contaminated samples or a single low-quality user can shift the estimated subspace $\{\hat{U}_n\}$ substantially. In the federated setting this is particularly damaging: the server aggregates only sufficient statistics ($Z^{m\Phi}_{(n)}$ in Algorithm 3 of Chapter 2) and cannot inspect raw data to identify and remove outliers.

The goal of this chapter is to develop **robust federated heterogeneous tensor learning** — variants of the Chapter 2 estimator that automatically down-weight contaminated samples and users while preserving the federated, privacy-preserving structure of Algorithms 1–4. We propose a unified family of three robust losses, derive the corresponding closed-form local updates and federated SVD-based global updates, and demonstrate empirically that the resulting estimator recovers a substantially cleaner subspace and lower TTF prediction error under sample- and user-level contamination.

### 3.1.1 Notation

We retain the notation of Chapter 2: $\mathcal{X}_j^m \in \mathbb{R}^{I_1^m \times I_2^m \times I_3^m}$ is the $j$-th image-stream tensor of user $m$; $V_n^m \in \mathbb{R}^{I_n^m \times I_n}$ are local projections; $U_n \in \mathbb{R}^{I_n \times P_n}$ are global projections; $\mathcal{X}_j^{\prime m} = \mathcal{X}_j^m \times_1 V_1^m \times_2 V_2^m \times_3 V_3^m$ is the locally aligned tensor; $X_{j(n)}^{\prime m}$ is its mode-$n$ unfolding; and $X_{j(n)}^{\prime m \Phi} = X_{j(n)}^{\prime m} U^{\Phi}_n$ where $U^\Phi_1 = U_2 \otimes U_3$, $U^\Phi_2 = U_1 \otimes U_3$, $U^\Phi_3 = U_1 \otimes U_2$.

For brevity we sometimes write the projected tensor as $\mathcal{Y}_j^m := \mathcal{X}_j^m \times_1 V_1^m \times_2 V_2^m \times_3 V_3^m \times_1 U_1 \times_2 U_2 \times_3 U_3 \in \mathbb{R}^{P_1 \times P_2 \times P_3}$, so the Chapter 2 criterion is $\Psi^{\mathrm{all}} = \sum_m \sum_j \|\mathcal{Y}_j^m\|_F^2$.

---

## 3.2 A Family of Robust Scatter Criteria

We propose three robust modifications of (2). Each replaces or reweights the contribution of individual samples or users; each preserves the *closed-form, SVD-based federated updates* that make Chapter 2's estimator practical. All three have the same outer block-coordinate-descent structure as Algorithm 4, with a single inner change.

### 3.2.1 Variant 1 — Sample-Weighted Scatter (RFTL-S)

We attach a non-negative weight $w_j^m \in [0, 1]$ to each sample and maximize the weighted scatter

$$
\boxed{\; \Psi^{\mathrm{all}}_{\mathrm{S}} \;=\; \sum_{m=1}^{M}\sum_{j=1}^{J_m} w_j^m \, \big\| \mathcal{Y}_j^m \big\|_F^2 \;} \tag{R1}
$$

The weights are *learned, not specified*: at iteration $\ell$ they are a function of the current per-sample reconstruction residual

$$
r_j^m \;=\; \big\| \mathcal{X}_j^m - \mathcal{X}_j^m \times_1 V_1^m V_1^{m\top} \times_2 V_2^m V_2^{m\top} \times_3 V_3^m V_3^{m\top} \times_1 U_1 U_1^{\top} \times_2 U_2 U_2^{\top} \times_3 U_3 U_3^{\top} \big\|_F.
$$

Specifically, we use the Huber-like rule

$$
w_j^m \;=\; \min\!\Big(1,\; c \, / \, r_j^m\Big), \qquad c = 1.345 \cdot \mathrm{MAD}_{m', j'}\big(r_{j'}^{m'}\big),
$$

where $\mathrm{MAD}$ is the median absolute deviation across all assets and users — computed in a federated manner (see §3.4.3). This yields full weight ($w_j^m = 1$) for samples with residual within typical range and $\mathcal{O}(1/r_j^m)$ down-weighting for samples in the tail. This is the well-known IRLS implementation of an M-estimator.

**Note (sign of the criterion).** Because (2) *maximizes* scatter, an outlier sample has *large* $\|\mathcal{Y}_j^m\|_F^2$ and inflates the objective. The reweighting rule above is based on *reconstruction* residual $r_j^m$, not on $\|\mathcal{Y}_j^m\|_F$, so a sample with large residual (i.e., poorly explained by the current subspace) is correctly identified as an outlier and down-weighted. This is the appropriate diagnostic for a PCA-style scatter-maximization objective.

### 3.2.2 Variant 2 — User-Weighted Aggregation (RFTL-U)

Where Variant 1 reweights samples within each user, Variant 2 reweights *whole users* in the server-side aggregation:

$$
\boxed{\; \Psi^{\mathrm{all}}_{\mathrm{U}} \;=\; \sum_{m=1}^{M} \omega_m \sum_{j=1}^{J_m} \big\| \mathcal{Y}_j^m \big\|_F^2 \;} \tag{R2}
$$

with $\omega_m \in [0, 1]$, $\sum_m \omega_m = M$ (preserving overall scale). The user weights are determined by user-level *consensus disagreement*: at each iteration $\ell$, each user $m$ computes its candidate global projection $\hat{U}_n^{(m)}$ as if it were the only contributor, and shares the principal-angle distance $d_m = \sum_n d_{\mathrm{Grass}}(\hat{U}_n^{(m)}, U_n^{(\ell)})$ between its candidate and the current consensus to the server. The server then sets

$$
\omega_m \;=\; \rho\!\left( \frac{d_m}{\mathrm{med}(d_1, \dots, d_M)} \right),
$$

with $\rho$ a bounded influence function (e.g., the Tukey biweight). Users whose locally-implied subspace agrees with the consensus are weighted near 1; users whose subspace disagrees badly receive $\omega_m \to 0$. This is a federated, Grassmann-manifold analogue of trimmed mean robust aggregation as used in robust federated learning [refs needed: Pillutla et al., Yin et al.].

**Practical note.** The principal-angle distance $d_{\mathrm{Grass}}(A, B)$ for matrices with orthonormal columns can be computed from $\sigma_{\min}(A^\top B)$ and reveals nothing about the user's data beyond the subspace they would propose — which is what we are aggregating anyway. This preserves the privacy guarantees of Chapter 2.

### 3.2.3 Variant 3 — $\ell_{2,1}$ Mode-Wise Robust Scatter (RFTL-21)

Both prior variants are *whole-tensor* outlier detectors. Variant 3 detects *partial* corruption — outliers in particular slabs (e.g., a single bad frame, or a single spatial row) of an otherwise valid tensor:

$$
\boxed{\; \Psi^{\mathrm{all}}_{2,1} \;=\; \sum_{m=1}^{M}\sum_{j=1}^{J_m} \big\| Y_{j(n)}^m \big\|_{2,1} \;} \tag{R3}
$$

where $\|A\|_{2,1} = \sum_i \|A_{i,:}\|_2$ is the row-wise $\ell_{2,1}$ norm of the mode-$n$ unfolding. (Mode $n$ is fixed during an inner update; the choice rotates as we update each $V_n^m$ and $U_n$.) This is the standard $\ell_{2,1}$ generalization of robust PCA [Nie et al. 2010]; rows of $Y_{j(n)}^m$ correspond to mode-$n$ slabs of the projected tensor, so a single bad frame ($n=3$) or row ($n=1$) is automatically given lower influence.

The $\ell_{2,1}$ norm admits the standard variational form $\|A\|_{2,1} = \min_{D \succeq 0,\, \mathrm{diag}} \tfrac{1}{2}\,\mathrm{tr}(A^\top D A) + \tfrac{1}{2}\,\mathrm{tr}(D^{-1})$, giving an IRLS update structurally identical to Variant 1, with weights $w_{j,i}^m = 1/\|Y_{j(n);i,:}^m\|_2$ on individual rows of the unfolded projected tensor.

### 3.2.4 Three Methods for Three Threat Models

The three variants are presented as **three distinct methods**, each targeted at a different contamination scenario the practitioner may face. They are not competitors to be ranked; they are siblings with non-overlapping use cases. A practitioner picks the variant that matches their threat model — or runs all three when the threat model is unknown.

| Method | Threat model addressed | Mechanism | Where the robustness lives |
|---|---|---|---|
| **RFTL-S** | One or more *individual assets* contributed corrupted image streams (sensor saturation, occlusion, mislabeled TTF) | IRLS over per-asset weights $w_j^m$ | Inside each user's local update |
| **RFTL-U** | One or more *whole organizations* contributed systematically miscalibrated data | Grassmann-distance trimming of user contributions $\omega_m$ | At the server-side aggregation |
| **RFTL-21** | Specific *frames or rows* within otherwise-valid streams are corrupted (single dropped frame, partial occlusion) | Mode-wise $\ell_{2,1}$ row weights | Inside each mode-$n$ inner update |

In §3.5 we evaluate each method against the contamination scenario it targets, plus cross-evaluate to characterize what happens when the practitioner's choice is mis-matched to the actual threat.

---

## 3.3 Robust Federated Updates: Modified Closed-Form Solutions

We now derive the analogues of Propositions 1 and 2 for the weighted scatter (R1). The corresponding results for (R3) follow immediately by the standard $\ell_{2,1}$–weighted-Frobenius equivalence; (R2) inherits the Chapter 2 derivations and only modifies the server-side aggregation step.

### 3.3.1 Updating Local Projections under (R1)

**Proposition 1′ (Weighted local update).** *Fix global projections $\{U_n\}_{n=1}^3$, fix $V_2^m, V_3^m$, and fix non-negative sample weights $\{w_j^m\}_{j=1}^{J_m}$. Then $V_1^m$ that maximizes the weighted criterion*

$$
\sum_{j=1}^{J_m} w_j^m \big\| \mathcal{X}_j^m \times_1 V_1^m \times_2 V_2^m \times_3 V_3^m \times_1 U_1 \times_2 U_2 \times_3 U_3 \big\|_F^2
$$

*is given by the $I_1$ leading eigenvectors of the **weighted** mode-1 scatter matrix*

$$
\Phi_{(1)}^{m, w} \;=\; \sum_{j=1}^{J_m} w_j^m \, X_{j(1)}^{m\Phi}\, X_{j(1)}^{m\Phi\,\top}, \tag{R1.1}
$$

*where $X_{j(1)}^{m\Phi} = X_{j(1)}^m \cdot \big(U_1 \otimes (V_2^m U_2) \otimes (V_3^m U_3)\big)$ as in Chapter 2, eq. (4).*

**Proof.** Substitute $\sqrt{w_j^m}\, \mathcal{X}_j^m$ for $\mathcal{X}_j^m$ in the proof of Chapter 2's Proposition 1 (Appendix); the squared Frobenius norm absorbs the square root, yielding $w_j^m \|\mathcal{Y}_j^m\|_F^2$. The remainder of the proof — Frobenius-norm-trace identity, eigenvalue–eigenvector argument — is unchanged. $\blacksquare$

The analogous statement for $V_2^m$ and $V_3^m$ holds by symmetry (rotating modes 1, 2, 3). The composite-matrix formulation $C_n^m = V_n^m U_n$ from Chapter 2, §3.4 carries through identically with $\Phi^{m, w}_{(n)}$ replacing $\Phi^m_{(n)}$ in line 13 of Algorithm 4.

### 3.3.2 Updating Global Projections under (R1)

**Proposition 2′ (Federated weighted SVD).** *Fix all local projections and weights. Then $U_n$ that maximizes the weighted scatter is given by the $P_n$ leading left singular vectors of*

$$
Z^{\Phi, w}_{(n)} \;=\; \big[\, \sqrt{w_1^1}\, X_{1(n)}^{\prime 1 \Phi},\; \sqrt{w_2^1}\, X_{2(n)}^{\prime 1 \Phi},\; \dots,\; \sqrt{w_{J_M}^M}\, X_{J_M(n)}^{\prime M \Phi}\, \big]. \tag{R1.2}
$$

**Proof.** As in Proposition 2 of Chapter 2, $\sum_m \sum_j w_j^m\, X_{j(n)}^{\prime m \Phi}\, (X_{j(n)}^{\prime m \Phi})^\top = Z^{\Phi, w}_{(n)} (Z^{\Phi, w}_{(n)})^\top$ by direct computation; the leading eigenvectors of the LHS are the leading left singular vectors of the RHS by SVD. $\blacksquare$

The crucial property — *user-separability* of the columns of $Z^{\Phi, w}_{(n)}$ — is preserved. Each user $m$ scales its own columns by $\sqrt{w_j^m}$ locally before sharing, then the federated incremental SVD (Algorithm 3 of Chapter 2) proceeds *unchanged*. **No new privacy considerations are introduced**: each user's $w_j^m$ is computed from local residuals, applied locally, and never shared.

### 3.3.3 Updating Global Projections under (R2)

For Variant 2, the weighted aggregate $\sum_m \omega_m \Phi_{(n)}^m$ is replaced in the federated SVD by

$$
Z^{\Phi, \omega}_{(n)} \;=\; \big[\, \sqrt{\omega_1}\, Z^{1\Phi}_{(n)},\; \sqrt{\omega_2}\, Z^{2\Phi}_{(n)},\; \dots,\; \sqrt{\omega_M}\, Z^{M\Phi}_{(n)} \,\big],
$$

where $Z^{m\Phi}_{(n)}$ is user $m$'s contribution as in Algorithm 3. The $\omega_m$ are computed by the server from the principal-angle distances received from each user (see §3.2.2), so each user need only scale their columns by $\sqrt{\omega_m}$ before passing them into the incremental update. Algorithm 3 is otherwise unchanged.

### 3.3.4 Computational Cost

The robust extensions add the following per-iteration overhead relative to Chapter 2:

- **RFTL-S:** one tensor reconstruction per asset to compute $r_j^m$ (which is dominated by the projection cost we already pay), plus one MAD across all $\sum_m J_m$ residuals (a single scalar per user shared with the server). Net overhead: $O(1)$ communication, $O(\sum_m J_m \prod_n I_n^m)$ compute — same order as one ordinary iteration.
- **RFTL-U:** $M$ Grassmann distances of size $I_n \times I_n$ — negligible.
- **RFTL-21:** IRLS over rows of the mode-$n$ unfolding — same order as Chapter 2, with a small per-iteration constant for the row weights.

In all three variants the algorithm remains a single-pass-per-iteration block coordinate ascent, with the same convergence guarantees as Chapter 2 §3.4 (monotone non-decreasing weighted scatter, bounded above, hence convergent to a stationary point of the weighted objective).

---

## 3.4 Algorithm

We present the full robust federated algorithm for Variant 1 (RFTL-S) — the other variants differ only in the weight-update step and a transparent change in line 5 of the server side.

### Algorithm 5: Robust Federated Tensor Learning — Sample-Weighted (RFTL-S)

**Input.** Local data $\{\mathcal{X}_j^m\}$ for users $m = 1, \dots, M$. Tuning constant $c > 0$ (default: data-driven, see below). Maximum iterations $L$, tolerance $\epsilon$.

**Output.** Local projections $\{\hat{V}_n^m\}$, global projections $\{\hat{U}_n\}$, sample weights $\{\hat{w}_j^m\}$.

**Initialization.** Run Algorithms 1 and 2 of Chapter 2 with all $w_j^m = 1$. Initialize residuals $r_j^m$ from this initial fit, set $w_j^m \leftarrow \min(1, c / r_j^m)$ with $c = 1.345 \cdot \mathrm{MAD}(\{r_{j'}^{m'}\})$.

**For** $\ell = 1, \dots, L$:

1. **(Server)** Receive scaled column blocks $\{\sqrt{w_j^m}\, X_{j(n)}^{\prime m \Phi}\}$ from each user; run Algorithm 3 of Chapter 2 *unchanged*; broadcast updated $U_n$ for $n = 1, 2, 3$.
2. **(Each user $m$, in parallel)** For $n = 1, 2, 3$, compute $\Phi_{(n)}^{m, w}$ via (R1.1) and update $V_n^m$ via Proposition 1′. Recover via $V_n^m = (U_n U_n^\top)^{-1} U_n C_n^{m\top}$ as in Chapter 2 §3.4.
3. **(Each user $m$)** Recompute residuals $r_j^m$ for $j = 1, \dots, J_m$ using updated projections.
4. **(Federated MAD)** Each user shares its empirical CDF of $\{r_j^m\}$ at a fixed quantile grid (or shares $\{r_j^m\}$ for a privacy-acceptable robust statistic) — see §3.4.3. Server returns $\mathrm{MAD}^{(\ell)}$.
5. **(Each user $m$)** Update weights: $w_j^m \leftarrow \min(1, c^{(\ell)} / r_j^m)$, $c^{(\ell)} = 1.345 \cdot \mathrm{MAD}^{(\ell)}$.
6. **(Server)** Compute $\Psi^{\mathrm{all}, w}_{\mathrm{S}, \ell}$. **If** $|\Psi^{\mathrm{all}, w}_{\mathrm{S}, \ell} - \Psi^{\mathrm{all}, w}_{\mathrm{S}, \ell-1}| < \epsilon$: **break**.

### 3.4.1 Privacy considerations

The only quantities crossing the server boundary in Algorithm 5 are: (i) $\sqrt{w_j^m}$-scaled column blocks of $X_{j(n)}^{\prime m \Phi}$ — same privacy posture as Chapter 2; (ii) per-user summary statistics of residuals $r_j^m$ (mean / quantile sketch / empirical CDF). The latter is a function of magnitude only, not of subspace direction, and reveals strictly less than the aligned tensors already shared. We discuss the privacy implication of sharing residual *magnitudes* in Appendix C.

### 3.4.2 Choice of tuning constant $c$

The constant $c = 1.345 \cdot \mathrm{MAD}$ is the standard Huber choice giving $\sim95\%$ asymptotic efficiency under Gaussian data while bounding the influence of outliers. We test sensitivity in §3.5.4.

### 3.4.3 Federated MAD

Computing $\mathrm{MAD}(\{r_j^m\}_{m, j})$ exactly requires either centralization of all residuals or a federated quantile primitive. We report results for two practical implementations: (i) each user shares its local median to the server, the server computes a median-of-medians estimator; (ii) each user shares a fixed-grid empirical CDF (e.g., 100 quantile bins). Both are differentially-private compatible. Variant (i) is sufficient for our experiments and is what we use unless noted.

---

## 3.5 Empirical Study (Outline — to be completed)

*The full experimental matrix and results are deferred to a companion document. The following is the planned structure.*

### 3.5.1 Contamination scenarios

We evaluate three contamination regimes mapped one-to-one to the three variants:

- **Sample-level contamination.** A fraction $\pi_S \in \{0, 0.05, 0.10, 0.20, 0.30\}$ of samples per user have a randomly chosen frame replaced by saturated (all-white) or zeroed (all-black) pixels, or have additive Gaussian noise of $10\times$ baseline variance.
- **User-level contamination.** A fraction $\pi_U \in \{0, 1/M, 2/M\}$ of *users* have all their samples corrupted by an arbitrary additional rotation in the projected space (mimicking miscalibration).
- **Slab-level contamination.** A fraction $\pi_F$ of *individual frames* across all assets are corrupted as above.

For each scenario, both the heat-transfer simulator from Chapter 2 §4 and the bearing IR data from Chapter 2 §5 are used.

### 3.5.2 Baselines

(a) Chapter 2's federated estimator (no robustness). (b) Per-user MFPCA + LLS regression (the Chapter 2 baseline). (c) A simple "filter-then-fit" baseline that drops samples with $\|\mathcal{X}_j^m\|_F$ above a fixed quantile, then runs Chapter 2's estimator. (d) The three RFTL variants.

### 3.5.3 Metrics

(i) Subspace recovery: principal angle between $\hat{U}_n$ and the noise-free $U_n^\star$ obtained from clean data. (ii) TTF prediction error (median, Q1, Q3) as in Chapter 2. (iii) Recall and precision of contaminated samples flagged by RFTL-S weights $w_j^m < 0.5$.

### 3.5.4 Sensitivity analyses

(i) Sensitivity of RFTL-S to $c$ across $\{0.5, 1.0, 1.345, 2.0\} \cdot \mathrm{MAD}$. (ii) Number of iterations to convergence vs. contamination level. (iii) Effect of federated MAD approximation vs. exact MAD.

---

## 3.6 Open Items / TODOs Before Submission

- [ ] Run §3.5.1 sample-level contamination pilot on the Chapter 2 simulator to confirm motivation: does Chapter 2's estimator's $\hat{U}_n$ degrade meaningfully (say, principal angle $> 10°$) at $\pi_S = 0.10$? If not, the motivation needs to be re-cast.
- [ ] Tighten privacy claim on residual sharing (Appendix C). Consider whether to add a differential-privacy story or leave as informal "reveals strictly less than current shared sufficient statistics."
- [ ] Confirm that the LLS regression stage of Chapter 2 carries over unchanged (it should — once we have clean low-dim features $\hat{\mathcal{Z}}_j^m$ from the robust fusion stage, the regression doesn't care).
