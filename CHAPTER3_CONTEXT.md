# Chapter 3 Context Primer (for Claude Code)

This file brings a fresh Claude session up to speed on Chapter 3 of the PhD. Read this first, then `chapter3_draft.md`, then look at the existing code in folder code (`onepass-real-03-21-2.py`, `my_mpca_02_27_nomean.py`, `onepass-02-13-0.py`).

## Project in one paragraph

Chapter 2 (already published / under review): a federated tensor learning model for prognostics from heterogeneous imaging data. $M$ organizations have image-stream tensors $\mathcal{X}_j^m \in \mathbb{R}^{I_1^m \times I_2^m \times I_3^m}$ with **different per-mode dimensions per user**. Local projections $V_n^m$ map each user's tensors into a common intermediate space; global projections $U_n$ extract shared low-dim features. Trained by maximizing total scatter via federated incremental SVD (privacy-preserving — no raw data leaves users). Features feed an LLS regression for time-to-failure. The idea of MPCA used was from paper `papers\MPCA_Multilinear_Principal_Component_Analysis_of_Tensor_Objects.pdf`.

Chapter 3 (this work): **Robust Federated Tensor Learning (RFTL)** — three named extensions of Chapter 2 that handle three contamination scenarios. They preserve the closed-form federated SVD updates of Chapter 2.

## The three methods

- **RFTL-S** (sample-level cleanness): IRLS reweighting of individual assets via Huber-like weights $w_j^m$ on reconstruction residuals. Catches: bad assets within a user.
- **RFTL-U** (user-level cleanness): server-side trimming via Grassmann-distance $\omega_m$ between each user's locally-implied subspace and the consensus. Catches: miscalibrated organizations.
- **RFTL-21** (frame/slab-level cleanness): mode-$n$ $\ell_{2,1}$ row weights. Catches: corrupted individual frames within otherwise-valid streams.

Each is a minimal, principled modification of Chapter 2's Propositions 1 and 2 — see §3.3 of `chapter3_draft.md` for the modified propositions ("Proposition 1′" and "Proposition 2′").

## Decisions already made (don't re-litigate)

- The three variants are presented as **three separate methods**, not as ablations of a single method. Each has a named threat model.
- Method name is **RFTL** (Robust Federated Tensor Learning).
- Reweighting diagnostic for RFTL-S is the **reconstruction residual** $r_j^m = \|\mathcal{X}_j^m - \hat{\mathcal{X}}_j^m\|_F$, not $\|\mathcal{Y}_j^m\|_F$. (Reasoning: we maximize scatter, so a large projection is signal not outlier; a poor reconstruction is the outlier indicator.)
- Tuning constant: $c = 1.345 \cdot \mathrm{MAD}$ (standard Huber, ~95% Gaussian efficiency).
- Federated MAD: median-of-medians is the default; quantile-grid CDF is an alternative.
- LLS regression stage from Chapter 2 carries over unchanged. The robustness only changes the feature-extraction stage.
- Scope is "minimum new code, strong experiments, modest methodology." Reuse Chapter 2's heat-transfer simulator and bearing IR data — don't introduce new datasets unless empirically necessary.

## Immediate next steps (pre-chapter writing)

1. **Motivation pilot.** Inject sample-level contamination into the Chapter 2 simulator at $\pi_S \in \{0, 0.05, 0.10, 0.20\}$. Fit Chapter 2's existing estimator (no robustness). Measure the principal angle between $\hat{U}_n$ and a clean reference $U_n^\star$. Goal: confirm that contamination at $\pi_S = 0.10$ produces a meaningful subspace shift (target: $> 10°$). If the shift is small, the motivation has to be reframed.
2. **Implement RFTL-S** as a minimal wrapper around the existing solver:
   - Add a `weights` argument to whatever function builds $Z^{m\Phi}_{(n)}$. Just multiply each column block by $\sqrt{w_j^m}$ before federated SVD. This is the entire change to the existing code.
   - Add an outer IRLS loop that recomputes residuals and weights at the end of each pass.
   - Add a federated MAD subroutine (median-of-medians is fine to start).
3. **Run the experimental matrix in §3.5 of `chapter3_draft.md`.** Sample-level contamination first (matched to RFTL-S). Then user-level (matched to RFTL-U). Then frame-level (matched to RFTL-21). Then cross-matrix (mismatched threat model vs. method).

## Data

- Real data that used for the case study is `ResampleDegImages.mat` in `data` folder. This is real bearing data.
- Simulated data used for the simulation study is in folder `Simulated Data` the samples individually exist there. 

## Existing code in the repo

- `onepass-real-03-21-2.py` — the primary Chapter 2 implementation using real data.
- `onepass-02-13-0.py` — Primary chapter 2 implementation and run for simulation data.
- `my_mpca_02_27_nomean.py` — MPCA implementation (probably the per-user baseline).
- `.mat` data files — These are the heat-transfer simulation outputs and the Gebraeel bearing IR data referenced in Chapter 2 §4 and §5.

There are some experimental code work  there as well:

-  `MPCA.py` implementing MPCA (double check the correctness before reusing it). 
- `Original_paper.py` implementing the work in `MPCA_Multilinear_Principal_Component_Analysis_of_Tensor_Objects.pdf` which is present in the papers folder (do double check the code before reusing it).
- Last piece of code is `Orthogonal_test.py` that was an experimental thing to see wheter after projections tensors stay orthogonal or not.

## Style/scope reminders

- **Don't rewrite Chapter 2 code.** Add a `rftl_s.py`, `rftl_u.py`, `rftl_21.py` that *import* the existing functions and inject the reweighting step. The robustness should be a wrapper, not a fork.
- **Match Chapter 2's notation in any new code.** Same variable names ($V_n^m$, $U_n$, $\Phi_{(n)}$, $Z^{m\Phi}_{(n)}$, $X_{j(n)}^{\prime m \Phi}$).
- **Three users, three different sizes** as in Chapter 2. Don't change the user count for the main experiments — only for the user-count sensitivity analysis.

## Open items still to resolve

- Privacy claim on residual sharing (Appendix C of `chapter3_draft.md`). Currently informal. Can be tightened with a DP argument or left informal.
- Whether the same MAD constant works across all three variants, or each needs its own.

## Files in this handoff

- `chapter3_draft.md` — methodology stub. Read this in full before writing code.
- `CHAPTER3_CONTEXT.md` — this file.
- `Image_tensors.pdf` — the Chapter 2 paper (the immediate prior work being extended).
