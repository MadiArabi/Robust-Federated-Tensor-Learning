# Chapter 3 — Experiment Log

## 2026-05-03: Session 1 — Motivation Pilot + RFTL-S Start

### What we did

1. **Reviewed existing codebase.** Read `CHAPTER3_CONTEXT.md`, `chapter3_draft.md`, and all code files. Confirmed that:
   - `my_mpca_02_27_nomean.py` is the core Chapter 2 implementation (MPCA, MPCA_FD, MPCA_beta classes)
   - `onepass-02-13-0.py` and `onepass-real-03-21-2.py` are the simulation and real-data runners
   - Data: 2000 simulated .mat files (each 10 frames of 21×21), plus `ResampleDegImages.mat` (real bearing IR)
   - No robust code exists yet — methodology is drafted but unimplemented

2. **Initialized git repo** with `.gitignore` excluding `data/` (large .mat binaries).

3. **Wrote and ran the motivation pilot** (`code/motivation_pilot.py`):
   - Injects sample-level contamination (additive Gaussian noise at 10× baseline std) at π_S ∈ {0, 0.05, 0.10, 0.20, 0.30}
   - Runs Chapter 2's MPCA_FD on clean vs. contaminated data
   - Measures max principal angle between clean U* and contaminated U across 3 modes
   - **Result (2 repeats, 310 samples):**
     - π_S=0.00: 0.6° (baseline noise from random splits)
     - π_S=0.05: **54.4°** 
     - π_S=0.10: **71.3°**
     - π_S=0.20: **69.9°**
     - π_S=0.30: **74.4°**
   - **Verdict: PASSED.** Even 5% contamination shifts the subspace by ~54°, far exceeding the 10° threshold. Motivation for RFTL is unequivocal.

4. **Started RFTL-S implementation** (see below).

### Practical findings

- Loading 2000 .mat files from Google Shared Drive is very slow (~2 min for 310 files). HPC is needed for full experiments.
- Python environment: use `C:\Users\sarabi\AppData\Local\anaconda3\envs\anomaly\python.exe` directly (conda run crashes).
- MPCA_FD with 30 iterations is sufficient for subspace direction convergence in the pilot.
- The 10× noise multiplier is aggressive — saturation at ~71° suggests the subspace is nearly orthogonal even at moderate contamination. Consider testing with 3× and 5× for a more gradual curve in the final paper.

5. **Implemented RFTL-S** (`code/rftl_s.py`):
   - `MPCA_FD_Weighted` class: extends Chapter 2's MPCA_FD with per-sample weights
     - V update (Proposition 1'): weighted scatter matrix via `einsum("n,nij,nik->jk", w, ...)`
     - U update (Proposition 2'): each column block scaled by sqrt(w_j^m) before incremental SVD
   - `RFTL_S` class: outer IRLS loop wrapping MPCA_FD_Weighted
     - Computes reconstruction residuals per sample
     - Federated MAD (median-of-medians)
     - Huber weights on standardized residuals: `w = min(1, k/|s_j|)` where `s_j = |r_j - median(r)| / MAD`
     - Converges when max weight change < tolerance
   - Helper functions: `federated_mad()`, `huber_weights()`, `reconstruction_residual()`
   - Experiment runner: compares baseline vs RFTL-S under contamination, reports principal angles + precision/recall of flagged samples
   - **Smoke test results (15 samples/user, 3 contaminated in user 0, 10x noise):**
     - Contaminated samples: weights = 0.002 (effectively zeroed)
     - Clean samples: weights 0.9-1.0
     - Clean users 1&2: mean weight ~0.91
   - Key design decision: weights are computed on standardized residuals (deviation from median / MAD), not raw residuals. Raw residuals are all large because low-rank projection can't explain all variance; standardization isolates the outliers.

### What's next

- [ ] Finalize motivation plot on HPC with 10 repeats (local 2-repeat plot already saved)
- [x] ~~Implement RFTL-S~~ 
- [ ] Run full RFTL-S experiments on HPC (subspace recovery + TTF prediction)
- [ ] Implement RFTL-U (`code/rftl_u.py`)
- [ ] Implement RFTL-21 (`code/rftl_21.py`)
- [ ] Full experimental matrix from §3.5 of `chapter3_draft.md`

### Files created this session

- `code/motivation_pilot.py` — motivation pilot script (args: --n-repeats, --max-files, --data-path)
- `code/rftl_s.py` — RFTL-S: weighted MPCA_FD + IRLS wrapper + experiment runner
- `code/test_rftl_s.py` — smoke test for RFTL-S
- `motivation_pilot_results.png` — motivation plot (2 repeats, 310 samples)
- `experiment_log.md` — this file
- `.gitignore` — excludes data/, __pycache__, etc.

---

## 2026-05-04 → 2026-05-16: Sessions 2–4 — RFTL-S Iteration & Variants

### What we did

1. **Iterated on RFTL-S** — fixed mean metric, switched to conservative Huber k=3.0, warm-started IRLS, parallelized experiment runner across repeats.
2. **Replaced IRLS with hard-trimming one-step estimator** — simpler and more stable.
3. **Added V-only weighting option** and multi-config comparison experiment.
4. **Increased iterations** to 200 for convergence; optimized by sharing the unweighted fit and using 50-iter warm-started re-fit.
5. **Fixed baseline** to use MPCA_FD (same implementation as clean reference) for fair comparison.
6. **Implemented RFTL-U** (`code/rftl_u.py`) and **RFTL-21** (`code/rftl_21.py`).
7. **Added cold re-fit and multi-noise RFTL-S** experiments; added cold-start results.

---

## 2026-06-06: Session 5 — Trajectory Change: Prediction-Based Evaluation

### Key decision

**We changed the evaluation approach.** Previously, we measured robustness by comparing subspace angles (principal angles between clean and contaminated factor matrices). We are now evaluating via **prediction performance** instead — specifically, MAPE on log(TTF) using Tucker regression on real degradation data.

**Why the change:** Subspace angles show that contamination distorts the factor matrices, which is useful as a motivation argument (and is kept for that purpose in the motivation pilot). However, the ultimate goal of RFTL is to produce better *predictions* under contamination, not just recover subspaces. Prediction-based comparison on real data is a stronger and more practical evaluation that directly measures what matters for prognostics.

### What we did

1. **Implemented `rftl_s_real.py`** — full prediction pipeline on real bearing degradation data (`ResampleDegImages.mat`), replicating the Chapter 2 pipeline (MPCA_FD → min-max normalize → Ridge → MPCA_beta → Tucker regression → MAPE on log-TTF) with a robustness layer on top.
2. **Three-way comparison:**
   - **Clean reference:** MPCA_FD on uncontaminated data (upper bound)
   - **Baseline:** MPCA_FD on contaminated data (no robustness)
   - **RFTL-S:** Huber-weighted MPCA_FD re-fit on contaminated data
3. **Added `tucker_regression0.py`** — Tucker tensor regressor used by the prediction pipeline.
4. Experiment sweeps multiple contamination levels and noise multipliers with AIC rank selection.

### Current evaluation plan

- **Motivation pilot** (subspace angles) is retained as §3.4.1 — shows contamination breaks the subspace, justifying the need for RFTL.
- **Prediction experiment** (MAPE on log-TTF) is now the primary evaluation in §3.5 — shows RFTL-S actually improves downstream task performance under contamination.

### What's next

- [x] ~~Run `rftl_s_real.py` on HPC with 50 repeats across full contamination/noise grid~~ (done 2026-06-30, see Session 6)
- [ ] Add RFTL-U and RFTL-21 to the prediction pipeline for full method comparison
- [ ] Compile results table: Clean vs Baseline vs RFTL-S vs RFTL-U vs RFTL-21
- [ ] Finalize motivation plot on HPC with 10 repeats

---

## 2026-07-05: Session 6 — HPC Results Analysis: Gaussian Contamination Doesn't Hurt Prediction

### The run

HPC job 631860 (Hazel, `long` queue, 10 workers) completed successfully: 50 repeats of `rftl_s_real.py` over ~2.6 days (Jun 27 → Jun 30). Full grid: noise multiplier {2, 3, 5, 10} × π_S {0.05, 0.1, 0.2, 0.3} × Huber k {1.0, 1.345, 2.0, 3.0}. Results in `output/rftl_s_real_results.csv` (10,450 rows, no NaNs) plus per-repeat checkpoints in `output/checkpoints/`.

### Headline finding: the experiment's premise did not hold

**Contamination does not degrade prediction MAPE — at 10× noise it *improves* it.**

- Clean reference: MAPE = 0.0689 ± 0.0167.
- Contaminated baseline: 0.054–0.072 across the grid. At noise 2–5×, statistically indistinguishable from clean (paired t-tests, same split per repeat). At 10× noise, the baseline is significantly **better** than clean (0.054–0.058, p < 0.001, baseline wins 70–78% of repeats).
- This is despite the motivation pilot showing the subspace rotates ~70° at 10× noise — the subspace is destroyed, yet prediction improves.

### Consequences for RFTL-S

- **RFTL-S ≈ clean everywhere** (0.062–0.075). It does exactly what it was designed to do: remove contamination and recover clean-level performance.
- But because the baseline *benefits* from the noise at 10×, RFTL-S **loses** significantly in all sixteen 10×-noise cells (6–28% worse MAPE, p < 0.01).
- At noise 2–5×: 61 of 64 (noise, π, k) comparisons are statistical ties. The lone significant RFTL-S win (noise=5, π=0.2, ~13%, p≈0.004) is almost certainly a multiple-testing artifact.
- No Huber k is meaningfully better than another for MAPE (grid-average 0.0660–0.0669 across k).

### The robustness mechanism itself works perfectly

- **Recall of contaminated samples = 1.0 in every cell** (all noise levels, all π, all k).
- Precision at k=3.0 is 0.988 (flags essentially exactly the injected outliers). Smaller k over-flags clean samples (precision 0.34 at k=1.0) without hurting MAPE.
- So the estimator is sound; the downstream task simply doesn't reward robustness against this contamination type.

### Why might Gaussian contamination help prediction? (hypotheses, being tested)

1. **AIC picks smaller ranks under contamination** — noise inflates RSS, pushing AIC toward lower-rank configs that generalize better. → Testing now with `code/diag_rank_selection.py` (logs per-rank MAPE/AIC/beta-ranks for clean vs baseline, 5 local repeats).
2. **Min-max normalization absorbs it** — contaminated projections inflate the train Min/Max range, shrinking clean features toward the middle = extra regularization for Ridge/Tucker.
3. **The prediction signal doesn't live in the leading MPCA subspace** — Tucker regression may read TTF from whatever subspace it is handed.

### Code observations made during analysis (inherited from Chapter 2, worth noting in the write-up)

- The "AIC" in `prediction_pipeline` is computed from **test-set** residuals (`rftl_s_real.py`), so rank selection already peeks at test RSS.
- The AIC penalty uses the MPCA_beta 90%-variance ranks (P1_b·P2_b·P3_b), not the projection rank config itself.

### Decision / path forward

**Gaussian sample-level noise is the wrong contamination model for the prediction-based evaluation** — it is too benign for this pipeline (isotropic white noise is exactly what regression + regularization shrugs off). Plan:

1. First, diagnose the mechanism via the AIC rank-selection diagnostic (in progress).
2. Then **switch the contamination model** to something that actually damages prediction, e.g. label contamination (wrong TTF), samples from a different degradation regime, or structured artifacts (dead pixels, stripes, sensor drift).
3. The motivation pilot (subspace angles) remains valid and unequivocal — contamination does destroy the subspace. The chapter narrative may become: robust estimation recovers the clean subspace; for the prediction task, contamination models beyond isotropic noise are where robustness pays off.

### New contamination model designed (same session)

**Chosen: structured artifacts (hot-pixel block + stripes), concentrated in one user.** Reasoning: the contamination model must satisfy two constraints — (a) damage baseline prediction (or we repeat the Gaussian null result) and (b) be *detectable from reconstruction residuals*, the only sensor RFTL-S has. Checked against (b):

- **Label contamination (wrong TTF): rejected** — tensor side is clean, residual-based detection is blind to it.
- **Regime mixing (frames from another degradation stage): rejected as primary** — swapped samples still lie in the shared low-rank subspace, so residuals stay small. Same blindness. Could appear in the chapter later as a *limitation* experiment ("residual-based robustness cannot detect in-subspace contamination").
- **Structured artifacts: chosen** — far from the degradation subspace (detectable) and spatially coherent (cannot average out the way isotropic Gaussian noise did).

**Key design decisions:**

1. **Fixed pattern across contaminated samples** (drawn once per run): a faulty camera produces the same artifact in every image, and correlated contamination looks like real structure to MPCA — it drags leading factors toward the artifact instead of averaging out. Independent per-sample patterns would partially cancel (the Gaussian failure mode).
2. **Contamination concentrated in user 0** (`target_user=0`, faulty-camera-site story) — matches the federated setting that federated MAD was designed for. `target_user=None` gives the old spread behavior for comparison. Later analysis should report user 0's test error separately (most impacted) in addition to overall.
3. **Physical story for the chapter:** IR camera fixed-pattern noise / hot-dead pixels / readout stripes are classic IR sensor failure modes — easy to motivate.
4. **Richer error metric:** don't just report the mean of |predicted−true|/|true| (MAPE) — also record **median, 25th, 75th percentile**, overall **and per user**. (Madi's request; gives distributional insight, esp. for the impacted user.)
5. **Pre-HPC gate rule (lesson from the Gaussian null result):** before any HPC submission, a ~5-repeat local pilot running *baseline only* at the harshest setting must show baseline MAPE clearly degraded vs clean (>20%). No degradation → no HPC run.

**Sample sizes** (relevant for choosing π_S): users A/B/C have 45/50/55 samples = 25/30/35 train + 20 test each (90 train, 60 test total, from 284 samples in `ResampleDegImages.mat`). At π_S=0.3 of user 0 that's only ⌈0.3·25⌉=8 of 90 training samples (<9% global). If the gate shows weak degradation, π_S up to 0.5–1.0 of user 0 is physically defensible (a truly faulty camera corrupts everything it sees).

### Files created this session

- `code/diag_rank_selection.py` — logs per-rank MAPE/AIC/beta-ranks for clean vs baseline (runs clean + 4 grid corners: noise {2,10} × π {0.1,0.3}), reusing seeds/functions from `rftl_s_real.py` so repeats match the HPC run
- `output/diag_rank_selection.csv` — diagnostic output
- `code/contamination.py` — shared injection module: `contaminate(train_users, pi_s, mode, rng, amplitude, block_size, n_stripes, target_user)` with modes `gaussian` / `hot_block` (b×b block saturated to ref_max + amplitude·std, same location all 16 frames) / `stripe` (amplitude·std offset on fixed rows-or-columns set). Smoke-tested: fixed pattern across samples confirmed, only target user touched, reproducible from RNG.
- `code/pilot_contamination_gate.py` — the pre-HPC gate: clean + baseline at 4 harsh conditions (hot_block/stripe × amplitude {5,10}, π_S=0.3 of user 0, block 5×5, 3 stripes), 5 repeats; reports MAPE/median/q25/q75 overall + per user, AIC-selected ranks, and PASS/fail verdict per condition. Note: uses condition *index* (not `hash()`) for the contamination seed — Python string hash is randomized per process.

### Rank diagnostic result (same session): mechanism identified — noise acts as regularization, NOT rank selection

`diag_rank_selection.py` (5 repeats, clean + baseline at noise {2,10} × π {0.1,0.3}) shows:

1. **Hypothesis 1 (AIC picks smaller ranks under contamination) is REJECTED.** AIC selects the same small ranks ([2,2,3]–[4,4,5]) for clean and contaminated data alike; the selection distribution barely moves.
2. **The improvement happens at *every* rank.** At 10× noise / π=0.3, per-rank MAPE beats clean at essentially all rank configs (e.g. rank [4,4,5]: clean 0.0680 vs contaminated 0.0489). Even the oracle best-rank-per-repeat MAPE is better contaminated (0.0489) than clean (0.0639). So the benefit is not a selection artifact — the contaminated features genuinely predict better.
3. **Smoking gun: contamination stabilizes the high-rank pipeline.** On clean data the pipeline explodes at ranks ≥ [5,5,6] (MAPE 2.4 → 25.6 at [9,9,11]). Under 10× contamination the blowup largely disappears (MAPE stays < 1.4 at every rank). This is the signature of hypothesis 2: contaminated projections inflate the train min-max range, so the scaled features are bounded and Ridge/Tucker are effectively regularized; clean data has near-degenerate feature dimensions (tiny min-max denominators → exploding scaled features) that noise papers over.

**Conclusion for the chapter:** Gaussian contamination was benign because it acts as implicit feature regularization in the min-max → Ridge → Tucker pipeline, at every rank. This both explains the null result and further motivates structured (correlated, non-averaging) contamination.

### Status right now (for resuming after a disconnect)

1. ~~`diag_rank_selection.py`~~ DONE — mechanism identified (see "Rank diagnostic result" above): noise = implicit regularization, not rank selection.
2. `pilot_contamination_gate.py` **running locally in the background** (5 repeats, 5 workers). Gate PASS = baseline degrades >20% under hot_block/stripe.
3. If the gate passes: fold `contamination.py` + quartile/per-user stats into `rftl_s_real.py` (note: the uncommitted edit in `rftl_s_real.py` already returns a stats dict from `best_rank_mape`, but `report_results` and the checkpoint writer still expect scalars — finish that integration in one pass), then HPC run with the new model.
4. If the gate fails: raise π_S for user 0 (0.5–1.0), bigger block / more stripes, before rethinking.

### What's next

- [x] ~~Analyze AIC rank-selection diagnostic~~ (mechanism: noise = implicit regularization at every rank; AIC selection unchanged)
- [x] ~~Design new contamination model(s) that plausibly damage prediction~~ (hot_block + stripe in `contamination.py`)
- [x] ~~Run `pilot_contamination_gate.py` locally~~ (July 5 background run died with the session; re-run 2026-07-07 — gate FAILED, see Session 7)
- [ ] Integrate contamination.py + quartile/per-user error stats into `rftl_s_real.py` (finish the half-done stats-dict edit)
- [ ] Re-run prediction experiment on HPC with the new contamination model (only if gate passes)
- [ ] Add RFTL-U and RFTL-21 to the prediction pipeline for full method comparison

---

## 2026-07-07: Session 7 — Gate Run 1 FAILED (structured artifacts also benign); escalated gate launched

### Housekeeping

The July 5 background run of `pilot_contamination_gate.py` never finished — it died when that session ended (no CSV was written). Re-launched from scratch this session; completed in well under an hour (5 repeats, 5 workers).

### Gate run 1 result: FAIL on all four conditions

Structured artifacts (π_S=0.3 of user 0, block 5×5, 3 stripes) do **not** hurt baseline prediction — they slightly improve it, exactly like Gaussian noise did. Clean MAPE 0.0649; contaminated:

| Condition | MAPE | Δ overall | Δ user 0 |
|---|---|---|---|
| hot_block amp=5 | 0.0605 | −6.8% | −11.4% |
| hot_block amp=10 | 0.0604 | −6.9% | −4.3% |
| stripe amp=5 | 0.0577 | −11.1% | −6.1% |
| stripe amp=10 | 0.0604 | −7.0% | −13.8% |

Even user 0 (the contaminated camera site) predicts *better* under its own contamination. Amplitude 10 was no worse than 5 — consistent with the Session 6 mechanism: min-max normalization absorbs large-amplitude contamination (inflated train range → clean test features squashed toward the middle → implicit regularization). AIC-selected ranks barely moved ([2,2,3]–[4,4,5]), same as the Gaussian diagnostic. Results in `output/pilot_contamination_gate.csv` (25 rows).

Per the pre-HPC gate rule: **no HPC submission** with this configuration.

### Escalation (running now)

Followed the pre-agreed fallback (raise π_S for user 0, bigger artifacts). Edited `PILOT_CONDITIONS` in `pilot_contamination_gate.py`:

- π_S ∈ {0.6, 1.0} of user 0 (at 1.0, every training sample from the faulty camera is corrupted — physically defensible)
- block 5→9 pixels (9×9 of 21×21 ≈ 18% of the image), stripes 3→6 (of 21 rows/cols)
- amplitude fixed at 10 (amplitude demonstrably doesn't matter)
- Output → `output/pilot_contamination_gate_escalated.csv` (run 1 CSV kept separately)

### If the escalated gate also fails

Amplitude/extent escalation is likely exhausted at π_S=1.0 — that would point to a structural property of the pipeline, not a weak contamination model: any train-only tensor-side corruption inflates the min-max range (regularizing) while Tucker regression reads TTF from whatever subspace it gets (Session 6 hypothesis 3, supported by the motivation pilot's 70° rotation not hurting MAPE). Candidate responses, in order of preference:

1. **Contaminate in-range** (artifacts that don't inflate min-max: e.g. dead/zeroed block instead of hot, or permuting frames) — attacks structure without triggering the regularization shield.
2. **Contaminate test-side too** (faulty camera corrupts *all* its images, train and test) — the deployed-sensor story; robustness then matters for cleaning the reference subspace used to project test data.
3. **Reframe the chapter's evaluation**: keep subspace recovery (motivation pilot, unequivocal) as the primary robustness evidence; report honestly that log-TTF Tucker prediction on this dataset is insensitive to tensor-side train contamination.

### Decision (same session): test-side contamination adopted (Madi)

While the escalated gate was running, **Madi decided the experiments should contaminate the test data too** — candidate response 2 above, promoted to the primary design. Physical story is stronger (a faulty camera corrupts every image it takes, train and test alike), and it removes the suspected structural shield (test features were always computed from clean images, so any subspace still transmitting TTF signal lets Ridge/Tucker recalibrate around the contamination).

**New evaluation structure** — with dirty test images, "clean reference" splits in two:

- **clean** (clean train → clean test): camera never broke; context upper bound
- **oracle** (clean-train factors → dirty test): best a *perfect* robust estimator can do — robustness fixes the subspace, not the test images
- **baseline** (dirty train → dirty test): no robustness

**Gate criterion changes accordingly: baseline must degrade >20% vs *oracle*** — that gap is what RFTL-S can actually close. (Baseline vs clean is reported for context.)

Implementation (smoke-tested, all pass):

- `contamination.py` refactored: `_draw_pattern()` / `_apply_pattern()` split out so one camera pattern applies identically to train and test; old `contaminate()` API and RNG call order unchanged (July gate runs stay reproducible). New `contaminate_train_and_test(..., pi_s_test=None)` — same fault rate in test as train by default; artifact scale (ref_std/ref_max) from clean *train* only (no test leakage).
- `code/pilot_gate_testside.py` — test-side gate: clean + (oracle, baseline) × {hot_block, stripe} × π_S {0.3, 1.0}, amp 10, block 9, 6 stripes, user 0 only; oracle rows reuse the clean fit (nearly free). Output → `output/pilot_gate_testside.csv`.

Open design question for the chapter (flagged, not yet decided): is RFTL-S judged on artifact-corrupted user-0 test images as-is (harsh, realistic), or does the story include fixing the camera at test time? Changes what "recovering clean performance" means for user 0.

### Escalated train-only gate result (completed 2026-07-08): first PASS, and a monotone trend

π_S {0.6, 1.0} of user 0, amp 10, block 9×9, 6 stripes. Clean MAPE 0.0649. Degradation is finally **positive and monotone in π_S**:

| Condition | MAPE | Δ overall | Δ user 0 | Verdict |
|---|---|---|---|---|
| hot_block π_S=0.6 | 0.0735 | +13.2% | +0.8% | fail |
| **hot_block π_S=1.0** | **0.0866** | **+33.5%** | **+21.3%** | **PASS** |
| stripe π_S=0.6 | 0.0684 | +5.3% | −11.2% | fail |
| stripe π_S=1.0 | 0.0721 | +11.1% | +6.0% | fail |

Reading: **extent, not amplitude, is the lever** (amplitude 5→10 did nothing in run 1; π_S 0.3→1.0 moves Δ from −7% to +33%). But it took *total* corruption of user 0's training set with the biggest artifact to get one PASS on a clean-test evaluation — quantifying how strong the min-max/regression shield is, and reinforcing the test-side pivot. Interesting detail: overall MAPE degrades more than user 0's own (the dragged shared subspace hurts everyone; user 0's clean test images don't carry the artifact). Results in `output/pilot_contamination_gate_escalated.csv`.

Note for later: at π_S=1.0, user 0 has zero clean training samples — RFTL-S downweighting would effectively remove user 0 from the subspace fit entirely (federated MAD still works: user 0 is 25 of 90 global samples, so the global median stays clean-dominated).

### Integration done (2026-07-08, while test-side gate runs): `rftl_s_real.py` rewritten for the new design

- Contamination: inline Gaussian injection replaced with `contaminate_train_and_test` (structured artifacts, user 0, train+test). Grid (PROVISIONAL until gate verdict): {hot_block, stripe} × π_S {0.3, 0.6, 1.0}, amp 10, block 9, 6 stripes.
- Four arms per condition: clean / **oracle** (clean-train factors → dirty test) / baseline / RFTL-S (k sweep unchanged). Result keys now (method, mode, pi_s[, k]).
- Stats-dict edit finished: `best_rank_mape` → `best_rank_stats` returning {selected_rank, mape, median, q25, q75, mape_user0-2, median_user0-2} of **absolute** relative errors (the half-done version used signed errors for the quantiles — fixed to match the pilots). Reporting + CSV expand the dict into columns; checkpoint JSON handles dicts as-is.
- ⚠ Old Gaussian-run checkpoints are NOT resume-compatible — the HPC run must use a fresh `--output-dir` (warned in docstring).
- End-to-end smoke test (3 iters, 2 ranks, 1 repeat): all keys/fields/CSV correct; detection at π_S=1.0 rec 1.0, prec 0.93.

**Smoke-test observation to check against the converged gate result:** the oracle arm exploded (MAPE 1.51 vs baseline 0.078) — clean-fit min-max has never seen the artifact, so dirty test features scale far out of range; the baseline calibrated to the artifact during training. If this holds at 150 iterations, "perfect robust estimation + discard" is *miscalibrated at deployment*, and the gate contrast to watch becomes RFTL-vs-baseline (RFTL-S keeps dirty features in the regression = calibrated, but with a robust subspace). Possible chapter angle rather than a bug.

### Test-side gate result (2026-07-08): ORACLE COLLAPSES, baseline self-calibrates — the finding inverts

`pilot_gate_testside.py` (5 repeats, converged 150-iter fits; `output/pilot_gate_testside.csv`):

| Condition | Oracle MAPE (user0) | Baseline MAPE (user0) | Baseline vs clean |
|---|---|---|---|
| hot_block π=0.3 | 0.1698 (0.3149) | 0.0713 (0.0688) | +9.8% |
| hot_block π=1.0 | 0.2670 (0.6064) | 0.0622 (0.0481) | −4.1% |
| stripe π=0.3 | 0.1145 (0.1954) | 0.0595 (0.0600) | −8.3% |
| stripe π=1.0 | 0.1667 (0.3392) | 0.0710 (0.0612) | +9.3% |

The smoke-test observation holds at convergence, decisively: **the "oracle" (perfect robust estimator: clean factors, discard all contamination) is the WORST method under persistent test-side contamination** — 4–9× clean MAPE for user 0 — because its min-max calibration has never seen the artifact. The baseline trains on artifact-bearing data and stays within ±10% of clean. Conclusion now supported across all three contamination designs: **this prediction pipeline does not reward robustness; under persistent deployment faults, idealized discard-robustness actively harms it.** This is the prediction-side finding for the chapter (with the regularization mechanism from Session 6), not a failed experiment. Open follow-up: where does actual RFTL-S land between baseline and oracle here (robust subspace but regression sees dirty features → possibly best of both; smoke test hinted rftl beat baseline 26%). The integrated `rftl_s_real.py` answers this.

### Monitoring task built (2026-07-08): SPE control chart = the subspace-native downstream task

Per discussion with Madi: the positive robustness demonstration moves to **degradation monitoring via SPE charts**, where the subspace and training-residual distribution are consumed directly (no regression shield). Mechanisms: limit inflation (masking) + subspace rotation (swamping/false alarms).

- `code/pilot_monitoring.py`: per user, median-TTF split into healthy/degraded; healthy → 70% train / 30% FAR holdout, centered by healthy-train mean; federated MPCA at rank [5,5,6]; SPE limit = (weighted) mean + 3σ of train SPE. Metrics per user: FAR, power, AUC, detection TTF (first alarm streaming degraded by descending TTF), limit value. Methods: clean / baseline / RFTL-S (k=3, weighted re-fit + weighted limits). Contamination: train-side only (hot_block/stripe × π_S {0.3, 0.6}, amp 10, block 9, 6 stripes, user 0).
- **Smoke test (1 repeat, 5 iters): hot_block π=0.3 is a textbook PASS** — user0 limit 61→1721, power 0.86→0.00, degradation never detected; clean users' limits also inflate via the shared subspace (federated coupling — one faulty site poisons everyone's charts); RFTL-S restores everything (prec/rec 1.0).
- **Two watch-items for the converged run:** (1) at π=0.6 and for stripes, detection recall = 0 — heavily-repeated identical patterns get absorbed into the non-robust initial fit, so residual-based detection goes blind (initial-estimator breakdown; potential limitation section / motivates better initial estimator); (2) clean-chart FAR ≈ 0.43 — mean+3σ limit from ~15 under-converged train residuals is optimistic; if it persists at 150 iters, fix limit calibration (e.g., holdout-calibrated or higher-percentile limit) before chapter use.
- Full pilot (5 repeats, 150 iters) launched 2026-07-08, output → `output/pilot_monitoring.csv`.

### Monitoring pilot v1 result (2026-07-08, converged 150-iter fits; `output/pilot_monitoring.csv`)

- **hot_block π=0.6: PASS** — user0 limit 59→582 (masking), power 0.89→0.54, RFTL-S recovers fully (power 0.896, limits back to clean) despite partial detection (prec 0.6 / rec 0.455).
- **hot_block π=0.3: masking confirmed (limit 59→121, power 0.89→0.45) but RFTL recovery only partial (0.56)** — detection recall fell to 0.3, vs 1.0 in the 5-iter smoke test. Diagnosis: **a converged non-robust initial fit absorbs the repeated artifact into the subspace, so contaminated samples reconstruct well and residual detection goes blind.** More iterations = blinder detection. Classic one-step-estimator failure: needs a weak/high-breakdown initial estimator.
- **Stripes: benign AND invisible at convergence** (limits/power/detection all flat) — MPCA absorbs the additive pattern entirely. Keep as the limitation/symmetry case: contamination the subspace can absorb is both harmless to the chart and undetectable from residuals.
- **Clean-chart FAR ≈ 0.5 persists** — mean+3σ limit from ~15 overfit train residuals is optimistic. Hits all methods equally (contrast valid), but needs a calibration fix for chapter-grade absolute numbers (options: holdout-calibrated limit, lower monitor rank, leave-one-out residuals). Decide with Madi.

**v2 launched same session:** detection residuals now computed from a separate early-stopped fit (`DETECT_ITERATIONS = 5`) that hasn't yet learned the artifact — the smoke test showed rec 1.0 at 5 iters. Output → `output/pilot_monitoring_v2.csv`.

### Monitoring pilot v2 result (2026-07-09): early stopping alone does NOT fix detection — one-step reweighting is the real culprit

`output/pilot_monitoring_v2.csv`. At π=0.3 detection unchanged (prec 0.4 / rec 0.3 — the smoke test's rec 1.0 was a lucky seed; artifact detectability varies per draw). At π=0.6 recall improved (0.455→0.6) **but recovery collapsed** (user0 power 0.896→0.532, rftl limit 58→252): unflagged contaminated samples at full weight have huge residuals under the partially-cleaned subspace and inflate the weighted limit. v1's apparent recovery there was partly the converged fit *absorbing* the artifact harmlessly. **Structural conclusion: one reweighting pass cannot handle correlated repeated artifacts** — what the initial fit absorbs stays invisible; what it half-absorbs poisons the limit.

**v3 launched: IRLS** (5 rounds of early-stopped weighted fit → residuals → Huber reweight, then converged final weighted fit). This is the original RFTL-S design (dropped in Sessions 2–4 for simplicity); correlated contamination is where it earns its keep — each round pushes the artifact further out of the subspace and exposes remaining contaminated samples to the next. Output → `output/pilot_monitoring_v3_irls.csv`. If v3 works, one-step vs IRLS is itself a chapter ablation (and the Sessions 2–4 simplification needs revisiting for RFTL-S generally).

### What's next

- [x] ~~Test-side gate~~ (oracle collapses; see above)
- [x] ~~Monitoring pilot v1~~ (masking + recovery demonstrated at π=0.6; detection-blindness diagnosed)
- [x] ~~Monitoring pilot v2~~ (one-step reweighting is structurally insufficient for correlated artifacts)
- [x] ~~Monitoring pilot v3 (IRLS)~~ — **π=0.6 PASS restored, better than v1** (rftl power 0.925 ≈ clean 0.892, limit 58.1 ≈ clean 58.8) with final recall only 0.26: IRLS's continuous partial downweighting cleans the subspace without flagging every sample. **π=0.3 still stuck** (power 0.587) — per-repeat analysis (`output/pilot_monitoring_v3_irls.csv`) shows bimodality: 1/5 repeats detection fires (rec 1.0) → perfect recovery; 4/5 the rank-[5,5,6] subspace absorbs the 5-sample artifact entirely (rec 0) → nothing to reweight. Root cause: capacity — 150 core dims fit from ~15 samples/user.
- [x] ~~Monitoring pilot v4 (rank [3,3,4] + IRLS)~~ — **backfired for detection; v3 config stands.** Clean users' FAR improved (0.45→0.17, confirming the overfitting diagnosis) but detection died everywhere (recall ≤ 0.04, RFTL ≡ baseline in every cell): at low rank the unexplained-variance noise floor is so high that artifact residuals no longer stand out from the federated MAD at k=3. Rank trade-off identified: high rank absorbs the artifact (invisible), low rank drowns it in residual noise (invisible). [5,5,6]+IRLS is the sweet spot found so far — full recovery at π=0.6, partial (1/5 repeats) at π=0.3. `output/pilot_monitoring_v4_rank334.csv`.

**Design fork for Madi (2026-07-09).** Options for the π=0.3 (low-fraction correlated artifact) blindness:
  - **(A) Accept v3 design**, set the HPC grid in the demonstrated-working regime (hot_block, π ≥ ~0.5, or bigger artifacts at lower π), document the absorption boundary honestly as a limitation, and fix FAR reporting via ROC-based operating points (power at matched FAR) rather than more rank tuning.
  - **(B) Prototype leave-one-user-out detection** (~1 pilot run): score user m's samples by their residuals under the subspace fitted on the *other* users only. An artifact absorbed into the global fit is NOT in the other users' consensus subspace, so it stands out regardless of absorption — attacks the root cause, is federated-native (cross-site validation), and would be a methodological contribution beyond the drafted RFTL-S. Adds a method component to the chapter.
  - Recommendation: (B) is cheap to test and mechanistically sound; if it works, π=0.3 becomes recoverable and the method story strengthens. But it extends the methodology, so Madi's call.

**Madi chose (B).** v5 launched (2026-07-09): leave-one-user-out detection — each user's train samples scored by residuals under the consensus subspace of the *other* users (implemented via MPCA_FD_Weighted with uniform ~0 weights for the held-out user: removes it from global U while its local V still adapts — uniform scaling leaves the local eigenvector problem unchanged). LOO residuals standardized per user (median/MAD within user; different fits + different crop-size scales). Rank back to [5,5,6] so v5 differs from v3 only in the detection mechanism. One-step from LOO weights (no IRLS — if LOO detection is accurate, one weighted re-fit suffices). Output → `output/pilot_monitoring_v5_loo.csv`.

### Monitoring pilot v5 result (2026-07-09): LOO helps detection but two flaws found — and the artifact itself is part of the problem

`output/pilot_monitoring_v5_loo.csv`:

- π=0.3: recall 0.30→**0.527** (LOO works as intended) but recovery barely moved (power 0.569) — flagged samples get only partial Huber weights because even LOO residuals of these artifacts are moderate, not gross.
- π=0.6: prec/rec **0.0** yet "PASS" (power 0.942) — **v5 design error**: within-user median/MAD standardization breaks down when the majority of a user's samples are contaminated (the median centers on the artifact cluster → flags the clean minority); the chart recovered by accidental majority-absorption, not by design.
- **Root-cause realization: user 0 is the 10×10 crop, so "block 9" saturates 81% of its image** (earlier "18% of 21×21" note was wrong — `prepare_data` crops per user before contamination). A near-constant frame is mostly DC component, which any smooth image subspace (including the LOO consensus) reconstructs well → residual detectability is intrinsically capped. The artifact is *too big to see*.

**v6 launched (same session):** (a) trimmed-core standardization — median/MAD of the lowest 40% of each user's LOO residuals (clean core survives up to ~60% contamination), fixing the v5 breakdown; (b) block-size axis {9, 5} × π {0.3, 0.6} to test the DC-absorption hypothesis (block 5 = 25% of user 0's image = localized, less absorbable, more physical as "hot-pixel cluster"); stripes dropped (consistently benign+invisible, documented). Output → `output/pilot_monitoring_v6_loo_core.csv`.

### Monitoring pilot v6 result + CONSOLIDATED DESIGN DECISION (2026-07-09)

v6 (`output/pilot_monitoring_v6_loo_core.csv`):

- **π=0.6 block 9: power 1.000 but FAR 0.94 — a false victory.** Trimmed-core scale over-flags clean samples (prec 0.19) → weighted limit over-shrinks (47 vs clean 59) → chart alarms on everything.
- **π=0.3 block 9: worse than baseline** (rftl limit 480 vs base 121): with no clean/dirty residual gap, the lowest-40% core has near-zero MAD → mass flagging everywhere (prec 0.014), degrading even clean users' charts.
- **Block 5: benign** (baseline power 0.92–0.95 ≈ clean) — the small block is absorbed harmlessly.

**Structural finding of the v1–v6 arc: damage and invisibility travel together.** Contamination hurts the chart only when the fit absorbs it as believed structure; what the fit absorbs is exactly what residual detection can't see. Small artifacts: visible but harmless. Large correlated artifacts: harmful but invisible — except at high π, where IRLS pries them out over rounds. This is a chapter-level insight, not a pilot failure.

**FINAL MONITORING DESIGN: v3 = rank [5,5,6], IRLS (5 rounds × 5-iter inner fits, converged final fit), pooled federated MAD, Huber k=3.** Only variant with well-calibrated full recovery (π=0.6: power 0.925, FAR ≈ clean, limit 58.1 ≈ clean 58.8). HPC grid: hot_block block 9, π ∈ {0.4, 0.5, 0.6, 0.8, 1.0} (working regime + boundary), π=0.3 reported as the documented breakdown boundary; LOO prototype written up as an ablation (recall ↑, scale estimation fragile, chart not better).
FAR calibration for chapter figures: report ROC operating points (power at matched FAR) alongside limit values (the masking mechanism evidence).

### RFTL-S placement result (2026-07-12, `output/testside_local/`): RFTL-S ≈ baseline, far from the oracle collapse — because detection blindness is *protective* for prediction

5 repeats, k=3.0, {hot_block, stripe} × π {0.3, 1.0}, train+test contamination of user 0. (Also fixed a latent Windows bug first: the pool in `rftl_s_real.py` had no initializer — worked on HPC via fork, crashed locally under spawn; now uses `_init_worker`, fork-safe.)

| Cell | Oracle | Baseline | RFTL-S k=3 |
|---|---|---|---|
| hot_block π=0.3 | 0.1698 | 0.0713 | **0.0669** (+6% vs base) |
| hot_block π=1.0 | 0.2670 | 0.0622 | **0.0618** (+1%) |
| stripe π=0.3 | 0.1145 | 0.0595 | 0.0768 (−29%, likely noise + false flags) |
| stripe π=1.0 | 0.1667 | 0.0710 | 0.0734 (−3%) |

Detection prec/rec ≈ 0 everywhere (converged one-step fit absorbs the artifacts — same as monitoring v1), so weights ≈ 1 and RFTL-S ≈ baseline nearly by construction. **The punchline: the same absorption that blinds detection is what makes the contamination harmless to prediction — so blind robustness = no harm, while the all-seeing oracle is catastrophic.** "Damage and invisibility travel together" holds for prediction too, with opposite sign to monitoring.

**⚠ Design tension for the chapter (IRLS is NOT unconditionally better):** folding IRLS into the prediction pipeline would improve detection → cleaner subspace + discarded contaminated samples → push RFTL-S *toward the collapsed oracle* under persistent test-side faults. Task-dependent conclusion: monitoring needs IRLS (pry the artifact out of the subspace); prediction under persistent faults is protected by keep-and-downweight/blindness. Proposal: include both estimator variants (one-step + IRLS) in the prediction HPC run as the ablation that demonstrates this tension empirically.

### HPC package built and smoke-tested (2026-07-12/13): ready to submit

Two production scripts, both checkpointed/resumable and using a fork-safe `_init_worker` pool initializer (fixes the Windows-only crash found during the placement run; harmless under HPC's fork-based multiprocessing):

- **`code/rftl_s_real.py`** (prediction). Full grid restored: {hot_block, stripe} × π_S {0.3, 0.6, 1.0}, Huber k ∈ {1.345, 3.0} (June run showed k doesn't move MAPE, so dropped {1.0, 2.0}). Four arms per condition: clean / oracle / baseline / **rftl_s (one-step)** / **rftl_s_irls (5-round IRLS ablation, new)** — the ablation exists specifically to demonstrate the task-dependence finding from the placement run (IRLS helps monitoring, risks pushing prediction toward the oracle collapse). Smoke-tested end-to-end (reduced config): all 9 result keys present, correct fields, CSV round-trips. `run_job_prediction.sh` → fresh `output_testside/` dir (June Gaussian checkpoints are schema-incompatible, must not reuse).
- **`code/monitoring_real.py`** (new, production version of the pilot_monitoring.py arc). Final design from pilot v3: rank [5,5,6], IRLS (5 rounds × 5-iter inner fits + converged final fit), pooled federated MAD, Huber k=3. Grid: hot_block π ∈ {0.3, 0.4, 0.5, 0.6, 0.8, 1.0} (0.3 = documented breakdown boundary, rest = working regime) + one stripe π=0.6 condition (documented absorbed-artifact limitation case). Smoke-tested: correct method sequence, all chart fields present per user, checkpoint/resume round-trips correctly. `run_job_monitoring.sh` → fresh `output_monitoring/` dir.

**Deferred, not blocking this HPC submission:**
- Chart FAR calibration (~0.5 on clean data) — affects all methods equally so contrasts are valid; will report ROC operating points (power at matched FAR) in the write-up rather than re-tuning before this run.
- RFTL-U / RFTL-21 into either pipeline — separate follow-on, not part of this submission.

### What's next

- [x] ~~RFTL-S placement run~~ (≈ baseline; see above)
- [x] ~~Build + smoke-test HPC package~~ (prediction + monitoring scripts, job files — see above)
- [x] ~~Submit `run_job_prediction.sh` and `run_job_monitoring.sh` to Hazel~~ (2026-07-13, both running — 50 repeats each)
- [ ] Fix chart FAR calibration for chapter-grade figures (post-HPC, doesn't block submission)
- [ ] Add RFTL-U and RFTL-21 to both pipelines for full method comparison (separate follow-on)

---

## 2026-07-13: Session 8 — Strategic reassessment while HPC jobs run: two parallel directions opened

### Context for the pivot

Both HPC jobs (prediction, monitoring) are submitted and running (50 repeats each). While waiting, reassessed the chapter's trajectory with Madi: every positive robustness result so far has needed real engineering to surface (Gaussian → null; train-only structured artifacts → null; only test-side contamination broke the baseline; the monitoring win only appears in a narrow regime after 6 pilot iterations, with caveats — FAR miscalibration, detection blindness at low π, AIC test-leakage). The results aren't wrong, but the narrative is fragile as the spine of a chapter. Discussed alternative/complementary directions that reuse the existing infrastructure (MPCA_FD, federated V-local/U-global split, `federated_mad`/`huber_weights`, Tucker regression, IR bearing dataset) rather than starting over.

**Decision: pursue two directions in parallel**, not as a full abandonment of the current work (the HPC results still matter and will be used either way) but as reframings/extensions that make the chapter more defensible:

1. **Byzantine-robust federated aggregation.** Reframe RFTL-S's federated-MAD + Huber-weighting machinery as a Byzantine-robust aggregation rule for federated tensor factorization, positioned against the Byzantine-robust FL literature (Krum, trimmed mean, coordinate-wise median, geometric median). Key move: the "small-fraction, spatially-correlated artifact evades residual-based detection" finding from the monitoring pilot arc (v1–v6) maps onto a *known open problem* in that literature — robust aggregators are provably vulnerable to small-scale/structured ("stealthy") attacks. This turns our hardest empirical finding from "our detector has a gap" into "we replicate a recognized hard regime in a new domain (tensor-valued federated prognostics)." Starting here first.
2. **Non-IID / statistical heterogeneity.** Reuse the same federated V-local/U-global architecture to study cross-site heterogeneity (different degradation regimes per user) instead of adversarial contamination. Revives "regime mixing," which was rejected as a contamination model specifically *because* it's undetectable in-subspace — that's exactly the right property for a heterogeneity study (no attacker to detect, just legitimate cross-client distribution shift). Standard, citable FL problem (FedProx-style literature); low technical risk since the architecture already supports it.

Both directions are logged now so that if the session breaks, the next one picks up knowing we are pursuing (1) and (2), not just monitoring the HPC queue.

### Direction 1 foundation built (2026-08-15/16): `byzantine_agg.py` + `mpca_byzantine.py` + `pilot_byzantine_agg.py`

Built the reframing's core pieces:

- **`code/byzantine_agg.py`** — aggregation rules operating on a list of n per-client matrices: `aggregate_sum` (non-robust baseline), `aggregate_coordinate_median`, `aggregate_trimmed_mean(n_trim)` (raises if n≤2·n_trim; at n=3,n_trim=1 it's mathematically identical to median — verified in smoke test), `aggregate_krum(f)` (raises if n<2f+3, so **not usable at our n=3, f=1** — needs n≥5). Citations for these methods are well-known but not yet pulled/verified — flagged, not blocking.
- **`code/mpca_byzantine.py`** — `MPCA_FD_Robust`: same local-V math as `MPCA_FD` (nothing to aggregate there — purely per-client), but makes the global-U aggregation step explicit: each client's local scatter-matrix contribution to each mode's shared subspace is computed separately, then combined via a pluggable `agg_fn` before eigendecomposition (batch eigendecomposition of the aggregated scatter, replacing the incremental-SVD algorithm `MPCA_FD` uses for the same target — see docstring for the math correspondence).
- **`code/pilot_byzantine_agg.py`** — real-data foundation pilot (reuses `prepare_data`/`load_data`/`contaminate` from the existing pipeline). Two checks: (1) sanity — does `agg_fn=sum` recover the same subspace as existing `MPCA_FD`; (2) robustness — does `median` resist a contaminated user 0 better than `sum`, using a clean reference matched to each rule (not one shared reference — see bug note below).

**Bug fixed during smoke-testing:** `rob.agg == agg_name` in the summary printer silently compared a bound `DataFrame.agg()` method to a string (always False) instead of filtering the `'agg'` column — `'agg'` collides with pandas' built-in method name. Root-caused via direct CSV inspection (data was fine, filter was broken) rather than assuming the fit itself was NaN. Fixed to `rob['agg']`.

**Design bug also fixed:** the first pilot version compared both `sum`-dirty and `median`-dirty fits against a single *median*-based clean reference, conflating contamination damage with the baseline difference between aggregation rules on clean (non-IID) data. Fixed to give each rule its own matched clean reference, plus a dedicated `sum-clean vs median-clean` check to measure that baseline difference directly.

### Foundation pilot result (5 repeats, `output/pilot_byzantine_agg.csv`): median aggregation is WORSE than sum here, and low n / non-IID clients is why

| Check | Result |
|---|---|
| 1: sum-agg vs existing `MPCA_FD` (clean data) | 2.60° ± 5.81° — small, confirms the batch-eigendecomposition reformulation targets the same subspace as the incremental-SVD original |
| 1b: sum-clean vs median-clean (**no attack at all**) | **20.73° ± 6.95°** — large. Median aggregation is already substantially biased relative to sum on clean data, purely from cross-user heterogeneity (different crop sizes, different degradation regimes) |
| 2: sum-dirty vs sum-clean, hot_block π=0.6/1.0 | **1.89° at both π levels** — essentially flat, contamination barely moves the sum-aggregated subspace |
| 2: median-dirty vs median-clean, hot_block π=0.6/1.0 | **38.19° → 60.61°**, worsening with π — median moves far more under the same attack |

**Reading:** this is the opposite of the naive expectation, and it's explainable, not a bug. Two compounding mechanisms:
1. Sum's near-flatness is very likely the *same absorption mechanism* documented throughout the monitoring pilot arc (v1–v6): a large, fixed, repeated (low-rank) artifact tends to get its own dedicated eigenvector slot in a summed scatter matrix rather than rotating the other top components — protective for the "top" subspace, at the cost of the artifact being invisible there (consistent with everything already found about this data/pipeline).
2. Median's large, worsening angle is very likely the well-documented weakness of *coordinate-wise* robust statistics: combining matrices entry-by-entry ignores cross-entry correlation, and can assemble an incoherent matrix that doesn't resemble any real client's structure — worse yet at **n=3**, where the "coordinate-wise middle" is a raw selection among only 3 values per entry (not a smoothed estimate), so heterogeneous non-IID clients (not just an attacker) can swing it substantially. This also means **Krum (needs n≥5) and genuinely-averaging trimmed-mean (needs n>2·n_trim with room to spare) aren't well-posed at our current n=3** — the classic FL robustness literature assumes far more clients than we have.

**This does not kill Direction 1** — it's exactly the kind of finding a foundation check should surface before betting a chapter section on it. It does mean the naive "swap in classic median/trimmed-mean baselines" plan needs a fork.

### Decision needed / proposed next steps (not yet chosen — flagged for Madi)

1. **Expand client count.** Split the 3 users into more synthetic sub-cohorts (n≥5, ideally more) — directly fixes the n=3 degeneracy for median/trimmed-mean/Krum, and is *also* exactly what Direction 2 (heterogeneity) needs anyway, so this is shared foundational work, not a detour. Needs checking that sub-cohorts still have enough samples per client for MPCA to fit sensibly (currently ~25–35 train samples/user).
2. **Try a distance-based (not coordinate-wise) robust rule** — e.g. geometric median (Weiszfeld iteration on the whole matrix) or an ad hoc Krum-at-n=3 (pick the single most "central" whole client contribution, clearly caveated as lacking Krum's formal guarantee at this n). Distance-based rules preserve each client's internal coordinate correlations instead of shredding them — directly targets mechanism 2 above. Cheap to add to `byzantine_agg.py`.
3. **Report the finding itself as a contribution** regardless of 1/2 — "naive coordinate-wise robust aggregation underperforms plain averaging under small-n, non-IID federated tensor learning, unlike the large-n/near-IID regime its literature guarantees assume" is a legitimate, citable point for the discussion section, and it directly motivates Direction 2 (the root cause is non-IID clients).

### Decision: both fixes at once (Madi, 2026-08-16) — n≥8 synthetic clients, coordinate-wise vs distance-based head-to-head

Chose to pursue both proposed next steps together rather than sequentially: expand client count to a minimum of 8, and add distance-based aggregation rules to compare directly against the coordinate-wise family.

**`code/byzantine_agg.py` extended:**
- `aggregate_geometric_median` — Weiszfeld iteration; treats each client's matrix as one point in R^(rows·cols) and returns a distance-weighted average, preserving cross-entry correlation (unlike coordinate-wise median's independent per-entry pick).
- `aggregate_multi_krum` — averages the `m` (default n−f) lowest-Krum-score clients instead of keeping only one; `_krum_scores` factored out and shared with `aggregate_krum`.
- `AGGREGATORS` dict now has 6 entries: `sum`, `median`, `trimmed_mean`, `krum`, `multi_krum`, `geometric_median`.

**`code/pilot_byzantine_agg.py` rewritten (v2) for n≥8:**
- `build_synthetic_clients()` splits each real user (A/B/C) into sub-cohorts: split plan `[('A',2),('B',3),('C',3)]` → **8 synthetic clients**, sizes 16–23 samples each (using each user's full sample pool, not just the train split, since this pilot needs no held-out test). Explicitly flagged as synthetic (same underlying sensor/site data, randomly subsampled) — not independent physical sites, but exactly the expansion Direction 2 will also need.
- One client (`A0`) is the sole Byzantine target (`F_BYZANTINE=1`), now satisfying Krum/Multi-Krum's `n≥2f+3=5` requirement (8≥5 ✓) and giving trimmed-mean genuine slack (drops 1-of-8 per side, averages the remaining 6 — no longer degenerate to median as it was at n=3).
- Same matched-clean-reference design as the n=3 fix: every rule gets its own clean fit; heterogeneity (attack-free rule vs sum) and robustness (rule's dirty vs that rule's own clean) are reported separately per rule, coordinate-wise and distance-based grouped for direct comparison.
- Smoke-tested (3 iterations): 8 clients build correctly, no NaN, and the pattern already looks structurally sane — median/trimmed-mean far closer to sum than at n=3 (5.3°/1.5° heterogeneity vs the earlier 20.7°), and Krum shows a large but *constant* angle across both π levels (22–28°, unmoved by how much the target client is contaminated) — consistent with Krum discarding 7 of 8 clients outright regardless of the attack, a known low-statistical-efficiency property of Krum, not a bug.
- Full run (5 repeats, 30 iterations, all 6 rules × 2 conditions) launched 2026-08-16.

### n=8, 6-rule pilot result (2026-08-16, 5 repeats, `output/pilot_byzantine_agg.csv`): trimmed_mean and multi_krum stand out; median stays unstable; the residual metric turned out to be nearly uninformative

**Headline numbers** (mean ± std across 5 repeats; robustness = hot_block, π=0.6, essentially identical at π=1.0 — see saturation note below):

| Rule | Heterogeneity (attack-free) | Robustness (dirty vs own clean) |
|---|---|---|
| sum | 0.00° ± 0.0004° | 1.07° ± 0.49° |
| median | 21.20° ± **19.44°** | 19.08° ± **20.27°** |
| trimmed_mean | **1.78° ± 0.47°** | **1.64° ± 0.93°** |
| geometric_median | 10.69° ± 4.06° | 15.34° ± 4.98° |
| krum | 30.27° ± 8.96° | 40.45° ± 10.34° |
| multi_krum | **4.23° ± 1.16°** | **4.48° ± 1.30°** |

**1. n=8 helped magnitude but not median's core problem.** Median's mean angle dropped from 38–61° (n=3) to ~19–21° (n=8) — real improvement — but its **standard deviation is nearly as large as its mean** in both checks (19.44°, 20.27°). Comparing the two: robustness angle (19.08°) is not meaningfully bigger than the attack-free heterogeneity angle (21.20°) — median isn't so much *reacting to the attack* as it is simply **noisy/high-variance across repeats regardless of whether there's an attacker**, driven by which random sub-sampling defines the 8 synthetic clients that repeat. Coordinate-wise median remains a poor choice at this scale, independent of the earlier n=3 degeneracy.

**2. trimmed_mean is the standout coordinate-wise rule at n=8** (finally genuinely differentiated from median — no longer degenerate as it was at n=3): small, *low-variance* deviation from sum both attack-free (1.78°±0.47°) and under attack (1.64°±0.93°) — it tracks sum's stability closely.

**3. Distance-based family is real and differentiated, not a monolith.** multi_krum performs about as well as trimmed_mean (4.2–4.5°, low variance) — the best distance-based rule. Plain krum is the *worst* performer of all six, both attack-free (30.27°) and under attack (40.45°) — expected: krum keeps only 1 of 8 clients and discards the rest, a well-known low-statistical-efficiency criticism of vanilla Krum in the literature, now reproduced empirically in a tensor-factorization setting. geometric_median sits in between (10.69° → 15.34°) — moderate baseline bias, but unlike median it shows a real, sensible marginal reaction *specifically attributable to the attack* (its robustness angle exceeds its heterogeneity angle by ~+4.6°, a cleaner "robust rule" signature than median's noisy, attack-independent behavior).

**4. Contamination saturates almost immediately** (all 6 rules show nearly identical angle at π=0.6 vs π=1.0, e.g. krum 40.45°/40.45°, sum 1.07°/1.02°) — with a fixed, repeated artifact pattern, corrupting 60% vs 100% of one client's ~22 samples barely differs, since the client's local scatter matrix is already dominated by the repeated pattern at π=0.6. Consistent with the contamination model design, not a bug.

**5. Sum's near-total insensitivity (1.0–1.07°) continues to look like the same absorption mechanism from the monitoring pilot arc (v1–v6)** — a large, low-rank, repeated artifact tends to get its own eigenvector slot rather than distorting the rest of the subspace. None of the 6 rules show a dramatic order-of-magnitude blow-up specifically attributable to *this* attack (on top of their own baseline heterogeneity noise) — raising the honest question of whether this fixed-pattern contamination is simply benign for subspace estimation broadly (mirroring the now-repeated project finding), regardless of aggregation rule. A harder, adversarially-optimized attack (aligned with the top eigendirections rather than pushed into a spare one — closer to the "worst-case Byzantine" the FL literature usually assumes) would be a natural stress test before concluding any rule is "robust enough."

**6. `clean_client_residual` turned out to be nearly uninformative — verified as real, not a bug.** Within one repeat/condition, residual values across all 6 wildly-different aggregation rules (sum vs krum, which discard 7/8 clients!) differ by only ~7×10⁻¹¹ relative. Likely mechanism: local V (fit per-client to explain that client's *own* data) can almost fully compensate for whatever rotation the shared global U has — a gauge-freedom effect in the bilinear (V, U) factorization, where reconstruction quality on a client's own data is far more identifiable than U's specific orientation alone. **Principal angle, not reconstruction residual, is the metric with real signal for comparing aggregation rules** — worth remembering if a downstream task (e.g. a future monitoring-style experiment for this direction) is built on top of this.

### Harder "hijack" attack built + a real methodological trap found and fixed (2026-08-16)

**Madi chose the harder-attack path** over pivoting straight to Direction 2. Rationale: none of the 6 rules showed damage clearly beyond their own baseline noise under the physically-motivated hot_block attack, so it's unclear whether Direction 1's comparison means anything yet — cheap to resolve given the infrastructure already built.

**New code:**
- `code/byzantine_attacks.py` — `hijack_attack(amplitude_mult, target)`: an *arbitrary* (not image-derived) malicious scatter-matrix injection. Eigendecomposes the honest clients' aggregated scatter, targets the honest data's own **weakest** eigendirection (or a random direction, as a sanity check), and injects a rank-1 malicious contribution there scaled to `amplitude_mult × honest top eigenvalue`. Classic "single large outlier hijacks unweighted averaging" attack, implemented directly in scatter-matrix space — bypasses the image-contamination pipeline that kept producing benign, absorbable artifacts.
- `code/mpca_byzantine.py` — `MPCA_FD_Robust` extended with `adversarial_client`/`adversarial_fn`: one client's per-mode scatter contribution can now be overridden with an arbitrary malicious matrix at the aggregation point, rather than derived from (even corrupted) real data — a more accurate implementation of "Byzantine" than the faulty-camera framing.
- `code/pilot_byzantine_hijack.py` — the stress-test pilot: same 8-client split, 6 rules, amplitude sweep {2×, 5×, 20×}.

**A major investigation, resolved — not a bug, a real evaluation-design trap.** First run showed sum hijacked to ~90° (as expected) but ALSO geometric_median hijacked almost as badly (~85–90°), contradicting geometric median's textbook 1/2 breakdown point. Root-caused via direct instrumentation (captured the exact matrices passed to `agg_fn` inside live `_aggregated_U`/`_initial_U` calls): **the aggregation step itself was correctly rejecting the malicious contribution at every single call** (≤0.1° from the 7 real honest clients' own aggregate, verified by capturing and replaying the exact live inputs). The apparent "hijack" was actually mostly something else: comparing a *defended* fit (which necessarily excludes the target client's real information once the defense works) against a clean reference that *includes* that client's real data conflates two different things — attack rejection vs. the unavoidable cost of losing a real, non-IID client's unique signal. Confirmed decisively with a no-attack control: **just removing client 0's real data (zero attacker) shifts geometric_median's converged fit by 33.87° vs sum's 1.25°** — geometric_median (and krum) have real, attack-independent optimization instability in this non-convex alternating (V,U) setting, from client-count/composition sensitivity alone.

**Fix:** every comparison now reports three numbers — (a) `angle_vs_loo_clean`: dirty fit vs. a clean fit trained on the 7 *other* clients only (target absent entirely) — the metric that actually isolates attack success; (b) `angle_vs_full_clean`: dirty fit vs. all-8-clients clean (total real-world cost, attack + exclusion); (c) a **no-attack baseline** — the SAME LOO comparison with zero attacker, giving each rule's own natural noise floor to judge the attack's *marginal* effect against.

**Smoke-test result (baseline-corrected residual attack effect, LOO metric − no-attack baseline):** sum ~88° (catastrophic, as predicted) · multi_krum +2–3° · trimmed_mean +1.5–2.5° · median +3–4° · krum +21° · geometric_median +~53° (worst of the five robust rules once corrected — contrary to naive expectation from its theoretical guarantee, likely because its higher baseline instability compounds with the attack across 30 rounds of the coupled optimization, not because any single aggregation call fails). Full 5-repeat run launched.

### Full hijack pilot result (5 repeats, `output/pilot_byzantine_hijack.csv`): multi_krum is the clear winner; geometric_median is the surprise loser

Baseline-corrected residual attack effect (LOO angle at amplitude=20× minus each rule's own no-attack baseline), 'weakest'-direction target:

| Rule | Baseline (no attack) | Attacked (LOO) | Residual attack effect |
|---|---|---|---|
| sum | 0.97° | 89.64° | **+88.66°** (catastrophic, as predicted) |
| median | 30.44° | 27.49° | −2.94° (noise-level; but baseline itself is huge — see caveat) |
| trimmed_mean | 1.58° | 4.89° | **+3.31°** |
| geometric_median | 18.55° | 75.53° | **+56.98°** (worst of the 5 robust rules) |
| krum | 19.51° | 33.61° | +14.09° |
| **multi_krum** | 1.27° | 4.22° | **+2.94°** (best) |

**multi_krum wins decisively and consistently.** Smallest residual effect, smallest baseline noise, and — critically — this held up under the **random-direction sanity check too** (4.22°±0.98° at amplitude 20×, matching the 'weakest'-direction result almost exactly): its robustness isn't an artifact of the specific attack direction tested.

**median's near-zero residual is a false positive, not real robustness** — its no-attack baseline (30.44°) is already so large (high inherent variance in this small-n, non-IID setting — consistent with the Direction-1 v1/v2 finding) that this specific attack simply doesn't add much on top. Confirmed by the random-direction check: 49.00°±**33.16°** — enormous variance, unusable as a robust rule regardless of attack presence.

**trimmed_mean is good against the targeted attack but fragile against a random direction**: residual shrinks to +3.31° at high amplitude under 'weakest' targeting, but explodes to 48.99°±**31.19°** under random-direction targeting at the same amplitude — direction-dependent, inconsistent robustness. A real weakness worth reporting honestly, not hiding.

**geometric_median is the standout disappointment** — the largest residual effect of any robust rule (+56.98°), consistent across both 'weakest' (75.53°) and random (79.07°±4.34°, low variance — so consistently bad, not noisy-bad) attack directions. This contradicts geometric median's textbook 1/2 breakdown-point guarantee, which is a **single-shot/static** guarantee — the investigation above showed the aggregation step itself correctly rejects the malicious contribution at every individual call (≤0.1° from the true honest aggregate). The likely mechanism: geometric_median's higher inherent optimization instability in this non-convex, iteratively-coupled (V,U) setting (33.87° baseline just from excluding one client) compounds over 30 rounds with an attacker that gets to adaptively retarget every round — a multi-round vulnerability the classical single-shot Byzantine-robustness literature doesn't cover. Worth stating explicitly as a finding: **theoretical single-aggregation-step robustness guarantees do not automatically transfer to iterative, coupled non-convex factorization settings.**

**krum: moderate, consistent residual** (+14.09°, same ~33.6° absolute value regardless of amplitude 2×/5×/20× or attack direction) — a real but bounded vulnerability, better than geometric_median, worse than multi_krum/trimmed_mean.

### Direction 1 headline result: **multi_krum is the carried-forward robust aggregation rule.** Across both stress tests run (physically-motivated hot_block artifacts AND the literature-standard arbitrary hijack attack, at multiple amplitudes and two attack directions), multi_krum is the only rule that is simultaneously low-baseline-noise, low-residual-attack-effect, and consistent across attack directions.

### Direction 1 write-up consolidated (2026-08-16)

Per Madi's request, the full Direction 1 narrative (motivation, n=3 degeneracy, n=8 fix, the hot_block/hijack experiments, the LOO-metric methodological fix, final results and recommendation) has been consolidated out of this chronological log into a single, self-contained document: **`direction1_byzantine_aggregation_findings.md`**. That document is now the primary reference for Direction 1's findings; this log retains the full session-by-session detail for provenance but new sessions should start from the consolidated write-up. `chapter3_draft.md` §3.6 was also updated with pointers into it (RFTL-U reconciliation flagged as an open scope question, citation-verification task noted).

### Direction 2 started (2026-08-16): does local V protect the shared subspace from non-IID heterogeneity?

**Design decided with Madi:** rather than reusing the n=8 A/B/C-derived split directly (which conflates degradation-regime heterogeneity with the sensor/crop-size heterogeneity already present between the real users), built a cleaner, isolated heterogeneity study: a single crop size (`prepare_data(..., 'C')`, 20×20) applied to the **full 284-sample pool**, partitioned into n=8 synthetic clients via **Dirichlet(α) allocation over TTF-tertile degradation-regime labels** — the standard non-IID FL benchmark construction (Hsu, Qi & Brown 2019; citation to verify). Low α → each client dominated by one regime (highly non-IID); high α → balanced/near-IID. This isolates regime heterogeneity as the only variable.

**Core comparison — the actual research question:** does the existing federated architecture (local `V` per client, shared `U`) protect the shared subspace from heterogeneity degradation better than a "no personalization" alternative (single shared `V` fit once across pooled data)? This is the standard personalized-FL hypothesis (local layers absorb client-specific structure, protecting a shared layer) applied to federated tensor factorization for the first time here.

**New code:**
- `code/heterogeneity.py` — `ttf_regime_labels` (TTF-tertile regime assignment), `dirichlet_partition` (with a retry-guarded `min_client_size` — a naive draw at α=0.1/n=8 produced a client with **zero samples**, which would break `Vinitial`'s eigendecomposition; fixed by resampling until every client has ≥5 samples, raising if infeasible after 50 attempts rather than silently proceeding with a degenerate client). Feasibility check: α ≥ ~0.15 reliably succeeds at n=8; swept α ∈ {0.2, 0.5, 1.0, 3.0, 10.0, 100.0}.
- `code/mpca_byzantine.py` — `MPCA_FD_Robust.train()` extended with `shared_v` (default False): when True, `V` is fit once on pooled data across all clients and applied identically to every client — the "no personalization" ablation. Smoke-tested: per-client-V path unchanged; shared-V path gives identical `V` across clients; at n=1 (no client boundaries), `shared_v=True` and `shared_v=False` reduce to the same subspace (principal angle ~10⁻⁶°, verified after correcting for eigenvector sign/rotation ambiguity within a degenerate eigenspace — not a bug).
- `code/pilot_heterogeneity.py` — compares `federated` (`shared_v=False`) vs `no_personalization` (`shared_v=True`) against a pooled reference (n=1, all 284 samples, no client boundaries — the "no privacy constraint" upper bound), via principal angle, across the α sweep.

**Smoke test (3 iterations, α ∈ {0.2, 1.0, 100.0}):** structurally clean (no NaN, size guard respected), and already directionally promising — `federated` beat `no_personalization` (smaller angle vs. pooled reference) at **every** α level tested, by a consistent ~5–7°. The α-trend itself (does the gap widen as heterogeneity increases) needs full iteration count to resolve — too noisy at 3 iterations to read. Full run (5 repeats, 30 iterations) launched.

### Full heterogeneity pilot result (5 repeats, `output/pilot_heterogeneity.csv`): the original hypothesis (gap widens with heterogeneity) did NOT hold — but a different, real finding emerged instead

Mean angle vs. pooled reference, by α (0.2 = most heterogeneous, 100 = near-IID):

| α | federated | no_personalization | gap |
|---|---|---|---|
| 0.2 | 8.81 | 11.39 | +2.58 |
| 0.5 | 9.60 | 7.69 | −1.91 |
| 1.0 | 9.84 | 12.12 | +2.28 |
| 3.0 | 8.30 | 13.00 | +4.71 |
| 10.0 | 9.75 | 9.28 | −0.46 |
| 100.0 | 8.67 | 13.86 | +5.19 |

**The gap does not widen monotonically as heterogeneity increases — if anything the largest federated wins are at the near-IID end (α=3, 100), the opposite of the original hypothesis.** Mean-only, this looked like a null/failed result.

**But the variance tells a clearer and more interesting story.** `federated`'s std is tight and stable at every α (0.63°–1.65°). `no_personalization`'s std is wildly unstable (1.0°–7.5°) and **bimodal**: three separate (repeat, α) combinations — at α=0.5 (×2) and α=10.0, different repeats and different α levels, ruling out a single artifact seed — landed at angle ≈0.000002–0.000003° (essentially exact agreement with the pooled reference), while other repeats at the *same* α landed 12–16° away. **Revised finding: local V gives a consistently stable, low-variance global subspace estimate regardless of client heterogeneity level; the no-personalization ablation is not just more biased on average, it's fundamentally less predictable — sometimes coincidentally near-perfect, sometimes far off, in the same non-convex optimization.**

This connects directly to a cross-cutting theme first seen in Direction 1: components that retain more per-client-specific structure during fitting (there: full per-client scatter contributions vs. Krum's single-client selection; here: local V vs. a single pooled V) tend to produce more stable, lower-variance convergence in this non-convex alternating (V,U) optimization — a generalizable point spanning both directions, not specific to either.

**A second candidate explanation for the flat mean-bias trend, worth testing:** TTF-regime heterogeneity may be structurally *in-subspace* — i.e., different degradation stages of the same bearing may still share the same dominant low-rank directions of image variation, varying smoothly within that subspace rather than requiring a different one. This is the exact mechanism that got "regime mixing" rejected as a contamination model back in Session 6 ("swapped samples still lie in the shared low-rank subspace, so residuals stay small — same blindness"). If true, TTF-regime partitioning may simply be a weak stressor for this pipeline regardless of architecture — a third instance of the "absorption" theme running through this chapter's whole empirical arc (Gaussian noise, train-only structured artifacts, and now regime heterogeneity all fail to perturb the directions the fit actually uses).

### HPC prediction run confirmed (2026-08-16): job 9370, 50 repeats, 5.39 hours — the oracle-collapse finding holds at full statistical power

Pulled from `origin/main` (Madi ran/pushed from Hazel). `output_testside/rftl_s_real_results.csv` (4250 rows) + `output_file.j9370` (LSF log) — job completed successfully, full grid: {hot_block, stripe} × π_S {0.3, 0.6, 1.0} × Huber k {1.345, 3.0}, clean/oracle/baseline/rftl_s(one-step)/rftl_s_irls.

**Clean reference: MAPE 0.0689 ± 0.0165** (median 0.0563, user0 0.0780) — matches the 5-repeat pilot closely.

**1. Oracle collapse confirmed decisively, and it's worse at n=50 than the pilot suggested.** Baseline beats oracle by **43.4% to 75.9%** across all six (mode, π) cells — e.g. hot_block π=1.0: oracle MAPE 0.2626±0.1822 (user0 0.6289 — 8× clean) vs. baseline 0.0632±0.0120 (essentially at clean). This is no longer pilot-level evidence — it's the headline finding of the whole prediction-side story at proper power: an idealized robust estimator that discards contamination gets catastrophically miscalibrated when the fault persists at deployment, while doing nothing self-calibrates fine.

**2. Baseline stays within ~3% of clean at every single condition** (0.0609–0.0703 vs. clean's 0.0689) — the absorption/self-calibration mechanism holds across the full grid, not just the pilot cells.

**3. RFTL-S/IRLS costs essentially nothing relative to baseline** — MAPE deltas range −5.7% to +5.4% across all conditions/k values, mostly within noise given per-condition std of 0.01–0.03. Robustness doesn't meaningfully help prediction here (as established), but critically it doesn't hurt either.

**4. A striking, 100%-consistent detection null: rftl_s (one-step) recall for stripe at k=3.0 is *exactly* 0.000000 — every one of 50 repeats, all three π levels.** (Verified directly against the raw CSV, not just the printed 3-decimal summary — genuinely exact zero, not a rounding artifact.) Precision is undefined/0 alongside it (nothing ever flagged). This is a clean, decisive, fully-powered confirmation of the stripe-invisibility theme already established in the monitoring pilot arc — at the conservative k=3.0 threshold, one-step residual detection never once catches a stripe artifact, regardless of contamination fraction.

**5. IRLS partially rescues stripe detection at k=3.0** (recall 0.04→0.19→0.20 as π increases 0.3→0.6→1.0) — better than one-step's hard zero, but still weak. Multi-round reweighting helps some; doesn't fully fix the underlying absorption.

**6. k=1.345 is the practically better choice for this task, contradicting the "prefer the conservative k=3.0" assumption carried over from the monitoring work.** Detection is dramatically better at k=1.345 for both artifact types (hot_block recall 0.94–1.00 vs. 0.06–0.17 at k=3.0; stripe recall 0.55–0.71 vs. ≈0 at k=3.0) with no meaningful MAPE cost either way. For prediction specifically — unlike monitoring, where a looser k traded detection for higher false-alarm rate on the control chart — there's no equivalent penalty visible here, so the looser threshold is close to a free win on detection quality.

**Relevance to the "is this worth a chapter" question:** this closes the main gap flagged earlier — finding #2 (naive robustness backfires under persistent deployment contamination) is now confirmed at n=50 across the full grid, not just a 5-repeat pilot. The monitoring HPC job (`run_job_monitoring.sh`) is still outstanding and would close the same gap for finding #3.

### HPC monitoring job crashed (2026-08-16): SVD non-convergence, root-caused and fixed

Madi reported a crash mid-run on Hazel: `numpy.linalg.LinAlgError: SVD did not converge`, raised inside `my_mpca_02_27_nomean.py::U()` (called from `monitoring_real.py::fit_mpca` — the **baseline** arm), which killed the entire `pool.map()` call and the whole job with it (one worker's exception propagates and terminates the pool).

**Root cause.** `U()` and `Uinitial()` (Chapter 2's original incremental-SVD implementation) both compute a new sample's orthogonal residual `W = sample - u@u.T@sample`, then normalize: `norm_w = W / np.linalg.norm(W, axis=0)`. If a sample's data already lies entirely within the subspace `u` has already seen (i.e. it's a near-duplicate of previously-processed data), the column norm of `W` is ≈0, and the division is 0/0 = NaN. That NaN gets folded into `u` via `np.hstack((u, norm_w)) @ u_prime`, poisoning every subsequent incremental-SVD call — which is exactly what surfaces downstream as "SVD did not converge" (LAPACK can't handle a NaN-contaminated matrix). The monitoring production grid is precisely where this triggers: `hot_block` at π=0.8/1.0 replaces most-or-all of a user's training samples with the *identical* fixed artifact (same block, same value, same location, every time) — once the incremental fit has absorbed that pattern from one contaminated sample, every subsequent duplicate contributes ≈0 new orthogonal information. Those high-π conditions were new to the production script's wider grid and were never exercised by the earlier local pilots (which only tested π ∈ {0.3, 0.6}) — a real gap in pre-submission testing.

**Confirms a prior, undocumented fix elsewhere in the codebase.** `rftl_s.py::MPCA_FD_Weighted` (the RFTL-S/IRLS estimator) already guards against exactly this (`norm_W[norm_W < 1e-10] = 1e-10`, from an earlier session) — the guard was never backported to the plain `MPCA_FD` baseline path in `my_mpca_02_27_nomean.py` that `monitoring_real.py`'s baseline arm still uses. This is why the RFTL-S/IRLS arms never crashed and only the baseline arm did.

**Fix applied (`code/my_mpca_02_27_nomean.py`, both `Uinitial()` and `U()`):** `norm_w = W / np.maximum(np.linalg.norm(W, axis=0), 1e-10)` — mathematically a strict no-op whenever the column norm exceeds 1e-10 (i.e. every previously-tested, well-conditioned case), and makes a near-duplicate sample contribute ~0 new basis direction (the correct behavior) instead of NaN. Verified: (a) an ordinary clean-data fit gives identical, NaN-free output after the fix; (b) searched the rest of the codebase for the same unguarded-division pattern — only `byzantine_attacks.py`'s random-direction draw has a bare `/np.linalg.norm`, but that's a random Gaussian vector with measure-zero collision probability, not a real risk.

**Also added defense-in-depth in `code/monitoring_real.py`:** wrapped each (repeat, condition) computation in its own try/except — a single pathological combination is now logged and skipped rather than crashing the entire 50-repeat job. Verified with a forced-failure test: one condition fails and is cleanly absent from that repeat's rows; the other 6 conditions and the checkpoint still complete normally. This is insurance against a *future* edge case beyond the one just fixed, not a substitute for the root-cause fix.

**Not yet confirmed:** a local reproduction sweep across all 50 production seeds × 7 conditions was still running (no crash found in the first repeat checked) when this was written — the root-cause diagnosis doesn't depend on reproducing it locally (LAPACK behavior can differ between the HPC's Linux BLAS and the local Windows conda env even for the same logical near-degeneracy), but worth checking the sweep's outcome for completeness.

### Monitoring HPC job succeeded after the fix — but the 5-repeat pilot's "recovery" finding does NOT hold at n=50 (2026-09-07)

Fix confirmed working: job 263309 completed cleanly (resumed the 49 repeats already checkpointed before the crash, ran the 1 remaining, no further SVD errors). Pulled `output_monitoring/monitoring_real_results.csv` (750 rows) and did a full per-repeat analysis — the console summary's 3-decimal means hid a much more important structural finding.

**Clean baseline (n=50):** power 0.923±0.096, limit 58.7±4.5, FAR 0.523±0.235 — FAR calibration issue persists as flagged, but the contrast machinery is still valid.

**Masking is confirmed, but recovery is much weaker than the pilot suggested, and the failure mode differs sharply by π.** At π=0.6 — the pilot's headline condition (previously reported power 0.925 vs clean 0.892, near-full recovery) — the n=50 average shows rftl_s power only 0.883 vs baseline 0.880, essentially no improvement, with the limit still at 137 vs clean's 58.7. **This does not replicate.** Across π ∈ {0.3, 0.4, 0.5, 0.8}, rftl_s beats baseline by only a few points of mean power (+0.01 to +0.06) — a far cry from "full recovery."

**The real finding, hiding behind those noisy means: hot_block detection is genuinely, near-perfectly BIMODAL at the repeat level, not a matter of degree.** Recall values are almost always exactly 0.0 or exactly 1.0 (checked directly, not an artifact of rounding): at π=0.3, 43/50 repeats get recall=0, 6/50 get recall=1, only 1/50 in between; at π=0.6, 42 zero / 8 one / **0 in between**; at π=1.0, 29 zero / 21 one / 0 in between. Detection either fully succeeds for a given repeat's random draw or fully fails — there is no "partial" detection regime. The success *rate* rises with π (≈12% at π=0.3 → ≈16% at π=0.5–0.6 → ≈42% at π=1.0), but even at total corruption, most repeats still fail to detect.

**Splitting by detection outcome reveals TWO qualitatively different mechanisms, not one:**
- **π ≤ 0.6, when detection succeeds:** RFTL-S genuinely helps, and by a lot. At π=0.6's "detected" subset (8/50 repeats): rftl power 0.846 vs baseline 0.714 (+0.132); rftl limit 133.6 vs baseline's **417.1** — a real, large recovery. The "missed" subset (42/50) shows rftl ≈ baseline, as expected when weights stay ≈1. So the underlying mechanism from the pilot is real — it just only fires in a small, unpredictable fraction of deployments.
- **π=1.0 (total corruption of user 0's training set): RFTL-S makes things WORSE, regardless of whether detection "succeeds."** Detected subset: rftl power 0.762 vs baseline 0.867 (−0.105); rftl limit 284.7 vs baseline's 161.4 (worse, not better). Missed subset: same direction, rftl power 0.793 vs baseline 0.862. **Likely mechanism:** at π=1.0 there are zero clean samples left for user 0 — every one of its training images carries the identical artifact. Huber weighting relative to the federated MAD then has no genuine clean/dirty distinction to exploit within that user; any residual variation it picks up is arbitrary rather than signal, so down-weighting an arbitrary subset of an already-uniformly-contaminated user actively degrades that user's effective local sample size and its contribution to the shared fit — a real, mechanistically-explainable boundary condition: **robust reweighting requires a genuine partial-corruption signal to exploit, and backfires once corruption is total.**

**Stripe (π=0.6) confirms the earlier, cleaner finding** — baseline is already near clean (power 0.889 vs clean's 0.923, limit 60.6 vs 58.7 — barely inflated), rftl_s doesn't move it (0.882), and unlike hot_block, variance is LOW (std 0.11) and recall is mostly zero (44/50) — stripe absorption is a consistent, low-variance phenomenon, not a bimodal one. Confirms stripes are reliably (not just occasionally) invisible/benign.

**Chapter-level implication — the "positive result" needs re-framing, not abandonment.** The story is no longer "robust re-estimation recovers monitoring performance." It's more precise and arguably more interesting: **detection of a structured, repeated artifact via residual-based robust reweighting is an unpredictable, bimodal phenomenon whose success probability scales with contamination severity — and even when it works, its benefit is conditional on partial (not total) corruption, with total corruption of a single client actively backfiring.** This is an honest, mechanistically-grounded, still-substantive finding, but it is NOT the clean "our method recovers performance" result the 5-repeat pilot implied, and the chapter text needs to reflect the real distribution, not the lucky mean.

### CRITICAL CORRECTION (2026-09-07): the n=50 monitoring analysis above is INVALID — 49/50 repeats used the pre-fix buggy code

Madi asked, correctly, whether the bimodal-detection finding could be an implementation artifact rather than real. Investigating that directly uncovered a serious problem with the run itself, not just the interpretation.

**Root cause:** job 263309 (the "successful" resubmission) resumed from checkpoints rather than running fresh — its own log said so plainly ("Resuming: 49 repeats already saved, 1 remaining"). Those 49 checkpoints were written by the *crashed* run, i.e. by the pre-fix code, before `7b3d45d` existed. Confirmed via git history: `output_monitoring/checkpoints/repeat_002_seed51809.json` (and the other 48) were introduced in the "cleaning" commit at 19:57; only `repeat_006_seed68031.json` — the one repeat actually recomputed — came from the later "monitoring_result" commit at 20:38. Resuming correctly avoided recomputing already-checkpointed repeats, exactly as designed — but checkpointing has no way to know a code fix invalidates old results, and resubmitting without clearing the checkpoint directory reused 49 repeats' worth of output from the very code just proven broken.

**This is worse than "some stale data crept in."** Directly reproduced repeat 2 (seed 51809, hot_block π=0.6) locally with the *current, fixed* code: baseline limit_user0 = 61.75 (clean-like). The CSV, from the stale checkpoint, reports 2523.27 for the identical repeat/seed/condition. Same input, different code version, wildly different answer — and the buggy run never raised an exception for this case. **The bug's silent failure mode (near-zero, not exactly-zero, residual norm) is likely far more common than its crashing failure mode**: LAPACK only raises `LinAlgError` when a matrix genuinely can't converge (essentially exact NaN/Inf); a merely-very-small norm still produces a huge but finite, non-NaN blowup that silently corrupts the subspace estimate with no error raised at all. A hard crash was the loud, rare case; quiet corruption like repeat 2's is probably the common one.

**Consequence: every number in the "bimodal detection" analysis above — the recall distribution, the catastrophic-masking subset, the π=1.0 backfire — is built on data where up to 49/50 repeats may be silently corrupted, not a real confirmation of anything.** The whole analysis needs to be treated as void pending a genuine fresh run.

**Action taken:** deleted `output_monitoring/checkpoints/` and `output_monitoring/monitoring_real_results.csv` entirely (not left for a future resume to reuse) and committed the deletion, so there is no way to accidentally resume from tainted checkpoints again. Full fresh 50-repeat run needed — not a resume — with the current fixed code.

**Open question, not yet resolved: did the PREDICTION job (9370) suffer the same silent corruption?** It ran as a single uninterrupted pass (no crash, no resume) entirely on the pre-fix code, and its baseline/oracle arms use the same vulnerable `MPCA_FD` path (`fit_all_ranks_mpca` → `my_mpca_02_27_nomean.py`) with the same high-π (up to 1.0) hot_block contamination that triggers near-duplicate samples. Never crashing does not rule out the same silent, non-crashing corruption found here — it only rules out the exact-zero case. Needs a systematic spot-check: reproduce a sample of `output_testside/checkpoints/*.json` repeats locally with the fixed code and compare against the committed baseline/oracle MAPE values.

### Job 9370 (prediction) spot-check result: NOT corrupted — the oracle-collapse finding stands (2026-09-07)

**First attempt (direct local reproduction) turned out to be an invalid method, discovered by testing it against itself.** Reproduced 6 repeats (seeds spanning the full range) of the highest-risk cell (baseline arm, hot_block π=1.0 — the same near-duplicate-sample scenario that broke monitoring) using the current fixed code: all 6 mismatched the HPC checkpoint, with the AIC-selected rank differing in 4/6 cases and the direction of the MAPE difference inconsistent (sometimes higher, sometimes lower). Before concluding corruption, tested the decisive control: extracted the exact pre-fix `my_mpca_02_27_nomean.py` from git history (`7b3d45d^`) into an isolated copy and reproduced the same 6 repeats with the *old, unfixed* code. **The old code did not reproduce the checkpoint either** (e.g. repeat 0: checkpoint 0.0768, old-code-local 0.0739, fixed-code-local 0.0510 — three different values from what should be two identical computations). This proves the mismatch is NOT attributable to the fix — it's genuine platform-level floating-point non-determinism (BLAS/LAPACK/threading differences between the HPC's Linux build and local Windows conda) compounding over 150 iterations of a sensitive non-convex alternating optimization. **Local reproduction is not a valid verification method for this codebase across platforms — noted for future reference, don't rely on it again.**

**Correct method: scan the actual HPC-computed data directly for the corruption's known signature.** The confirmed monitoring corruption was a ~40× blow-up in one specific direction (2523 vs. 61.75) — a statistically unmistakable spike, not normal noise. Scanned all 1,550 real MAPE values in `output_testside/rftl_s_real_results.csv` (clean/baseline/oracle/rftl_s/rftl_s_irls arms — every arm with a numeric mape, prec/rec rows correctly excluded as NaN-by-design) for any value exceeding its (mode, π_s[, k]) group's mean + 8σ: **zero outliers found anywhere.** Baseline (the vulnerable unweighted `MPCA_FD` path, same as monitoring's baseline) shows a tight, well-behaved distribution at every condition (std ~15–25% of mean, max ~1.5–2× mean). Oracle's wider spread is smooth and continuous, fully consistent with its already-documented, independently-derived miscalibration mechanism — not an unexplained anomaly.

**Verdict: job 9370's results are trustworthy. The oracle-collapse finding does NOT need to be re-run.** This makes mechanistic sense too: even if the near-duplicate-sample bug fired occasionally in the prediction pipeline's baseline/oracle fits, prediction's min-max→Ridge→Tucker regression stack has been shown throughout this chapter to absorb/regularize away extreme upstream feature values — the same buffering that makes prediction largely insensitive to real contamination would likely also dampen a numerical artifact, unlike monitoring's direct-residual metric which has no such buffer and amplified the bug into an obvious spike.

### Genuinely fresh monitoring run verified and analyzed (2026-09-12): job 275630's checkpoints are correct, corruption theme replicates, mechanism now understood

**Verifying freshness.** Job 275630's log said "Resuming: 50 repeats already saved, 0 remaining" and ran in 33 seconds — meaning the real computation happened in an earlier, unlogged submission (whose `output_file` was never pushed), and this job only re-packaged already-complete checkpoints. Confirmed those checkpoints are genuinely the fixed-code output, not a repeat of the earlier corruption: the 4 repeats previously proven corrupted (2, 8, 12, 47 — baseline limit 341–2523 at hot_block π=0.6) now all show normal values (54–62), matching what independent local reproduction with the fixed code gave for repeat 2 (61.75, exact match). Confirmed trustworthy.

**But new large outliers appear elsewhere (repeats 4, 17, 18, 20 — baseline limits up to 2770) — verified genuine, not a residual bug**, using the same isolated-vs-correlated diagnostic developed for job 9370: checked each flagged repeat's *own* clean arm (always normal, 54–60) and its *own* π-sweep (spikes are non-monotonic and scattered — e.g. repeat 17 spikes at π=0.5 and 0.8 but is normal at 0.3/0.4/0.6/1.0; different repeats spike at different π levels). A residual bug tied to contamination density would correlate with π; this doesn't. Consistent instead with a well-understood statistical mechanism: **the control limit (mean + 3σ of ~15–16 training residuals) is a non-robust dispersion estimate at a very small sample size, so a single high-leverage residual can dominate it and produce an order-of-magnitude swing** — clean data shows a mild version of this (max limit 207 vs typical ~59, ~3.5×) from pure sampling luck, while contamination amplifies it severely (2000–2800, ~35–45×) because contaminated samples are specifically the kind of high-residual point this estimator is fragile to.

**Full re-analysis on verified data (n=50):**

| pi_s | baseline power | rftl_s power | baseline limit (median) | rftl_s limit (median) |
|---|---|---|---|---|
| clean | 0.920±0.123 | — | 61.2±21.3 | — |
| 0.3 | 0.692±0.197 | 0.720±0.177 | 69.0 | 68.7 |
| 0.4 | 0.654±0.300 | 0.744±0.207 | 67.0 | 66.3 |
| 0.5 | 0.741±0.304 | 0.840±0.176 | 63.8 | 63.0 |
| 0.6 | 0.900±0.235 | 0.883±0.267 | 57.0 | 57.0 |
| 0.8 | 0.818±0.387 | 0.836±0.369 | 47.1 | 49.6 |
| 1.0 | 0.840±0.370 | 0.780±0.418 | 25.9 | 25.9 |

Medians (robust to the isolated spikes) show baseline's *typical* masking is mild — most repeats aren't severely masked at all, consistent with the "small minority of unlucky draws" mechanism above, not universal, guaranteed masking. Means are dominated by that minority.

**Detected-vs-missed split (rftl_s recall ≥ 0.5) replicates the earlier qualitative shape, though with different specific numbers (as expected, since the data itself changed):** at π=0.3, detected repeats (7/50) show real recovery (power 0.661→0.780); at π=1.0, detected repeats (21/50) show baseline was *already fine* (power 0.952) and RFTL-S made it *worse* (0.762) — the backfire mechanism (total corruption of one user leaves no clean/dirty distinction for Huber weighting to exploit) replicates on genuinely correct data, not just the corrupted run. Detection recall remains sharply bimodal (almost always exactly 0 or 1, not graded) at every π level — also replicates.

**Overall verdict:** the qualitative story from the invalidated analysis holds up on real data — masking is real but conditional on an unlucky small-sample draw (not universal), detection is bimodal, and RFTL-S's benefit is inconsistent and actively backfires at total (π=1.0) corruption. The specific numbers differ from the earlier (voided) run, as they should, but the mechanism-level conclusions are now on solid ground. This is ready to write up, with the small-sample control-limit fragility explicitly named as part of the mechanism (not just "masking happens sometimes").

### What's next

- [ ] Write up the monitoring finding for the chapter using the verified table and mechanism above — do not use any number from the voided 2026-09-07 run
- [ ] Consider whether the control-limit estimator itself should be made more robust (e.g. MAD-based rather than mean+3σ) given the small-sample fragility identified — this is a natural, well-motivated follow-up experiment, not just a caveat
- [ ] Decide with Madi: (a) accept the Direction 2 stability finding as its headline result and move to connecting it to a downstream task, or (b) try a heterogeneity source more likely to be out-of-subspace as a harder stress test
- [ ] Reconcile RFTL-U (§3.2.2 of `chapter3_draft.md`, Grassmann-distance user trimming) with the empirical benchmark — does it supersede, extend as a 7th candidate rule, or sit alongside Multi-Krum? (Madi's call — see `direction1_byzantine_aggregation_findings.md` §1 and §6)
- [ ] Adopt multi_krum as the RFTL-S-adjacent aggregation rule for any further Direction 1 experiments (e.g. folding into a production-style comparison, or directly into the monitoring/prediction pipelines as a fourth method arm)
- [ ] Pull and verify actual citations for Krum / coordinate-median / trimmed-mean / geometric-median / small-magnitude-attack literature before chapter use, plus the Dirichlet-partition non-IID citation for Direction 2
