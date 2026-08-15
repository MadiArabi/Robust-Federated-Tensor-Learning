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
- [ ] Submit `run_job_prediction.sh` and `run_job_monitoring.sh` to Hazel (50 repeats each, fresh output dirs)
- [ ] Fix chart FAR calibration for chapter-grade figures (post-HPC, doesn't block submission)
- [ ] Add RFTL-U and RFTL-21 to both pipelines for full method comparison (separate follow-on)
