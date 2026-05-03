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
