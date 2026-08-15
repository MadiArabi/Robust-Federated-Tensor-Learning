"""
Pilot: SPE control-chart monitoring as the subspace-native downstream task.

Motivation (2026-07-08): both contamination designs showed that the
prediction pipeline (min-max -> Ridge -> Tucker) largely absorbs train-side
contamination — the regression re-calibrates around whatever subspace it is
handed. Monitoring has no such shield: the chart consumes the subspace and
the training-residual distribution directly, and contamination attacks both:

  1. MASKING: contaminated training samples inflate the SPE control limit
     (mean + 3 std of train SPE), so real degradation slips under it.
  2. SWAMPING: the rotated subspace raises SPE on healthy data -> false alarms.

Task: per user, split samples at the median TTF into healthy / degraded.
Fit the federated subspace on healthy training samples (user 0's healthy
train contaminated with structured artifacts — the camera was faulty during
the subspace-building phase; the monitored stream itself is clean). Then:

  - FAR:    fraction of held-out healthy samples with SPE above the limit
  - power:  fraction of degraded samples with SPE above the limit
  - AUC:    healthy-holdout vs degraded separation by raw SPE score
  - det_ttf: TTF at the first alarm streaming degraded samples in
             TTF-descending order (higher = earlier warning; NaN = missed)

Methods: clean / baseline / RFTL-S (Huber k=3, weighted re-fit; the control
limit uses the weighted mean/std of train SPE — same limit rule as the
baseline, only the weights differ).

Usage:
    python pilot_monitoring.py --n-repeats 5 --n-workers 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
import multiprocessing
import pandas as pd
from sklearn.metrics import roc_auc_score

from my_mpca_02_27_nomean import MPCA_FD
from rftl_s import (MPCA_FD_Weighted, reconstruction_residual,
                    federated_mad, huber_weights)
from contamination import contaminate
from rftl_s_real import prepare_data, load_data, SIZES, I_COMMON

MONITOR_RANK = [5, 5, 6]   # v4 showed low rank drowns artifacts in noise; back to v3
FIT_ITERATIONS = 150
# v5: LEAVE-ONE-USER-OUT detection. The whole v1-v4 arc showed the global
# fit absorbs low-fraction repeated artifacts (nothing to detect) while a
# weak fit drowns them in noise. Fix the root cause: score user m's samples
# under the subspace of the OTHER users only — an artifact local to user m
# cannot be in the others' consensus, so it stands out regardless of what
# the global fit absorbs. Implemented with MPCA_FD_Weighted: giving user m
# uniform weight ~0 removes it from the global U while its local V still
# adapts to the consensus (uniform scaling leaves its eigenvectors
# unchanged) — exactly what a site validating its own data against the
# federation would compute.
LOO_DETECT_ITERATIONS = 20
HUBER_K = 3.0
TRAIN_FRAC = 0.7          # of each user's healthy samples; rest -> FAR holdout
LIMIT_SIGMA = 3.0         # limit = (weighted) mean + LIMIT_SIGMA * std

# v6: block-size axis added. User 0 is the 10x10 crop, so block 9 saturates
# 81% of its image — a near-constant (DC-like) frame that ANY smooth image
# subspace reconstructs well, capping residual detectability (v5 recall
# ~0.5 despite LOO). Block 5 = 25% of the image: localized structure, less
# absorbable, and more physical ("cluster of hot pixels"). Stripes dropped:
# consistently benign AND invisible (documented limitation).
PILOT_CONDITIONS = [
    # (mode, amplitude, pi_s, block_size)
    ('hot_block', 10.0, 0.30, 9),
    ('hot_block', 10.0, 0.60, 9),
    ('hot_block', 10.0, 0.30, 5),
    ('hot_block', 10.0, 0.60, 5),
]
N_STRIPES = 6
TARGET_USER = 0

_GLOBAL_DATA = None
_GLOBAL_Y = None


def weighted_mean_std(x, w):
    mean = np.sum(w * x) / np.sum(w)
    var = np.sum(w * (x - mean) ** 2) / np.sum(w)
    return mean, np.sqrt(var)


def fit_mpca(train_users, iterations):
    import copy
    mpca = MPCA_FD(I_COMMON, MONITOR_RANK)
    mpca.iterations = iterations
    mpca.train(*[copy.deepcopy(u) for u in train_users])
    return mpca.V_mat, mpca.U_mat


def chart_metrics(train_users, holdout_users, degraded_users, degraded_ttf,
                  V_mat, U_mat, weights=None):
    """SPE chart per user: limit from (weighted) train SPE, then FAR / power
    / AUC / detection TTF on that user's holdout + degraded stream."""
    out = {}
    for m in range(3):
        spe_train = reconstruction_residual(train_users[m], V_mat[m], U_mat)
        w = np.ones_like(spe_train) if weights is None else weights[m]
        mu, sd = weighted_mean_std(spe_train, w)
        limit = mu + LIMIT_SIGMA * sd

        spe_hold = reconstruction_residual(holdout_users[m], V_mat[m], U_mat)
        spe_degr = reconstruction_residual(degraded_users[m], V_mat[m], U_mat)

        far = float(np.mean(spe_hold > limit))
        power = float(np.mean(spe_degr > limit))
        labels = np.r_[np.zeros(len(spe_hold)), np.ones(len(spe_degr))]
        auc = float(roc_auc_score(labels, np.r_[spe_hold, spe_degr]))

        # degraded stream in TTF-descending order (approach to failure)
        order = np.argsort(-degraded_ttf[m])
        alarms = spe_degr[order] > limit
        det_ttf = float(degraded_ttf[m][order][np.argmax(alarms)]) \
            if alarms.any() else np.nan

        out[f'limit_user{m}'] = float(limit)
        out[f'far_user{m}'] = far
        out[f'power_user{m}'] = power
        out[f'auc_user{m}'] = auc
        out[f'det_ttf_user{m}'] = det_ttf
    return out


def _run_single_repeat(args):
    rep_idx, rep_seed = args
    data, y = _GLOBAL_DATA, _GLOBAL_Y

    # Same user construction as rftl_s_real / the gate pilots
    np.random.seed(rep_seed)
    random.seed(rep_seed)
    sample = np.arange(len(data))

    users, ttfs = [], []
    for size, which in zip(SIZES, ['A', 'B', 'C']):
        np.random.shuffle(sample)
        users.append(prepare_data(data[sample], size, which))
        ttfs.append(y[sample][:size])

    # Per user: median-TTF split into healthy / degraded; healthy split into
    # train / FAR holdout; center everything by the healthy-train mean.
    train_users, holdout_users, degraded_users, degraded_ttf = [], [], [], []
    for m in range(3):
        healthy = np.where(ttfs[m] >= np.median(ttfs[m]))[0]
        degraded = np.where(ttfs[m] < np.median(ttfs[m]))[0]
        np.random.shuffle(healthy)
        n_train = int(round(TRAIN_FRAC * len(healthy)))
        tr, ho = healthy[:n_train], healthy[n_train:]
        mean = np.mean(users[m][tr], axis=0)
        train_users.append(users[m][tr] - mean)
        holdout_users.append(users[m][ho] - mean)
        degraded_users.append(users[m][degraded] - mean)
        degraded_ttf.append(ttfs[m][degraded])

    rows = []

    # ─── Clean chart ───
    V_clean, U_clean = fit_mpca(train_users, FIT_ITERATIONS)
    rows.append({'repeat': rep_idx, 'method': 'clean', 'mode': 'none',
                 'amplitude': 0.0, 'pi_s': 0.0, 'block': 0,
                 **chart_metrics(train_users, holdout_users, degraded_users,
                                 degraded_ttf, V_clean, U_clean)})

    for cond_idx, (mode, amplitude, pi_s, blk) in enumerate(PILOT_CONDITIONS):
        contam_rng = np.random.RandomState(
            rep_seed + 1000 * (cond_idx + 1) + int(pi_s * 100))
        train_dirty, contam_indices = contaminate(
            train_users, pi_s, mode, contam_rng, amplitude=amplitude,
            block_size=blk, n_stripes=N_STRIPES,
            target_user=TARGET_USER)

        # Baseline chart: subspace and limits from dirty train
        V_base, U_base = fit_mpca(train_dirty, FIT_ITERATIONS)
        rows.append({'repeat': rep_idx, 'method': 'baseline', 'mode': mode,
                     'amplitude': amplitude, 'pi_s': pi_s, 'block': blk,
                     **chart_metrics(train_dirty, holdout_users,
                                     degraded_users, degraded_ttf,
                                     V_base, U_base)})

        # RFTL-S chart: leave-one-user-out detection (see note at
        # LOO_DETECT_ITERATIONS), then a fully converged weighted fit +
        # weighted limits. Standardization uses a TRIMMED-CORE scale per
        # user — median/MAD of the lowest 40% of that user's LOO residuals.
        # v5 used the plain within-user median, which breaks down once a
        # majority of the user's samples are contaminated (it centered on
        # the artifact cluster at pi_s=0.6 and flagged the clean minority);
        # the low-residual core is clean for contamination up to ~60%.
        import copy
        weights = []
        for m in range(3):
            w_loo = [np.full(d.shape[0], 1e-6) if j == m
                     else np.ones(d.shape[0])
                     for j, d in enumerate(train_dirty)]
            model = MPCA_FD_Weighted(I_COMMON, MONITOR_RANK,
                                     iterations=LOO_DETECT_ITERATIONS,
                                     weight_U=False)
            _, U_loo, V_loo = model.train(
                [copy.deepcopy(d) for d in train_dirty], w_loo)
            res_m = reconstruction_residual(train_dirty[m], V_loo[m], U_loo)
            n_core = max(3, int(0.4 * len(res_m)))
            core = np.sort(res_m)[:n_core]
            med_m = np.median(core)
            mad_m = max(np.median(np.abs(core - med_m)), 1e-10)
            weights.append(huber_weights(res_m, med_m, mad_m, k=HUBER_K))

        tp = sum(len(set(np.where(weights[m] < 1.0)[0]) & contam_indices[m])
                 for m in range(3))
        n_flag = sum(int(np.sum(weights[m] < 1.0)) for m in range(3))
        n_contam = sum(len(s) for s in contam_indices)

        import copy
        model = MPCA_FD_Weighted(I_COMMON, MONITOR_RANK,
                                 iterations=FIT_ITERATIONS, weight_U=False)
        _, U_rftl, V_rftl = model.train(
            [copy.deepcopy(d) for d in train_dirty],
            [w.copy() for w in weights])
        rows.append({'repeat': rep_idx, 'method': 'rftl_s', 'mode': mode,
                     'amplitude': amplitude, 'pi_s': pi_s, 'block': blk,
                     'precision': tp / n_flag if n_flag else 0.0,
                     'recall': tp / n_contam if n_contam else 0.0,
                     **chart_metrics(train_dirty, holdout_users,
                                     degraded_users, degraded_ttf,
                                     V_rftl, U_rftl, weights=weights)})

    print(f"  Repeat {rep_idx + 1} done (seed={rep_seed}).", flush=True)
    return rows


def _init_worker(data, y):
    global _GLOBAL_DATA, _GLOBAL_Y
    _GLOBAL_DATA, _GLOBAL_Y = data, y


def _fmt(df, field):
    users = [f'{field}_user{m}' for m in range(3)]
    vals = df[users].mean()
    return (f"u0 {vals[users[0]]:.3f}  "
            f"u1 {vals[users[1]]:.3f}  u2 {vals[users[2]]:.3f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=5)
    parser.add_argument('--n-workers', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=FIT_ITERATIONS)
    parser.add_argument('--output', default='output/pilot_monitoring.csv')
    args = parser.parse_args()
    FIT_ITERATIONS = args.iterations

    data_path = args.data_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    print("Loading real degradation data...", flush=True)
    data, y = load_data(data_path)
    print(f"  Loaded {len(data)} samples", flush=True)

    master_rng = np.random.RandomState(2024)
    rep_seeds = [int(master_rng.randint(1, 100000))
                 for _ in range(args.n_repeats)]
    worker_args = list(enumerate(rep_seeds))

    print(f"Running {args.n_repeats} repeats x "
          f"{1 + 2 * len(PILOT_CONDITIONS)} fits (SPE monitoring)...",
          flush=True)

    if args.n_workers > 1:
        pool = multiprocessing.Pool(processes=args.n_workers,
                                    initializer=_init_worker,
                                    initargs=(data, y))
        all_rows = pool.map(_run_single_repeat, worker_args)
        pool.close()
        pool.join()
    else:
        _init_worker(data, y)
        all_rows = [_run_single_repeat(a) for a in worker_args]

    df = pd.DataFrame([r for rep in all_rows for r in rep])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}", flush=True)

    # ─── Summary ───
    print("\n" + "=" * 96, flush=True)
    print("MONITORING PILOT: does contamination mask degradation, and does "
          "RFTL-S recover the chart?", flush=True)
    print("=" * 96, flush=True)

    clean = df[df.method == 'clean']
    print(f"\nClean chart:", flush=True)
    for f in ['far', 'power', 'auc', 'det_ttf', 'limit']:
        print(f"    {f:>8}: {_fmt(clean, f)}", flush=True)

    for mode, amplitude, pi_s, blk in PILOT_CONDITIONS:
        cond = (df['mode'] == mode) & (df.amplitude == amplitude) \
            & (df.pi_s == pi_s) & (df.block == blk)
        base = df[cond & (df.method == 'baseline')]
        rftl = df[cond & (df.method == 'rftl_s')]
        if base.empty:
            continue
        print(f"\n{mode} amp={amplitude} pi_s={pi_s} block={blk}:", flush=True)
        for f in ['far', 'power', 'auc', 'det_ttf', 'limit']:
            print(f"    {f:>8}:  base  {_fmt(base, f)}", flush=True)
            print(f"    {'':>8}   rftl  {_fmt(rftl, f)}", flush=True)
        print(f"    detection: precision {rftl['precision'].mean():.3f}  "
              f"recall {rftl['recall'].mean():.3f}", flush=True)

        # Verdict on user 0 (the contaminated site): masking = power drop
        p_clean = clean['power_user0'].mean()
        p_base = base['power_user0'].mean()
        p_rftl = rftl['power_user0'].mean()
        masked = p_clean - p_base
        recovered = p_rftl - p_base
        verdict = 'PASS' if (masked > 0.15 and recovered > 0.5 * masked) \
            else ('masking weak' if masked <= 0.15 else 'rftl fails to recover')
        print(f"    user0 power: clean {p_clean:.3f} -> base {p_base:.3f} "
              f"-> rftl {p_rftl:.3f}   => {verdict}", flush=True)
