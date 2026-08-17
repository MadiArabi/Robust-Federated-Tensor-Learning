"""
Monitoring experiment on real degradation data — PRODUCTION (HPC) version.

SPE control-chart monitoring is the subspace-native downstream task: the
chart consumes the federated subspace and the training-residual distribution
directly, so contamination attacks it via limit inflation (masking) and
subspace rotation (swamping) with no regression layer to absorb the damage.
See experiment_log.md Sessions 7+ (pilot arc v1-v6) for the design history.

Final estimator (pilot v3): rank [5,5,6], IRLS — 5 rounds of early-stopped
(5-iter) weighted fits alternating with Huber reweighting on pooled
federated median/MAD residuals — then one fully converged weighted fit.
The only pilot variant with well-calibrated full recovery.

Task setup per user: median-TTF split into healthy/degraded; healthy ->
70% train / 30% FAR holdout (centered by healthy-train mean). Contamination
(fixed-pattern structured artifacts, user 0 only, TRAIN side) at
hot_block pi_S {0.3..1.0} — pi_S=0.3 is the documented detection-breakdown
boundary — plus one stripe condition for the absorbed-artifact limitation.

Metrics per user: FAR, power, AUC, detection TTF, limit value.
Methods: clean / baseline / RFTL-S (IRLS).

Usage:
    python monitoring_real.py --data-path /path/to/data --n-repeats 50 \
        --n-workers 10 --output-dir output_monitoring/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import json
import time
import random
import multiprocessing

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from my_mpca_02_27_nomean import MPCA_FD
from rftl_s import (MPCA_FD_Weighted, reconstruction_residual,
                    federated_mad, huber_weights)
from contamination import contaminate
from rftl_s_real import prepare_data, load_data, SIZES, I_COMMON

# ─── Final design constants (pilot v3) ──────────────────────────────────────

MONITOR_RANK = [5, 5, 6]
FIT_ITERATIONS = 150
IRLS_ROUNDS = 5
IRLS_INNER_ITERATIONS = 5
HUBER_K = 3.0
TRAIN_FRAC = 0.7
LIMIT_SIGMA = 3.0

CONDITIONS = [
    # (mode, amplitude, pi_s, block_size)
    ('hot_block', 10.0, 0.30, 9),   # documented breakdown boundary
    ('hot_block', 10.0, 0.40, 9),
    ('hot_block', 10.0, 0.50, 9),
    ('hot_block', 10.0, 0.60, 9),
    ('hot_block', 10.0, 0.80, 9),
    ('hot_block', 10.0, 1.00, 9),
    ('stripe', 10.0, 0.60, 9),      # absorbed-artifact limitation case
]
N_STRIPES = 6
TARGET_USER = 0

_GLOBAL_DATA = None
_GLOBAL_Y = None
_GLOBAL_OUTPUT_DIR = None


def _init_worker(data, y, output_dir):
    """Pool initializer — required on Windows (spawn), harmless under fork."""
    global _GLOBAL_DATA, _GLOBAL_Y, _GLOBAL_OUTPUT_DIR
    _GLOBAL_DATA, _GLOBAL_Y, _GLOBAL_OUTPUT_DIR = data, y, output_dir


# ─── Chart helpers (identical to pilot_monitoring) ──────────────────────────

def weighted_mean_std(x, w):
    mean = np.sum(w * x) / np.sum(w)
    var = np.sum(w * (x - mean) ** 2) / np.sum(w)
    return mean, np.sqrt(var)


def fit_mpca(train_users, iterations):
    mpca = MPCA_FD(I_COMMON, MONITOR_RANK)
    mpca.iterations = iterations
    mpca.train(*[copy.deepcopy(u) for u in train_users])
    return mpca.V_mat, mpca.U_mat


def chart_metrics(train_users, holdout_users, degraded_users, degraded_ttf,
                  V_mat, U_mat, weights=None):
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


# ─── Checkpointing (same pattern as rftl_s_real) ────────────────────────────

def _ckpt_path(output_dir, rep_idx, rep_seed):
    return os.path.join(output_dir, 'checkpoints',
                        f'repeat_{rep_idx:03d}_seed{rep_seed}.json')


def _save_repeat(rep_idx, rep_seed, rows, output_dir):
    if output_dir is None:
        return
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    with open(_ckpt_path(output_dir, rep_idx, rep_seed), 'w') as f:
        json.dump(rows, f)


def _load_checkpoints(output_dir):
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    rows, completed = [], set()
    if not os.path.isdir(ckpt_dir):
        return rows, completed
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith('.json'):
            continue
        completed.add(int(fname.split('_')[1]))
        with open(os.path.join(ckpt_dir, fname)) as f:
            rows.extend(json.load(f))
    return rows, completed


# ─── Worker ──────────────────────────────────────────────────────────────────

def _run_single_repeat(args):
    rep_idx, rep_seed = args
    data, y = _GLOBAL_DATA, _GLOBAL_Y

    np.random.seed(rep_seed)
    random.seed(rep_seed)
    sample = np.arange(len(data))

    users, ttfs = [], []
    for size, which in zip(SIZES, ['A', 'B', 'C']):
        np.random.shuffle(sample)
        users.append(prepare_data(data[sample], size, which))
        ttfs.append(y[sample][:size])

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

    V_clean, U_clean = fit_mpca(train_users, FIT_ITERATIONS)
    rows.append({'repeat': rep_idx, 'method': 'clean', 'mode': 'none',
                 'amplitude': 0.0, 'pi_s': 0.0, 'block': 0,
                 **chart_metrics(train_users, holdout_users, degraded_users,
                                 degraded_ttf, V_clean, U_clean)})

    for cond_idx, (mode, amplitude, pi_s, blk) in enumerate(CONDITIONS):
      try:
        contam_rng = np.random.RandomState(
            rep_seed + 1000 * (cond_idx + 1) + int(pi_s * 100))
        train_dirty, contam_indices = contaminate(
            train_users, pi_s, mode, contam_rng, amplitude=amplitude,
            block_size=blk, n_stripes=N_STRIPES, target_user=TARGET_USER)

        V_base, U_base = fit_mpca(train_dirty, FIT_ITERATIONS)
        rows.append({'repeat': rep_idx, 'method': 'baseline', 'mode': mode,
                     'amplitude': amplitude, 'pi_s': pi_s, 'block': blk,
                     **chart_metrics(train_dirty, holdout_users,
                                     degraded_users, degraded_ttf,
                                     V_base, U_base)})

        # RFTL-S (IRLS, pilot v3): rounds of early-stopped weighted fits +
        # pooled federated median/MAD Huber reweighting
        weights = [np.ones(d.shape[0]) for d in train_dirty]
        for _ in range(IRLS_ROUNDS):
            model = MPCA_FD_Weighted(I_COMMON, MONITOR_RANK,
                                     iterations=IRLS_INNER_ITERATIONS,
                                     weight_U=False)
            _, U_it, V_it = model.train(
                [copy.deepcopy(d) for d in train_dirty],
                [w.copy() for w in weights])
            residuals = [reconstruction_residual(train_dirty[m], V_it[m], U_it)
                         for m in range(3)]
            median_r = np.median(np.concatenate(residuals))
            mad = federated_mad(residuals)
            weights = [huber_weights(residuals[m], median_r, mad, k=HUBER_K)
                       for m in range(3)]

        tp = sum(len(set(np.where(weights[m] < 1.0)[0]) & contam_indices[m])
                 for m in range(3))
        n_flag = sum(int(np.sum(weights[m] < 1.0)) for m in range(3))
        n_contam = sum(len(s) for s in contam_indices)

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
      except Exception as e:
          # A single pathological (repeat, condition) combination -- e.g.
          # an incremental-SVD non-convergence from an unusually degenerate
          # sample draw -- must not lose the other 6 conditions' results
          # for this repeat, let alone kill the whole 50-repeat pool.map()
          # job (this is exactly what happened on HPC job 9370's monitoring
          # run before this guard existed). Logged, not silently swallowed:
          # visible in the job's stdout, and this condition is simply
          # absent from the CSV for this repeat rather than corrupted.
          print(f"  Repeat {rep_idx + 1} (seed={rep_seed}) condition "
               f"{cond_idx} (mode={mode}, pi_s={pi_s}) FAILED: "
               f"{type(e).__name__}: {e} -- skipping this condition",
               flush=True)

    _save_repeat(rep_idx, rep_seed, rows, _GLOBAL_OUTPUT_DIR)
    print(f"  Repeat {rep_idx + 1} done (seed={rep_seed}).", flush=True)
    return rows


# ─── Runner ──────────────────────────────────────────────────────────────────

def _fmt(df, field):
    users = [f'{field}_user{m}' for m in range(3)]
    vals = df[users].mean()
    return (f"u0 {vals[users[0]]:.3f}  "
            f"u1 {vals[users[1]]:.3f}  u2 {vals[users[2]]:.3f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=50)
    parser.add_argument('--n-workers', type=int, default=10)
    parser.add_argument('--output-dir', default='output_monitoring')
    args = parser.parse_args()

    data_path = args.data_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    print("Loading real degradation data...", flush=True)
    data, y = load_data(data_path)
    print(f"  Loaded {len(data)} samples", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Same seed derivation as rftl_s_real so repeats match across experiments
    master_rng = np.random.RandomState(2024)
    rep_seeds = [int(master_rng.randint(1, 100000))
                 for _ in range(args.n_repeats)]

    _, completed = _load_checkpoints(args.output_dir)
    worker_args = [(i, s) for i, s in enumerate(rep_seeds)
                   if i not in completed]
    if completed:
        print(f"  Resuming: {len(completed)} repeats already saved, "
              f"{len(worker_args)} remaining.", flush=True)

    print(f"Running {len(worker_args)} repeats x "
          f"{1 + 2 * len(CONDITIONS)} converged fits (+IRLS inner fits), "
          f"{args.n_workers} workers...", flush=True)

    start = time.time()
    if worker_args:
        if args.n_workers > 1:
            pool = multiprocessing.Pool(
                processes=args.n_workers, initializer=_init_worker,
                initargs=(data, y, args.output_dir))
            pool.map(_run_single_repeat, worker_args)
            pool.close()
            pool.join()
        else:
            _init_worker(data, y, args.output_dir)
            for a in worker_args:
                _run_single_repeat(a)
    print(f"\nDone in {(time.time() - start) / 3600:.2f} hours.", flush=True)

    all_rows, _ = _load_checkpoints(args.output_dir)
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.output_dir, 'monitoring_real_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}", flush=True)

    # ─── Summary ───
    print("\n" + "=" * 96, flush=True)
    print("MONITORING RESULTS (SPE chart): clean / baseline / RFTL-S (IRLS)",
          flush=True)
    print("=" * 96, flush=True)
    clean = df[df.method == 'clean']
    print("\nClean chart:", flush=True)
    for f in ['far', 'power', 'auc', 'det_ttf', 'limit']:
        print(f"    {f:>8}: {_fmt(clean, f)}", flush=True)

    for mode, amplitude, pi_s, blk in CONDITIONS:
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
        p_clean = clean['power_user0'].mean()
        p_base = base['power_user0'].mean()
        p_rftl = rftl['power_user0'].mean()
        print(f"    user0 power: clean {p_clean:.3f} -> base {p_base:.3f} "
              f"-> rftl {p_rftl:.3f}", flush=True)
