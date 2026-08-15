"""
RFTL-S Prediction Experiment on Real Degradation Data
(structured test-side contamination — July 2026 redesign)

Identical prediction pipeline to Chapter 2 (onepass-real-03-21-2.py), with a
faulty-camera contamination scenario: user 0's camera produces a fixed
structured artifact (hot-pixel block or readout stripes) that corrupts BOTH
its training and its test images (contamination.contaminate_train_and_test).
The June 2026 run showed train-only contamination is largely absorbed by the
min-max -> Ridge -> Tucker stack; test-side contamination is where robustness
can pay off.

Four-way comparison per condition:
  - clean:    clean train -> clean test   (camera never broke; upper bound)
  - oracle:   clean-train factors -> dirty test  (perfect robust estimator:
              robustness can fix the subspace, not the test images)
  - baseline: MPCA_FD on dirty train -> dirty test  (no robustness)
  - rftl_s:   Huber-weighted MPCA_FD re-fit on dirty train -> dirty test

Evaluation: full error distribution on log-TTF — mean (MAPE), median,
q25/q75 of |predicted - true| / |true| — overall and per user, with AIC rank
selection, same as Chapter 2.

NOTE: checkpoints from the June 2026 Gaussian run use different keys/values
and are NOT resume-compatible — always use a fresh --output-dir.

Usage:
    python rftl_s_real.py --data-path /path/to/data --n-repeats 50
"""

import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import copy
import random
import time
import json
import multiprocessing
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from scipy.io import loadmat
from tensorly import unfold
from tensorly.tenalg import multi_mode_dot

from my_mpca_02_27_nomean import MPCA_FD, MPCA_beta, train_test
from rftl_s import (MPCA_FD_Weighted, reconstruction_residual,
                     federated_mad, huber_weights)
from contamination import contaminate_train_and_test
import tucker_regression0


# ─── Configuration ───────────────────────────────────────────────────────────

TEST_SIZE = 20
SIZES = [45, 50, 55]
I_COMMON = [10, 10, 16]
WEIGHT_RANK = [5, 5, 6]

RANK_CONFIGS = [
    [2, 2, 3], [3, 3, 4], [4, 4, 5], [5, 5, 6],
    [6, 6, 7], [7, 7, 9], [8, 8, 9], [9, 9, 11], [10, 10, 11]
]

# Contamination grid — HPC configuration (2026-07-12).
# k sweep reduced to {1.345, 3.0}: the June run showed k does not move MAPE.
CONTAM_MODES = ['hot_block', 'stripe']
PI_S_LEVELS = [0.3, 0.6, 1.0]   # fault rate of user 0, train AND test
AMPLITUDE = 10.0
BLOCK_SIZE = 9
N_STRIPES = 6
TARGET_USER = 0

HUBER_K_VALUES = [1.345, 3.0]

# RFTL-S-IRLS ablation (task-dependence of robustness): IRLS improves
# detection, which monitoring needs — but a cleaner subspace pushes the
# estimator toward the collapsed oracle under persistent test-side faults.
# Run both variants to demonstrate the tension empirically.
IRLS_ROUNDS = 5
IRLS_INNER_ITERATIONS = 5
FIT_ITERATIONS = 150

_GLOBAL_DATA = None
_GLOBAL_Y = None
_GLOBAL_OUTPUT_DIR = None


def _init_worker(data, y, output_dir):
    """Pool initializer — required on Windows (spawn), harmless under fork."""
    global _GLOBAL_DATA, _GLOBAL_Y, _GLOBAL_OUTPUT_DIR
    _GLOBAL_DATA, _GLOBAL_Y, _GLOBAL_OUTPUT_DIR = data, y, output_dir


def _save_repeat_result(rep_idx, rep_seed, rep_results, output_dir):
    """Save a single repeat's results to a JSON file immediately."""
    if output_dir is None:
        return
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    serializable = {}
    for key, val in rep_results.items():
        str_key = '|'.join(str(k) for k in key)
        serializable[str_key] = val
    path = os.path.join(checkpoint_dir, f'repeat_{rep_idx:03d}_seed{rep_seed}.json')
    with open(path, 'w') as f:
        json.dump(serializable, f)


def _load_checkpoint_results(output_dir):
    """Load all saved per-repeat checkpoint files and return aggregated results."""
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_dir):
        return {}, set()
    results = {}
    completed = set()
    for fname in sorted(os.listdir(checkpoint_dir)):
        if not fname.endswith('.json'):
            continue
        rep_idx = int(fname.split('_')[1])
        completed.add(rep_idx)
        with open(os.path.join(checkpoint_dir, fname)) as f:
            data = json.load(f)
        for str_key, val in data.items():
            parts = str_key.split('|')
            key = []
            for p in parts:
                try:
                    key.append(int(p))
                except ValueError:
                    try:
                        key.append(float(p))
                    except ValueError:
                        key.append(p)
            key = tuple(key)
            results.setdefault(key, []).append(val)
    return results, completed


# ─── Data helpers (same as Chapter 2) ────────────────────────────────────────

def prepare_data(X, size, which):
    if which == 'A':
        start_x = start_y = 0
        end_x = end_y = 10
    elif which == 'B':
        start_x, start_y = 0, 0
        end_x, end_y = 15, 15
    else:
        start_x, start_y = 0, 0
        end_x, end_y = 20, 20
    userdata = [[image[0, i][start_x:end_x, start_y:end_y]
                 for i in range(16)] for image in X[:size]]
    return np.transpose(np.array(userdata), (0, 2, 3, 1))


def load_data(path):
    mat_data = loadmat(os.path.join(path, 'ResampleDegImages.mat'))
    data = mat_data['ResampleDegImages'][0]
    y = np.array([i[0] for i in mat_data['ResampleTTF']])
    return np.array(data), y


# ─── Projection helpers ─────────────────────────────────────────────────────

def project_data(users, V_mat, U_mat):
    """Project through V (per-user local) then U (global), concatenate."""
    features = []
    for m, data_m in enumerate(users):
        proj_v = multi_mode_dot(data_m, [v.T for v in V_mat[m]], modes=[1, 2, 3])
        proj_u = multi_mode_dot(proj_v, [u.T for u in U_mat], modes=[1, 2, 3])
        features.extend(proj_u)
    return np.array(features)


# ─── Prediction pipeline (same as Chapter 2) ────────────────────────────────

def prediction_pipeline(prime_train, prime_test, y_train, y_test):
    """
    Min-max scale -> Ridge -> MPCA_beta -> Tucker regression -> predict -> MAPE.
    Returns (mape, aic, rel_errors) or (None, inf, None) on failure.
    """
    Min = np.min(prime_train, axis=0)
    Max = np.max(prime_train, axis=0)
    denom = Max - Min
    denom[denom == 0] = 1e-10
    prime_scaled = (prime_train - Min) / denom
    prime_test_scaled = (prime_test - Min) / denom

    P1, P2, P3 = prime_train.shape[1], prime_train.shape[2], prime_train.shape[3]

    try:
        # Ridge on raw y for Tucker weight initialization (same as Chapter 2)
        clf = Ridge(alpha=0.0001)
        clf.fit(prime_scaled.reshape(len(prime_scaled), -1), y_train)
        beta = clf.coef_.reshape(1, P1, P2, P3)

        mpca_beta = MPCA_beta([P1, P2, P3], 1, 0.90)
        G, U_beta = mpca_beta.train(beta)
        P1_b = U_beta[0].shape[1]
        P2_b = U_beta[1].shape[1]
        P3_b = U_beta[2].shape[1]

        estimator = tucker_regression0.TuckerRegressor(
            weight_ranks=[P1_b, P2_b, P3_b],
            G=np.squeeze(G), U=U_beta,
            tol=10e-7, n_iter_max=100, reg_W=0, verbose=0
        )
        estimator.fit(prime_scaled, np.log(y_train))
        predicted = estimator.predict(prime_test_scaled)

        abs_diff = np.abs(predicted) - np.abs(np.log(y_test))
        RSS = np.mean(abs_diff ** 2)
        AIC = len(predicted) * np.log(RSS) + P1_b * P2_b * P3_b
        rel_errors = abs_diff / np.abs(np.log(y_test))
        mape = np.mean(np.abs(rel_errors))

        return mape, AIC, rel_errors
    except Exception:
        return None, np.inf, None


def error_stats(rel_errors):
    """Distribution of absolute relative errors, overall and per user.

    rel_errors is ordered [user A test (20), user B test (20), user C (20)].
    """
    abs_rel = np.abs(rel_errors)
    stats = {
        'mape': float(np.mean(abs_rel)),
        'median': float(np.median(abs_rel)),
        'q25': float(np.percentile(abs_rel, 25)),
        'q75': float(np.percentile(abs_rel, 75)),
    }
    for u in range(3):
        chunk = abs_rel[u * TEST_SIZE:(u + 1) * TEST_SIZE]
        stats[f'mape_user{u}'] = float(np.mean(chunk))
        stats[f'median_user{u}'] = float(np.median(chunk))
    return stats


def best_rank_stats(train_users, test_users, V_dict, U_dict, y_train, y_test):
    """Try all rank configs, return error stats dict of the AIC-selected one."""
    best_aic = np.inf
    best_stats = None

    for rank in RANK_CONFIGS:
        rk = tuple(rank)
        if rk not in V_dict:
            continue
        prime_train = project_data(train_users, V_dict[rk], U_dict[rk])
        prime_test = project_data(test_users, V_dict[rk], U_dict[rk])
        mape, aic, rel_errors = prediction_pipeline(prime_train, prime_test, y_train, y_test)
        if mape is not None and aic < best_aic:
            best_aic = aic
            best_stats = {'selected_rank': str(rank), **error_stats(rel_errors)}

    return best_stats


# ─── Subspace fitting helpers ────────────────────────────────────────────────

MAX_RANK = RANK_CONFIGS[-1]  # [10, 10, 11] — fit once, truncate U for smaller ranks


def fit_all_ranks_mpca(train_users):
    """Fit MPCA_FD once at max rank, then truncate U for all rank configs."""
    mpca = MPCA_FD(I_COMMON, MAX_RANK)
    mpca.iterations = FIT_ITERATIONS
    mpca.train(
        copy.deepcopy(train_users[0]),
        copy.deepcopy(train_users[1]),
        copy.deepcopy(train_users[2])
    )
    V_full = [[v.copy() for v in mpca.V_mat[m]] for m in range(3)]
    U_full = [u.copy() for u in mpca.U_mat]

    V_dict, U_dict = {}, {}
    for rank in RANK_CONFIGS:
        rk = tuple(rank)
        V_dict[rk] = V_full
        U_dict[rk] = [U_full[mode][:, :rank[mode]] for mode in range(3)]
    return V_dict, U_dict


def fit_all_ranks_weighted(train_users, weights):
    """Fit weighted MPCA_FD once at max rank, then truncate U for all rank configs."""
    model = MPCA_FD_Weighted(
        I_COMMON, MAX_RANK,
        iterations=FIT_ITERATIONS,
        weight_U=False
    )
    _, U_full, V_full = model.train(
        [copy.deepcopy(d) for d in train_users],
        [w.copy() for w in weights]
    )
    V_full = [[v.copy() for v in V_full[m]] for m in range(3)]
    U_full = [u.copy() for u in U_full]

    V_dict, U_dict = {}, {}
    for rank in RANK_CONFIGS:
        rk = tuple(rank)
        V_dict[rk] = V_full
        U_dict[rk] = [U_full[mode][:, :rank[mode]] for mode in range(3)]
    return V_dict, U_dict


# ─── Worker function ─────────────────────────────────────────────────────────

def _run_single_repeat(args):
    rep_idx, rep_seed = args
    data = _GLOBAL_DATA
    y = _GLOBAL_Y

    # Same random setup as Chapter 2
    np.random.seed(rep_seed)
    random.seed(rep_seed)
    sample = np.arange(len(data))

    np.random.shuffle(sample)
    user1 = prepare_data(data[sample], SIZES[0], 'A')
    y_A = y[sample][:SIZES[0]]

    np.random.shuffle(sample)
    user2 = prepare_data(data[sample], SIZES[1], 'B')
    y_B = y[sample][:SIZES[1]]

    np.random.shuffle(sample)
    user3 = prepare_data(data[sample], SIZES[2], 'C')
    y_C = y[sample][:SIZES[2]]

    # Train/test split with mean-centering (same as Chapter 2)
    Abar, A_test, y_A_train, y_A_test = train_test(user1, y_A, TEST_SIZE)
    Bbar, B_test, y_B_train, y_B_test = train_test(user2, y_B, TEST_SIZE)
    Cbar, C_test, y_C_train, y_C_test = train_test(user3, y_C, TEST_SIZE)

    train_clean = [Abar.copy(), Bbar.copy(), Cbar.copy()]
    test_clean = [A_test, B_test, C_test]
    y_train = np.concatenate([y_A_train, y_B_train, y_C_train])
    y_test = np.concatenate([y_A_test, y_B_test, y_C_test])

    rep_results = {}

    # ─── Clean reference (camera never broke) ───
    V_clean, U_clean = fit_all_ranks_mpca(train_clean)
    rep_results[('clean', 'none', 0.0)] = best_rank_stats(
        train_clean, test_clean, V_clean, U_clean, y_train, y_test
    )

    # ─── Contamination experiments ───
    conditions = [(m, p) for m in CONTAM_MODES for p in PI_S_LEVELS]
    for cond_idx, (mode, pi_s) in enumerate(conditions):
        # Separate RNG so main state is unaffected; keyed on the condition
        # index (str hash is randomized per process)
        contam_rng = np.random.RandomState(
            rep_seed + 1000 * (cond_idx + 1) + int(pi_s * 100)
        )
        train_dirty, contam_indices, test_dirty, _ = contaminate_train_and_test(
            train_clean, test_clean, pi_s, mode, contam_rng,
            amplitude=AMPLITUDE, block_size=BLOCK_SIZE, n_stripes=N_STRIPES,
            target_user=TARGET_USER
        )

        # ─── Oracle: clean-train factors, contaminated test images ───
        rep_results[('oracle', mode, pi_s)] = best_rank_stats(
            train_clean, test_dirty, V_clean, U_clean, y_train, y_test
        )

        # ─── Baseline: unweighted MPCA_FD on dirty data ───
        V_base, U_base = fit_all_ranks_mpca(train_dirty)
        rep_results[('baseline', mode, pi_s)] = best_rank_stats(
            train_dirty, test_dirty, V_base, U_base, y_train, y_test
        )

        # ─── RFTL-S: compute residuals once, then sweep k values ───
        wr = tuple(WEIGHT_RANK)
        V_init = V_base[wr]
        U_init = U_base[wr]

        residuals = [
            reconstruction_residual(train_dirty[m], V_init[m], U_init)
            for m in range(3)
        ]
        all_residuals = np.concatenate(residuals)
        median_r = np.median(all_residuals)
        mad = federated_mad(residuals)

        for huber_k in HUBER_K_VALUES:
            weights = [
                huber_weights(residuals[m], median_r, mad, k=huber_k)
                for m in range(3)
            ]

            # Detection metrics (train side — that is where weights act)
            tp, fp, fn = 0, 0, 0
            for m in range(3):
                flagged = set(np.where(weights[m] < 1.0)[0])
                tp += len(flagged & contam_indices[m])
                fp += len(flagged - contam_indices[m])
                fn += len(contam_indices[m] - flagged)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            rep_results[('rftl_s_prec', mode, pi_s, huber_k)] = prec
            rep_results[('rftl_s_rec', mode, pi_s, huber_k)] = rec

            # Weighted re-fit for all ranks
            V_rftl, U_rftl = fit_all_ranks_weighted(train_dirty, weights)
            rep_results[('rftl_s', mode, pi_s, huber_k)] = best_rank_stats(
                train_dirty, test_dirty, V_rftl, U_rftl, y_train, y_test
            )

        # ─── RFTL-S-IRLS ablation: iterated reweighting (monitoring's
        # final estimator) instead of the one-step weights above ───
        for huber_k in HUBER_K_VALUES:
            weights = [np.ones(d.shape[0]) for d in train_dirty]
            for _ in range(IRLS_ROUNDS):
                model = MPCA_FD_Weighted(I_COMMON, WEIGHT_RANK,
                                         iterations=IRLS_INNER_ITERATIONS,
                                         weight_U=False)
                _, U_it, V_it = model.train(
                    [copy.deepcopy(d) for d in train_dirty],
                    [w.copy() for w in weights])
                res_it = [
                    reconstruction_residual(train_dirty[m], V_it[m], U_it)
                    for m in range(3)
                ]
                med_it = np.median(np.concatenate(res_it))
                mad_it = federated_mad(res_it)
                weights = [
                    huber_weights(res_it[m], med_it, mad_it, k=huber_k)
                    for m in range(3)
                ]

            tp, fp, fn = 0, 0, 0
            for m in range(3):
                flagged = set(np.where(weights[m] < 1.0)[0])
                tp += len(flagged & contam_indices[m])
                fp += len(flagged - contam_indices[m])
                fn += len(contam_indices[m] - flagged)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            rep_results[('rftl_s_irls_prec', mode, pi_s, huber_k)] = prec
            rep_results[('rftl_s_irls_rec', mode, pi_s, huber_k)] = rec

            V_ri, U_ri = fit_all_ranks_weighted(train_dirty, weights)
            rep_results[('rftl_s_irls', mode, pi_s, huber_k)] = best_rank_stats(
                train_dirty, test_dirty, V_ri, U_ri, y_train, y_test
            )

    _save_repeat_result(rep_idx, rep_seed, rep_results, _GLOBAL_OUTPUT_DIR)
    print(f"  Repeat {rep_idx + 1} done (seed={rep_seed}).", flush=True)
    return rep_results


# ─── Experiment runner ───────────────────────────────────────────────────────

def run_experiment(data_path, n_repeats=50, n_workers=4, seed=2024,
                   output_dir=None):
    global _GLOBAL_DATA, _GLOBAL_Y, _GLOBAL_OUTPUT_DIR
    _GLOBAL_OUTPUT_DIR = output_dir

    print("Loading real degradation data...", flush=True)
    _GLOBAL_DATA, _GLOBAL_Y = load_data(data_path)
    n_total = len(_GLOBAL_DATA)
    print(f"  Loaded {n_total} samples, {len(_GLOBAL_Y)} TTF values", flush=True)

    min_needed = max(SIZES) + TEST_SIZE
    if n_total < min_needed:
        print(f"  ERROR: Need at least {min_needed} samples, got {n_total}",
              flush=True)
        sys.exit(1)

    master_rng = np.random.RandomState(seed)
    rep_seeds = [int(master_rng.randint(1, 100000)) for _ in range(n_repeats)]

    # Resume: skip repeats that already have saved checkpoints
    already_done = set()
    if output_dir:
        _, already_done = _load_checkpoint_results(output_dir)
        if already_done:
            print(f"  Resuming: {len(already_done)} repeats already saved, "
                  f"{n_repeats - len(already_done)} remaining.", flush=True)

    worker_args = [(i, s) for i, s in enumerate(rep_seeds)
                   if i not in already_done]

    if not worker_args:
        print("  All repeats already completed! Loading from checkpoints.",
              flush=True)
        results, _ = _load_checkpoint_results(output_dir)
        return results

    print(f"Running {len(worker_args)} repeats across {n_workers} workers...",
          flush=True)
    print(f"  Sizes: {SIZES}, Ranks: {len(RANK_CONFIGS)} configs", flush=True)
    print(f"  Modes: {CONTAM_MODES}, pi_S (user {TARGET_USER}, train+test): "
          f"{PI_S_LEVELS}", flush=True)
    print(f"  Artifact: amp={AMPLITUDE}, block={BLOCK_SIZE}, "
          f"stripes={N_STRIPES}", flush=True)
    print(f"  Huber k={HUBER_K_VALUES}, weight_U=False, iterations={FIT_ITERATIONS}",
          flush=True)

    start = time.time()
    if n_workers > 1:
        pool = multiprocessing.Pool(
            processes=n_workers, initializer=_init_worker,
            initargs=(_GLOBAL_DATA, _GLOBAL_Y, _GLOBAL_OUTPUT_DIR))
        all_rep = pool.map(_run_single_repeat, worker_args)
        pool.close()
        pool.join()
    else:
        all_rep = [_run_single_repeat(a) for a in worker_args]
    elapsed = (time.time() - start) / 3600

    # Load all results (checkpointed + just completed)
    if output_dir:
        results, _ = _load_checkpoint_results(output_dir)
    else:
        results = {}
        for rep in all_rep:
            for key, val in rep.items():
                results.setdefault(key, []).append(val)

    print(f"\nDone in {elapsed:.2f} hours.", flush=True)
    return results


# ─── Reporting ───────────────────────────────────────────────────────────────

def _agg(vals, field='mape'):
    """Mean and std of one stats field over repeats (ignoring failed fits)."""
    xs = [v[field] for v in vals if v is not None]
    if not xs:
        return None, None
    return float(np.mean(xs)), float(np.std(xs))


def report_results(results, output_dir=None):
    print("\n" + "=" * 105, flush=True)
    print("RFTL-S PREDICTION RESULTS — REAL DATA, TEST-SIDE CONTAMINATION "
          "(errors on log-TTF)", flush=True)
    print("=" * 105, flush=True)

    clean_vals = results.get(('clean', 'none', 0.0), [])
    if clean_vals:
        c_mean, c_std = _agg(clean_vals)
        c_med, _ = _agg(clean_vals, 'median')
        c_u0, _ = _agg(clean_vals, 'mape_user0')
        print(f"\nClean reference: MAPE = {c_mean:.4f} +/- {c_std:.4f}  "
              f"median {c_med:.4f}  user0-MAPE {c_u0:.4f}", flush=True)

    for mode in CONTAM_MODES:
        print(f"\n--- Mode: {mode} (amp={AMPLITUDE}, user {TARGET_USER}, "
              f"train+test) ---", flush=True)
        for pi_s in PI_S_LEVELS:
            o_vals = results.get(('oracle', mode, pi_s), [])
            b_vals = results.get(('baseline', mode, pi_s), [])
            if not (o_vals and b_vals):
                continue
            o_mean, o_std = _agg(o_vals)
            o_u0, _ = _agg(o_vals, 'mape_user0')
            b_mean, b_std = _agg(b_vals)
            b_u0, _ = _agg(b_vals, 'mape_user0')
            gap = (b_mean - o_mean) / o_mean * 100 if o_mean else 0
            print(f"\n  pi_S={pi_s:.2f}", flush=True)
            print(f"    {'oracle':>8}: MAPE {o_mean:.4f}+/-{o_std:.4f}  "
                  f"user0 {o_u0:.4f}", flush=True)
            print(f"    {'baseline':>8}: MAPE {b_mean:.4f}+/-{b_std:.4f}  "
                  f"user0 {b_u0:.4f}  ({gap:+.1f}% vs oracle)", flush=True)
            for method, tag in [('rftl_s', 'rftl'), ('rftl_s_irls', 'irls')]:
                for huber_k in HUBER_K_VALUES:
                    r_vals = results.get((method, mode, pi_s, huber_k), [])
                    if not r_vals:
                        continue
                    r_mean, r_std = _agg(r_vals)
                    r_u0, _ = _agg(r_vals, 'mape_user0')
                    improv = (b_mean - r_mean) / b_mean * 100 if b_mean else 0
                    prec = np.mean(results.get(
                        (f'{method}_prec', mode, pi_s, huber_k), [0]))
                    rec = np.mean(results.get(
                        (f'{method}_rec', mode, pi_s, huber_k), [0]))
                    print(f"    {tag} k={huber_k:<5}: "
                          f"MAPE {r_mean:.4f}+/-{r_std:.4f}  "
                          f"user0 {r_u0:.4f}  ({improv:+.1f}% vs baseline)  "
                          f"prec {prec:.3f} rec {rec:.3f}", flush=True)

    print("\n" + "=" * 105, flush=True)

    # Save to CSV — stats dicts expand to columns; prec/rec use 'value'
    if output_dir:
        rows = []
        for key, vals in results.items():
            if len(key) == 3:
                method, mode, pi_s = key
                huber_k = None
            else:
                method, mode, pi_s, huber_k = key
            for rep_idx, val in enumerate(vals):
                if val is None:
                    continue
                row = {'method': method, 'mode': mode,
                       'pi_s': pi_s, 'huber_k': huber_k, 'repeat': rep_idx}
                if isinstance(val, dict):
                    row.update(val)
                else:
                    row['value'] = val
                rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, 'rftl_s_real_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}", flush=True)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RFTL-S Prediction on Real Data')
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=50)
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--output-dir', default=None,
                        help='Directory for CSV output')
    args = parser.parse_args()

    if args.data_path:
        data_path = args.data_path
    else:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data')

    if not os.path.exists(data_path):
        print(f"ERROR: Data path not found: {data_path}", flush=True)
        sys.exit(1)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    results = run_experiment(
        data_path, n_repeats=args.n_repeats, n_workers=args.n_workers,
        output_dir=args.output_dir
    )
    report_results(results, output_dir=args.output_dir)
