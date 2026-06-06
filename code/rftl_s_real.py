"""
RFTL-S Prediction Experiment on Real Degradation Data

Identical pipeline to Chapter 2 (onepass-real-03-21-2.py) with:
  1. Contamination injection into training samples
  2. Huber-weighted MPCA_FD to handle compromised samples
  3. Prediction comparison: baseline vs RFTL-S

Comparison:
  - Clean: MPCA_FD on clean data (upper bound)
  - Baseline: MPCA_FD on contaminated data (no robustness)
  - RFTL-S: Huber weights -> Weighted MPCA_FD re-fit (with robustness)

Evaluation: MAPE on log(TTF) with AIC rank selection, same as Chapter 2.

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

PI_S_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
NOISE_MULTIPLIERS = [2, 3, 5, 10]
HUBER_K = 1.345
FIT_ITERATIONS = 150

_GLOBAL_DATA = None
_GLOBAL_Y = None


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
    mat_data = loadmat(os.path.join(path, 'ResampleDegImages (1).mat'))
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
    Returns (mape, aic) or (None, inf) on failure.
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
        mape = np.mean(np.abs(abs_diff) / np.abs(np.log(y_test)))

        return mape, AIC
    except Exception:
        return None, np.inf


def best_rank_mape(train_users, test_users, V_dict, U_dict, y_train, y_test):
    """Try all rank configs, return best MAPE (AIC-selected)."""
    best_aic = np.inf
    best_mape = None

    for rank in RANK_CONFIGS:
        rk = tuple(rank)
        if rk not in V_dict:
            continue
        prime_train = project_data(train_users, V_dict[rk], U_dict[rk])
        prime_test = project_data(test_users, V_dict[rk], U_dict[rk])
        mape, aic = prediction_pipeline(prime_train, prime_test, y_train, y_test)
        if mape is not None and aic < best_aic:
            best_aic = aic
            best_mape = mape

    return best_mape


# ─── Subspace fitting helpers ────────────────────────────────────────────────

def fit_all_ranks_mpca(train_users):
    """Train MPCA_FD for all rank configs. Returns V_dict, U_dict keyed by rank tuple."""
    V_dict, U_dict = {}, {}
    for rank in RANK_CONFIGS:
        mpca = MPCA_FD(I_COMMON, rank)
        mpca.iterations = FIT_ITERATIONS
        mpca.train(
            copy.deepcopy(train_users[0]),
            copy.deepcopy(train_users[1]),
            copy.deepcopy(train_users[2])
        )
        rk = tuple(rank)
        V_dict[rk] = [[v.copy() for v in mpca.V_mat[m]] for m in range(3)]
        U_dict[rk] = [u.copy() for u in mpca.U_mat]
    return V_dict, U_dict


def fit_all_ranks_weighted(train_users, weights):
    """Train MPCA_FD_Weighted for all rank configs. Returns V_dict, U_dict."""
    V_dict, U_dict = {}, {}
    for rank in RANK_CONFIGS:
        model = MPCA_FD_Weighted(
            I_COMMON, rank,
            iterations=FIT_ITERATIONS,
            weight_U=False
        )
        _, U_mat, V_mat = model.train(
            [copy.deepcopy(d) for d in train_users],
            [w.copy() for w in weights]
        )
        rk = tuple(rank)
        V_dict[rk] = [[v.copy() for v in V_mat[m]] for m in range(3)]
        U_dict[rk] = [u.copy() for u in U_mat]
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
    test_users = [A_test, B_test, C_test]
    y_train = np.concatenate([y_A_train, y_B_train, y_C_train])
    y_test = np.concatenate([y_A_test, y_B_test, y_C_test])

    rep_results = {}

    # ─── Clean reference ───
    V_clean, U_clean = fit_all_ranks_mpca(train_clean)
    clean_mape = best_rank_mape(
        train_clean, test_users, V_clean, U_clean, y_train, y_test
    )
    rep_results[('clean', 0, 0.0)] = clean_mape

    # ─── Contamination experiments ───
    for noise_mult in NOISE_MULTIPLIERS:
        for pi_s in PI_S_LEVELS:
            if pi_s == 0.0:
                continue

            # Contaminate (separate RNG so main state is unaffected)
            contam_rng = np.random.RandomState(
                rep_seed + int(noise_mult * 100) + int(pi_s * 1000)
            )
            train_dirty = []
            contam_indices = []
            for m in range(3):
                dirty = train_clean[m].copy()
                n_samples = dirty.shape[0]
                n_contam = int(np.ceil(pi_s * n_samples))
                if n_contam > 0:
                    idx = contam_rng.choice(n_samples, size=n_contam, replace=False)
                    noise_std = noise_mult * np.std(train_clean[m])
                    for i in idx:
                        dirty[i] += contam_rng.randn(*dirty[i].shape) * noise_std
                    contam_indices.append(set(idx))
                else:
                    contam_indices.append(set())
                train_dirty.append(dirty)

            # ─── Baseline: unweighted MPCA_FD on dirty data ───
            V_base, U_base = fit_all_ranks_mpca(train_dirty)

            baseline_mape = best_rank_mape(
                train_dirty, test_users, V_base, U_base, y_train, y_test
            )
            rep_results[('baseline', noise_mult, pi_s)] = baseline_mape

            # ─── RFTL-S: compute Huber weights, then weighted re-fit ───
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

            weights = [
                huber_weights(residuals[m], median_r, mad, k=HUBER_K)
                for m in range(3)
            ]

            # Detection metrics
            tp, fp, fn = 0, 0, 0
            for m in range(3):
                flagged = set(np.where(weights[m] < 1.0)[0])
                tp += len(flagged & contam_indices[m])
                fp += len(flagged - contam_indices[m])
                fn += len(contam_indices[m] - flagged)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            rep_results[('rftl_s_prec', noise_mult, pi_s)] = prec
            rep_results[('rftl_s_rec', noise_mult, pi_s)] = rec

            # Weighted re-fit for all ranks
            V_rftl, U_rftl = fit_all_ranks_weighted(train_dirty, weights)

            rftl_mape = best_rank_mape(
                train_dirty, test_users, V_rftl, U_rftl, y_train, y_test
            )
            rep_results[('rftl_s', noise_mult, pi_s)] = rftl_mape

    print(f"  Repeat {rep_idx + 1} done (seed={rep_seed}).", flush=True)
    return rep_results


# ─── Experiment runner ───────────────────────────────────────────────────────

def run_experiment(data_path, n_repeats=50, n_workers=4, seed=2024):
    global _GLOBAL_DATA, _GLOBAL_Y

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
    worker_args = [(i, s) for i, s in enumerate(rep_seeds)]

    print(f"Running {n_repeats} repeats across {n_workers} workers...", flush=True)
    print(f"  Sizes: {SIZES}, Ranks: {len(RANK_CONFIGS)} configs", flush=True)
    print(f"  Noise: {NOISE_MULTIPLIERS}x, pi_S: {PI_S_LEVELS}", flush=True)
    print(f"  Huber k={HUBER_K}, weight_U=False, iterations={FIT_ITERATIONS}",
          flush=True)

    start = time.time()
    if n_workers > 1:
        pool = multiprocessing.Pool(processes=n_workers)
        all_rep = pool.map(_run_single_repeat, worker_args)
        pool.close()
        pool.join()
    else:
        all_rep = [_run_single_repeat(a) for a in worker_args]
    elapsed = (time.time() - start) / 3600

    # Aggregate
    results = {}
    for rep in all_rep:
        for key, val in rep.items():
            results.setdefault(key, []).append(val)

    print(f"\nDone in {elapsed:.2f} hours.", flush=True)
    return results


# ─── Reporting ───────────────────────────────────────────────────────────────

def report_results(results, output_dir=None):
    print("\n" + "=" * 95, flush=True)
    print("RFTL-S PREDICTION RESULTS — REAL DATA (MAPE on log-TTF)", flush=True)
    print("=" * 95, flush=True)

    clean_vals = results.get(('clean', 0, 0.0), [])
    if clean_vals:
        clean_mean = np.mean([v for v in clean_vals if v is not None])
        clean_std = np.std([v for v in clean_vals if v is not None])
        print(f"\nClean reference: MAPE = {clean_mean:.4f} +/- {clean_std:.4f}",
              flush=True)

    for noise_mult in NOISE_MULTIPLIERS:
        print(f"\n--- Noise: {noise_mult}x std ---", flush=True)
        header = f"  {'pi_S':>5} | {'Baseline':>16} | {'RFTL-S':>16} | {'Improv':>7} | {'Prec':>5} {'Rec':>5}"
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)

        for pi_s in PI_S_LEVELS:
            if pi_s == 0.0:
                continue

            b_key = ('baseline', noise_mult, pi_s)
            r_key = ('rftl_s', noise_mult, pi_s)
            p_key = ('rftl_s_prec', noise_mult, pi_s)
            rc_key = ('rftl_s_rec', noise_mult, pi_s)

            b_vals = [v for v in results.get(b_key, []) if v is not None]
            r_vals = [v for v in results.get(r_key, []) if v is not None]
            p_vals = results.get(p_key, [])
            rc_vals = results.get(rc_key, [])

            if b_vals and r_vals:
                b_mean = np.mean(b_vals)
                b_std = np.std(b_vals)
                r_mean = np.mean(r_vals)
                r_std = np.std(r_vals)
                improv = (b_mean - r_mean) / b_mean * 100 if b_mean > 0 else 0
                prec = np.mean(p_vals) if p_vals else 0
                rec = np.mean(rc_vals) if rc_vals else 0
                print(f"  {pi_s:>5.2f} | {b_mean:.4f}+/-{b_std:.4f}"
                      f" | {r_mean:.4f}+/-{r_std:.4f}"
                      f" | {improv:>+6.1f}% | {prec:.3f} {rec:.3f}",
                      flush=True)

    print("\n" + "=" * 95, flush=True)

    # Save to CSV
    if output_dir:
        rows = []
        for key, vals in results.items():
            method, noise, pi_s = key
            for rep_idx, val in enumerate(vals):
                if val is not None:
                    rows.append({
                        'method': method, 'noise_mult': noise,
                        'pi_s': pi_s, 'repeat': rep_idx, 'value': val
                    })
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

    results = run_experiment(
        data_path, n_repeats=args.n_repeats, n_workers=args.n_workers
    )
    report_results(results, output_dir=args.output_dir)
