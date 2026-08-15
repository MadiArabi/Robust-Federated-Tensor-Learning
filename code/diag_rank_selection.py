"""
Diagnostic: which rank does AIC select for clean vs contaminated (baseline) fits?

Background (from the 50-repeat HPC run, June 2026): the contaminated baseline
matches or beats the clean reference on MAPE — at 10x noise it is significantly
BETTER. One hypothesis is that contamination shifts the AIC rank selection
toward smaller (better-generalizing) models. This script logs, for every rank
config, the MAPE / AIC / MPCA_beta ranks for clean and baseline fits, so we can
see whether the improvement comes from rank selection or from every rank
improving.

Only clean + baseline are run (no weighted re-fits), so a repeat costs
~5 MPCA fits instead of ~81.

Usage:
    python diag_rank_selection.py --n-repeats 5 --n-workers 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
import multiprocessing
import pandas as pd
from sklearn.linear_model import Ridge

from my_mpca_02_27_nomean import MPCA_beta, train_test
import tucker_regression0
from rftl_s_real import (
    prepare_data, load_data, project_data, fit_all_ranks_mpca,
    RANK_CONFIGS, SIZES, TEST_SIZE,
)

# Focus on the corners of the grid: mild vs extreme noise, low vs high pi_S
DIAG_NOISE = [2, 10]
DIAG_PI = [0.10, 0.30]

_GLOBAL_DATA = None
_GLOBAL_Y = None


def instrumented_pipeline(prime_train, prime_test, y_train, y_test):
    """Same as rftl_s_real.prediction_pipeline but also returns beta ranks."""
    Min = np.min(prime_train, axis=0)
    Max = np.max(prime_train, axis=0)
    denom = Max - Min
    denom[denom == 0] = 1e-10
    prime_scaled = (prime_train - Min) / denom
    prime_test_scaled = (prime_test - Min) / denom

    P1, P2, P3 = prime_train.shape[1], prime_train.shape[2], prime_train.shape[3]

    try:
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

        return mape, AIC, RSS, (P1_b, P2_b, P3_b)
    except Exception:
        return None, np.inf, None, None


def all_rank_stats(train_users, test_users, V_dict, U_dict, y_train, y_test):
    """Run the pipeline at every rank config, return one row per config."""
    rows = []
    for rank in RANK_CONFIGS:
        rk = tuple(rank)
        prime_train = project_data(train_users, V_dict[rk], U_dict[rk])
        prime_test = project_data(test_users, V_dict[rk], U_dict[rk])
        mape, aic, rss, beta_ranks = instrumented_pipeline(
            prime_train, prime_test, y_train, y_test)
        rows.append({
            'rank': str(rank), 'mape': mape, 'aic': aic, 'rss': rss,
            'beta_ranks': str(beta_ranks),
            'beta_dim': (np.prod(beta_ranks) if beta_ranks else None),
        })
    return rows


def _run_single_repeat(args):
    rep_idx, rep_seed = args
    data, y = _GLOBAL_DATA, _GLOBAL_Y

    # Same random setup as rftl_s_real._run_single_repeat
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

    Abar, A_test, y_A_train, y_A_test = train_test(user1, y_A, TEST_SIZE)
    Bbar, B_test, y_B_train, y_B_test = train_test(user2, y_B, TEST_SIZE)
    Cbar, C_test, y_C_train, y_C_test = train_test(user3, y_C, TEST_SIZE)

    train_clean = [Abar.copy(), Bbar.copy(), Cbar.copy()]
    test_users = [A_test, B_test, C_test]
    y_train = np.concatenate([y_A_train, y_B_train, y_C_train])
    y_test = np.concatenate([y_A_test, y_B_test, y_C_test])

    rows = []

    V_clean, U_clean = fit_all_ranks_mpca(train_clean)
    for r in all_rank_stats(train_clean, test_users, V_clean, U_clean,
                            y_train, y_test):
        rows.append({'repeat': rep_idx, 'method': 'clean',
                     'noise_mult': 0, 'pi_s': 0.0, **r})

    for noise_mult in DIAG_NOISE:
        for pi_s in DIAG_PI:
            # Identical contamination RNG to rftl_s_real
            contam_rng = np.random.RandomState(
                rep_seed + int(noise_mult * 100) + int(pi_s * 1000))
            train_dirty = []
            for m in range(3):
                dirty = train_clean[m].copy()
                n_samples = dirty.shape[0]
                n_contam = int(np.ceil(pi_s * n_samples))
                idx = contam_rng.choice(n_samples, size=n_contam, replace=False)
                noise_std = noise_mult * np.std(train_clean[m])
                for i in idx:
                    dirty[i] += contam_rng.randn(*dirty[i].shape) * noise_std
                train_dirty.append(dirty)

            V_base, U_base = fit_all_ranks_mpca(train_dirty)
            for r in all_rank_stats(train_dirty, test_users, V_base, U_base,
                                    y_train, y_test):
                rows.append({'repeat': rep_idx, 'method': 'baseline',
                             'noise_mult': noise_mult, 'pi_s': pi_s, **r})

    print(f"  Repeat {rep_idx + 1} done (seed={rep_seed}).", flush=True)
    return rows


def _init_worker(data, y):
    global _GLOBAL_DATA, _GLOBAL_Y
    _GLOBAL_DATA, _GLOBAL_Y = data, y


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=5)
    parser.add_argument('--n-workers', type=int, default=5)
    parser.add_argument('--output', default='output/diag_rank_selection.csv')
    args = parser.parse_args()

    data_path = args.data_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    print("Loading real degradation data...", flush=True)
    data, y = load_data(data_path)
    print(f"  Loaded {len(data)} samples", flush=True)

    # Same seed derivation as rftl_s_real so repeats match the HPC run
    master_rng = np.random.RandomState(2024)
    rep_seeds = [int(master_rng.randint(1, 100000))
                 for _ in range(args.n_repeats)]
    worker_args = list(enumerate(rep_seeds))

    print(f"Running {args.n_repeats} repeats x "
          f"(1 clean + {len(DIAG_NOISE) * len(DIAG_PI)} baseline fits)...",
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

    # Quick summary: AIC-selected rank per (method, noise, pi, repeat)
    print("\nAIC-selected rank per condition/repeat:", flush=True)
    ok = df.dropna(subset=['mape'])
    sel = ok.loc[ok.groupby(['method', 'noise_mult', 'pi_s', 'repeat'])['aic']
                 .idxmin()]
    print(sel.groupby(['method', 'noise_mult', 'pi_s'])['rank']
          .value_counts().to_string(), flush=True)
