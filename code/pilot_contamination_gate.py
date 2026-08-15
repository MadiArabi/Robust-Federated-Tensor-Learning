"""
Pilot gate for structured contamination models (hot_block, stripe).

Lesson from the June 2026 HPC run: Gaussian sample contamination never hurt
downstream prediction, so 2.6 days of HPC bought a null result. This pilot is
the gate we now require BEFORE any HPC submission: run clean vs baseline
(no robustness) at harsh contamination settings for a few repeats, and check
that the baseline actually degrades. Only if it does is the full experiment
worth running.

Runs clean + baseline only (no weighted re-fits). Contamination is
concentrated in user 0 (faulty-camera scenario). For each condition we record
the full error distribution — mean (MAPE), median, 25th/75th percentile of
|predicted - true| / |true| on log-TTF — overall and per user, since user 0
should be hit hardest.

Usage:
    python pilot_contamination_gate.py --n-repeats 5 --n-workers 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
import multiprocessing
import pandas as pd

from my_mpca_02_27_nomean import train_test
from contamination import contaminate
from rftl_s_real import (
    prepare_data, load_data, project_data, fit_all_ranks_mpca,
    prediction_pipeline, RANK_CONFIGS, SIZES, TEST_SIZE,
)

# Escalated settings (2026-07-07): the first gate run (amp {5,10}, pi_s=0.3,
# block 5, 3 stripes) came back benign on all four conditions — amplitude had
# no effect, consistent with min-max absorbing large-amplitude artifacts. So
# escalate the *extent* instead: contaminate most/all of user 0's training
# samples with much larger artifacts.
PILOT_CONDITIONS = [
    # (mode, amplitude, pi_s)
    ('hot_block', 10.0, 0.60),
    ('hot_block', 10.0, 1.00),
    ('stripe', 10.0, 0.60),
    ('stripe', 10.0, 1.00),
]
BLOCK_SIZE = 9
N_STRIPES = 6
TARGET_USER = 0

_GLOBAL_DATA = None
_GLOBAL_Y = None


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
    """AIC-select over rank configs; return error stats + selected rank."""
    best_aic = np.inf
    best = None
    for rank in RANK_CONFIGS:
        rk = tuple(rank)
        prime_train = project_data(train_users, V_dict[rk], U_dict[rk])
        prime_test = project_data(test_users, V_dict[rk], U_dict[rk])
        mape, aic, rel_errors = prediction_pipeline(
            prime_train, prime_test, y_train, y_test)
        if mape is not None and aic < best_aic:
            best_aic = aic
            best = {'selected_rank': str(rank), **error_stats(rel_errors)}
    return best


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
    stats = best_rank_stats(train_clean, test_users, V_clean, U_clean,
                            y_train, y_test)
    rows.append({'repeat': rep_idx, 'method': 'clean', 'mode': 'none',
                 'amplitude': 0.0, 'pi_s': 0.0, **stats})

    for cond_idx, (mode, amplitude, pi_s) in enumerate(PILOT_CONDITIONS):
        # str hash is randomized per process — use the condition index instead
        contam_rng = np.random.RandomState(
            rep_seed + 1000 * (cond_idx + 1) + int(pi_s * 100))
        train_dirty, _ = contaminate(
            train_clean, pi_s, mode, contam_rng, amplitude=amplitude,
            block_size=BLOCK_SIZE, n_stripes=N_STRIPES,
            target_user=TARGET_USER)

        V_base, U_base = fit_all_ranks_mpca(train_dirty)
        stats = best_rank_stats(train_dirty, test_users, V_base, U_base,
                                y_train, y_test)
        rows.append({'repeat': rep_idx, 'method': 'baseline', 'mode': mode,
                     'amplitude': amplitude, 'pi_s': pi_s, **stats})

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
    parser.add_argument('--output', default='output/pilot_contamination_gate.csv')
    args = parser.parse_args()

    data_path = args.data_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    print("Loading real degradation data...", flush=True)
    data, y = load_data(data_path)
    print(f"  Loaded {len(data)} samples", flush=True)

    # Same seed derivation as rftl_s_real so repeats match across experiments
    master_rng = np.random.RandomState(2024)
    rep_seeds = [int(master_rng.randint(1, 100000))
                 for _ in range(args.n_repeats)]
    worker_args = list(enumerate(rep_seeds))

    n_fits = 1 + len(PILOT_CONDITIONS)
    print(f"Running {args.n_repeats} repeats x {n_fits} fits "
          f"(contamination in user {TARGET_USER} only)...", flush=True)

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

    # ─── Gate summary ───
    print("\n" + "=" * 90, flush=True)
    print("GATE CHECK: does contamination hurt the baseline?", flush=True)
    print("=" * 90, flush=True)
    clean = df[df.method == 'clean']
    print(f"\nClean:  MAPE {clean['mape'].mean():.4f}  "
          f"median {clean['median'].mean():.4f}  "
          f"[q25 {clean['q25'].mean():.4f}, q75 {clean['q75'].mean():.4f}]  "
          f"user0-MAPE {clean['mape_user0'].mean():.4f}", flush=True)

    for mode, amplitude, pi_s in PILOT_CONDITIONS:
        sub = df[(df['mode'] == mode) & (df.amplitude == amplitude)
                 & (df.pi_s == pi_s)]
        if sub.empty:
            continue
        degr = (sub['mape'].mean() - clean['mape'].mean()) \
            / clean['mape'].mean() * 100
        degr_u0 = (sub['mape_user0'].mean() - clean['mape_user0'].mean()) \
            / clean['mape_user0'].mean() * 100
        verdict = 'PASS (hurts)' if degr > 20 else 'fail (benign)'
        print(f"\n{mode} amp={amplitude} pi_s={pi_s}:  "
              f"MAPE {sub['mape'].mean():.4f}  "
              f"median {sub['median'].mean():.4f}  "
              f"[q25 {sub['q25'].mean():.4f}, q75 {sub['q75'].mean():.4f}]  "
              f"user0-MAPE {sub['mape_user0'].mean():.4f}", flush=True)
        print(f"    degradation vs clean: overall {degr:+.1f}%  "
              f"user0 {degr_u0:+.1f}%   -> {verdict}", flush=True)
        print(f"    AIC-selected ranks: "
              f"{sub['selected_rank'].value_counts().to_dict()}", flush=True)
