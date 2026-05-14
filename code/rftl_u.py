"""
RFTL-U: Robust Federated Tensor Learning — User-Level Variant

Detects and excludes corrupted organizations by comparing each user's
locally-implied subspace to the consensus via Grassmann distance.

Usage:
    python rftl_u.py --data-path /path/to/SimulatedData --n-repeats 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import copy
from tensorly import unfold
from tensorly.tenalg import multi_mode_dot


def grassmann_distance(A, B):
    """Geodesic distance on the Grassmann manifold between column spans of A and B."""
    svs = np.linalg.svd(A.T @ B, compute_uv=False)
    svs = np.clip(svs, -1, 1)
    angles = np.arccos(svs)
    return np.sqrt(np.sum(angles ** 2))


def user_implied_U(user_projected, U_mat, mode, P_n):
    """Compute what U_n would be if only this user's data contributed.

    Projects user's V-projected data through the other modes' U, stacks
    across samples, and takes the leading left singular vectors.
    """
    first = min((mode + 1) % 3, (mode + 2) % 3)
    second = max((mode + 1) % 3, (mode + 2) % 3)
    kron_u = np.kron(U_mat[first], U_mat[second])

    all_cols = []
    for j in range(user_projected.shape[0]):
        sample = unfold(user_projected[j], mode=mode) @ kron_u
        all_cols.append(sample)

    stacked = np.hstack(all_cols)
    u, _, _ = np.linalg.svd(stacked, full_matrices=False)
    return u[:, :P_n]


# ─── RFTL-U class ────────────────────────────────────────────────────────────

class RFTL_U:
    """
    RFTL-U: User-level robustness via Grassmann distance.

    1. Fit MPCA_FD on all users
    2. Compute each user's locally-implied U_n from their data alone
    3. Grassmann distance d_m = sum_n d(U_n^(m), U_n^consensus)
    4. Exclude users with d_m > c * median(d_1,...,d_M)
    5. Cold re-fit on remaining users' pooled data
    """

    def __init__(self, I, P, threshold_c=3.0, inner_iterations=200):
        self.I = I
        self.P = P
        self.threshold_c = threshold_c
        self.inner_iterations = inner_iterations

    def fit(self, user_data):
        from my_mpca_02_27_nomean import MPCA_FD
        M = len(user_data)

        mpca = MPCA_FD(self.I, self.P)
        mpca.iterations = self.inner_iterations
        mpca.train(
            copy.deepcopy(user_data[0]),
            copy.deepcopy(user_data[1]),
            copy.deepcopy(user_data[2])
        )
        U_mat = [u.copy() for u in mpca.U_mat]
        V_mat = [[v.copy() for v in mpca.V_mat[m]] for m in range(M)]

        projected = [multi_mode_dot(user_data[m], [v.T for v in V_mat[m]],
                     modes=[1, 2, 3]) for m in range(M)]

        distances = np.zeros(M)
        for m in range(M):
            for n in range(3):
                U_n_m = user_implied_U(projected[m], U_mat, n, self.P[n])
                distances[m] += grassmann_distance(U_mat[n], U_n_m)

        median_d = np.median(distances)
        if median_d < 1e-10:
            median_d = 1e-10
        threshold = self.threshold_c * median_d
        user_weights = np.where(distances <= threshold, 1.0, 0.0)
        n_excluded = int(np.sum(user_weights == 0))

        if 0 < n_excluded < M:
            good_data = np.concatenate(
                [user_data[m] for m in range(M) if user_weights[m] == 1.0])
            n = len(good_data)
            s1, s2 = n // 3, 2 * n // 3
            mpca_trim = MPCA_FD(self.I, self.P)
            mpca_trim.iterations = self.inner_iterations
            mpca_trim.train(
                copy.deepcopy(good_data[:s1]),
                copy.deepcopy(good_data[s1:s2]),
                copy.deepcopy(good_data[s2:])
            )
            U_mat = [u.copy() for u in mpca_trim.U_mat]

        self.U_mat = U_mat
        self.V_mat = V_mat
        self.distances = distances
        self.user_weights = user_weights
        self.n_excluded = n_excluded

        return self


# ─── Experiment runner ────────────────────────────────────────────────────────

N_CORRUPT_LEVELS = [0, 1, 2]
NOISE_MULTIPLIERS = [2, 3, 5, 10]
SIZES = [70, 100, 130]
I_COMMON = [21, 21, 10]
RANK = [5, 5, 4]

_GLOBAL_DATA = None

CONFIGS = [
    ('c=2', 2.0),
    ('c=3', 3.0),
    ('c=5', 5.0),
]


def _run_single_repeat(args):
    """Worker: for each (noise_mult, n_corrupt) pair, corrupt n_corrupt users,
    run baseline MPCA_FD, compute Grassmann distances, and cold re-fit
    excluding detected bad users.
    """
    rep_idx, rep_seed = args
    from motivation_pilot import setup_users, principal_angles_deg
    from my_mpca_02_27_nomean import MPCA_FD

    data = _GLOBAL_DATA
    rng = np.random.RandomState(rep_seed)
    sample = np.arange(len(data))
    rng.shuffle(sample)

    user1, user2, user3 = setup_users(data, sample, SIZES, seed=rep_seed)
    users_centered = [u - np.mean(u, axis=0) for u in [user1, user2, user3]]

    mpca_clean = MPCA_FD(I_COMMON, RANK)
    mpca_clean.iterations = 200
    mpca_clean.train(
        copy.deepcopy(users_centered[0]),
        copy.deepcopy(users_centered[1]),
        copy.deepcopy(users_centered[2])
    )
    U_star = [u.copy() for u in mpca_clean.U_mat]

    rep_results = {}

    # n_corrupt=0: clean baseline (noise irrelevant)
    mpca_base0 = MPCA_FD(I_COMMON, RANK)
    mpca_base0.iterations = 200
    mpca_base0.train(
        copy.deepcopy(users_centered[0]),
        copy.deepcopy(users_centered[1]),
        copy.deepcopy(users_centered[2])
    )
    U_b0 = [u.copy() for u in mpca_base0.U_mat]
    angle_clean = np.mean([a for n in range(3)
                           for a in principal_angles_deg(U_star[n], U_b0[n])])
    rep_results[('baseline', 0, 0)] = angle_clean

    for noise_mult in NOISE_MULTIPLIERS:
        for n_corrupt in N_CORRUPT_LEVELS:
            if n_corrupt == 0:
                continue

            contam_rng = np.random.RandomState(rep_seed + n_corrupt * 1000)
            corrupt_users = set(contam_rng.choice(3, size=n_corrupt, replace=False))

            users_dirty = []
            for m in range(3):
                dirty = users_centered[m].copy()
                if m in corrupt_users:
                    noise_std = noise_mult * np.std(users_centered[m])
                    dirty += contam_rng.randn(*dirty.shape) * noise_std
                users_dirty.append(dirty)

            mpca_baseline = MPCA_FD(I_COMMON, RANK)
            mpca_baseline.iterations = 200
            mpca_baseline.train(
                copy.deepcopy(users_dirty[0]),
                copy.deepcopy(users_dirty[1]),
                copy.deepcopy(users_dirty[2])
            )
            U_unw = [u.copy() for u in mpca_baseline.U_mat]
            V_unw = [[v.copy() for v in mpca_baseline.V_mat[m]] for m in range(3)]

            angle_base = np.mean([a for n in range(3)
                                  for a in principal_angles_deg(U_star[n], U_unw[n])])
            rep_results[('baseline', noise_mult, n_corrupt)] = angle_base

            projected = [multi_mode_dot(users_dirty[m],
                         [v.T for v in V_unw[m]], modes=[1, 2, 3])
                         for m in range(3)]

            distances = np.zeros(3)
            for m in range(3):
                for n in range(3):
                    U_n_m = user_implied_U(projected[m], U_unw, n, RANK[n])
                    distances[m] += grassmann_distance(U_unw[n], U_n_m)

            median_d = np.median(distances)
            if median_d < 1e-10:
                median_d = 1e-10

            for label, threshold_c in CONFIGS:
                threshold = threshold_c * median_d
                user_weights = np.where(distances <= threshold, 1.0, 0.0)
                n_excluded = int(np.sum(user_weights == 0))

                if 0 < n_excluded < 3:
                    good_data = np.concatenate(
                        [users_dirty[m] for m in range(3)
                         if user_weights[m] == 1.0])
                    n = len(good_data)
                    s1, s2 = n // 3, 2 * n // 3
                    mpca_trim = MPCA_FD(I_COMMON, RANK)
                    mpca_trim.iterations = 200
                    mpca_trim.train(
                        copy.deepcopy(good_data[:s1]),
                        copy.deepcopy(good_data[s1:s2]),
                        copy.deepcopy(good_data[s2:])
                    )
                    U_result = [u.copy() for u in mpca_trim.U_mat]
                else:
                    U_result = U_unw

                angle_trim = np.mean([a for n in range(3)
                                      for a in principal_angles_deg(
                                          U_star[n], U_result[n])])
                rep_results[(label, noise_mult, n_corrupt)] = angle_trim

                excluded_set = set(
                    m for m in range(3) if user_weights[m] == 0.0)
                tp = len(excluded_set & corrupt_users)
                fp = len(excluded_set - corrupt_users)
                fn = len(corrupt_users - excluded_set)
                rep_results[(f'{label}_prec', noise_mult, n_corrupt)] = (
                    tp / (tp + fp) if (tp + fp) > 0 else 0.0)
                rep_results[(f'{label}_rec', noise_mult, n_corrupt)] = (
                    tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    print(f"  Repeat {rep_idx + 1} done (seed={rep_seed}).", flush=True)
    return rep_results


def run_rftl_u_experiment(data_path, n_repeats=10, max_files=350,
                          n_workers=4, seed=2024):
    import multiprocessing
    from motivation_pilot import load_simulated_data

    global _GLOBAL_DATA

    print("Loading simulated data...", flush=True)
    min_needed = sum(SIZES)
    _GLOBAL_DATA = load_simulated_data(data_path,
                                       max_files=max(max_files, min_needed + 10))
    n_total = len(_GLOBAL_DATA)
    print(f"  Loaded {n_total} samples", flush=True)
    if n_total < min_needed:
        print(f"  ERROR: Need at least {min_needed} samples, got {n_total}",
              flush=True)
        sys.exit(1)

    master_rng = np.random.RandomState(seed)
    rep_seeds = [int(master_rng.randint(1, 100000)) for _ in range(n_repeats)]
    worker_args = [(i, s) for i, s in enumerate(rep_seeds)]

    print(f"Running {n_repeats} repeats across {n_workers} workers...",
          flush=True)
    print(f"Configs: {[c[0] for c in CONFIGS]}, Noise: {NOISE_MULTIPLIERS}x",
          flush=True)

    if n_workers > 1:
        pool = multiprocessing.Pool(processes=n_workers)
        all_rep = pool.map(_run_single_repeat, worker_args)
        pool.close()
        pool.join()
    else:
        all_rep = [_run_single_repeat(a) for a in worker_args]

    results = {}
    for rep in all_rep:
        for key, val in rep.items():
            results.setdefault(key, []).append(val)

    return results


def report_rftl_u_results(results):
    config_labels = [c[0] for c in CONFIGS]

    print("\n" + "=" * 90, flush=True)
    print("RFTL-U EXPERIMENT RESULTS — USER-LEVEL COLD RE-FIT", flush=True)
    print("=" * 90, flush=True)

    clean_key = ('baseline', 0, 0)
    if clean_key in results:
        print(f"\nClean data (0 corrupt users): baseline = "
              f"{np.mean(results[clean_key]):.2f} deg", flush=True)

    for noise_mult in NOISE_MULTIPLIERS:
        print(f"\n--- Noise: {noise_mult}x baseline std ---", flush=True)

        header = f"  {'#bad':>4} | {'Baseline':>8}"
        for label in config_labels:
            header += f" | {label:>14}"
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)

        for n_corrupt in N_CORRUPT_LEVELS:
            if n_corrupt == 0:
                continue
            b_key = ('baseline', noise_mult, n_corrupt)
            b_mean = np.mean(results.get(b_key, [0]))
            row = f"  {n_corrupt:>4} | {b_mean:>8.2f}"
            for label in config_labels:
                r_key = (label, noise_mult, n_corrupt)
                r_mean = np.mean(results.get(r_key, [0]))
                improvement = b_mean - r_mean
                row += f" | {r_mean:>6.2f} ({improvement:>+5.1f})"
            print(row, flush=True)

        print(f"\n  Detection (noise={noise_mult}x):", flush=True)
        det_hdr = f"  {'#bad':>4}"
        for label in config_labels:
            det_hdr += f" |  {label} P  {label} R"
        print(det_hdr, flush=True)
        print("  " + "-" * (len(det_hdr) - 2), flush=True)

        for n_corrupt in N_CORRUPT_LEVELS:
            if n_corrupt == 0:
                continue
            row = f"  {n_corrupt:>4}"
            for label in config_labels:
                p_key = (f'{label}_prec', noise_mult, n_corrupt)
                r_key = (f'{label}_rec', noise_mult, n_corrupt)
                p = np.mean(results.get(p_key, [0]))
                r = np.mean(results.get(r_key, [0]))
                row += f" | {p:>6.3f} {r:>6.3f}"
            print(row, flush=True)

    print("\n" + "=" * 90, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RFTL-U Experiment')
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=10)
    parser.add_argument('--max-files', type=int, default=350)
    parser.add_argument('--n-workers', type=int, default=4)
    args = parser.parse_args()

    if args.data_path:
        data_path = args.data_path
    else:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data', 'Simulated Data')

    if not os.path.exists(data_path):
        print(f"ERROR: Data path not found: {data_path}", flush=True)
        sys.exit(1)

    results = run_rftl_u_experiment(
        data_path, n_repeats=args.n_repeats, max_files=args.max_files,
        n_workers=args.n_workers
    )
    report_rftl_u_results(results)
