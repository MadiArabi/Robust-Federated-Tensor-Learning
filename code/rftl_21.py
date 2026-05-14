"""
RFTL-21: Robust Federated Tensor Learning — Frame-Level Variant

Detects corrupted individual frames (slabs) via per-frame reconstruction
residuals, zeros them out, and cold re-fits on cleaned data.

The ℓ₂,₁ norm naturally downweights rows (frames) with large norms.
This implementation uses the hard-threshold cold-re-fit analog:
flag frames with residual > c * median, zero them, re-fit from scratch.

Usage:
    python rftl_21.py --data-path /path/to/SimulatedData --n-repeats 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import copy
from tensorly.tenalg import multi_mode_dot


def per_frame_residuals(X, V_mat, U_mat, mode=2):
    """Per-frame reconstruction residuals along the given mode.

    For each sample j, reconstructs via C_n = V_n @ U_n projection and
    measures error at each frame (slab along `mode`).

    Returns:
        residuals: (J, n_frames) array of per-frame Frobenius norms
    """
    J = X.shape[0]
    C = [V_mat[n] @ U_mat[n] for n in range(3)]
    recon_mats = [c @ c.T for c in C]

    n_frames = X.shape[mode + 1]
    residuals = np.zeros((J, n_frames))

    for j in range(J):
        X_recon = multi_mode_dot(X[j], recon_mats, modes=[0, 1, 2])
        error = X[j] - X_recon

        for k in range(n_frames):
            if mode == 0:
                frame_err = error[k, :, :]
            elif mode == 1:
                frame_err = error[:, k, :]
            else:
                frame_err = error[:, :, k]
            residuals[j, k] = np.sqrt(np.sum(frame_err ** 2))

    return residuals


# ─── RFTL-21 class ───────────────────────────────────────────────────────────

class RFTL_21:
    """
    RFTL-21: Frame-level robustness via per-frame reconstruction residuals.

    1. Fit MPCA_FD on all data
    2. Per-frame reconstruction residuals along detect_mode
    3. Flag frames with residual > c * median(all frame residuals)
    4. Zero out flagged frames, cold re-fit MPCA_FD
    """

    def __init__(self, I, P, threshold_c=3.0, inner_iterations=200,
                 detect_mode=2):
        self.I = I
        self.P = P
        self.threshold_c = threshold_c
        self.inner_iterations = inner_iterations
        self.detect_mode = detect_mode

    def fit(self, user_data):
        from my_mpca_02_27_nomean import MPCA_FD
        M = len(user_data)

        mpca_init = MPCA_FD(self.I, self.P)
        mpca_init.iterations = self.inner_iterations
        mpca_init.train(
            copy.deepcopy(user_data[0]),
            copy.deepcopy(user_data[1]),
            copy.deepcopy(user_data[2])
        )
        U_mat = [u.copy() for u in mpca_init.U_mat]
        V_mat = [[v.copy() for v in mpca_init.V_mat[m]] for m in range(M)]

        frame_residuals = [per_frame_residuals(user_data[m], V_mat[m], U_mat,
                           self.detect_mode) for m in range(M)]
        all_res = np.concatenate([r.ravel() for r in frame_residuals])
        median_r = np.median(all_res)
        threshold = self.threshold_c * median_r

        frame_masks = [fr <= threshold for fr in frame_residuals]
        n_flagged = sum(int(np.sum(~mask)) for mask in frame_masks)

        if n_flagged > 0:
            users_cleaned = []
            for m in range(M):
                cleaned = user_data[m].copy()
                bad_j, bad_k = np.where(~frame_masks[m])
                for j, k in zip(bad_j, bad_k):
                    if self.detect_mode == 0:
                        cleaned[j, k, :, :] = 0.0
                    elif self.detect_mode == 1:
                        cleaned[j, :, k, :] = 0.0
                    else:
                        cleaned[j, :, :, k] = 0.0
                users_cleaned.append(cleaned)

            mpca_clean = MPCA_FD(self.I, self.P)
            mpca_clean.iterations = self.inner_iterations
            mpca_clean.train(
                copy.deepcopy(users_cleaned[0]),
                copy.deepcopy(users_cleaned[1]),
                copy.deepcopy(users_cleaned[2])
            )
            U_mat = [u.copy() for u in mpca_clean.U_mat]
            V_mat = [[v.copy() for v in mpca_clean.V_mat[m]] for m in range(M)]

        self.U_mat = U_mat
        self.V_mat = V_mat
        self.frame_residuals = frame_residuals
        self.frame_masks = frame_masks
        self.threshold = threshold
        self.n_flagged = n_flagged

        return self


# ─── Experiment runner ────────────────────────────────────────────────────────

PI_F_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
NOISE_MULTIPLIERS = [2, 3, 5, 10]
SIZES = [70, 100, 130]
I_COMMON = [21, 21, 10]
RANK = [5, 5, 4]
DETECT_MODE = 2

_GLOBAL_DATA = None

CONFIGS = [
    ('c=2', 2.0),
    ('c=3', 3.0),
    ('c=5', 5.0),
]


def _run_single_repeat(args):
    """Worker: for each (noise_mult, pi_F) pair, corrupt individual frames,
    run baseline MPCA_FD, detect bad frames via per-frame residuals,
    zero them out, and cold re-fit.
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
    rep_results[('baseline', 0, 0.0)] = angle_clean

    n_frames = users_centered[0].shape[DETECT_MODE + 1]

    for noise_mult in NOISE_MULTIPLIERS:
        for pi_f in PI_F_LEVELS:
            if pi_f == 0.0:
                continue

            contam_rng = np.random.RandomState(rep_seed + int(pi_f * 1000))

            corrupted_frames = []
            users_dirty = []
            for m in range(3):
                n_samples = users_centered[m].shape[0]
                dirty = users_centered[m].copy()
                frames_m = set()

                for j in range(n_samples):
                    for k in range(n_frames):
                        if contam_rng.random() < pi_f:
                            noise_std = noise_mult * np.std(users_centered[m])
                            if DETECT_MODE == 0:
                                dirty[j, k, :, :] += contam_rng.randn(
                                    *dirty[j, k, :, :].shape) * noise_std
                            elif DETECT_MODE == 1:
                                dirty[j, :, k, :] += contam_rng.randn(
                                    *dirty[j, :, k, :].shape) * noise_std
                            else:
                                dirty[j, :, :, k] += contam_rng.randn(
                                    *dirty[j, :, :, k].shape) * noise_std
                            frames_m.add((j, k))

                corrupted_frames.append(frames_m)
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
                                  for a in principal_angles_deg(U_star[n],
                                                                U_unw[n])])
            rep_results[('baseline', noise_mult, pi_f)] = angle_base

            frame_res = [per_frame_residuals(users_dirty[m], V_unw[m], U_unw,
                         DETECT_MODE) for m in range(3)]
            all_res = np.concatenate([r.ravel() for r in frame_res])
            median_r = np.median(all_res)

            for label, threshold_c in CONFIGS:
                threshold = threshold_c * median_r
                frame_masks = [frame_res[m] <= threshold for m in range(3)]
                n_flagged = sum(int(np.sum(~mask)) for mask in frame_masks)

                if n_flagged > 0:
                    users_cleaned = []
                    for m in range(3):
                        cleaned = users_dirty[m].copy()
                        bad_j, bad_k = np.where(~frame_masks[m])
                        for j, k in zip(bad_j, bad_k):
                            if DETECT_MODE == 0:
                                cleaned[j, k, :, :] = 0.0
                            elif DETECT_MODE == 1:
                                cleaned[j, :, k, :] = 0.0
                            else:
                                cleaned[j, :, :, k] = 0.0
                        users_cleaned.append(cleaned)

                    mpca_trim = MPCA_FD(I_COMMON, RANK)
                    mpca_trim.iterations = 200
                    mpca_trim.train(
                        copy.deepcopy(users_cleaned[0]),
                        copy.deepcopy(users_cleaned[1]),
                        copy.deepcopy(users_cleaned[2])
                    )
                    U_result = [u.copy() for u in mpca_trim.U_mat]
                else:
                    U_result = U_unw

                angle_trim = np.mean([a for n in range(3)
                                      for a in principal_angles_deg(
                                          U_star[n], U_result[n])])
                rep_results[(label, noise_mult, pi_f)] = angle_trim

                tp, fp, fn = 0, 0, 0
                for m in range(3):
                    bad_j, bad_k = np.where(~frame_masks[m])
                    flagged_set = set(zip(bad_j.tolist(), bad_k.tolist()))
                    true_bad = corrupted_frames[m]
                    tp += len(flagged_set & true_bad)
                    fp += len(flagged_set - true_bad)
                    fn += len(true_bad - flagged_set)

                rep_results[(f'{label}_prec', noise_mult, pi_f)] = (
                    tp / (tp + fp) if (tp + fp) > 0 else 0.0)
                rep_results[(f'{label}_rec', noise_mult, pi_f)] = (
                    tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    print(f"  Repeat {rep_idx + 1} done (seed={rep_seed}).", flush=True)
    return rep_results


def run_rftl_21_experiment(data_path, n_repeats=10, max_files=350,
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


def report_rftl_21_results(results):
    config_labels = [c[0] for c in CONFIGS]

    print("\n" + "=" * 90, flush=True)
    print("RFTL-21 EXPERIMENT RESULTS — FRAME-LEVEL COLD RE-FIT", flush=True)
    print("=" * 90, flush=True)

    clean_key = ('baseline', 0, 0.0)
    if clean_key in results:
        print(f"\nClean data (pi_F=0.00): baseline = "
              f"{np.mean(results[clean_key]):.2f} deg", flush=True)

    for noise_mult in NOISE_MULTIPLIERS:
        print(f"\n--- Noise: {noise_mult}x baseline std ---", flush=True)

        header = f"  {'pi_F':>5} | {'Baseline':>8}"
        for label in config_labels:
            header += f" | {label:>14}"
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)

        for pi_f in PI_F_LEVELS:
            if pi_f == 0.0:
                continue
            b_key = ('baseline', noise_mult, pi_f)
            b_mean = np.mean(results.get(b_key, [0]))
            row = f"  {pi_f:>5.2f} | {b_mean:>8.2f}"
            for label in config_labels:
                r_key = (label, noise_mult, pi_f)
                r_mean = np.mean(results.get(r_key, [0]))
                improvement = b_mean - r_mean
                row += f" | {r_mean:>6.2f} ({improvement:>+5.1f})"
            print(row, flush=True)

        print(f"\n  Frame detection (noise={noise_mult}x):", flush=True)
        det_hdr = f"  {'pi_F':>5}"
        for label in config_labels:
            det_hdr += f" |  {label} P  {label} R"
        print(det_hdr, flush=True)
        print("  " + "-" * (len(det_hdr) - 2), flush=True)

        for pi_f in PI_F_LEVELS:
            if pi_f == 0.0:
                continue
            row = f"  {pi_f:>5.2f}"
            for label in config_labels:
                p_key = (f'{label}_prec', noise_mult, pi_f)
                r_key = (f'{label}_rec', noise_mult, pi_f)
                p = np.mean(results.get(p_key, [0]))
                r = np.mean(results.get(r_key, [0]))
                row += f" | {p:>6.3f} {r:>6.3f}"
            print(row, flush=True)

    print("\n" + "=" * 90, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RFTL-21 Experiment')
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

    results = run_rftl_21_experiment(
        data_path, n_repeats=args.n_repeats, max_files=args.max_files,
        n_workers=args.n_workers
    )
    report_rftl_21_results(results)
