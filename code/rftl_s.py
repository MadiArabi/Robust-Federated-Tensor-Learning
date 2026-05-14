"""
RFTL-S: Robust Federated Tensor Learning — Sample-Weighted Variant

IRLS wrapper around Chapter 2's MPCA_FD that down-weights contaminated samples
via Huber-like weights on reconstruction residuals.

Key modifications vs. Chapter 2:
  - Scatter matrix in V update: weighted by w_j^m per sample
  - Incremental SVD in U update: each column block scaled by sqrt(w_j^m)
  - Outer IRLS loop: residuals → MAD → Huber weights → repeat

Usage:
    python rftl_s.py --data-path /path/to/SimulatedData --n-repeats 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import copy
from tensorly import unfold
from tensorly.tenalg import multi_mode_dot
from scipy.io import loadmat


# ─── Federated MAD ───────────────────────────────────────────────────────────

def federated_mad(residuals_per_user):
    """
    Compute MAD (median absolute deviation) across all users via
    median-of-medians approximation.

    Args:
        residuals_per_user: list of arrays, one per user, each containing
                           per-sample residuals for that user.
    Returns:
        mad: scalar MAD estimate
    """
    all_residuals = np.concatenate(residuals_per_user)
    median_r = np.median(all_residuals)
    mad = np.median(np.abs(all_residuals - median_r))
    if mad < 1e-10:
        mad = 1e-10
    return mad


def huber_weights(residuals, median_r, mad, k=1.345):
    """
    Compute Huber-like weights based on standardized residuals.
    Standardized residual: s_j = |r_j - median(r)| / MAD
    Weight: w_j = min(1, k / s_j)

    Samples near the median get w=1; samples far from it get downweighted.
    """
    standardized = np.abs(residuals - median_r) / max(mad, 1e-10)
    weights = np.where(standardized <= k, 1.0,  k / np.maximum(standardized, 1e-10))
    return weights


# ─── Reconstruction residual ─────────────────────────────────────────────────

def reconstruction_residual(X, V_mat, U_mat):
    """
    Compute per-sample reconstruction residual:
        r_j = ||X_j - X_j x_1 V1 V1^T x_2 V2 V2^T x_3 V3 V3^T x_1 U1 U1^T x_2 U2 U2^T x_3 U3 U3^T||_F

    Simplified: project to low-dim, reconstruct, measure error.

    Args:
        X: (J, I1, I2, I3) tensor data for one user (mean-centered)
        V_mat: list of 3 local projection matrices [V1, V2, V3]
        U_mat: list of 3 global projection matrices [U1, U2, U3]

    Returns:
        residuals: (J,) array of per-sample Frobenius residuals
    """
    J = X.shape[0]
    residuals = np.zeros(J)

    # Combined projection matrices C_n = V_n @ U_n
    C = [V_mat[n] @ U_mat[n] for n in range(3)]

    # Reconstruction matrices: C_n @ C_n^T projects and reconstructs in mode n
    # Full reconstruction: X_hat = X x_1 (C1 C1^T) x_2 (C2 C2^T) x_3 (C3 C3^T)
    recon_mats = [c @ c.T for c in C]

    for j in range(J):
        X_recon = multi_mode_dot(X[j], [r for r in recon_mats], modes=[0, 1, 2])
        residuals[j] = np.sqrt(np.sum((X[j] - X_recon) ** 2))

    return residuals


# ─── Weighted MPCA_FD ─────────────────────────────────────────────────────────

class MPCA_FD_Weighted:
    """
    Federated MPCA with per-sample weights. Implements Propositions 1' and 2'
    from Chapter 3.

    Structurally identical to MPCA_FD from my_mpca_02_27_nomean.py, but:
    - V update uses weighted scatter: Phi^{m,w}_{(n)} = sum_j w_j X_{j(n)}^{Phi} X_{j(n)}^{Phi T}
    - U update: optionally scales columns by sqrt(w_j) before incremental SVD
    """

    def __init__(self, I, P, iterations=30, weight_U=False):
        self.I = I
        self.P = P
        self.iterations = iterations
        self.weight_U = weight_U
        self.lam = 0.00001

    def projection(self, data, matrix):
        return multi_mode_dot(data, [m.T for m in matrix], modes=[1, 2, 3])

    def train(self, user_data, weights_per_user, init_V=None, init_U=None):
        """
        Train weighted federated MPCA.

        Args:
            user_data: list of 3 arrays, each (J_m, I1_m, I2_m, I3_m)
            weights_per_user: list of 3 arrays, each (J_m,) non-negative weights
            init_V: optional warm-start V_mat from previous IRLS iteration
            init_U: optional warm-start U_mat from previous IRLS iteration

        Returns:
            prime: projected features (concatenated across users)
            U_mat: list of 3 global projection matrices
            V_mat: list of lists of 3 local projection matrices per user
        """
        M = len(user_data)

        # ─── Initialization (unweighted, as in Chapter 2) ───
        def Vinitial(x, MODE, S):
            unfolded_x = np.array([unfold(x[i], mode=MODE).T for i in range(x.shape[0])])
            scatter = np.einsum("nij,nik->jk", unfolded_x, unfolded_x)
            eigenvalues, u = np.linalg.eig(scatter)
            u = np.real_if_close(u, tol=1)
            eigenvalues = np.real_if_close(eigenvalues, tol=1)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            return u[:, sorted_indices][:, :S]

        def Uinitial(projected_data, MODE, P):
            """Incremental SVD initialization across all users."""
            all_samples = []
            for user_proj in projected_data:
                for i in range(user_proj.shape[0]):
                    all_samples.append(unfold(user_proj[i], mode=MODE))

            I1 = all_samples[0].shape[0]
            X = all_samples[0]
            u, s, _ = np.linalg.svd(X, full_matrices=True)
            sigma = np.zeros_like(X, dtype=float)
            k = min(X.shape[0], X.shape[1])
            sigma[:k, :k] = np.diag(s[:k])

            for sample in all_samples[1:]:
                W = sample - u @ u.T @ sample
                norm_W = np.linalg.norm(W, axis=0)
                norm_W[norm_W < 1e-10] = 1e-10
                norm_w = W / norm_W
                M_mat = np.block([
                    [sigma, u.T @ sample],
                    [np.zeros((sample.shape[1], sigma.shape[1])),
                     np.diag(norm_W ** 2)]
                ])
                u_prime, sigma_prime, _ = np.linalg.svd(M_mat, full_matrices=True)
                u = (np.hstack((u, norm_w)) @ u_prime)[:, :I1]
                sigma = np.zeros_like(sample, dtype=float)
                k = min(sample.shape[0], sample.shape[1])
                sigma[:k, :k] = np.diag(sigma_prime[:k])

            return u[:, :P]

        # ─── Weighted V update (Proposition 1') ───
        def V_weighted(x, w, U_mat, V_mat_user, P_n, MODE, S):
            """Weighted local projection update for one user, one mode."""
            u1, u2, u3 = U_mat
            first = min((MODE + 1) % 3, (MODE + 2) % 3)
            second = max((MODE + 1) % 3, (MODE + 2) % 3)
            v1, v2 = V_mat_user[first], V_mat_user[second]

            if MODE == 0:
                c = np.kron(v1 @ u2, v2 @ u3)
            elif MODE == 1:
                c = np.kron(v1 @ u1, v2 @ u3)
            else:
                c = np.kron(v1 @ u1, v2 @ u2)

            # Weighted scatter: sum_j w_j * X_{j(n)}^Phi * X_{j(n)}^{Phi T}
            unfolded_x = np.array([(unfold(x[i], mode=MODE) @ c).T
                                   for i in range(x.shape[0])])
            # unfolded_x shape: (J, P_other, I_n^m) — apply weights along axis 0
            scatter = np.einsum("n,nij,nik->jk", w, unfolded_x, unfolded_x)

            eigenvalues, u = np.linalg.eig(scatter)
            u = np.real_if_close(u, tol=1)
            eigenvalues = np.real_if_close(eigenvalues, tol=1)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            u = u[:, sorted_indices][:, :P_n]

            # Recover V from composite C = V @ U via V = C @ U^T @ (U @ U^T)^{-1}
            if MODE == 0:
                v = u @ u1.T @ np.linalg.inv(u1 @ u1.T + self.lam * np.eye(S))
            elif MODE == 1:
                v = u @ u2.T @ np.linalg.inv(u2 @ u2.T + self.lam * np.eye(S))
            else:
                v = u @ u3.T @ np.linalg.inv(u3 @ u3.T + self.lam * np.eye(S))

            return v

        # ─── U update via incremental SVD ───
        def U_update(projected_data, weights_list, U_mat, MODE, P, use_weights):
            """
            Global projection update via incremental SVD.
            If use_weights=True, each sample's column block is scaled by sqrt(w_j^m)
            (Proposition 2'). If False, uses standard unweighted incremental SVD
            (same as Chapter 2) — V weighting alone catches contaminated samples.
            """
            first = min((MODE + 1) % 3, (MODE + 2) % 3)
            second = max((MODE + 1) % 3, (MODE + 2) % 3)
            u1, u3 = U_mat[first], U_mat[second]
            kron_u = np.kron(u1, u3)

            all_samples = []
            for m, (user_proj, w_m) in enumerate(zip(projected_data, weights_list)):
                for j in range(user_proj.shape[0]):
                    sample = unfold(user_proj[j], mode=MODE) @ kron_u
                    if use_weights:
                        sample = np.sqrt(w_m[j]) * sample
                    all_samples.append(sample)

            I1 = all_samples[0].shape[0]
            X = all_samples[0]
            u, s, _ = np.linalg.svd(X, full_matrices=True)
            sigma = np.zeros_like(X, dtype=float)
            k = min(X.shape[0], X.shape[1])
            sigma[:k, :k] = np.diag(s[:k])

            for sample in all_samples[1:]:
                W = sample - u @ u.T @ sample
                norm_W = np.linalg.norm(W, axis=0)
                norm_W[norm_W < 1e-10] = 1e-10
                norm_w = W / norm_W
                M_mat = np.block([
                    [sigma, u.T @ sample],
                    [np.zeros((sample.shape[1], sigma.shape[1])),
                     np.diag(norm_W ** 2)]
                ])
                u_prime, sigma_prime, _ = np.linalg.svd(M_mat, full_matrices=True)
                u = (np.hstack((u, norm_w)) @ u_prime)[:, :I1]
                sigma = np.zeros_like(sample, dtype=float)
                k = min(sample.shape[0], sample.shape[1])
                sigma[:k, :k] = np.diag(sigma_prime[:k])

            return u[:, :P]

        # ─── Objective function ───
        def phi_weighted(projected_data, U_mat, weights_list):
            phi = 0.0
            for user_proj, w_m in zip(projected_data, weights_list):
                proj = self.projection(user_proj, U_mat)
                for j in range(proj.shape[0]):
                    phi += w_m[j] * np.sum(proj[j] ** 2)
            return phi

        # ═══ Main training loop ═══

        if init_V is not None and init_U is not None:
            V_mat = [v_list[:] for v_list in init_V]
            U_mat = [u.copy() for u in init_U]
            projected = [self.projection(user_data[i], V_mat[i]) for i in range(M)]
        else:
            V_mat = [None] * M
            for i in range(M):
                V_mat[i] = [None] * 3
                for n in range(3):
                    V_mat[i][n] = Vinitial(user_data[i], n, self.I[n])
            projected = [self.projection(user_data[i], V_mat[i]) for i in range(M)]
            U_mat = [None] * 3
            for n in range(3):
                U_mat[n] = Uinitial(projected, n, self.P[n])

        # Main iterations (weighted)
        for iteration in range(self.iterations):
            # Update U
            for n in range(3):
                U_mat[n] = U_update(projected, weights_per_user, U_mat, n, self.P[n],
                                    use_weights=self.weight_U)

            # Update V (weighted) for each user
            for i in range(M):
                for n in range(3):
                    V_mat[i][n] = V_weighted(
                        user_data[i], weights_per_user[i],
                        U_mat, V_mat[i], self.P[n], n, self.I[n]
                    )

            # Re-project
            projected = [self.projection(user_data[i], V_mat[i]) for i in range(M)]

        # Compact output
        prime = []
        for i in range(M):
            proj = self.projection(projected[i], U_mat)
            prime.extend(proj)
        prime = np.array(prime)

        self.U_mat = U_mat
        self.V_mat = V_mat
        return prime, U_mat, V_mat


# ─── RFTL-S: Trimmed Re-fit ──────────────────────────────────────────────────

class RFTL_S:
    """
    Robust Federated Tensor Learning — Sample-Weighted (Algorithm 5).

    Two-step approach:
      1. Fit standard MPCA_FD (unweighted, fully converged)
      2. Compute reconstruction residuals to identify outliers
      3. Hard-threshold: samples with r_j > c * median(r) are excluded
      4. Cold re-fit: remove flagged samples, run fresh MPCA_FD from scratch

    Cold re-fit avoids anchoring to the contaminated subspace that a
    warm-started approach would inherit from the initial fit.
    """

    def __init__(self, I, P, threshold_c=3.0, inner_iterations=200):
        self.I = I
        self.P = P
        self.threshold_c = threshold_c
        self.inner_iterations = inner_iterations

    def fit(self, user_data):
        from my_mpca_02_27_nomean import MPCA_FD
        M = len(user_data)

        # Step 1: unweighted fit on full data
        mpca_init = MPCA_FD(self.I, self.P)
        mpca_init.iterations = self.inner_iterations
        mpca_init.train(
            copy.deepcopy(user_data[0]),
            copy.deepcopy(user_data[1]),
            copy.deepcopy(user_data[2])
        )
        U_mat = [u.copy() for u in mpca_init.U_mat]
        V_mat = [[v.copy() for v in mpca_init.V_mat[m]] for m in range(M)]

        # Step 2: compute residuals and identify outliers
        residuals = [reconstruction_residual(user_data[m], V_mat[m], U_mat)
                     for m in range(M)]
        all_residuals = np.concatenate(residuals)
        median_r = np.median(all_residuals)
        threshold = self.threshold_c * median_r

        weights = [np.where(residuals[m] <= threshold, 1.0, 0.0) for m in range(M)]
        n_flagged = sum(np.sum(w == 0) for w in weights)

        # Step 3: cold re-fit — remove outliers, fresh MPCA_FD from scratch
        if n_flagged > 0:
            users_trimmed = [user_data[m][weights[m] == 1.0] for m in range(M)]
            mpca_trimmed = MPCA_FD(self.I, self.P)
            mpca_trimmed.iterations = self.inner_iterations
            mpca_trimmed.train(
                copy.deepcopy(users_trimmed[0]),
                copy.deepcopy(users_trimmed[1]),
                copy.deepcopy(users_trimmed[2])
            )
            U_mat = [u.copy() for u in mpca_trimmed.U_mat]
            V_mat = [[v.copy() for v in mpca_trimmed.V_mat[m]] for m in range(M)]

        self.U_mat = U_mat
        self.V_mat = V_mat
        self.weights = weights
        self.residuals = residuals
        self.threshold = threshold
        self.median_r = median_r
        self.n_flagged = n_flagged

        return self

    def predict_features(self, test_data):
        """
        Project test data through learned V_mat and U_mat.

        Args:
            test_data: list of M arrays for test samples per user

        Returns:
            features: array of projected low-dim features
        """
        M = len(test_data)
        features = []
        for m in range(M):
            proj = multi_mode_dot(test_data[m], [v.T for v in self.V_mat[m]], modes=[1, 2, 3])
            proj = multi_mode_dot(proj, [u.T for u in self.U_mat], modes=[1, 2, 3])
            features.extend(proj)
        return np.array(features)


# ─── Experiment runner (parallelized across repeats) ──────────────────────────

PI_S_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
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
    """Worker function for one repeat.

    For each (noise_multiplier, pi_S) pair:
      1. Inject contamination at the given noise level
      2. Run baseline MPCA_FD on dirty data (200 iters)
      3. For each threshold c: identify outliers, remove them, cold re-fit
         with fresh MPCA_FD (200 iters)
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

    # pi_s=0: no contamination, noise irrelevant — run once
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

    for noise_mult in NOISE_MULTIPLIERS:
        for pi_s in PI_S_LEVELS:
            if pi_s == 0.0:
                continue

            contam_rng = np.random.RandomState(rep_seed + int(pi_s * 1000))

            contaminated_indices = []
            users_dirty = []
            for m in range(3):
                n_samples = users_centered[m].shape[0]
                n_contam = int(np.ceil(pi_s * n_samples))
                indices = (contam_rng.choice(n_samples, size=n_contam, replace=False)
                           if n_contam > 0 else np.array([], dtype=int))
                contaminated_indices.append(set(indices))

                dirty = users_centered[m].copy()
                if n_contam > 0:
                    noise_std = noise_mult * np.std(users_centered[m])
                    for idx in indices:
                        dirty[idx] += contam_rng.randn(*dirty[idx].shape) * noise_std
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
            rep_results[('baseline', noise_mult, pi_s)] = angle_base

            residuals = [reconstruction_residual(users_dirty[m], V_unw[m], U_unw)
                         for m in range(3)]
            all_residuals = np.concatenate(residuals)
            median_r = np.median(all_residuals)

            for label, threshold_c in CONFIGS:
                threshold = threshold_c * median_r
                weights = [np.where(residuals[m] <= threshold, 1.0, 0.0)
                           for m in range(3)]
                n_flagged = sum(np.sum(w == 0) for w in weights)

                if n_flagged > 0:
                    users_trimmed = [users_dirty[m][weights[m] == 1.0]
                                     for m in range(3)]
                    mpca_trimmed = MPCA_FD(I_COMMON, RANK)
                    mpca_trimmed.iterations = 200
                    mpca_trimmed.train(
                        copy.deepcopy(users_trimmed[0]),
                        copy.deepcopy(users_trimmed[1]),
                        copy.deepcopy(users_trimmed[2])
                    )
                    U_result = [u.copy() for u in mpca_trimmed.U_mat]
                else:
                    U_result = U_unw

                angle_trim = np.mean([a for n in range(3)
                                      for a in principal_angles_deg(U_star[n],
                                                                    U_result[n])])
                rep_results[(label, noise_mult, pi_s)] = angle_trim

                tp, fp, fn = 0, 0, 0
                for m in range(3):
                    flagged = set(np.where(weights[m] == 0.0)[0])
                    true_contam = contaminated_indices[m]
                    tp += len(flagged & true_contam)
                    fp += len(flagged - true_contam)
                    fn += len(true_contam - flagged)
                rep_results[(f'{label}_prec', noise_mult, pi_s)] = (
                    tp / (tp + fp) if (tp + fp) > 0 else 0.0)
                rep_results[(f'{label}_rec', noise_mult, pi_s)] = (
                    tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    print(f"  Repeat {rep_idx + 1} done (seed={rep_seed}).", flush=True)
    return rep_results


def run_rftl_s_experiment(data_path, n_repeats=10, max_files=350,
                          n_workers=4, seed=2024):
    import multiprocessing
    from motivation_pilot import load_simulated_data

    global _GLOBAL_DATA

    print("Loading simulated data...", flush=True)
    min_needed = sum(SIZES)
    _GLOBAL_DATA = load_simulated_data(data_path, max_files=max(max_files, min_needed + 10))
    n_total = len(_GLOBAL_DATA)
    print(f"  Loaded {n_total} samples", flush=True)
    if n_total < min_needed:
        print(f"  ERROR: Need at least {min_needed} samples, got {n_total}", flush=True)
        sys.exit(1)

    master_rng = np.random.RandomState(seed)
    rep_seeds = [int(master_rng.randint(1, 100000)) for _ in range(n_repeats)]
    worker_args = [(i, s) for i, s in enumerate(rep_seeds)]

    print(f"Running {n_repeats} repeats across {n_workers} workers...", flush=True)
    print(f"Configs: {[c[0] for c in CONFIGS]}, Noise: {NOISE_MULTIPLIERS}x",
          flush=True)

    if n_workers > 1:
        pool = multiprocessing.Pool(processes=n_workers)
        all_rep_results = pool.map(_run_single_repeat, worker_args)
        pool.close()
        pool.join()
    else:
        all_rep_results = [_run_single_repeat(a) for a in worker_args]

    results = {}
    for rep in all_rep_results:
        for key, val in rep.items():
            results.setdefault(key, []).append(val)

    return results


def report_rftl_results(results):
    config_labels = [c[0] for c in CONFIGS]

    print("\n" + "=" * 90, flush=True)
    print("RFTL-S EXPERIMENT RESULTS — COLD RE-FIT", flush=True)
    print("=" * 90, flush=True)

    clean_key = ('baseline', 0, 0.0)
    if clean_key in results:
        print(f"\nClean data (pi_S=0.00): baseline = "
              f"{np.mean(results[clean_key]):.2f} deg", flush=True)

    for noise_mult in NOISE_MULTIPLIERS:
        print(f"\n--- Noise: {noise_mult}x baseline std ---", flush=True)

        header = f"  {'pi_S':>5} | {'Baseline':>8}"
        for label in config_labels:
            header += f" | {label:>14}"
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)

        for pi_s in PI_S_LEVELS:
            if pi_s == 0.0:
                continue
            b_key = ('baseline', noise_mult, pi_s)
            b_mean = np.mean(results.get(b_key, [0]))
            row = f"  {pi_s:>5.2f} | {b_mean:>8.2f}"
            for label in config_labels:
                r_key = (label, noise_mult, pi_s)
                r_mean = np.mean(results.get(r_key, [0]))
                improvement = b_mean - r_mean
                row += f" | {r_mean:>6.2f} ({improvement:>+5.1f})"
            print(row, flush=True)

        print(f"\n  Detection (noise={noise_mult}x):", flush=True)
        det_hdr = f"  {'pi_S':>5}"
        for label in config_labels:
            det_hdr += f" |  {label} P  {label} R"
        print(det_hdr, flush=True)
        print("  " + "-" * (len(det_hdr) - 2), flush=True)

        for pi_s in PI_S_LEVELS:
            if pi_s == 0.0:
                continue
            row = f"  {pi_s:>5.2f}"
            for label in config_labels:
                p_key = (f'{label}_prec', noise_mult, pi_s)
                r_key = (f'{label}_rec', noise_mult, pi_s)
                p = np.mean(results.get(p_key, [0]))
                r = np.mean(results.get(r_key, [0]))
                row += f" | {p:>6.3f} {r:>6.3f}"
            print(row, flush=True)

    print("\n" + "=" * 90, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RFTL-S Experiment')
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--n-repeats', type=int, default=10)
    parser.add_argument('--max-files', type=int, default=350)
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    args = parser.parse_args()

    if args.data_path:
        data_path = args.data_path
    else:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data', 'Simulated Data')

    if not os.path.exists(data_path):
        print(f"ERROR: Data path not found: {data_path}", flush=True)
        sys.exit(1)

    results = run_rftl_s_experiment(
        data_path, n_repeats=args.n_repeats, max_files=args.max_files,
        n_workers=args.n_workers
    )
    report_rftl_results(results)
