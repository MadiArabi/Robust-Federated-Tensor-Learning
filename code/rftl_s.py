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
    - U update scales columns by sqrt(w_j) before incremental SVD
    """

    def __init__(self, I, P, iterations=30):
        self.I = I
        self.P = P
        self.iterations = iterations
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

        # ─── Weighted U update (Proposition 2') ───
        def U_weighted(projected_data, weights_list, U_mat, MODE, P):
            """
            Weighted global projection update via incremental SVD.
            Each sample's column block is scaled by sqrt(w_j^m).
            """
            first = min((MODE + 1) % 3, (MODE + 2) % 3)
            second = max((MODE + 1) % 3, (MODE + 2) % 3)
            u1, u3 = U_mat[first], U_mat[second]
            kron_u = np.kron(u1, u3)

            # Build list of weighted column blocks across all users
            all_samples = []
            for m, (user_proj, w_m) in enumerate(zip(projected_data, weights_list)):
                for j in range(user_proj.shape[0]):
                    # Scale by sqrt(w_j^m) — Proposition 2'
                    scaled = np.sqrt(w_m[j]) * unfold(user_proj[j], mode=MODE) @ kron_u
                    all_samples.append(scaled)

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
            # Update U (weighted)
            for n in range(3):
                U_mat[n] = U_weighted(projected, weights_per_user, U_mat, n, self.P[n])

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


# ─── RFTL-S: Full IRLS Algorithm ─────────────────────────────────────────────

class RFTL_S:
    """
    Robust Federated Tensor Learning — Sample-Weighted (Algorithm 5).

    Wraps MPCA_FD_Weighted in an outer IRLS loop that updates per-sample
    weights based on reconstruction residuals and Huber-like reweighting.
    """

    def __init__(self, I, P, irls_iterations=10, inner_iterations=30,
                 huber_k=3.0, tol=1e-4):
        """
        Args:
            I: list of 3 intermediate dimensions [I1, I2, I3]
            P: list of 3 target dimensions [P1, P2, P3]
            irls_iterations: max outer IRLS iterations
            inner_iterations: iterations per MPCA_FD_Weighted call
            huber_k: Huber tuning constant (default: 3.0, only downweights
                     samples >3 MADs from median — conservative for tensor setting)
            tol: convergence tolerance for weighted scatter
        """
        self.I = I
        self.P = P
        self.irls_iterations = irls_iterations
        self.inner_iterations = inner_iterations
        self.huber_k = huber_k
        self.tol = tol

    def fit(self, user_data):
        """
        Fit RFTL-S model.

        Args:
            user_data: list of M arrays, each (J_m, I1_m, I2_m, I3_m),
                      mean-centered tensor data per user.

        Returns:
            self (fitted model with U_mat, V_mat, weights, residuals, history)
        """
        M = len(user_data)

        self.history = {
            'mad': [],
            'mean_weight': [],
            'n_downweighted': []
        }

        # Step 0: unweighted initialization (equivalent to Chapter 2)
        weights = [np.ones(user_data[m].shape[0]) for m in range(M)]
        model = MPCA_FD_Weighted(self.I, self.P, iterations=self.inner_iterations)
        prime, U_mat, V_mat = model.train(
            [copy.deepcopy(d) for d in user_data],
            weights
        )

        # Compute initial residuals and weights from the unweighted fit
        residuals = [reconstruction_residual(user_data[m], V_mat[m], U_mat)
                     for m in range(M)]
        all_residuals = np.concatenate(residuals)
        median_r = np.median(all_residuals)
        mad = federated_mad(residuals)
        weights = [huber_weights(residuals[m], median_r, mad, k=self.huber_k)
                   for m in range(M)]

        # IRLS iterations with warm-starting
        for irls_iter in range(self.irls_iterations):
            model = MPCA_FD_Weighted(self.I, self.P, iterations=self.inner_iterations)
            prime, U_mat, V_mat = model.train(
                [copy.deepcopy(d) for d in user_data],
                [w.copy() for w in weights],
                init_V=[v[:] for v in V_mat],
                init_U=[u.copy() for u in U_mat]
            )

            residuals = [reconstruction_residual(user_data[m], V_mat[m], U_mat)
                         for m in range(M)]
            all_residuals = np.concatenate(residuals)
            median_r = np.median(all_residuals)
            mad = federated_mad(residuals)

            new_weights = [huber_weights(residuals[m], median_r, mad, k=self.huber_k)
                           for m in range(M)]

            all_w = np.concatenate(new_weights)
            self.history['mad'].append(mad)
            self.history['mean_weight'].append(np.mean(all_w))
            self.history['n_downweighted'].append(np.sum(all_w < 0.99))

            weight_change = max(
                np.max(np.abs(new_weights[m] - weights[m])) for m in range(M)
            )
            weights = new_weights

            if weight_change < self.tol:
                break

        self.U_mat = U_mat
        self.V_mat = V_mat
        self.weights = weights
        self.residuals = residuals
        self.prime = prime

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
SIZES = [70, 100, 130]
I_COMMON = [21, 21, 10]
RANK = [5, 5, 4]

# Global data array set by the main process before forking workers
_GLOBAL_DATA = None


def _run_single_repeat(args):
    """Worker function for one repeat. Designed for multiprocessing.Pool.map."""
    rep_idx, rep_seed = args
    from motivation_pilot import setup_users, principal_angles_deg
    from my_mpca_02_27_nomean import MPCA_FD

    data = _GLOBAL_DATA
    n_total = len(data)

    rng = np.random.RandomState(rep_seed)
    sample = np.arange(n_total)
    rng.shuffle(sample)

    user1, user2, user3 = setup_users(data, sample, SIZES, seed=rep_seed)
    users_centered = [u - np.mean(u, axis=0) for u in [user1, user2, user3]]

    # Clean reference
    mpca_clean = MPCA_FD(I_COMMON, RANK)
    mpca_clean.iterations = 30
    mpca_clean.train(
        copy.deepcopy(users_centered[0]),
        copy.deepcopy(users_centered[1]),
        copy.deepcopy(users_centered[2])
    )
    U_star = [u.copy() for u in mpca_clean.U_mat]

    rep_results = {
        'baseline': {},
        'rftl_s': {},
        'precision': {},
        'recall': {},
    }

    for pi_s in PI_S_LEVELS:
        contam_rng = np.random.RandomState(rep_seed + int(pi_s * 1000))

        contaminated_indices = []
        users_dirty = []
        for m in range(3):
            n_samples = users_centered[m].shape[0]
            n_contam = int(np.ceil(pi_s * n_samples))
            indices = contam_rng.choice(n_samples, size=n_contam, replace=False) if n_contam > 0 else np.array([], dtype=int)
            contaminated_indices.append(set(indices))

            dirty = users_centered[m].copy()
            if n_contam > 0:
                noise_std = 10.0 * np.std(users_centered[m])
                for idx in indices:
                    dirty[idx] += contam_rng.randn(*dirty[idx].shape) * noise_std
            users_dirty.append(dirty)

        # Baseline (Chapter 2, no robustness)
        mpca_baseline = MPCA_FD(I_COMMON, RANK)
        mpca_baseline.iterations = 30
        mpca_baseline.train(
            copy.deepcopy(users_dirty[0]),
            copy.deepcopy(users_dirty[1]),
            copy.deepcopy(users_dirty[2])
        )
        # Mean principal angle across all modes (more stable than max)
        all_angles_baseline = []
        for n in range(3):
            angles = principal_angles_deg(U_star[n], mpca_baseline.U_mat[n])
            all_angles_baseline.extend(angles)
        rep_results['baseline'][pi_s] = np.mean(all_angles_baseline)

        # RFTL-S
        rftl = RFTL_S(I_COMMON, RANK, irls_iterations=8, inner_iterations=30)
        rftl.fit(users_dirty)

        all_angles_rftl = []
        for n in range(3):
            angles = principal_angles_deg(U_star[n], rftl.U_mat[n])
            all_angles_rftl.extend(angles)
        rep_results['rftl_s'][pi_s] = np.mean(all_angles_rftl)

        # Precision/recall
        if pi_s > 0:
            tp, fp, fn = 0, 0, 0
            for m in range(3):
                flagged = set(np.where(rftl.weights[m] < 0.5)[0])
                true_contam = contaminated_indices[m]
                tp += len(flagged & true_contam)
                fp += len(flagged - true_contam)
                fn += len(true_contam - flagged)
            rep_results['precision'][pi_s] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rep_results['recall'][pi_s] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"  Repeat {rep_idx + 1} done (seed={rep_seed}).", flush=True)
    return rep_results


def run_rftl_s_experiment(data_path, n_repeats=10, max_files=350,
                          n_workers=4, seed=2024):
    """
    Run RFTL-S vs. Chapter 2 baseline under sample contamination.
    Parallelized across repeats using multiprocessing.
    """
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

    if n_workers > 1:
        pool = multiprocessing.Pool(processes=n_workers)
        all_rep_results = pool.map(_run_single_repeat, worker_args)
        pool.close()
        pool.join()
    else:
        all_rep_results = [_run_single_repeat(a) for a in worker_args]

    # Aggregate results
    results = {
        'baseline': {pi_s: [] for pi_s in PI_S_LEVELS},
        'rftl_s': {pi_s: [] for pi_s in PI_S_LEVELS},
        'precision': {pi_s: [] for pi_s in PI_S_LEVELS},
        'recall': {pi_s: [] for pi_s in PI_S_LEVELS},
    }
    for rep in all_rep_results:
        for pi_s in PI_S_LEVELS:
            results['baseline'][pi_s].append(rep['baseline'][pi_s])
            results['rftl_s'][pi_s].append(rep['rftl_s'][pi_s])
            if pi_s > 0 and pi_s in rep['precision']:
                results['precision'][pi_s].append(rep['precision'][pi_s])
                results['recall'][pi_s].append(rep['recall'][pi_s])

    return PI_S_LEVELS, results


def report_rftl_results(pi_s_levels, results):
    print("\n" + "=" * 70, flush=True)
    print("RFTL-S EXPERIMENT RESULTS", flush=True)
    print("=" * 70, flush=True)

    print("\nMean principal angle (degrees) -- lower is better:", flush=True)
    print(f"{'pi_S':>6} | {'Baseline':>12} | {'RFTL-S':>12} | {'Improvement':>12}", flush=True)
    print("-" * 70, flush=True)

    for pi_s in pi_s_levels:
        b_mean = np.mean(results['baseline'][pi_s])
        r_mean = np.mean(results['rftl_s'][pi_s])
        improvement = b_mean - r_mean
        print(f"{pi_s:>6.2f} | {b_mean:>10.2f}  | {r_mean:>10.2f}  | {improvement:>+10.2f} ",
              flush=True)

    print("\nContamination detection (w < 0.5 threshold):", flush=True)
    print(f"{'pi_S':>6} | {'Precision':>10} | {'Recall':>10}", flush=True)
    print("-" * 40, flush=True)
    for pi_s in pi_s_levels:
        if pi_s > 0 and results['precision'][pi_s]:
            p = np.mean(results['precision'][pi_s])
            r = np.mean(results['recall'][pi_s])
            print(f"{pi_s:>6.2f} | {p:>10.3f} | {r:>10.3f}", flush=True)

    print("=" * 70, flush=True)


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

    pi_s_levels, results = run_rftl_s_experiment(
        data_path, n_repeats=args.n_repeats, max_files=args.max_files,
        n_workers=args.n_workers
    )
    report_rftl_results(pi_s_levels, results)
